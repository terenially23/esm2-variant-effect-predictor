# ESM2 Variant Effect Predictor

Zero-shot prediction of whether a missense mutation is pathogenic or benign,
using [ESM2](https://huggingface.co/facebook/esm2_t12_35M_UR50D) — a protein
language model that has never seen a disease label — deployed as a Docker
container on AWS Lambda.

**Result:** AUROC 0.784 separating 176 pathogenic from 94 benign real ClinVar
missense variants across five genes (BRCA1, TP53, CFTR, MLH1, PTEN), with no
fine-tuning at all.

**[See the results page](https://claude.ai/artifact/Bs97MfZqyBFwpGY7DgJXBT)**
for the method, the score distribution, individual examples (including the
ones it gets wrong), the deployment pipeline, and the two real AWS bugs hit
along the way.

## How it works

For each variant, the wild-type residue is masked out of the full protein
sequence and ESM2 predicts a distribution over all 20 amino acids at that
position — the same task it was pretrained on, just never on this data:

```
score = log P(mutant | context) − log P(wild-type | context)
```

A more negative score means the model finds the mutation less plausible given
the rest of the sequence — the working definition of "likely damaging" used
here. This is a zero-shot proxy for pathogenicity, not a validated clinical
predictor.

## Project layout

```
scripts/fetch_variants.py   Pulls real missense variants from ClinVar (NCBI
                             eutils) and canonical sequences from UniProt,
                             writes data/variants.csv
src/scoring.py               Shared ESM2 masked-marginal scoring logic
src/score_variants.py        CLI: scores data/variants.csv, reports AUROC
src/lambda_handler.py        AWS Lambda entry point (single variant or S3
                             batch mode), imports src/scoring.py
Dockerfile                   Container image for AWS Lambda
.github/workflows/deploy.yml CI/CD: build, smoke-test, push to ECR, deploy
```

## Running it yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_variants.py          # writes data/variants.csv
python src/score_variants.py              # writes data/variant_scores.csv, prints AUROC
```

## Deployment

Every push to `main` that touches the model code runs a full pipeline
(`.github/workflows/deploy.yml`):

1. **Build** the Docker image (Lambda-compatible, `python:3.11-slim` +
   [`awslambdaric`](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html#images-create-from-alt) —
   AWS's own pattern for a non-AWS base image)
2. **Smoke-test** it with the real [AWS Lambda Runtime Interface
   Emulator](https://github.com/aws/aws-lambda-runtime-interface-emulator),
   POSTing an actual variant and checking for a real score back — the build
   fails if this doesn't pass
3. **Push** to Amazon ECR
4. **Deploy** — auto-updates the live Lambda function

The deployed function is invoked with:

```json
{"sequence": "MDLSALRVEEV...", "pos": 1713, "wt": "V", "mt": "G"}
```

and returns:

```json
{"statusCode": 200, "body": "{\"esm2_score\": -5.53, \"predicted_effect\": \"likely_damaging\"}"}
```

A second event shape scores a whole CSV of variants via S3
(`{"input_s3_uri": "...", "output_s3_uri": "..."}`) — see
`src/lambda_handler.py`.

### Two real production bugs

Both only appeared on actual AWS infrastructure, not in local Docker testing:

- **Cold-start timeout.** Lambda caps a container's *initialization* phase at
  10 seconds before retrying against the function's own timeout. A 9.7GB
  image with torch and its CUDA runtime libraries took longer than that to
  unpack and import cold. Fixed by raising the Lambda timeout — warm
  invocations stay fast.
- **Read-only filesystem.** `OSError: Read-only file system:
  '/home/sbx_user1051'` — Hugging Face's model cache defaults to a path under
  `$HOME`, but Lambda's filesystem is read-only everywhere except `/tmp`.
  Fixed with `ENV HF_HOME=/tmp/huggingface` in the Dockerfile.

## Why the image is ~9.7GB

PyPI's Linux `torch` wheel links its native library against `libcudart` at
import time even for CPU-only inference, so the NVIDIA/CUDA packages it
depends on can't be stripped without breaking `import torch` outright. The
CPU-only wheel from `download.pytorch.org` avoids this, but wasn't reachable
from the environment this was built in. Lambda's image limit is 10GB, so this
was accepted rather than worked around — a real angle for further
optimization if this were headed to production.

## Caveats

This is a portfolio project, not a clinical tool. The `-1.0` classification
threshold is this project's own choice for computing an evaluation accuracy,
not a validated clinical cutoff. Ground truth comes from ClinVar's own
`germline_classification`; not every gene has enough labelled benign
missense variants for a balanced count (CFTR: 31 pathogenic / 3 benign;
PTEN: 36 / 1) — see the results page for the full breakdown.
