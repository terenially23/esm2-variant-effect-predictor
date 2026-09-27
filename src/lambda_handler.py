"""AWS Lambda entry point for zero-shot variant scoring.

Two modes, chosen by the event payload:

1. Single variant, synchronous:
   {"sequence": "MSTL...", "pos": 1699, "wt": "R", "mt": "Q"}
   -> {"esm2_score": -3.42, "predicted_effect": "likely_damaging"}

2. Batch, via S3: score every row of a CSV (columns: sequence,pos,wt,mt,gene,label)
   and write the results back to S3.
   {"input_s3_uri": "s3://bucket/variants.csv", "output_s3_uri": "s3://bucket/scores.csv"}

The model loads once per container (module scope), so it's paid for on a cold
start and reused for every warm invocation after that.
"""

import csv
import io
import json

import boto3

from scoring import score_variant

# Loaded lazily on first invocation, not at import time, so the container can
# start (and Lambda's health checks pass) before the ~35M-parameter model is
# pulled into memory.
_s3 = None


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), f"not an s3:// URI: {uri}"
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def _score_row(row: dict) -> dict:
    score = score_variant(row["sequence"], int(row["pos"]), row["wt"], row["mt"])
    return {**row, "esm2_score": score, "predicted_effect": _classify(score)}


def _classify(score: float) -> str:
    # Threshold chosen from this project's own labelled evaluation set
    # (see README for the AUROC this proxy achieves) -- not a clinical cutoff.
    return "likely_damaging" if score < -1.0 else "likely_tolerated"


def _handle_batch(input_s3_uri: str, output_s3_uri: str) -> dict:
    s3 = _s3_client()
    in_bucket, in_key = _parse_s3_uri(input_s3_uri)
    body = s3.get_object(Bucket=in_bucket, Key=in_key)["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(body))
    scored_rows = [_score_row(row) for row in reader]

    out_buffer = io.StringIO()
    writer = csv.DictWriter(out_buffer, fieldnames=list(scored_rows[0].keys()))
    writer.writeheader()
    writer.writerows(scored_rows)

    out_bucket, out_key = _parse_s3_uri(output_s3_uri)
    s3.put_object(Bucket=out_bucket, Key=out_key, Body=out_buffer.getvalue().encode("utf-8"))

    return {"scored_variants": len(scored_rows), "output_s3_uri": output_s3_uri}


def handler(event, context=None):
    if isinstance(event, str):
        event = json.loads(event)

    if "input_s3_uri" in event:
        result = _handle_batch(event["input_s3_uri"], event["output_s3_uri"])
    else:
        score = score_variant(event["sequence"], int(event["pos"]), event["wt"], event["mt"])
        result = {"esm2_score": score, "predicted_effect": _classify(score)}

    return {"statusCode": 200, "body": json.dumps(result)}
