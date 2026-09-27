FROM python:3.11-slim

# AWS's official Lambda base image pulls through an ECR-Public-backed CloudFront
# CDN whose hostname isn't stable in some network environments, so this follows
# AWS's documented alternative for a non-AWS base image: install the Lambda
# Runtime Interface Client directly.
# https://docs.aws.amazon.com/lambda/latest/dg/images-create.html#images-create-from-alt
RUN pip install --no-cache-dir awslambdaric

WORKDIR /var/task

COPY requirements.txt .
# PyPI's Linux "torch" wheel links its native lib against libcudart at load
# time even for CPU-only inference, so the NVIDIA/CUDA packages it depends on
# (~3GB) can't be skipped without breaking `import torch`. Lambda's image
# limit is 10GB, so this is accepted rather than worked around.
RUN pip install --no-cache-dir -r requirements.txt

COPY src/scoring.py src/lambda_handler.py ./

# Lambda's filesystem is read-only outside /tmp, but huggingface_hub defaults
# its cache to a path under $HOME -- redirect it so model downloads succeed.
ENV HF_HOME=/tmp/huggingface

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_handler.handler"]
