# The tag matches the torch/CUDA pin in pixi.toml. Bump both in the same commit
# or the container links against a different CUDA runtime than the environment
# every number in RESULTS.md was measured in.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg \
    # Keep the InceptionV3 weights inside the image rather than re-fetching them
    # on every container start.
    TORCH_HOME=/app/.torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# torch and torchvision ship in the base image; reinstalling pulls a second copy
# and a second CUDA runtime with it.
RUN grep -vE '^(torch|torchvision)' /app/requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests
COPY pyproject.toml /app/pyproject.toml

# Bake the FID feature extractor in, so a container with no network can still
# score a checkpoint.
RUN python -c "from torchvision.models import inception_v3, Inception_V3_Weights; \
               inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)"

# Proves the image works without needing the dataset mounted.
RUN python -m pgan.smoke --steps 8 --batch-size 4 --device cpu

# The dataset is not in the image. Mount it:
#   docker run --gpus all -v $PWD/data:/app/data -v $PWD/checkpoints:/app/checkpoints <image>
CMD ["python", "scripts/run_ablation.py"]
