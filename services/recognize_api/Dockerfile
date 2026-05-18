# syntax=docker/dockerfile:1.6
#
# Two-stage build for the Phase 6.1 recognize() FastAPI service.
#
# Stage 1 (builder) builds a virtualenv with torch (CPU wheel),
# open_clip, FastAPI and the project itself (for the
# car_lense_engine.* imports the app uses). Stage 2 (runtime) is a slim
# image with only the venv and the app code.
#
# Image base: python:3.11-slim (Debian). NOT Alpine -- torch wheels are
# manylinux + don't currently ship for musl.
#
# GPU note: the default build is CPU-only. To run on a CUDA host, build
# with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121`
# and run with `--gpus all -e DEVICE=cuda`.

ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

#####################################################################
# Stage 1 -- builder
#####################################################################
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_INDEX
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install service-level Python deps. We keep this list narrow (no
# datasets / playwright / sqlite-utils / etc.) because the recognize
# service only needs the inference + HTTP slice.
COPY requirements.txt /build/requirements.txt
RUN pip install --upgrade pip \
 && pip install --extra-index-url "${TORCH_INDEX}" -r /build/requirements.txt

# Install the local car_lense_engine package -- the app imports a few
# helpers from it (canonical label rendering, baseline checkpoint
# loader). We copy only what's needed for the install, not the full
# repo, so build context churn doesn't invalidate this layer.
COPY pyproject.toml /build/pkg/pyproject.toml
COPY src /build/pkg/src
RUN pip install --no-deps /build/pkg


#####################################################################
# Stage 2 -- runtime
#####################################################################
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libjpeg62-turbo libpng16-16 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app.py /app/app.py
COPY build_prototypes.py /app/build_prototypes.py

# Mountpoints for the prototypes, model checkpoint, and DB (RO). The
# Compose file binds the host's ./cache, ./models/checkpoints, ./db
# into these paths.
RUN mkdir -p /app/cache /app/models /app/db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fs http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
