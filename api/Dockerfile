# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/translator/.cache/huggingface
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 translator \
    && useradd --uid 10001 --gid translator --create-home translator

COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt
RUN mkdir -p /home/translator/.cache/huggingface \
    && chown -R translator:translator /home/translator/.cache

FROM base AS gpu-runtime
USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip cache purge \
    && python -m pip uninstall -y onnxruntime \
    && python -m pip install --default-timeout=1800 --retries=10 "onnxruntime-gpu[cuda,cudnn]==1.26.0"

FROM gpu-runtime AS gpu
COPY --chown=translator:translator app ./app
COPY --chown=translator:translator docker-entrypoint.py ./docker-entrypoint.py
USER translator
EXPOSE 8000
CMD ["python", "docker-entrypoint.py"]
