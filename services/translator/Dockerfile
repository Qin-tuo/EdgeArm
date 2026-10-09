# Multi-stage build for NLLB-200 translator service with CTranslate2
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build stage: install dependencies
FROM base AS builder

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
RUN pip install --upgrade pip wheel && \
    pip install -e .

# Runtime stage
FROM base AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    TRANSLATOR_DEVICE=cuda \
    TRANSLATOR_DEVICE_INDEX=0

COPY --from=builder /opt/venv /opt/venv
COPY server.py /app/server.py

# Models mounted at /models (host volume)
# Expected structure: /models/nllb-200-distilled-600m-ct2-int8/
VOLUME /models

EXPOSE 9001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:9001/health').read()" || exit 1

CMD ["python", "/app/server.py"]
