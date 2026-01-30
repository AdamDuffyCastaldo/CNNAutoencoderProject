# SAR Codec Docker Image
# Multi-stage build for minimal production image
#
# Usage:
#   docker build -t sarcodec .
#   docker run sarcodec --help
#   docker run -p 8000:8000 --gpus all sarcodec --serve

# Stage 1: Builder (includes dev tools for pip compilation)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Stage 2: Runtime (minimal image)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    libgdal30 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY scripts/docker-entrypoint.sh /entrypoint.sh

# Copy models (if bundled) - optional, can mount at runtime
# COPY models/ ./models/

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Default environment
ENV SARCODEC_CHECKPOINT_DIR=/app/models
ENV SARCODEC_HOST=0.0.0.0
ENV SARCODEC_PORT=8000

# Expose API port
EXPOSE 8000

# Entrypoint handles CLI vs API mode
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
