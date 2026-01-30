# SAR Codec Docker Image
# Uses official PyTorch image to avoid CUDA dependency issues
#
# Usage:
#   docker build -t sarcodec .
#   docker run -p 8000:8000 --gpus all sarcodec --serve

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch already installed in base image)
WORKDIR /app
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY scripts/docker-entrypoint.sh /entrypoint.sh

# Fix Windows line endings and make executable
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Default environment
ENV SARCODEC_CHECKPOINT_DIR=/app/models
ENV SARCODEC_HOST=0.0.0.0
ENV SARCODEC_PORT=8000

# Expose API port
EXPOSE 8000

# Entrypoint handles CLI vs API mode
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
