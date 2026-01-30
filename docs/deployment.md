# Deployment Guide

Comprehensive guide for deploying SAR Codec in production environments.

## Table of Contents

- [Python Package](#python-package)
- [Docker Deployment](#docker-deployment)
- [ONNX Export](#onnx-export)
- [REST API](#rest-api)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Python Package

### Installation

```bash
# Basic installation (PyTorch inference)
pip install sarcodec

# With GPU support via ONNX Runtime
pip install sarcodec[gpu]

# With REST API server
pip install sarcodec[api]

# Full installation with all features
pip install sarcodec[full]
```

### Basic Usage

```python
from sarcodec import SARCompressor

# Initialize with model checkpoint
compressor = SARCompressor(
    model_path="models/resnet_8x.pth",
    overlap=64,      # Tile overlap in pixels
    batch_size=None  # Auto-detect based on GPU memory
)

# Check device
print(f"Using device: {compressor.device}")
print(f"Batch size: {compressor.batch_size}")
```

### Batch Processing

```python
from pathlib import Path
from sarcodec import SARCompressor
from sarcodec.geotiff import read_geotiff, write_geotiff

compressor = SARCompressor("models/resnet_8x.pth")

# Process all TIFFs in a directory
for tif_path in Path("input/").glob("*.tif"):
    # Read
    data, metadata = read_geotiff(tif_path)

    # Compress
    latent, tile_meta = compressor.compress(data)

    # Save compressed
    import numpy as np
    np.savez_compressed(
        tif_path.with_suffix(".npz"),
        latent=latent,
        tile_metadata=tile_meta
    )

    print(f"Compressed: {tif_path.name}")
```

### Progress Callbacks

```python
def progress_callback(current: int, total: int) -> None:
    percent = (current / total) * 100
    print(f"Progress: {percent:.1f}%")

latent, metadata = compressor.compress(
    data,
    progress_callback=progress_callback
)
```

---

## Docker Deployment

### Building the Image

```bash
# Build the Docker image
docker build -t sarcodec:latest .

# Verify build
docker images sarcodec
```

### Running CLI Commands

```bash
# Show help
docker run sarcodec --help

# Compress a file (mount volumes for data access)
docker run -v $(pwd)/data:/data sarcodec \
    compress /data/input.tif -o /data/output.npz \
    --model /app/models/resnet_8x.pth

# Decompress
docker run -v $(pwd)/data:/data sarcodec \
    decompress /data/output.npz -o /data/reconstructed.tif
```

### Running the API Server

```bash
# Start API with GPU support
docker run -d \
    --name sarcodec-api \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    sarcodec --serve

# Check health
curl http://localhost:8000/health
```

### Using Docker Compose

```yaml
# docker-compose.yml
services:
  sarcodec:
    build: .
    image: sarcodec:latest
    command: ["--serve"]
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: '2g'
```

```bash
# Start the service
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SARCODEC_CHECKPOINT_DIR` | `/app/models` | Directory containing model checkpoints |
| `SARCODEC_HOST` | `0.0.0.0` | API server bind address |
| `SARCODEC_PORT` | `8000` | API server port |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

### CPU-Only Mode

```bash
# Run without GPU
docker compose --profile cpu-only up

# Or with docker run
docker run -d \
    -e CUDA_VISIBLE_DEVICES="" \
    -p 8001:8000 \
    sarcodec --serve
```

---

## ONNX Export

Export models for deployment outside PyTorch (e.g., ONNX Runtime, TensorRT).

### Export Script

```bash
# Export a single model
python scripts/export_onnx.py \
    notebooks/checkpoints/resnet_c32_b64_cr8x_*/best.pth \
    models/onnx/8x/

# Export all ResNet models (4x, 8x, 16x)
python scripts/export_onnx.py --all models/onnx/

# Skip validation for faster export
python scripts/export_onnx.py --no-validate checkpoint.pth output/
```

### Output Structure

```
models/onnx/
  resnet_4x/
    encoder.onnx    # Raw SAR -> latent
    decoder.onnx    # Latent -> raw SAR
    metadata.json   # Model info and checksums
  resnet_8x/
    ...
  resnet_16x/
    ...
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np
import json

# Load metadata
with open("models/onnx/8x/metadata.json") as f:
    meta = json.load(f)

# Initialize sessions
encoder = ort.InferenceSession(
    "models/onnx/8x/encoder.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
decoder = ort.InferenceSession(
    "models/onnx/8x/decoder.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Prepare input (NCHW format, float32)
# Note: Preprocessing is embedded in the encoder
input_data = np.random.rand(1, 1, 256, 256).astype(np.float32) * 300

# Encode
latent = encoder.run(None, {"input": input_data})[0]
print(f"Latent shape: {latent.shape}")  # (1, 32, 16, 16) for 8x

# Decode
reconstructed = decoder.run(None, {"latent": latent})[0]
print(f"Reconstructed shape: {reconstructed.shape}")  # (1, 1, 256, 256)
```

### Verifying Model Integrity

```python
import hashlib
import json

# Load expected checksums
with open("models/onnx/8x/metadata.json") as f:
    meta = json.load(f)

# Verify encoder
with open("models/onnx/8x/encoder.onnx", "rb") as f:
    actual_hash = hashlib.sha256(f.read()).hexdigest()

expected_hash = meta["checksums"]["encoder_sha256"]
assert actual_hash == expected_hash, "Encoder checksum mismatch!"
print("Encoder integrity verified")
```

---

## REST API

### Starting the Server

```bash
# Using Docker (recommended for production)
docker compose up -d

# Using Python directly (development)
pip install sarcodec[api]
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with loaded models |
| `/encode` | POST | Compress GeoTIFF to NPZ |
| `/decode` | POST | Decompress NPZ to GeoTIFF |
| `/compress` | POST | Alias for `/encode` |
| `/decompress` | POST | Alias for `/decode` |

### curl Examples

```bash
# Health check
curl http://localhost:8000/health

# Compress an image
curl -X POST "http://localhost:8000/encode?model=8x" \
    -F "file=@input.tif" \
    -o compressed.npz

# Decompress
curl -X POST "http://localhost:8000/decode?model=8x" \
    -F "file=@compressed.npz" \
    -o reconstructed.tif
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())
# {'status': 'healthy', 'version': '1.0.0', 'models_loaded': ['4x', '8x', '16x'], ...}

# Compress
with open("input.tif", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/encode?model=8x",
        files={"file": f}
    )
with open("compressed.npz", "wb") as f:
    f.write(response.content)
print(f"Processing time: {response.headers['X-Processing-Time-Ms']} ms")

# Decompress
with open("compressed.npz", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/decode?model=8x",
        files={"file": f}
    )
with open("reconstructed.tif", "wb") as f:
    f.write(response.content)
```

### Interactive API Documentation

The FastAPI server includes automatic documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Performance Tuning

### GPU Memory Optimization

```python
from sarcodec import SARCompressor

# For 8GB VRAM (RTX 3070)
compressor = SARCompressor(
    model_path="models/resnet_8x.pth",
    batch_size=16,  # Reduce if OOM
    overlap=64      # Reduce for speed (may affect quality)
)

# For large images (>10000x10000)
compressor = SARCompressor(
    model_path="models/resnet_8x.pth",
    batch_size=4,   # Very conservative
    overlap=128     # Larger overlap for better blending
)
```

### Tile Overlap Selection

| Image Size | Recommended Overlap | Notes |
|------------|---------------------|-------|
| < 2000x2000 | 64 | Default, good quality |
| 2000-5000 | 64-128 | Increase if boundary artifacts visible |
| > 5000 | 128 | Smoother blending for large images |

### Batch Size Guidelines

| VRAM | Recommended Batch Size |
|------|------------------------|
| 4 GB | 4 |
| 6 GB | 8 |
| 8 GB | 16 |
| 12+ GB | 32 |

### AMP (Automatic Mixed Precision)

AMP is enabled by default when CUDA is available, reducing memory usage and improving speed.

```python
# Check if AMP is being used
compressor = SARCompressor("models/resnet_8x.pth")
print(f"Device: {compressor.device}")  # Should show cuda:0
```

---

## Troubleshooting

### Common Issues

#### Model not loading

```bash
# Verify checkpoint exists
ls -la models/resnet_8x.pth

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

**Solution:** Ensure checkpoint matches PyTorch version and architecture.

#### CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce batch size: `--batch-size 4` or `--batch-size 1`
2. Process smaller image tiles
3. Close other GPU applications
4. Use CPU fallback: `CUDA_VISIBLE_DEVICES="" sarcodec compress ...`

#### Docker GPU not detected

```bash
# Verify nvidia-container-toolkit is installed
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

**Solution:** Install nvidia-container-toolkit:
```bash
# Ubuntu
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### API returns 404 for model

```json
{"detail": "Model '4x' not loaded. Available models: 8x, 16x"}
```

**Solution:** Ensure all checkpoint files are in the mounted models directory:
```bash
ls -la models/
# Should show: resnet_4x.pth, resnet_8x.pth, resnet_16x.pth
```

#### ONNX export validation fails

```
Decoder validation failed: max_diff = 0.05
```

**Note:** Decoder validation uses looser tolerance (1e-2) due to BatchNorm and inverse preprocessing. Small differences are expected and acceptable for most applications.

### Getting Help

1. Check the [GitHub Issues](https://github.com/username/sarcodec/issues)
2. Review the [API Reference](api-reference.md)
3. See example usage in [notebooks/usage_example.ipynb](../notebooks/usage_example.ipynb)

### Reporting Bugs

Include the following information:
- Python and PyTorch versions
- GPU model and VRAM
- Full error traceback
- Minimal reproduction script
