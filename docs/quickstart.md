# Quickstart Guide

Get started with SAR image compression in under 5 minutes.

## Installation

### Option 1: pip install (recommended)

```bash
pip install sarcodec
```

### Option 2: From source

```bash
git clone https://github.com/username/sarcodec.git
cd sarcodec
pip install -e .
```

### Optional dependencies

```bash
# GPU acceleration with ONNX Runtime
pip install sarcodec[gpu]

# REST API server
pip install sarcodec[api]

# ONNX export tools
pip install sarcodec[export]

# Everything
pip install sarcodec[full]
```

## Download Models

Download pre-trained model checkpoints (choose one):

| Model | Compression | PSNR | File Size | Best For |
|-------|-------------|------|-----------|----------|
| ResNet 4x | 4:1 | 24.95 dB | ~6 MB | Quality-critical applications |
| ResNet 8x | 8:1 | 23.13 dB | ~3 MB | Balanced quality/size (recommended) |
| ResNet 16x | 16:1 | 20.52 dB | ~1.5 MB | Maximum compression |

```bash
# Create models directory
mkdir -p models

# Download 8x model (recommended)
curl -L -o models/resnet_8x.pth \
  https://github.com/username/sarcodec/releases/download/v1.0.0/resnet_8x.pth
```

## First Compression

### Using CLI

```bash
# Compress a GeoTIFF
sarcodec compress sentinel1.tif -o compressed.npz --model models/resnet_8x.pth

# Decompress back to GeoTIFF
sarcodec decompress compressed.npz -o reconstructed.tif --model models/resnet_8x.pth
```

### Using Python

```python
from sarcodec import SARCompressor

# Load model
compressor = SARCompressor("models/resnet_8x.pth")

# Read and compress
import rasterio
with rasterio.open("sentinel1.tif") as src:
    data = src.read(1).astype("float32")

latent, metadata = compressor.compress(data)
print(f"Compressed to {latent.shape} latent representation")

# Decompress
reconstructed = compressor.decompress(latent, metadata)
print(f"Reconstructed shape: {reconstructed.shape}")
```

## Batch Processing

```bash
# Compress all TIFFs in a directory
sarcodec compress *.tif --model models/resnet_8x.pth

# Output files are saved with .npz extension
```

## Verify Installation

```bash
# Check version and GPU availability
sarcodec --version

# Expected output:
# sarcodec version 1.0.0
#
# PyTorch:
#   Version: 2.x.x
#   CUDA: 12.x
#   GPU: NVIDIA RTX 3070
#   VRAM: 8192 MB
```

## Next Steps

- [Deployment Guide](deployment.md) - Docker, API, and production deployment
- [API Reference](api-reference.md) - REST API documentation
- [Example Notebook](../notebooks/usage_example.ipynb) - Interactive examples

## Troubleshooting

### Model not found

Ensure the model path is correct:

```bash
# Check if model exists
ls -la models/resnet_8x.pth
```

### GPU not detected

Verify PyTorch CUDA installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Out of memory

Reduce batch size for large images:

```bash
sarcodec compress large_image.tif --batch-size 1
```
