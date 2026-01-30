# SAR Codec: Neural Compression for Sentinel-1 SAR Imagery

A CNN autoencoder for compressing Sentinel-1 SAR (Synthetic Aperture Radar) satellite imagery. Achieves up to 16x compression while preserving image quality for downstream analysis.

## Features

- **Multiple compression ratios**: 4x, 8x, 16x with trained models
- **GeoTIFF support**: Preserves CRS, transform, GCPs, and nodata values
- **REST API**: FastAPI server for compression/decompression
- **Docker deployment**: GPU-enabled container with CUDA support
- **ONNX export**: Deploy models outside PyTorch ecosystem
- **CLI tool**: Simple command-line compression workflow

## Installation

```bash
# Basic installation
pip install -e .

# With API support
pip install -e ".[api]"

# Full installation (GPU + API + ONNX)
pip install -e ".[full]"
```

## Quick Start

### Command Line

```bash
# Compress a GeoTIFF (uses default 8x compression)
sarcodec compress input.tif -o output.npz

# Compress with specific compression ratio (4x = best quality, 16x = smallest)
sarcodec compress input.tif -o output.npz -c 4x

# Compress with custom model checkpoint
sarcodec compress input.tif -o output.npz --model notebooks/checkpoints/resnet_c64_b64_cr4x_*/best.pth

# Decompress back to GeoTIFF
sarcodec decompress output.npz -o reconstructed.tif

# Decompress as Cloud Optimized GeoTIFF
sarcodec decompress output.npz -o reconstructed.tif --cog
```

### Python API

```python
from src.inference import SARCompressor
from src.models import ResNetAutoencoder

# Load model
model = ResNetAutoencoder(latent_channels=64)  # 4x compression
compressor = SARCompressor(model, checkpoint_path="path/to/best.pth")

# Compress
latent, metadata = compressor.compress("input.tif")
compressor.save_compressed(latent, metadata, "output.npz")

# Decompress
compressor.decompress("output.npz", "reconstructed.tif")
```

### REST API

```bash
# Start server
python -m src.api

# Compress (curl)
curl -X POST http://localhost:8000/encode \
  -F "file=@input.tif" \
  -F "compression_ratio=4" \
  -o output.npz

# Decompress
curl -X POST http://localhost:8000/decode \
  -F "file=@output.npz" \
  -o reconstructed.tif
```

## Results

### Model Performance

| Model | Compression | PSNR | SSIM | BPP |
|-------|-------------|------|------|-----|
| ResNet 4x | 4x | 25.56 dB | 0.899 | 1.53 |
| ResNet 8x | 8x | 23.78 dB | 0.847 | 0.89 |
| ResNet 16x | 16x | 21.20 dB | 0.740 | 0.44 |

### Comparison with JPEG-2000 (at matched bitrates)

| Model | vs JPEG-2000 PSNR | vs JPEG-2000 SSIM |
|-------|-------------------|-------------------|
| ResNet 4x | -3.22 dB | -0.035 |
| ResNet 8x | -1.42 dB | ~0.000 |
| ResNet 16x | -1.33 dB | **+0.040** |

At high compression (16x), the autoencoder achieves better perceptual quality (SSIM) than JPEG-2000 at the same bitrate.

### SAR-Specific Metrics

- **ENL Ratio**: Equivalent Number of Looks preservation (0.85)
- **EPI**: Edge Preservation Index (0.88)

## Project Structure

```
├── src/                    # Core library
│   ├── api/               # REST API (FastAPI)
│   ├── data/              # Data loading and preprocessing
│   ├── evaluation/        # Metrics and codec baselines
│   ├── export/            # ONNX export utilities
│   ├── inference/         # Full image compression pipeline
│   ├── losses/            # Loss functions (MSE + SSIM)
│   ├── models/            # Neural network architectures
│   └── training/          # Training loop
│
├── scripts/               # CLI tools
│   ├── sarcodec.py       # Main CLI
│   ├── train.py          # Training script
│   └── export_onnx.py    # ONNX export script
│
├── notebooks/             # Training notebooks
├── learningnotebooks/     # Educational compression notebooks
├── configs/               # Training configurations
├── docs/                  # Documentation
└── reports/               # Evaluation results
```

## Documentation

- [Quickstart Guide](docs/quickstart.md) - Get started in < 5 minutes
- [Deployment Guide](docs/deployment.md) - Docker and production deployment
- [API Reference](docs/api-reference.md) - REST API documentation
- [Codebase Reference](CODEBASE.md) - Complete file documentation

## Docker

```bash
# Build image
docker build -t sarcodec .

# Run with GPU
docker compose up

# API available at http://localhost:8000
```

## Training

Train your own models with custom data:

```bash
# Using notebook
jupyter notebook notebooks/train_resnet.ipynb

# Using script
python scripts/train.py --config configs/default.yaml
```

See `notebooks/` for training examples with different architectures.

## Learning Resources

The `learningnotebooks/` directory contains educational notebooks covering:

1. **Compression Fundamentals**: Information theory, Huffman/arithmetic coding, quantization, DCT, wavelets
2. **Learned Compression**: Autoencoders, quantization in neural networks, entropy models
3. **SAR-Specific**: Processing Sentinel-1 data, SAR quality metrics

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependencies.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@software{sarcodec2026,
  title = {SAR Codec: Neural Compression for Sentinel-1 SAR Imagery},
  year = {2026},
  url = {https://github.com/username/sarcodec}
}
```
