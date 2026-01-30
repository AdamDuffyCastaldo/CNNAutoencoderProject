# Codebase Reference

Complete documentation of all files in the SAR Codec repository.

## Project Overview

This repository implements a CNN autoencoder for compressing Sentinel-1 SAR (Synthetic Aperture Radar) satellite imagery. The system achieves up to 16x compression while preserving image quality sufficient for downstream analysis.

---

## Directory Structure

```
sarcodec/
├── src/                    # Core library code
│   ├── api/               # REST API (FastAPI)
│   ├── compression/       # Entropy coding utilities
│   ├── data/              # Data loading and preprocessing
│   ├── evaluation/        # Metrics and evaluation tools
│   ├── export/            # ONNX export utilities
│   ├── inference/         # Full image compression pipeline
│   ├── losses/            # Loss functions
│   ├── models/            # Neural network architectures
│   ├── sarcodec/          # Installable package wrapper
│   ├── training/          # Training loop
│   └── utils/             # Utilities
├── scripts/               # CLI scripts and tools
├── configs/               # Training configurations (YAML)
├── notebooks/             # Training and evaluation notebooks
├── learningnotebooks/     # Educational compression notebooks
├── docs/                  # Documentation
├── reports/               # Final evaluation results
├── data/                  # Data directory (mostly gitignored)
└── results/               # Experiment outputs (mostly gitignored)
```

---

## Source Code (`src/`)

### API Module (`src/api/`)

REST API for serving compression/decompression over HTTP.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports with lazy loading |
| `__main__.py` | Entry point for `python -m src.api` |
| `app.py` | FastAPI application with lifespan model loading |
| `endpoints.py` | API endpoints: `/health`, `/encode`, `/decode` |
| `models.py` | Pydantic request/response models |

### Compression Module (`src/compression/`)

Entropy coding utilities for learned compression research.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `entropy.py` | Entropy estimation functions |
| `histogram.py` | Histogram-based probability modeling |

### Data Module (`src/data/`)

Data loading, preprocessing, and PyTorch datasets.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `datamodule.py` | `SARDataModule` - PyTorch Lightning-style data handling |
| `dataset.py` | `SARPatchDataset`, `LazyPatchDataset` for memory-efficient loading |
| `preprocessing.py` | SAR preprocessing (dB conversion, normalization, inverse) |

### Evaluation Module (`src/evaluation/`)

Metrics, codec baselines, and evaluation tools.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `bitrate.py` | Entropy-based bitrate estimation for learned codecs |
| `codec_baselines.py` | JPEG-2000 and JPEG codec wrappers for comparison |
| `evaluator.py` | `Evaluator` class for batch model evaluation |
| `metrics.py` | `SARMetrics` - PSNR, SSIM, ENL ratio, EPI |
| `visualizer.py` | `Visualizer` for generating comparison figures |

### Export Module (`src/export/`)

ONNX export utilities for deployment.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `onnx_export.py` | ONNX export with validation and checksums |
| `wrapper.py` | `SARAutoencoderWithPreprocess` - model with embedded preprocessing |

### Inference Module (`src/inference/`)

Full image compression pipeline with tiling and GeoTIFF I/O.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `compressor.py` | `SARCompressor` - end-to-end compression/decompression |
| `geotiff.py` | GeoTIFF I/O with metadata preservation (CRS, GCPs) |
| `tiling.py` | Tile extraction, cosine-squared blending, reconstruction |

### Losses Module (`src/losses/`)

Loss functions for training.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `combined.py` | `CombinedLoss` - weighted MSE + SSIM |
| `mse.py` | `MSELoss` wrapper |
| `ssim.py` | `SSIMLoss` using pytorch-msssim |

### Models Module (`src/models/`)

Neural network architectures.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `autoencoder.py` | `SARAutoencoder` - baseline architecture |
| `blocks.py` | Building blocks: `ConvBlock`, `ResidualBlock`, `PreActResidualBlock`, `CBAM` |
| `decoder.py` | `SARDecoder` - baseline decoder |
| `encoder.py` | `SAREncoder` - baseline encoder |
| `residual_autoencoder.py` | `ResidualAutoencoder` - with skip connections |
| `resnet_autoencoder.py` | `ResNetAutoencoder` - **best performing architecture** |
| `attention_autoencoder.py` | `AttentionAutoencoder` - with CBAM attention |

### SAR Codec Package (`src/sarcodec/`)

Installable package wrapper for pip installation.

| File | Purpose |
|------|---------|
| `__init__.py` | Path injection and public API exports |
| `cli.py` | CLI entry point (`sarcodec compress/decompress`) |
| `py.typed` | PEP 561 marker for type checking |

### Training Module (`src/training/`)

Training loop implementation.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `trainer.py` | `Trainer` - training loop with checkpointing, early stopping, AMP |

### Utils Module (`src/utils/`)

General utilities.

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `io.py` | I/O utilities (config loading, checkpoint management) |

---

## Scripts (`scripts/`)

Command-line tools for training, evaluation, and deployment.

| Script | Purpose |
|--------|---------|
| `sarcodec.py` | Main CLI: `python scripts/sarcodec.py compress/decompress` |
| `train.py` | General training script |
| `train_baseline.py` | Train baseline autoencoder |
| `train_sweep.py` | Hyperparameter sweep training |
| `evaluate.py` | Model evaluation script |
| `evaluate_model.py` | Detailed single-model evaluation |
| `run_evaluation_sweep.py` | Evaluate all models systematically |
| `run_evaluation_sweep_8bit.py` | Evaluation with 8-bit quantization |
| `run_bitrate_matched_evaluation.py` | Fair comparison at matched bitrates |
| `generate_final_report.py` | Generate final comparison report |
| `export_onnx.py` | Export models to ONNX format |
| `download_sentinel_data.py` | Download Sentinel-1 data (placeholder) |
| `quick_train_attention.py` | Quick attention model training |
| `docker-entrypoint.sh` | Docker container entrypoint |

---

## Configuration (`configs/`)

YAML configuration files for training.

| File | Purpose |
|------|---------|
| `default.yaml` | Default training configuration |
| `sweep_all_16x.yaml` | 16x compression sweep (baseline vs ResNet) |
| `sweep_baseline_ratios.yaml` | Baseline model at 4x/8x/16x |
| `sweep_resnet_ratios.yaml` | ResNet model at 4x/8x/16x |

---

## Notebooks (`notebooks/`)

Interactive notebooks for training and analysis.

| Notebook | Purpose |
|----------|---------|
| `train_baseline.ipynb` | Train baseline autoencoder |
| `train_resnet.ipynb` | Train ResNet autoencoder |
| `train_residual.ipynb` | Train residual autoencoder |
| `train_attention.ipynb` | Train attention autoencoder |
| `sweep_all_16x.ipynb` | Compare architectures at 16x |
| `sweep_baseline_ratios.ipynb` | Baseline at multiple ratios |
| `compare_architectures.ipynb` | Architecture comparison analysis |
| `test_full_inference.ipynb` | Test full image inference pipeline |
| `usage_example.ipynb` | User-facing usage demonstration |

### Notebook Outputs (`notebooks/results/`, `notebooks/evaluations/`)

Training results and evaluation outputs stored alongside notebooks.

---

## Learning Notebooks (`learningnotebooks/`)

Educational notebooks covering compression fundamentals. Organized by topic:

### Phase 1: Compression Fundamentals

| Section | Topics |
|---------|--------|
| 1.1 Information Theory | Entropy calculation, real data entropy, conditional entropy |
| 1.2 Lossless Compression | Huffman coding, arithmetic coding, comparison |
| 1.3 Quantization | Uniform quantization, Lloyd-Max algorithm |
| 1.4 Transform Coding | DCT from scratch, wavelet transform, simple JPEG |
| 1.5 Complete Codecs | Codec comparison, R-D curve plotting |

### Phase 2: Learned Compression

| Section | Topics |
|---------|--------|
| 2.1 Autoencoder Foundations | Basic AE, convolutional AE, denoising AE |
| 2.2 Quantization Problem | Visualizing quantization, training with quantization |
| 2.3 Entropy Models | Entropy estimation, factorized prior, hyperprior |
| 2.4 Full Model | Complete model, multiple rate points, actual pipeline |

### Phase 4: SAR Codec

| Notebook | Purpose |
|----------|---------|
| `day1.ipynb` | Initial SAR codec development |
| `day2_no_references.ipynb` | Continued development |
| `processingalldata.ipynb` | Full dataset processing |

---

## Documentation (`docs/`)

User-facing documentation.

| File | Purpose |
|------|---------|
| `quickstart.md` | Getting started in < 5 minutes |
| `deployment.md` | Production deployment guide (Docker, API) |
| `api-reference.md` | REST API endpoint documentation |

---

## Reports (`reports/`)

Final evaluation results and figures.

| Path | Purpose |
|------|---------|
| `final_comparison.md` | Final comparison report with executive summary |
| `final_comparison.ipynb` | Reproducible analysis notebook |
| `data/*.json` | Raw evaluation data (BPP, R-D curves) |
| `figures/*.png` | Rate-distortion curves, comparison figures |
| `tables/*.csv` | Summary tables |

---

## Build & Deployment

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python package configuration (PEP 621, hatchling) |
| `requirements.txt` | Core dependencies |
| `requirements-api.txt` | API-specific dependencies |
| `Dockerfile` | Multi-stage Docker build with CUDA support |
| `docker-compose.yml` | Docker Compose with GPU and volume mounts |
| `LICENSE` | Apache 2.0 license |

---

## Data Directory (`data/`)

Data files (mostly gitignored, only structure tracked).

| Path | Purpose |
|------|---------|
| `README.md` | Data directory documentation |
| `images/` | Visualization outputs from notebooks |

---

## Key Entry Points

### For Users

```bash
# Install package
pip install -e .

# Compress a GeoTIFF
sarcodec compress input.tif output.npz --checkpoint path/to/best.pth

# Decompress back to GeoTIFF
sarcodec decompress output.npz reconstructed.tif

# Start REST API
python -m src.api
```

### For Developers

```python
# Load and use a model
from src.models import ResNetAutoencoder
from src.inference import SARCompressor

model = ResNetAutoencoder(latent_channels=16)
compressor = SARCompressor(model, checkpoint_path="path/to/best.pth")
latent, metadata = compressor.compress("input.tif")
compressor.decompress(latent, metadata, "output.tif")
```

---

## Model Checkpoints

Best performing models (not tracked in git, stored in `notebooks/checkpoints/`):

| Model | Checkpoint | PSNR | SSIM | Compression |
|-------|------------|------|------|-------------|
| ResNet 4x | `resnet_c64_b64_cr4x_*/best.pth` | 24.95 dB | 0.907 | 4x |
| ResNet 8x | `resnet_c32_b64_cr8x_*/best.pth` | 23.13 dB | 0.857 | 8x |
| ResNet 16x | `resnet_c16_b64_cr16x_*/best.pth` | 20.52 dB | 0.754 | 16x |

---

*Generated: 2026-01-30*
