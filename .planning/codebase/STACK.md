# Technology Stack

**Analysis Date:** 2026-01-21

## Languages

**Primary:**
- Python 3.11.8 - Core application language
- YAML - Configuration files

## Runtime

**Environment:**
- Python 3.11.8 (via venv)
- Virtual environment location: `venv/`

**Package Manager:**
- pip
- Lockfile: requirements.txt present

## Frameworks

**Core:**
- PyTorch 2.0.0+ - Deep learning framework for SAR autoencoder model
- torchvision 0.15.0+ - Computer vision utilities and transforms

**Testing:**
- pytest (referenced in codebase but not explicitly in requirements)

**Visualization & Notebooks:**
- Jupyter - Interactive notebooks for learning and experimentation
- ipykernel - Jupyter kernel for Python
- Matplotlib - Plotting and visualization
- TensorBoard 2.12.0+ - Training monitoring and metrics visualization

## Key Dependencies

**Critical:**
- torch>=2.0.0 - Neural network model development, training, inference
- torchvision>=0.15.0 - Image preprocessing and model utilities
- numpy - Array operations and numerical computing
- scipy - Scientific computing (signal processing, statistics)
- rasterio>=1.3.0 - GeoTIFF file I/O for SAR data loading from Sentinel-1

**Specialized:**
- scikit-image - Image processing and quality metrics (SSIM, PSNR)
- Pillow - Image file handling
- PyWavelets - Wavelet transforms for compression analysis
- pytorch-msssim>=1.0.0 - SSIM loss computation for training
- tqdm>=4.65.0 - Progress bars for training loops
- pyyaml>=6.0 - YAML configuration file parsing

**Compression & Analysis:**
- Custom modules in `src/compression/` for entropy and histogram analysis

## Configuration

**Environment:**
- Configuration via `configs/default.yaml`
- Key configs:
  - Model parameters (latent_channels, base_channels, batch_norm usage)
  - Training hyperparameters (learning rate, epochs, early stopping patience)
  - Data settings (batch size, patch size, validation split)
  - Loss function weights (MSE vs SSIM balance)
  - Logging directories and checkpoint intervals

**Build:**
- No build configuration file detected
- Direct Python execution model

**VS Code Settings:**
- `.vscode/settings.json` - Python interpreter path and Jupyter settings

## Platform Requirements

**Development:**
- Windows 10/11 (based on venv configuration)
- CUDA-capable GPU recommended (PyTorch will auto-detect)
- At least 8GB RAM for training 256×256 SAR patches

**Production:**
- Linux or Windows
- Python 3.11+
- GPU optional but recommended for inference speed

---

*Stack analysis: 2026-01-21*
