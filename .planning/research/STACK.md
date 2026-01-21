# Technology Stack for SAR Autoencoder Compression

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression
**Researched:** 2026-01-21
**Constraints:** RTX 3070 8GB VRAM, training under 1 day, PyTorch 2.0+

---

## Executive Summary

The existing stack is well-suited for this project. **Keep PyTorch 2.0+ as the core** with targeted additions for SSIM computation and SAR data handling. No major stack changes needed; focus on completing the implementation with existing dependencies.

**Confidence:** HIGH for core stack, MEDIUM for newer PyTorch features, MEDIUM for alternative libraries.

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **PyTorch** | >=2.0.0 | Model development, training | Already in use. PyTorch 2.0+ includes `torch.compile()` for potential 20-40% speedup on compatible models. Native autograd, excellent GPU support, and the dominant framework for image compression research. |
| **torchvision** | >=0.15.0 | Image transforms, utilities | Already in use. Provides optimized image operations and model utilities. Compatible with PyTorch 2.0+. |

**Confidence:** HIGH - These are the established choices, already in the project.

### SSIM/Perceptual Loss Libraries

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **pytorch-msssim** | >=1.0.0 | SSIM and MS-SSIM loss | **Already in requirements.** Provides GPU-accelerated SSIM loss that works with autograd. Clean API: `ssim(x, y, data_range=1.0)`. Essential for perceptual quality in compression. |
| **piqa** (alternative) | >=0.3.0 | Multi-scale perceptual losses | Optional alternative with more loss functions (LPIPS, DISTS). Only add if MS-SSIM alone proves insufficient. |

**Recommendation:** Use pytorch-msssim (already included). It covers SSIM and MS-SSIM which are the most relevant for SAR compression.

**Confidence:** HIGH - pytorch-msssim is mature and already specified.

### SAR Data Handling

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **rasterio** | >=1.3.0 | GeoTIFF loading | **Already in requirements.** The standard for loading geospatial raster data including Sentinel-1 GeoTIFFs. Handles projections, metadata, and multi-band reads. |
| **numpy** | >=1.24.0 | Array operations | Core dependency. Used for patch extraction, preprocessing, and data manipulation. |
| **scipy** | >=1.10.0 | Scientific computing | **Already in requirements.** Provides `ndimage` for local statistics (ENL computation), filtering, and signal processing. |

**Confidence:** HIGH - These are standard, already in the project.

### Image Processing and Metrics

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **scikit-image** | >=0.20.0 | Quality metrics, image ops | **Already in requirements.** Provides `structural_similarity()` for evaluation (separate from training loss), along with other image processing utilities. |
| **Pillow** | >=9.0.0 | Image file I/O | **Already in requirements.** Handles PNG/JPEG visualization outputs. |

**Confidence:** HIGH - Standard choices, already present.

### Experiment Tracking and Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **TensorBoard** | >=2.12.0 | Training visualization | **Already in requirements.** Perfect for tracking losses, metrics, and visualizing reconstruction samples. Works seamlessly with `torch.utils.tensorboard.SummaryWriter`. |
| **matplotlib** | >=3.7.0 | Plots and figures | **Already in requirements.** Essential for rate-distortion curves, comparison plots, and publication figures. |
| **tqdm** | >=4.65.0 | Progress bars | **Already in requirements.** Clean progress indication for training loops and data preprocessing. |

**Confidence:** HIGH - Standard, already configured.

### Configuration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **PyYAML** | >=6.0 | YAML config files | **Already in requirements.** Clean configuration management. The project already uses YAML for hyperparameters. |

**Confidence:** HIGH - Already in use.

---

## Optional Additions (Evaluate During Implementation)

### For Advanced Learned Compression

| Library | Version | Purpose | When to Add |
|---------|---------|---------|-------------|
| **CompressAI** | >=1.2.0 | State-of-art compression | Only if you want to compare against published models (Balle, Cheng, etc.). Adds significant complexity. NOT recommended for this project's scope. |
| **torchac** | >=0.9.3 | Arithmetic coding | Only if implementing entropy coding stage. Currently out of scope per PROJECT.md. |

**Recommendation:** Do not add these. Focus on autoencoder quality first. Entropy coding is explicitly out of scope.

**Confidence:** MEDIUM - These are well-maintained but add complexity beyond current scope.

### For Hyperparameter Tuning

| Library | Version | Purpose | When to Add |
|---------|---------|---------|-------------|
| **Optuna** | >=3.0.0 | Hyperparameter optimization | Only if manual tuning proves insufficient. Can automate learning rate, loss weights, architecture choices. |

**Recommendation:** Start with manual experiments. Add Optuna only if systematic search becomes necessary.

**Confidence:** MEDIUM - Well-established but may be overkill for comparison study.

---

## What NOT to Use (and Why)

### Avoid: PyTorch Lightning

**Why not:** The existing codebase has a clean manual training loop with TensorBoard integration. PyTorch Lightning would require restructuring the trainer, dataset, and model classes. The overhead doesn't justify the benefits for this project size.

**Confidence:** HIGH - Unnecessary refactoring for no significant gain.

### Avoid: TensorFlow/Keras

**Why not:** Project is already PyTorch. Mixing frameworks creates dependency conflicts and maintenance burden. PyTorch has equal or better support for image compression research.

**Confidence:** HIGH - Wrong choice given existing codebase.

### Avoid: Hugging Face Diffusers

**Why not:** This is for diffusion models, not autoencoders. Would fundamentally change the architecture away from the compression-focused approach.

**Confidence:** HIGH - Different problem domain.

### Avoid: Custom CUDA Kernels

**Why not:** PyTorch's built-in operations are sufficient for this architecture. Custom kernels add compilation complexity and maintenance burden without clear performance benefits for standard conv/deconv operations.

**Confidence:** HIGH - Unnecessary complexity.

### Avoid: JAX/Flax

**Why not:** While JAX offers benefits for certain research workflows, it would require complete rewrite. PyTorch is sufficient and already in use.

**Confidence:** HIGH - Wrong choice given existing codebase.

### Avoid: Weights & Biases (wandb)

**Why not:** TensorBoard is already configured and sufficient for this project's needs. W&B adds external dependencies and account requirements without clear benefits for a local comparison study.

**Confidence:** MEDIUM - Good tool but unnecessary given TensorBoard is working.

---

## PyTorch 2.0+ Features to Leverage

### torch.compile() (Optional Speedup)

PyTorch 2.0 introduced `torch.compile()` which can provide 20-40% training speedup.

```python
# After model is defined, optionally compile
model = SARAutoencoder(latent_channels=64)
model = torch.compile(model)  # Optional speedup
```

**When to use:** After model is working correctly. Compile adds overhead to first iteration but speeds up subsequent iterations.

**Caveat:** May not work with all model architectures. Test before relying on it.

**Confidence:** MEDIUM - Depends on model compatibility.

### Native Scaled Dot-Product Attention

If implementing attention mechanisms:

```python
# PyTorch 2.0+ has optimized attention
from torch.nn.functional import scaled_dot_product_attention
```

**When to use:** If adding self-attention blocks (CBAM already sketched in codebase).

**Confidence:** HIGH - Stable PyTorch feature.

### BFloat16 Mixed Precision

For potential memory savings:

```python
# In training loop
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x_hat, z = model(x)
    loss, metrics = loss_fn(x_hat, x)
```

**When to use:** If VRAM becomes a bottleneck with larger batch sizes.

**Caution:** Test quality metrics carefully; some precision-sensitive operations may need exclusion.

**Confidence:** MEDIUM - Standard technique but needs validation for compression quality.

---

## Installation Commands

### Core Installation (Already Configured)

```bash
# The existing requirements.txt covers all core needs
pip install -r requirements.txt
```

### Verify PyTorch GPU Support

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Optional Additions (If Needed Later)

```bash
# Only if MS-SSIM proves insufficient
pip install piqa>=0.3.0

# Only if systematic hyperparameter search needed
pip install optuna>=3.0.0
```

---

## Requirements.txt Recommendations

The current requirements.txt is adequate. One minor cleanup recommendation:

```txt
# Core (keep as-is)
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
torch>=2.0.0
torchvision>=0.15.0

# Image Processing (keep as-is)
Pillow>=9.0.0
scikit-image>=0.20.0

# SAR/Geospatial (keep as-is)
rasterio>=1.3.0

# Training Support (keep as-is)
tensorboard>=2.12.0
tqdm>=4.65.0
pyyaml>=6.0
pytorch-msssim>=1.0.0

# Development (keep as-is)
jupyter>=1.0.0
ipykernel

# Remove PyWavelets if not using wavelet transforms (currently unused)
# PyWavelets  # <- Consider removing if not used
```

**Note:** The duplicate entries (torch listed twice, tqdm listed twice) in the current requirements.txt should be cleaned up but don't cause functional issues.

---

## Hardware Optimization for RTX 3070

### VRAM Budget (8GB)

| Component | Estimated Memory |
|-----------|-----------------|
| Model parameters (~5M) | ~20 MB |
| Optimizer states (Adam) | ~40 MB |
| Input batch (16 x 256 x 256) | ~4 MB |
| Activations + gradients | ~2-4 GB |
| **Total per batch** | ~2.5-4.5 GB |

**Recommendation:** Batch size 8-16 should fit comfortably. Start with 8, increase if VRAM permits.

### Tensor Cores

RTX 3070 has Tensor Cores. To leverage them:
- Keep tensor dimensions multiples of 8 (already the case with 64/128/256 channels)
- Use FP16 mixed precision if needed

**Confidence:** HIGH - Standard GPU optimization.

---

## Version Pinning Strategy

For reproducibility:

```txt
# Pin major.minor, allow patch updates
torch>=2.0.0,<3.0.0
torchvision>=0.15.0,<1.0.0
pytorch-msssim>=1.0.0,<2.0.0
rasterio>=1.3.0,<2.0.0
```

**Rationale:** Allows bug fixes while preventing breaking changes.

---

## Summary

| Category | Decision | Confidence |
|----------|----------|------------|
| Core Framework | PyTorch 2.0+ (keep existing) | HIGH |
| SSIM Loss | pytorch-msssim (keep existing) | HIGH |
| SAR I/O | rasterio (keep existing) | HIGH |
| Experiment Tracking | TensorBoard (keep existing) | HIGH |
| Configuration | PyYAML (keep existing) | HIGH |
| Add CompressAI? | NO - out of scope | HIGH |
| Add Lightning? | NO - unnecessary refactor | HIGH |
| Add Optuna? | MAYBE - only if manual tuning fails | MEDIUM |

**Bottom Line:** The existing stack is well-chosen. No major additions needed. Focus effort on implementing the skeleton code, not on stack changes.

---

## Sources

- PyTorch official documentation (torch.compile, autocast)
- pytorch-msssim GitHub repository
- rasterio documentation
- Existing project requirements.txt and knowledge documents
- Project constraints from PROJECT.md

**Note:** WebSearch and WebFetch were unavailable during research. Recommendations are based on training data knowledge (cutoff May 2025) combined with existing project documentation. Version recommendations should be validated against current PyPI releases.
