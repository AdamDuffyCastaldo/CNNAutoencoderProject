# SAR Autoencoder Compression

## What This Is

A CNN-based autoencoder for compressing Sentinel-1 SAR (Synthetic Aperture Radar) satellite imagery. The system encodes SAR patches into compact latent representations and reconstructs them with minimal quality loss. Designed to explore feasibility for satellite downlink scenarios where bandwidth is severely constrained.

## Core Value

Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis — the encoder must compress aggressively, the decoder must reconstruct accurately.

## Requirements

### Validated

- ✓ Project structure with modular architecture — existing
- ✓ Configuration system (YAML-based hyperparameters) — existing
- ✓ PyTorch + TensorBoard stack established — existing
- ✓ Sentinel-1 training data available — existing

### Active

- [ ] Complete preprocessing pipeline (approach open to experimentation)
- [ ] Implement model building blocks (ConvBlock, ResidualBlock, Attention)
- [ ] Implement encoder architecture (spatial compression to latent)
- [ ] Implement decoder architecture (latent to spatial reconstruction)
- [ ] Implement loss functions (MSE, SSIM, combined)
- [ ] Implement training loop with checkpointing and early stopping
- [ ] Implement evaluation metrics (PSNR, SSIM, ENL, EPI)
- [ ] Implement inference pipeline (full image compression/decompression)
- [ ] Train multiple architecture variants for comparison
- [ ] Train multiple compression ratio variants for comparison
- [ ] Generate comparison study with metrics across variants
- [ ] Compare against traditional codecs (JPEG-2000) at same compression ratios
- [ ] End-to-end pipeline: raw Sentinel-1 GeoTIFF → compress → decompress → GeoTIFF
- [ ] Production deployment: ONNX/TorchScript export, Docker container, REST API

### Out of Scope

- Real-time streaming compression — focus is batch processing for research
- On-device deployment optimization — exploring feasibility first
- Lossy entropy coding stage — focusing on autoencoder quality first
- Multi-polarization support — single channel (VV or VH) for v1

## Context

**Existing Codebase:**
The project has a complete skeleton structure with all modules defined but most core functionality unimplemented (NotImplementedError stubs). The architecture is documented in learning notebooks and docstrings. Key patterns are established but need implementation.

**SAR Domain:**
SAR images have unique characteristics — speckle noise, high dynamic range, multiplicative noise model. Preprocessing approach is open for experimentation (dB conversion, normalization strategies, etc.). Quality metrics should include SAR-specific measures like ENL (Equivalent Number of Looks) and EPI (Edge Preservation Index).

**Training Data:**
Sentinel-1 SAR patches are available. Data pipeline expects .npy files with preprocessed patches.

**Architecture Baseline:**
Current skeleton uses 4-layer encoder design with configurable latent channels. Decoder mirrors encoder. Building blocks include residual connections and optional attention mechanisms. Patch size and architecture details are open for experimentation.

## Constraints

- **GPU Memory**: RTX 3070 with 8GB VRAM — batch size and model size must fit
- **Training Time**: Prefer experiments under 1 day each — affects epoch count and model complexity
- **Tech Stack**: PyTorch 2.0+, existing project structure — maintain current patterns

## Open for Experimentation

- **Preprocessing**: If different preprocessing yields better results, explore it
- **Patch Size**: 256×256 is baseline but not fixed — other sizes welcome if beneficial
- **Architecture**: Residual blocks, attention mechanisms, depth, width all variable
- **Compression Ratio**: Latent space dimensions are variable

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Explore architecture variants | User wants to compare different designs (residual, attention) | — Pending |
| Explore compression ratios | User wants to understand quality/size tradeoff | — Pending |
| Loss function not a variable | Use whatever works best (MSE+SSIM likely) | — Pending |
| Preprocessing open | User wants suggestions if better approaches exist | — Pending |
| Patch size open | 256×256 baseline but flexible | — Pending |

---
*Last updated: 2026-01-21 after initialization*
