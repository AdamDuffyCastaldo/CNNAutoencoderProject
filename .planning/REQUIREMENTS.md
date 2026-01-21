# Requirements Specification

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression
**Version:** 1.0
**Generated:** 2026-01-21

---

## Scope Summary

Build a complete SAR image compression system exploring the tradeoff between compression ratio and reconstruction quality. Implement all existing code stubs, compare multiple architecture variants, and produce a comprehensive comparison study suitable for evaluating satellite downlink feasibility.

---

## Functional Requirements

### FR1: Data Pipeline

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR1.1 | Load Sentinel-1 GeoTIFF images using rasterio | Must | Existing stub |
| FR1.2 | Convert linear intensity to dB scale: `dB = 10 * log10(intensity + noise_floor)` | Must | Research: Pitfall #1 |
| FR1.3 | Handle invalid values (zeros, NaN) with noise floor substitution | Must | Research: Pitfall #5 |
| FR1.4 | Clip dynamic range using percentile-based bounds (e.g., 1st-99th percentile) | Must | Knowledge: 05_SAR_PREPROCESSING |
| FR1.5 | Normalize to [0,1] range using training-set computed bounds | Must | Research: Pitfall #2 |
| FR1.6 | Extract 256x256 patches with configurable overlap | Must | Memory constraint |
| FR1.7 | Filter patches by quality (variance threshold, invalid pixel ratio) | Must | Knowledge: 05_SAR_PREPROCESSING |
| FR1.8 | Save preprocessing parameters (vmin, vmax, noise_floor) with checkpoint | Must | Research: Pitfall #2 |
| FR1.9 | Support data augmentation (horizontal flip, random crop only) | Should | Research: Pitfall #12 |
| FR1.10 | Create PyTorch Dataset and DataLoader with configurable batch size | Must | Existing stub |

### FR2: Model Architecture

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR2.1 | Implement ConvBlock: Conv2d + BatchNorm + LeakyReLU(0.2) | Must | Existing stub |
| FR2.2 | Implement DeconvBlock: ConvTranspose2d + BatchNorm + ReLU | Must | Existing stub |
| FR2.3 | Implement ResidualBlock with projection shortcut for channel mismatch | Must | Knowledge: 02_RESIDUAL_BLOCKS |
| FR2.4 | Implement 4-layer encoder: 256x256x1 -> 16x16xC latent | Must | Research: Architecture |
| FR2.5 | Implement 4-layer decoder: 16x16xC -> 256x256x1 reconstructed | Must | Research: Architecture |
| FR2.6 | Use 5x5 kernels for improved receptive field | Should | Research: Architecture |
| FR2.7 | Configurable latent channels (4, 8, 16, 32, 64) for compression ratio control | Must | PROJECT.md |
| FR2.8 | Sigmoid output activation for bounded [0,1] output | Must | Research: Pitfall #8 |
| FR2.9 | No activation on final encoder layer (unbounded latent) | Should | Research: Architecture |
| FR2.10 | Implement CBAM attention module for bottleneck | Should | Knowledge: 03_ATTENTION_MECHANISMS |

### FR3: Training

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR3.1 | Combined loss function: configurable weights for MSE + SSIM | Must | Research: Pitfall #4 |
| FR3.2 | Default loss weights: 0.5 MSE + 0.5 SSIM | Must | Research: FEATURES.md |
| FR3.3 | Adam optimizer with configurable learning rate (default 1e-4) | Must | Research: Pitfall #13 |
| FR3.4 | Learning rate scheduler (ReduceLROnPlateau or cosine annealing) | Should | Best practice |
| FR3.5 | Gradient clipping (max norm 1.0) | Should | Research: Pitfall #13 |
| FR3.6 | Training loop with configurable epochs, early stopping | Must | Existing stub |
| FR3.7 | Checkpoint saving (model state, optimizer state, config, preprocessing params) | Must | Research: Pitfall #10 |
| FR3.8 | Best model tracking based on validation loss | Must | Research: Pitfall #15 |
| FR3.9 | Resume training from checkpoint | Should | Best practice |
| FR3.10 | TensorBoard logging (scalars, images, histograms) | Must | Existing stub |

### FR4: Evaluation

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR4.1 | PSNR metric computation | Must | Knowledge: 06_SAR_QUALITY_METRICS |
| FR4.2 | SSIM metric computation | Must | Knowledge: 06_SAR_QUALITY_METRICS |
| FR4.3 | MS-SSIM metric computation | Should | pytorch-msssim available |
| FR4.4 | Compression ratio calculation | Must | Knowledge: 07_COMPRESSION_TRADEOFFS |
| FR4.5 | Bits per pixel (BPP) calculation | Should | Standard metric |
| FR4.6 | ENL ratio metric for speckle preservation | Must | Knowledge: 06_SAR_QUALITY_METRICS |
| FR4.7 | Edge Preservation Index (EPI) metric | Must | Knowledge: 06_SAR_QUALITY_METRICS |
| FR4.8 | Visual comparison generation (original, reconstructed, difference) | Must | Existing stub |
| FR4.9 | Rate-distortion curve generation | Should | Research: FEATURES.md |
| FR4.10 | Batch evaluation across test set with statistics | Must | Best practice |

### FR5: Inference

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR5.1 | Single patch encode/decode | Must | Core functionality |
| FR5.2 | Full image tiled inference with configurable overlap | Should | Research: Pitfall #9 |
| FR5.3 | Cosine ramp blending for tile boundaries | Should | Research: ARCHITECTURE.md |
| FR5.4 | Memory-efficient processing for large images | Should | RTX 3070 constraint |
| FR5.5 | Inverse preprocessing (restore linear SAR values) | Should | Completeness |
| FR5.6 | Preserve GeoTIFF metadata in output | Could | Nice to have |

### FR6: Comparison Study

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR6.1 | Train plain architecture at multiple compression ratios (8x, 16x, 32x) | Must | PROJECT.md |
| FR6.2 | Train residual architecture at multiple compression ratios | Must | PROJECT.md |
| FR6.3 | Train residual+CBAM architecture at multiple compression ratios | Should | PROJECT.md |
| FR6.4 | Generate rate-distortion curves comparing all variants | Must | PROJECT.md |
| FR6.5 | Statistical analysis of results (mean, std across test set) | Should | Best practice |
| FR6.6 | Document findings with visual examples | Must | PROJECT.md |

---

## Non-Functional Requirements

### NFR1: Performance

| ID | Requirement | Target | Rationale |
|----|-------------|--------|-----------|
| NFR1.1 | Training fits in 8GB VRAM | Batch size 8 minimum | RTX 3070 constraint |
| NFR1.2 | Single experiment completes in <8 hours | Per architecture/CR combo | Total study <1 week |
| NFR1.3 | Inference latency <1s per 256x256 patch | Real-time preview | Usability |
| NFR1.4 | Full image (10000x10000) inference <5 minutes | - | Practical deployment |

### NFR2: Quality Targets

| ID | Requirement | Target | Rationale |
|----|-------------|--------|-----------|
| NFR2.1 | PSNR at 16x compression | >30 dB | Acceptable quality threshold |
| NFR2.2 | SSIM at 16x compression | >0.85 | Structural preservation |
| NFR2.3 | ENL ratio | 0.8-1.2 | Speckle texture preserved |
| NFR2.4 | EPI | >0.85 | Edge quality preserved |

### NFR3: Code Quality

| ID | Requirement | Rationale |
|----|-------------|-----------|
| NFR3.1 | Follow existing code conventions (type hints, docstrings) | Consistency |
| NFR3.2 | Configurable via YAML/JSON, not hardcoded | Reproducibility |
| NFR3.3 | Deterministic training with seed control | Reproducibility |
| NFR3.4 | Clear separation of concerns (data, models, training, evaluation) | Maintainability |

---

## Architecture Variants

### Variant A: Plain (Baseline)

```
Encoder: Conv(1,64) -> Conv(64,128) -> Conv(128,256) -> Conv(256,C_latent)
Decoder: Deconv(C_latent,256) -> Deconv(256,128) -> Deconv(128,64) -> Deconv(64,1)
```

- Parameters: ~3-5M
- Purpose: Establish baseline metrics

### Variant B: Residual

```
Encoder: Conv(1,64) -> Res(64) -> Conv(64,128) -> Res(128) -> Conv(128,256) -> Res(256) -> Conv(256,C_latent)
Decoder: Deconv(C_latent,256) -> Res(256) -> Deconv(256,128) -> Res(128) -> Deconv(128,64) -> Res(64) -> Deconv(64,1)
```

- Parameters: ~5-8M
- Purpose: Improved detail preservation

### Variant C: Residual + CBAM

```
Same as Variant B but with CBAM attention applied at bottleneck (16x16xC)
```

- Parameters: ~6-9M
- Purpose: Best quality/cost tradeoff

---

## Compression Ratios

| Latent Channels | Latent Size | Compression Ratio | Study Priority |
|-----------------|-------------|-------------------|----------------|
| 32 | 16x16x32 | 8x | Must |
| 16 | 16x16x16 | 16x | Must (recommended start) |
| 8 | 16x16x8 | 32x | Must |
| 64 | 16x16x64 | 4x | Should (if time permits) |
| 4 | 16x16x4 | 64x | Could (likely too aggressive) |

---

## Experiment Matrix

| Variant | 8x | 16x | 32x |
|---------|-----|-----|-----|
| Plain (A) | Must | Must | Must |
| Residual (B) | Must | Must | Must |
| Res+CBAM (C) | Should | Should | Should |

**Total experiments:** 6-9 trained models

---

## Acceptance Criteria

1. **Data pipeline produces valid normalized patches** - Visual inspection + value range verification
2. **Models train to convergence** - Loss plateaus, no NaN/explosion
3. **PSNR >30 dB at 16x compression** - At least one architecture achieves this
4. **ENL ratio between 0.8-1.2** - Speckle not over-smoothed or amplified
5. **Rate-distortion curves generated** - Quality vs compression visualized
6. **Visual comparison available** - Side-by-side original vs reconstructed at each compression level
7. **Full image inference works** - Can process a complete Sentinel-1 scene

---

## Out of Scope (v1)

- Entropy coding (Huffman, arithmetic, learned)
- Variable-rate single model
- Multi-polarization (VV+VH jointly)
- Phase preservation
- Real-time streaming
- ONNX/TensorRT export
- Web interface

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FR1.1 | Phase 1: Data Pipeline | Pending |
| FR1.2 | Phase 1: Data Pipeline | Pending |
| FR1.3 | Phase 1: Data Pipeline | Pending |
| FR1.4 | Phase 1: Data Pipeline | Pending |
| FR1.5 | Phase 1: Data Pipeline | Pending |
| FR1.6 | Phase 1: Data Pipeline | Pending |
| FR1.7 | Phase 1: Data Pipeline | Pending |
| FR1.8 | Phase 1: Data Pipeline | Pending |
| FR1.9 | Phase 1: Data Pipeline | Pending |
| FR1.10 | Phase 1: Data Pipeline | Pending |
| FR2.1 | Phase 2: Baseline Model | Pending |
| FR2.2 | Phase 2: Baseline Model | Pending |
| FR2.3 | Phase 4: Architecture Enhancement | Pending |
| FR2.4 | Phase 2: Baseline Model | Pending |
| FR2.5 | Phase 2: Baseline Model | Pending |
| FR2.6 | Phase 2: Baseline Model | Pending |
| FR2.7 | Phase 2: Baseline Model | Pending |
| FR2.8 | Phase 2: Baseline Model | Pending |
| FR2.9 | Phase 2: Baseline Model | Pending |
| FR2.10 | Phase 4: Architecture Enhancement | Pending |
| FR3.1 | Phase 2: Baseline Model | Pending |
| FR3.2 | Phase 2: Baseline Model | Pending |
| FR3.3 | Phase 2: Baseline Model | Pending |
| FR3.4 | Phase 2: Baseline Model | Pending |
| FR3.5 | Phase 2: Baseline Model | Pending |
| FR3.6 | Phase 2: Baseline Model | Pending |
| FR3.7 | Phase 2: Baseline Model | Pending |
| FR3.8 | Phase 2: Baseline Model | Pending |
| FR3.9 | Phase 2: Baseline Model | Pending |
| FR3.10 | Phase 2: Baseline Model | Pending |
| FR4.1 | Phase 2: Baseline Model | Pending |
| FR4.2 | Phase 2: Baseline Model | Pending |
| FR4.3 | Phase 3: SAR Evaluation | Pending |
| FR4.4 | Phase 3: SAR Evaluation | Pending |
| FR4.5 | Phase 3: SAR Evaluation | Pending |
| FR4.6 | Phase 3: SAR Evaluation | Pending |
| FR4.7 | Phase 3: SAR Evaluation | Pending |
| FR4.8 | Phase 3: SAR Evaluation | Pending |
| FR4.9 | Phase 3: SAR Evaluation | Pending |
| FR4.10 | Phase 3: SAR Evaluation | Pending |
| FR5.1 | Phase 5: Full Image Inference | Pending |
| FR5.2 | Phase 5: Full Image Inference | Pending |
| FR5.3 | Phase 5: Full Image Inference | Pending |
| FR5.4 | Phase 5: Full Image Inference | Pending |
| FR5.5 | Phase 5: Full Image Inference | Pending |
| FR5.6 | Phase 5: Full Image Inference | Pending |
| FR6.1 | Phase 6: Final Experiments | Pending |
| FR6.2 | Phase 4: Architecture Enhancement | Pending |
| FR6.3 | Phase 4: Architecture Enhancement | Pending |
| FR6.4 | Phase 6: Final Experiments | Pending |
| FR6.5 | Phase 6: Final Experiments | Pending |
| FR6.6 | Phase 6: Final Experiments | Pending |

---

*Requirements derived from: PROJECT.md, research synthesis, knowledge documents, and existing codebase analysis.*
*Traceability added: 2026-01-21*
