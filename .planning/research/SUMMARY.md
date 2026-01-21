# Research Summary

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression
**Synthesized:** 2026-01-21
**Overall Confidence:** MEDIUM-HIGH

---

## Executive Summary

This project implements a learned compression system for Sentinel-1 SAR satellite imagery using a CNN-based autoencoder architecture. The research reveals that the existing technology stack (PyTorch 2.0+, pytorch-msssim, rasterio, TensorBoard) is well-suited and requires no major changes. The critical insight is that SAR compression differs fundamentally from natural image compression due to multiplicative speckle noise and extreme dynamic range (40-60 dB), requiring mandatory preprocessing (dB conversion, noise floor handling) and SAR-specific quality metrics (ENL ratio, Edge Preservation Index) beyond standard PSNR/SSIM.

The recommended architecture follows a progressive enhancement strategy: start with a plain 4-layer strided convolutional encoder-decoder, then add residual blocks for detail preservation (+2-3 dB PSNR expected), and finally integrate CBAM attention at the bottleneck for edge quality improvement. This approach fits comfortably within the RTX 3070's 8GB VRAM constraint with batch size 8. The target compression ratio should be conservative initially (16x with 16x16x16 latent) rather than aggressive, since SAR's speckle texture requires more latent capacity than natural images.

The most dangerous pitfalls are all related to preprocessing and evaluation: training on linear intensity values (causes complete model failure), inconsistent normalization parameters across train/val/test (corrupts all metrics silently), and using only PSNR/SSIM for evaluation (misses over-smoothing that destroys radiometric calibration). These must be addressed in the earliest phases before any meaningful training can occur.

---

## Stack Recommendations

### Core Technologies (Keep Existing)

| Technology | Version | Purpose | Decision |
|------------|---------|---------|----------|
| PyTorch | >=2.0.0 | Model development, training | **KEEP** - Excellent GPU support, torch.compile() for speedup |
| torchvision | >=0.15.0 | Image transforms, utilities | **KEEP** - Required companion to PyTorch |
| pytorch-msssim | >=1.0.0 | SSIM and MS-SSIM loss | **KEEP** - GPU-accelerated, autograd-compatible |
| rasterio | >=1.3.0 | GeoTIFF loading | **KEEP** - Standard for Sentinel-1 data |
| TensorBoard | >=2.12.0 | Training visualization | **KEEP** - Already configured |

### Libraries to Avoid

- **PyTorch Lightning:** Unnecessary refactoring for project size
- **CompressAI:** Out of scope, entropy coding not needed for v1
- **Custom CUDA kernels:** Standard ops are sufficient
- **Weights & Biases:** TensorBoard already adequate

### Optional Future Additions

- **Optuna:** Only if manual hyperparameter tuning proves insufficient
- **piqa:** Only if MS-SSIM alone proves insufficient for perceptual quality

### Hardware Optimization (RTX 3070, 8GB)

- Batch size 8-16 fits comfortably
- Keep tensor dimensions as multiples of 8 for Tensor Core utilization
- Mixed precision (BFloat16) available if VRAM becomes constrained

**Confidence:** HIGH - Stack is proven and already in use

---

## Feature Priorities

### Table Stakes (Must Have for MVP)

| Feature | Complexity | Rationale |
|---------|------------|-----------|
| dB transform (linear to log) | Low | SAR dynamic range incompatible with neural networks without this |
| Invalid value handling (noise floor) | Low | log(0) = -inf crashes training |
| Dynamic range clipping | Low | Outliers destabilize training |
| Normalization to [0,1] | Low | NN weight initialization assumes bounded inputs |
| Patch extraction (256x256) | Low | Full SAR images exceed GPU memory |
| Quality filtering | Low | Remove no-data and corrupted patches |
| CNN encoder/decoder | Medium | Core functionality |
| Configurable latent dimensions | Low | Control compression ratio |
| MSE + SSIM combined loss | Low | Balanced pixel accuracy and structure |
| PSNR, SSIM metrics | Low | Standard quality measurement |
| Checkpointing | Low | Resume training, save best models |
| TensorBoard logging | Low | Training visibility |

### Should Have (Quality Gate)

| Feature | Complexity | Rationale |
|---------|------------|-----------|
| Residual/skip connections | Medium | +2-3 dB PSNR improvement |
| ENL ratio metric | Medium | SAR-specific speckle validation |
| EPI (Edge Preservation Index) | Medium | SAR-specific edge validation |
| Patch-based inference with tiling | Medium | Process full satellite images |
| Patch blending (cosine ramp) | Medium | Avoid visible tile seams |

### Nice to Have (Differentiators for v1)

| Feature | Complexity | Rationale |
|---------|------------|-----------|
| CBAM attention at bottleneck | Medium | Edge quality improvement |
| Multiple compression ratio variants | Low | Comparison study |
| Rate-distortion curves | Medium | Professional evaluation |

### Defer to Post-MVP (Anti-Features for v1)

| Feature | Why Defer |
|---------|-----------|
| End-to-end entropy coding | Massive complexity, marginal gains for research |
| Variable-rate single model | High complexity |
| Multi-polarization (VV+VH) | Doubles complexity, prove concept with single channel first |
| Phase preservation | Rarely needed, huge complexity |
| Real-time streaming | Different architecture requirements |
| ONNX/TensorRT deployment | Optimize after quality is proven |

**Confidence:** HIGH - Based on comprehensive project knowledge documents

---

## Architecture Decisions

### Recommended Architecture Progression

1. **Baseline (Variant A):** Plain 4-layer strided convolutions
   - Parameters: ~3-5M
   - Memory (batch 8): ~2GB
   - Purpose: Establish baseline, fast debugging

2. **Enhanced (Variant B):** Add residual blocks at each resolution level
   - Parameters: ~5-8M
   - Memory (batch 8): ~3GB
   - Expected: +2-3 dB PSNR over baseline

3. **Recommended (Variant C):** Residual + CBAM attention at bottleneck
   - Parameters: ~6-9M
   - Memory (batch 8): ~3.5GB
   - Expected: +1-2 dB PSNR over Variant B
   - Best quality/cost tradeoff for 8GB VRAM

### Key Architecture Parameters

| Parameter | Recommendation | Rationale |
|-----------|----------------|-----------|
| Kernel size | 5x5 | Better receptive field for SAR structures |
| Stride | 2 at each layer | 4 layers: 256x256 -> 16x16 |
| Encoder activation | LeakyReLU(0.2) | Prevents dead neurons |
| Decoder activation | ReLU | Standard choice |
| Output activation | Sigmoid | Bounds output to [0,1] for normalized input |
| Normalization | BatchNorm | Training stability (consider GroupNorm for inference) |
| Final encoder layer | No activation | Allow unbounded latent values |

### Compression Ratio Targets

| Latent Channels | Latent Size | Compression Ratio | Use Case |
|-----------------|-------------|-------------------|----------|
| 64 | 16x16x64 | 4x | High quality baseline |
| 32 | 16x16x32 | 8x | Good quality |
| **16** | **16x16x16** | **16x** | **Balanced (start here)** |
| 8 | 16x16x8 | 32x | High compression |
| 4 | 16x16x4 | 64x | Extreme (risky for SAR) |

**Recommendation:** Start with 16x compression (16 latent channels). Only increase compression after validating quality at conservative setting.

**Confidence:** HIGH for structure, MEDIUM for expected PSNR improvements (need experimental validation)

---

## Critical Pitfalls

### Severity: CRITICAL (Must Address in Phase 1)

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| **Training on linear intensity values** | Model learns nothing useful; focuses only on brightest pixels | Always convert to dB before training: `dB = 10 * log10(intensity + 1e-10)` |
| **Inconsistent preprocessing parameters** | Silent metric corruption; val/test metrics meaningless | Compute vmin/vmax from training set ONLY; save with checkpoint |
| **Not handling invalid values** | Training crashes with NaN | Apply noise floor before dB: `max(intensity, 1e-10)`; filter >1% invalid patches |
| **Missing SAR-specific metrics** | Over-smoothing goes undetected | Always report ENL ratio (target ~1.0) and EPI (target >0.9) |
| **MSE-only loss** | Model becomes denoiser, destroys speckle | Use balanced loss: 0.5*MSE + 0.5*SSIM (tune based on ENL ratio) |

### Severity: MAJOR (Address in Early Phases)

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| Latent space too small | Cannot represent SAR detail | Start with 16x compression, not 32x or 64x |
| BatchNorm inference issues | Inconsistent deployment | Call model.eval(); consider GroupNorm; test single-sample inference |
| Output activation mismatch | Clipping or unbounded outputs | Sigmoid for [0,1] normalized input |
| Memory overflow on full images | Cannot process real data | Implement tiled inference with overlap and cosine blending |
| Checkpoint incompatibility | Lost training progress | Save config dict and preprocessing params with model state |

### Severity: MODERATE (Address During Training)

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| Learning rate too high | Training instability | Start with lr=1e-4, use warmup and gradient clipping |
| No latent space monitoring | Undetected capacity issues | Log mean, std, dead channels every N epochs |
| Evaluating on same-region data | Overfitting goes undetected | Geographic train/val/test splits |
| Inappropriate augmentations | Broken speckle statistics | Use only flip and crop; avoid vertical flip and elastic deformation |

**Confidence:** MEDIUM-HIGH - Based on established SAR processing principles and common failure modes

---

## Implementation Roadmap Implications

Based on the research synthesis, the roadmap should follow this phase structure:

### Phase 1: Data Pipeline (Highest Priority)

**Rationale:** Cannot train any model without correct preprocessing. Critical pitfalls #1, #2, #5 must be addressed here.

**Features:**
- dB transform with noise floor handling
- Dynamic range clipping (percentile-based)
- Normalization to [0,1] with training-set bounds
- Patch extraction (256x256)
- Quality filtering (variance threshold, invalid ratio)
- Save preprocessing parameters to checkpoint

**Pitfalls to avoid:** #1 (linear values), #2 (inconsistent params), #5 (invalid values), #12 (bad augmentation)

**Research flag:** Standard patterns, no additional research needed

---

### Phase 2: Baseline Model

**Rationale:** Establish working end-to-end pipeline before adding complexity.

**Features:**
- Plain 4-layer encoder/decoder (ConvBlock, DeconvBlock)
- Configurable latent channels
- MSE + SSIM combined loss (balanced 0.5/0.5)
- Basic training loop with checkpointing
- PSNR, SSIM, compression ratio metrics
- TensorBoard scalar and image logging

**Pitfalls to avoid:** #4 (MSE-only), #8 (output activation), #10 (checkpoint compat), #13 (high LR), #15 (best model saving)

**Research flag:** Standard patterns, no additional research needed

---

### Phase 3: SAR Evaluation Framework

**Rationale:** Cannot make architecture decisions without SAR-specific quality assessment.

**Features:**
- ENL ratio computation (requires homogeneous region detection)
- Edge Preservation Index (EPI)
- Rate-distortion curve generation
- Geographic train/val/test split validation
- Visual comparison with difference images

**Pitfalls to avoid:** #3 (missing SAR metrics), #14 (same-region eval)

**Research flag:** May need phase research for ENL computation on homogeneous regions

---

### Phase 4: Architecture Enhancement

**Rationale:** Add residual blocks and attention after baseline is validated.

**Features:**
- Residual blocks (ResidualBlock with projection shortcut)
- CBAM attention at bottleneck
- Architecture variant comparison (Plain vs Residual vs Res+CBAM)
- Multiple compression ratio experiments (8x, 16x, 32x)

**Pitfalls to avoid:** #6 (small latent), #7 (BatchNorm issues), #11 (latent collapse)

**Research flag:** Standard patterns from ARCHITECTURE.md, no additional research needed

---

### Phase 5: Full Image Inference

**Rationale:** Required for real-world deployment on Sentinel-1 scenes.

**Features:**
- Tiled inference with overlap
- Cosine ramp blending weights
- Memory-efficient processing
- Inverse preprocessing (restore linear SAR values)
- Metadata preservation

**Pitfalls to avoid:** #7 (BatchNorm eval mode), #9 (memory/tiling)

**Research flag:** Standard patterns, may need research if seam artifacts persist

---

### Phase 6: Final Experiments and Documentation

**Rationale:** Comprehensive comparison study with all variants and compression levels.

**Features:**
- Full experiment matrix (architectures x compression ratios)
- Rate-distortion curves
- Statistical analysis of results
- Final model selection
- Documentation and reproducibility

**Pitfalls to avoid:** All moderate/minor pitfalls should be resolved by this phase

**Research flag:** No research needed, execution phase

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing stack is well-suited, no changes needed |
| Features | HIGH | Based on comprehensive project knowledge documents |
| Architecture | HIGH for patterns, MEDIUM for expected gains | Need experimental validation of PSNR improvements |
| Pitfalls | MEDIUM-HIGH | Based on established SAR principles, WebSearch unavailable for verification |

### Gaps to Address During Planning

1. **ENL computation methodology:** How to identify homogeneous regions automatically for ENL ratio calculation
2. **Expected PSNR improvements:** Architecture gain estimates need experimental validation
3. **Geographic split strategy:** Need to define which Sentinel-1 scenes go in train/val/test
4. **Optimal loss weights:** Starting point is 0.5/0.5 MSE/SSIM but may need tuning based on ENL monitoring

### Research Flags by Phase

| Phase | Research Needed |
|-------|-----------------|
| Phase 1: Data Pipeline | No - well-documented in knowledge base |
| Phase 2: Baseline Model | No - standard patterns |
| Phase 3: SAR Evaluation | Maybe - ENL homogeneous region detection |
| Phase 4: Architecture | No - comprehensive ARCHITECTURE.md |
| Phase 5: Full Image Inference | Maybe - if seam artifacts persist |
| Phase 6: Final Experiments | No - execution only |

---

## Sources

**Project Knowledge (HIGH confidence):**
- `.planning/knowledge/05_SAR_PREPROCESSING.md`
- `.planning/knowledge/06_SAR_QUALITY_METRICS.md`
- `.planning/knowledge/07_COMPRESSION_TRADEOFFS.md`
- `.planning/PROJECT.md`
- Existing codebase (src/models, src/evaluation)

**Domain Knowledge (MEDIUM confidence):**
- Learned image compression patterns
- SAR imaging fundamentals
- CNN architecture best practices

**Note:** WebSearch was unavailable during research. State-of-the-art comparison with published methods (Balle 2018, Cheng 2020) would strengthen recommendations.

---

*Research synthesis completed: 2026-01-21*
