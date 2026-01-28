# Project State

## Project Reference

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression

**Core Value:** Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis.

**Current Focus:** Phase 4 Complete - Ready for Phase 6 (Final Experiments)

---

## Current Position

**Phase:** 4 of 7 (Architecture Exploration) - COMPLETE
**Plan:** Architecture comparison sweep complete
**Status:** ResNet selected as best architecture, ready for Phase 6

**Progress:**
```
Phase 1: Data Pipeline      [##########] 100%
Phase 2: Baseline Model     [##########] 100%
Phase 3: SAR Evaluation     [##########] 100%
Phase 4: Architecture       [##########] 100%   <- COMPLETE (ResNet selected)
Phase 5: Full Inference     [##########] 100%
Phase 6: Final Experiments  [----------] 0%     <- NEXT
Phase 7: Deployment         [----------] 0%
```

**Phase 4 Final Results:**
- [x] Architecture implementations (baseline, resnet, residual, attention)
- [x] Baseline ratios sweep (4x, 8x, 12x)
- [x] Architecture comparison @ 16x (baseline vs ResNet)
- [x] Architecture selection: **ResNet b=64**

---

## Performance Metrics

| Metric | Target | Current (ResNet) | Status |
|--------|--------|------------------|--------|
| PSNR @ 16x | >25 dB | 21.13 dB | Below target (expected at 16x) |
| SSIM @ 16x | >0.85 | 0.739 | Below target |
| ENL ratio | 0.8-1.2 | ~0.85 | OK |
| EPI | >0.85 | ~0.88 | OK |

### Architecture Comparison @ 16x (Final)

| Model | Params | Val Loss | Val PSNR | Val SSIM | Status |
|-------|--------|----------|----------|----------|--------|
| Baseline (b=64) | 2.3M | 0.2201 | 19.09 dB | 0.572 | Undertrained |
| **ResNet (b=64)** | **22.4M** | **0.1342** | **21.13 dB** | **0.739** | **SELECTED** |

**Winner: ResNet b=64** — +2.04 dB PSNR, +29% SSIM over baseline at 16x.

### Rate-Distortion Curve (Baseline)

| Ratio | Latent Ch | PSNR | SSIM | Notes |
|-------|-----------|------|------|-------|
| 4x | 64 | 24.15 dB | 0.855 | Meets SSIM target |
| 8x | 32 | 21.34 dB | 0.675 | Similar to ResNet@16x |
| 12x | 21 | 19.48 dB | 0.595 | Below targets |
| 16x | 16 | 19.09 dB | 0.572 | Below targets |

**Key insight:** ResNet @16x (21.13 dB) achieves similar quality to Baseline @8x (21.34 dB).

### Best Checkpoints

| Model | Checkpoint | PSNR | SSIM |
|-------|------------|------|------|
| **ResNet 16x** | `notebooks/checkpoints/resnet_c16_b64_cr16x_20260128_003926/best.pth` | 21.13 dB | 0.739 |
| Baseline 4x | `notebooks/checkpoints/baseline_c64_b64_cr4x_20260127_195355/best.pth` | 24.15 dB | 0.855 |
| Baseline 8x | `notebooks/checkpoints/baseline_c32_b64_cr8x_20260127_205741/best.pth` | 21.34 dB | 0.675 |
| Baseline 12x | `notebooks/checkpoints/baseline_c21_b64_cr12x_20260127_220001/best.pth` | 19.48 dB | 0.595 |
| Baseline 16x | `notebooks/checkpoints/baseline_c16_b64_cr16x_20260127_231730/best.pth` | 19.09 dB | 0.572 |

### Codec Baselines (Random Noise Test @ 16x)

| Codec | PSNR | SSIM | Notes |
|-------|------|------|-------|
| JPEG-2000 | 18.6 dB | 0.914 | Wavelet-based, best traditional |
| JPEG | 17.2 dB | 0.886 | DCT-based, blocking artifacts |

---

## Accumulated Context

### Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 7-phase structure | Derived from requirements + deployment needs | Roadmap created |
| Start at 16x compression | Conservative, recommended by research | Implemented - latent_channels=16 |
| Use existing preprocessing | preprocess_sar_complete() already working | Extended with utilities |
| batch_size=32 with AMP | AMP allows larger batches on 8GB VRAM | Implemented |
| Lazy loading as default | 182GB dataset too large for RAM | Implemented via LazyPatchDataset |
| pytorch-msssim for SSIM | GPU-optimized, well-tested | Implemented in SSIMLoss |
| 0.5/0.5 MSE/SSIM weights | Balanced weighting per CONTEXT.md | Default in CombinedLoss |
| U-Net abandoned | Skip connections bypass bottleneck | Not suitable for compression |
| EPI as correlation not ratio | More robust, bounded output [0, 1] | Implemented in metrics.py |
| WebP codec excluded | JPEG-2000+JPEG sufficient per FR4.11 | Implemented in codec_baselines.py |
| Pre-activation residual blocks | Cleaner gradient flow for deeper networks | Implemented in blocks.py |
| CBAM 1x1 conv MLP | More efficient than Linear for attention | Implemented in blocks.py |
| GeoMetadata dataclass | Clean container for CRS, transform, nodata, tags | Implemented in geotiff.py |
| COG as optional | Not all users need COG, keep core deps minimal | rio-cogeo commented in requirements |
| rich for CLI | Better user experience for progress bars | Added to requirements.txt |
| Offset padding for tiling | Ensures boundary tiles have proper weight coverage | Implemented in tiling.py |
| Cosine-squared blending | Guarantees overlapping tiles sum to 1.0 | Implemented in tiling.py |
| Preprocess params from checkpoint | Extract from config['preprocessing_params'] | Implemented in SARCompressor |
| Auto-detect batch size | 70% VRAM, 3MB per tile estimate | Implemented in SARCompressor |
| GeoMetadata JSON serialization | CRS as WKT, transform as tuple for NPZ compatibility | Implemented in sarcodec.py |
| CLI exit codes | Distinct codes enable scripting/automation | 0=success, 1=file, 2=model, 3=OOM, 4=general |
| Nodata handling in compression | Replace with median, store mask separately | Enables lossless nodata preservation |
| **ResNet b=64 for 16x** | +2 dB over baseline, proper hyperparameters work | **Selected as best architecture** |

### Technical Notes

- **Data pipeline:** Complete - SARDataModule delivers (32, 1, 256, 256) batches to GPU
- **Dataset:** 696,277 patches across 44 .npy files (182GB), lazy loaded via mmap
- **Preprocessing params:** vmin=14.7688, vmax=24.5407
- **Hardware:** RTX 3070 with 8GB VRAM, batch_size=16 for ResNet with AMP
- **Best model:** ResNet b=64 (22.4M params) — 21.13 dB @ 16x
- **Compression:** 16x (256x256x1 -> 16x16x16 latent)
- **Training:** 35 epochs, 10% data subset, LR=1e-4, AdamW, ReduceLROnPlateau
- **All models undertrained:** No early stopping triggered, still improving at epoch 35
- **Building blocks ready:** PreActResidualBlock, PreActResidualBlockDown, PreActResidualBlockUp, CBAM
- **GeoTIFF I/O:** read_geotiff, write_geotiff, write_cog with metadata preservation
- **Tiling:** Cosine-squared blending with offset padding, <1e-7 reconstruction error
- **SARCompressor:** Full pipeline with batched GPU inference, AMP support, progress callbacks
- **CLI:** sarcodec compress/decompress with rich progress bars, exit codes

### Blockers

- None

### TODOs (Deferred Items)

- [ ] Longer training run for ResNet (model still improving at epoch 35)
- [ ] Full dataset training (currently using 10% subset)
- [ ] Consider ResNet at other ratios (4x, 8x) if needed

---

## Session Continuity

### Last Session

- **Date:** 2026-01-28
- **Activity:** Analyzed architecture comparison sweep results
- **Outcome:**
  - ResNet training COMPLETE: 21.13 dB PSNR, 0.739 SSIM @ 16x
  - ResNet beats baseline by +2.04 dB (+29% SSIM)
  - Architecture selected: ResNet b=64
  - Phase 4 marked COMPLETE
  - Charts generated: `architecture_comparison_16x_20260128.png`, `combined_rate_distortion_20260128.png`

### Next Session

- **Priority:** Plan and execute Phase 6 (Final Experiments)
- **Context needed:**
  - Full evaluation with codec baselines on real SAR data
  - Rate-distortion comparison across compression methods
  - SAR-specific metrics evaluation (ENL, EPI)
- **Commands:**
  - `/gsd:plan-phase 6` to create Phase 6 execution plan
  - Or `/gsd:execute-phase 6` if plans already exist

---

## Quick Reference

**Key Files:**
- Project definition: `.planning/PROJECT.md`
- Requirements: `.planning/REQUIREMENTS.md`
- Research: `.planning/research/SUMMARY.md`
- Roadmap: `.planning/ROADMAP.md`
- Phase 4 Summaries: `.planning/phases/04-architecture/04-*-SUMMARY.md`
- Phase 5 Summaries: `.planning/phases/05-inference/05-*-SUMMARY.md`

**Codebase Entry Points:**
- Preprocessing: `src/data/preprocessing.py`
- Dataset classes: `src/data/dataset.py`
- DataModule: `src/data/datamodule.py`
- Models: `src/models/` (SARAutoencoder, ResNetAutoencoder, ResidualAutoencoder, AttentionAutoencoder)
- Building blocks: `src/models/blocks.py` (includes PreActResidual*, CBAM)
- Training: `src/training/trainer.py`
- Evaluation metrics: `src/evaluation/metrics.py`
- Codec baselines: `src/evaluation/codec_baselines.py`
- Evaluator: `src/evaluation/evaluator.py`
- Visualizer: `src/evaluation/visualizer.py`
- CLI evaluation: `scripts/evaluate_model.py`
- **SARCompressor: `src/inference/compressor.py`**
- **GeoTIFF I/O: `src/inference/geotiff.py`**
- **Tiling: `src/inference/tiling.py`**
- **CLI: `scripts/sarcodec.py`** (compress/decompress commands)

**Best Checkpoint:**
- **ResNet 16x:** `notebooks/checkpoints/resnet_c16_b64_cr16x_20260128_003926/best.pth`

**Evaluation:**
- Run: `python scripts/evaluate_model.py --checkpoint path/to/model.pth`
- With codecs: `python scripts/evaluate_model.py --checkpoint path/to/model.pth --compare-codecs`
- Output: `evaluations/{model_name}/` with JSON and visualizations

**Full Image Compression:**
- Compress: `python scripts/sarcodec.py compress input.tif -o output.npz`
- Decompress: `python scripts/sarcodec.py decompress output.npz -o reconstructed.tif`

---

*State updated: 2026-01-28 (Phase 4 complete — ResNet b=64 selected)*
