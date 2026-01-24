# Project State

## Project Reference

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression

**Core Value:** Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis.

**Current Focus:** Phase 4 - Architecture Improvements (Plan 2 of 3 complete)

---

## Current Position

**Phase:** 4 of 7 (Architecture Improvements)
**Plan:** 2 of 3 complete
**Status:** In progress

**Progress:**
```
Phase 1: Data Pipeline      [##########] 100%
Phase 2: Baseline Model     [##########] 100%
Phase 3: SAR Evaluation     [##########] 100%
Phase 4: Architecture       [######----] 67%    <- IN PROGRESS
Phase 5: Full Inference     [----------] 0%
Phase 6: Final Experiments  [----------] 0%
Phase 7: Deployment         [----------] 0%
```

---

## Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| PSNR @ 16x | >25 dB | 21.2 dB | Below target (expected at 16x) |
| SSIM @ 16x | >0.85 | 0.726 | Below target |
| ENL ratio | 0.8-1.2 | Ready | Evaluation pipeline ready |
| EPI | >0.85 | Ready | Evaluation pipeline ready |

### Training Results

| Model | Params | Best Loss | Best PSNR | Best SSIM |
|-------|--------|-----------|-----------|-----------|
| Baseline | 2.3M | 0.1813 | 20.47 dB | 0.646 |
| ResNet-Lite v1 | 5.6M | 0.1415 | 21.24 dB | 0.725 |
| **ResNet-Lite v2** | 5.6M | **0.1410** | **21.20 dB** | **0.726** |
| ResidualAutoencoder (new) | 23.8M | -- | -- | -- |

**Best Checkpoint:** `notebooks/checkpoints/resnet_lite_v2_c16/best.pth`

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
| ResNet-Lite over full ResNet | 5.6M params sufficient, 22M unnecessary | +0.77 dB over baseline |
| U-Net abandoned | Skip connections bypass bottleneck | Not suitable for compression |
| Accept 21 dB at 16x | Within expected range for SAR at 16x | Proceed to evaluation |
| EPI as correlation not ratio | More robust, bounded output [0, 1] | Implemented in metrics.py |
| WebP codec excluded | JPEG-2000+JPEG sufficient per FR4.11 | Implemented in codec_baselines.py |
| compute_all_metrics unified | Consistent metrics across autoencoder/codec | Evaluator uses single call |
| JSON split summary/detailed | Summary for review, detailed for analysis | save_results() pattern |
| R-D format standardized | name, bpp, psnr, ssim for all methods | Enables unified plotting |
| Pre-activation residual blocks | Cleaner gradient flow for deeper networks | Implemented in blocks.py |
| CBAM 1x1 conv MLP | More efficient than Linear for attention | Implemented in blocks.py |
| Bilinear upsample for PreActUp | Cleaner than transposed conv | Implemented in blocks.py |
| 2 blocks per stage for ResidualAutoencoder | Deeper architecture for better quality | 23.8M params, 16 residual blocks total |

### Technical Notes

- **Data pipeline:** Complete - SARDataModule delivers (32, 1, 256, 256) batches to GPU
- **Dataset:** 696,277 patches across 44 .npy files (182GB), lazy loaded via mmap
- **Preprocessing params:** vmin=14.7688, vmax=24.5407
- **Hardware:** RTX 3070 with 8GB VRAM, batch_size=32 with AMP
- **Best model:** ResNet-Lite (5.6M params) with residual blocks
- **Compression:** 16x (256x256x1 -> 16x16x16 latent)
- **Training:** 30 epochs, ~10 hours, 20% data subset
- **NaN fix:** float32 cast + batch skipping in trainer.py
- **Metrics module:** 872 lines, 11 metric functions implemented
- **Codec baselines:** JPEG-2000, JPEG with binary search calibration
- **Evaluation pipeline:** 738 lines evaluator, 1099 lines visualizer, 396 lines CLI script
- **Building blocks:** PreActResidualBlock, PreActResidualBlockDown, PreActResidualBlockUp, CBAM ready
- **ResidualAutoencoder:** 23.8M params, uses 4GB VRAM at batch=8

### Blockers

None currently.

### TODOs (Deferred Items)

- Consider 8x compression variant if 16x insufficient for downstream tasks
- Full dataset training (currently using 20% subset)
- Run full evaluation with real SAR data (pipeline ready)
- Train ResidualAutoencoder to compare with ResNet-Lite

---

## Session Continuity

### Last Session

- **Date:** 2026-01-24
- **Activity:** Phase 4 Plan 02 completed (ResidualAutoencoder)
- **Outcome:**
  - PreActResidualEncoder: 4 stages, 2 blocks each (8 total)
  - PreActResidualDecoder: mirrors encoder with bilinear upsample
  - ResidualAutoencoder wrapper with same interface as existing models
  - Exported from src.models package
  - 23.8M parameters, fits in 8GB VRAM

### Next Session

- **Priority:** Execute Phase 4 Plan 03 (Attention Autoencoder - Variant C)
- **Command:** `/gsd:execute-plan 04-03`
- **Context needed:**
  - ResidualAutoencoder available as baseline for attention variant
  - CBAM attention module ready from 04-01
  - Will integrate attention into residual architecture

---

## Quick Reference

**Key Files:**
- Project definition: `.planning/PROJECT.md`
- Requirements: `.planning/REQUIREMENTS.md`
- Research: `.planning/research/SUMMARY.md`
- Roadmap: `.planning/ROADMAP.md`
- Phase 2 Training Summary: `.planning/phases/02-baseline-model/02-04-SUMMARY.md`
- Phase 3 Plan 03 Summary: `.planning/phases/03-sar-evaluation/03-03-SUMMARY.md`
- Phase 4 Plan 01 Summary: `.planning/phases/04-architecture/04-01-SUMMARY.md`
- Phase 4 Plan 02 Summary: `.planning/phases/04-architecture/04-02-SUMMARY.md`

**Codebase Entry Points:**
- Preprocessing: `src/data/preprocessing.py`
- Dataset classes: `src/data/dataset.py`
- DataModule: `src/data/datamodule.py`
- Models: `src/models/` (SARAutoencoder, ResNetAutoencoder, ResidualAutoencoder)
- Building blocks: `src/models/blocks.py` (includes PreActResidual*, CBAM)
- Training: `src/training/trainer.py`
- Evaluation metrics: `src/evaluation/metrics.py`
- Codec baselines: `src/evaluation/codec_baselines.py`
- Evaluator: `src/evaluation/evaluator.py`
- Visualizer: `src/evaluation/visualizer.py`
- CLI evaluation: `scripts/evaluate_model.py`

**Checkpoints:**
- Baseline: `notebooks/checkpoints/baseline_c16_fast/best.pth`
- ResNet-Lite v2 (best): `notebooks/checkpoints/resnet_lite_v2_c16/best.pth`

**Evaluation:**
- Run: `python scripts/evaluate_model.py --checkpoint path/to/model.pth`
- With codecs: `python scripts/evaluate_model.py --checkpoint path/to/model.pth --compare-codecs`
- Output: `evaluations/{model_name}/` with JSON and visualizations

---

*State updated: 2026-01-24 (04-02 complete)*
