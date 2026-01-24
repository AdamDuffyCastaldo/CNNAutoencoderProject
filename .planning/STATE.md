# Project State

## Project Reference

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression

**Core Value:** Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis.

**Current Focus:** Phase 2 Complete - Ready for Phase 3 (SAR Evaluation)

---

## Current Position

**Phase:** 2 of 7 (Baseline Model) - COMPLETE
**Plan:** 4 of 4 complete
**Status:** Ready for Phase 3

**Progress:**
```
Phase 1: Data Pipeline      [##########] 100%
Phase 2: Baseline Model     [##########] 100%  <- COMPLETE
Phase 3: SAR Evaluation     [----------] 0%   <- NEXT
Phase 4: Architecture       [----------] 0%
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
| ENL ratio | 0.8-1.2 | - | Phase 3 |
| EPI | >0.85 | - | Phase 3 |

### Training Results

| Model | Params | Best Loss | Best PSNR | Best SSIM |
|-------|--------|-----------|-----------|-----------|
| Baseline | 2.3M | 0.1813 | 20.47 dB | 0.646 |
| ResNet-Lite v1 | 5.6M | 0.1415 | 21.24 dB | 0.725 |
| **ResNet-Lite v2** | 5.6M | **0.1410** | **21.20 dB** | **0.726** |

**Best Checkpoint:** `notebooks/checkpoints/resnet_lite_v2_c16/best.pth`

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

### Technical Notes

- **Data pipeline:** Complete - SARDataModule delivers (32, 1, 256, 256) batches to GPU
- **Dataset:** 696,277 patches across 44 .npy files (182GB), lazy loaded via mmap
- **Preprocessing params:** vmin=14.7688, vmax=24.5407
- **Hardware:** RTX 3070 with 8GB VRAM, batch_size=32 with AMP
- **Best model:** ResNet-Lite (5.6M params) with residual blocks
- **Compression:** 16x (256x256x1 → 16x16x16 latent)
- **Training:** 30 epochs, ~10 hours, 20% data subset
- **NaN fix:** float32 cast + batch skipping in trainer.py

### Blockers

None currently.

### TODOs (Deferred Items)

- Consider 8x compression variant if 16x insufficient for downstream tasks
- Full dataset training (currently using 20% subset)

---

## Session Continuity

### Last Session

- **Date:** 2026-01-24
- **Activity:** Phase 2 Plan 04 completed (Training)
- **Outcome:**
  - Baseline, ResNet-Lite v1, v2 trained
  - Best: 21.2 dB PSNR, 0.726 SSIM at 16x compression
  - U-Net abandoned (skip connections bypass bottleneck)
  - NaN stability fix applied to trainer

### Next Session

- **Priority:** Plan and execute Phase 3 (SAR Evaluation)
- **Command:** `/gsd:plan-phase 3`
- **Context needed:**
  - Best checkpoint at `notebooks/checkpoints/resnet_lite_v2_c16/best.pth`
  - Need SAR-specific metrics: ENL, EPI
  - JPEG-2000 comparison at 16x

---

## Quick Reference

**Key Files:**
- Project definition: `.planning/PROJECT.md`
- Requirements: `.planning/REQUIREMENTS.md`
- Research: `.planning/research/SUMMARY.md`
- Roadmap: `.planning/ROADMAP.md`
- Phase 2 Training Summary: `.planning/phases/02-baseline-model/02-04-SUMMARY.md`

**Codebase Entry Points:**
- Preprocessing: `src/data/preprocessing.py`
- Dataset classes: `src/data/dataset.py`
- DataModule: `src/data/datamodule.py`
- Models: `src/models/` (SARAutoencoder, ResNetAutoencoder)
- Training: `src/training/trainer.py`
- Evaluation: `src/evaluation/` (to be implemented in Phase 3)

**Checkpoints:**
- Baseline: `notebooks/checkpoints/baseline_c16_fast/best.pth`
- ResNet-Lite v2 (best): `notebooks/checkpoints/resnet_lite_v2_c16/best.pth`

---

*State updated: 2026-01-24 (Phase 2 complete)*
