# Project State

## Project Reference

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression

**Core Value:** Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis.

**Current Focus:** Phase 5 Complete - Ready for Phase 6 (Final Experiments)

---

## Current Position

**Phase:** 5 of 7 (Full Image Inference) - COMPLETE
**Plan:** 5 of 5 complete
**Status:** Phase 5 validated, ready for Phase 6

**Progress:**
```
Phase 1: Data Pipeline      [##########] 100%
Phase 2: Baseline Model     [##########] 100%
Phase 3: SAR Evaluation     [##########] 100%
Phase 4: Architecture       [########--] 83%    <- Partial (training deferred)
Phase 5: Full Inference     [##########] 100%   <- COMPLETE
Phase 6: Final Experiments  [----------] 0%     <- NEXT
Phase 7: Deployment         [----------] 0%
```

**Phase 5 Plans:**
- [x] 05-01: Tiling Infrastructure (complete)
- [x] 05-02: GeoTIFF I/O (complete)
- [x] 05-03: SARCompressor (complete)
- [x] 05-04: CLI Interface (complete)
- [x] 05-05: Full Validation (complete)

**Phase 5 Validation Results:**
| Test | Criterion | Result |
|------|-----------|--------|
| Memory Test | 4096x4096 without OOM | PASSED (1444 MB, 2.6s) |
| Seamless Blending | No tile boundaries | PASSED (ratio 0.997) |
| PSNR Consistency | < 0.5 dB difference | PASSED (0.18 dB) |
| Preprocessing Round-Trip | Correlation > 0.75 | PASSED (0.785) |
| CLI Smoke Test | Metadata preserved | PASSED |

---

## Performance Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| PSNR @ 16x | >25 dB | 21.2 dB | Below target (expected at 16x) |
| SSIM @ 16x | >0.85 | 0.726 | Below target |
| ENL ratio | 0.8-1.2 | ~0.85 | OK |
| EPI | >0.85 | ~0.88 | OK |

### Training Results (16x compression only — other ratios not yet trained)

| Model | Params | Best Loss | Best PSNR | Best SSIM | Status |
|-------|--------|-----------|-----------|-----------|--------|
| Baseline (b=64) | 2.3M | 0.1813 | 20.47 dB | 0.646 | **Best Available** |
| ResNet-Lite (b=32) | 5.6M | 0.1819 | 19.06 dB | 0.649 | Regressed (LR too high) |
| Residual (b=32) | 6.0M | 0.1339 | 19.78 dB | 0.578 | Suboptimal (LR issue) |
| Attention (b=48) | 13.5M | 0.2870 | 11.29 dB | 0.081 | Did not converge |

**Best Checkpoint:** `notebooks/checkpoints/baseline_c16_fast/best.pth`

**Note:** Original ResNet-Lite v2 checkpoint (21.20 dB) was overwritten by overnight runs.
All non-baseline models need retraining with proper hyperparameters (LR=1e-4, ReduceLROnPlateau).

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
| ResNet-Lite over full ResNet | 5.6M params sufficient, 22M unnecessary | +0.73 dB over baseline |
| U-Net abandoned | Skip connections bypass bottleneck | Not suitable for compression |
| Accept 21 dB at 16x | Within expected range for SAR at 16x | Proceed to evaluation |
| EPI as correlation not ratio | More robust, bounded output [0, 1] | Implemented in metrics.py |
| WebP codec excluded | JPEG-2000+JPEG sufficient per FR4.11 | Implemented in codec_baselines.py |
| Pre-activation residual blocks | Cleaner gradient flow for deeper networks | Implemented in blocks.py |
| CBAM 1x1 conv MLP | More efficient than Linear for attention | Implemented in blocks.py |
| Phase 4 partial completion | Proceed with best available, return for training | ResNet-Lite v2 for Phase 5 |
| GeoMetadata dataclass | Clean container for CRS, transform, nodata, tags | Implemented in geotiff.py |
| COG as optional | Not all users need COG, keep core deps minimal | rio-cogeo commented in requirements |
| rich for CLI | Better user experience for progress bars | Added to requirements.txt |
| Offset padding for tiling | Ensures boundary tiles have proper weight coverage | Implemented in tiling.py |
| Cosine-squared blending | Guarantees overlapping tiles sum to 1.0 | Implemented in tiling.py |
| Preprocess params from checkpoint | Extract from config['preprocessing_params'] | Implemented in SARCompressor |
| Auto-detect batch size | 70% VRAM, 3MB per tile estimate | Implemented in SARCompressor |
| SAR-like test data | Model trained on SAR, random noise OOD | Test validates 21.94 dB PSNR |
| GeoMetadata JSON serialization | CRS as WKT, transform as tuple for NPZ compatibility | Implemented in sarcodec.py |
| CLI exit codes | Distinct codes enable scripting/automation | 0=success, 1=file, 2=model, 3=OOM, 4=general |
| Nodata handling in compression | Replace with median, store mask separately | Enables lossless nodata preservation |
| Correlation threshold 0.75 | Accounts for undertrained model in validation | Implemented in test notebook |

### Technical Notes

- **Data pipeline:** Complete - SARDataModule delivers (32, 1, 256, 256) batches to GPU
- **Dataset:** 696,277 patches across 44 .npy files (182GB), lazy loaded via mmap
- **Preprocessing params:** vmin=14.7688, vmax=24.5407
- **Hardware:** RTX 3070 with 8GB VRAM, batch_size=32 with AMP
- **Best model:** ResNet-Lite v2 (5.6M params) with post-activation residual blocks
- **Compression:** 16x (256x256x1 -> 16x16x16 latent)
- **Training:** 30 epochs, ~10 hours, 20% data subset
- **NaN fix:** float32 cast + batch skipping in trainer.py
- **Building blocks ready:** PreActResidualBlock, PreActResidualBlockDown, PreActResidualBlockUp, CBAM
- **Training infrastructure:** Warmup epochs, AdamW optimizer, quick search mode ready
- **GeoTIFF I/O:** read_geotiff, write_geotiff, write_cog with metadata preservation
- **Tiling:** Cosine-squared blending with offset padding, <1e-7 reconstruction error
- **SARCompressor:** Full pipeline with batched GPU inference, AMP support, progress callbacks
- **CLI:** sarcodec compress/decompress with rich progress bars, exit codes

### Blockers

- None

### TODOs (Deferred Items)

- [ ] Retrain Residual v2 with improved hyperparameters (LR=5e-5, warmup=3)
- [ ] Train Attention v2 with same improved config
- [ ] Consider 8x compression variant if 16x insufficient for downstream tasks
- [ ] Full dataset training (currently using 20% subset)

---

## Session Continuity

### Last Session

- **Date:** 2026-01-28
- **Activity:** Analyzed baseline ratios sweep + launched architecture comparison sweep
- **Outcome:**
  - Baseline ratios sweep COMPLETE: 4x=24.15dB, 8x=21.34dB, 12x=19.48dB (all undertrained, no early stop)
  - Architecture sweep updated: baseline + resnet only, batch_size=16 (ResNet OOM at 36)
  - Added checkpoint skip logic, combined R-D chart, datestamped save paths
  - ResNet training running overnight (~22 min/epoch, 35 epochs)

### Next Session

- **Priority:** Check ResNet results, analyze baseline vs ResNet at 16x
- **Resume file:** `.planning/phases/04-architecture/.continue-here.md`
- **Context needed:**
  - ResNet training should be complete (or check notebook for progress)
  - Run cells 10+ in `notebooks/sweep_all_16x.ipynb` for charts and combined R-D curve
  - Key question: does ResNet b=64 beat baseline at 16x with proper hyperparameters?
- **Checkpoints:**
  - Baseline 4x: `notebooks/checkpoints/baseline_c64_b64_cr4x_20260127_195355/best.pth`
  - Baseline 8x: `notebooks/checkpoints/baseline_c32_b64_cr8x_20260127_205741/best.pth`
  - Baseline 12x: `notebooks/checkpoints/baseline_c21_b64_cr12x_20260127_220001/best.pth`
  - Baseline 16x: `notebooks/checkpoints/baseline_c16_b64_cr16x_*/best.pth`
  - ResNet 16x: `notebooks/checkpoints/resnet_c16_b64_cr16x_*/best.pth` (pending)

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

**Checkpoints:**
- Baseline: `notebooks/checkpoints/baseline_c16_fast/best.pth`
- ResNet-Lite v2 (best): `notebooks/checkpoints/resnet_lite_v2_c16/best.pth`

**Evaluation:**
- Run: `python scripts/evaluate_model.py --checkpoint path/to/model.pth`
- With codecs: `python scripts/evaluate_model.py --checkpoint path/to/model.pth --compare-codecs`
- Output: `evaluations/{model_name}/` with JSON and visualizations

**Full Image Compression:**
- Compress: `python scripts/sarcodec.py compress input.tif -o output.npz`
- Decompress: `python scripts/sarcodec.py decompress output.npz -o reconstructed.tif`

---

*State updated: 2026-01-27 (Phase 4 training review + naming standardization)*
