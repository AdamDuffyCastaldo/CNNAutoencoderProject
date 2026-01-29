# Project State

## Project Reference

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression

**Core Value:** Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis.

**Current Focus:** Phase 6 (Final Experiments) - Plan 02 complete, ready for report generation

---

## Current Position

**Phase:** 6 of 7 (Final Experiments) - IN PROGRESS
**Plan:** 02 of 03 complete (Systematic Evaluation)
**Status:** Ready for Plan 06-03 (Technical Report Generation)

**Progress:**
```
Phase 1: Data Pipeline      [##########] 100%
Phase 2: Baseline Model     [##########] 100%
Phase 3: SAR Evaluation     [##########] 100%
Phase 4: Architecture       [##########] 100%
Phase 5: Full Inference     [##########] 100%
Phase 6: Final Experiments  [######----] 67%    <- Plan 02 complete
Phase 7: Deployment         [----------] 0%
```

**Phase 6 Progress:**
- [x] Plan 01: Checkpoint verification and training setup
- [x] Plan 02: Comprehensive evaluation (9 models)
- [ ] Plan 03: Technical report generation

---

## Performance Metrics (Updated from Evaluation Sweep)

| Metric | Target | ResNet 4x | ResNet 8x | ResNet 16x | Status |
|--------|--------|-----------|-----------|------------|--------|
| PSNR | >25 dB | 24.95 dB | 23.13 dB | 20.52 dB | 4x meets target |
| SSIM | >0.85 | 0.9074 | 0.8568 | 0.7536 | 4x, 8x meet target |
| ENL ratio | 0.8-1.2 | 1.03 | 1.10 | 1.01 | All OK |
| EPI | >0.85 | 0.955 | 0.929 | 0.886 | All meet target |

### Rate-Distortion Comparison (Evaluation Sweep Results)

| Model | Ratio | PSNR (dB) | SSIM | EPI | Notes |
|-------|-------|-----------|------|-----|-------|
| resnet_4x | 4x | 24.95 | 0.9074 | 0.9550 | Best autoencoder |
| baseline_4x | 4x | 24.41 | 0.8714 | 0.9460 | |
| jpeg2000_4x | 4x | 53.21 | 0.9999 | 1.0000 | Near-lossless |
| resnet_8x | 8x | 23.13 | 0.8568 | 0.9291 | Good quality |
| baseline_8x | 8x | 21.27 | 0.7182 | 0.8774 | |
| jpeg2000_8x | 8x | 41.24 | 0.9975 | 0.9993 | |
| resnet_16x | 16x | 20.52 | 0.7536 | 0.8865 | Acceptable |
| baseline_16x | 16x | 18.81 | 0.6095 | 0.8434 | Below target |
| jpeg2000_16x | 16x | 30.92 | 0.9691 | 0.9909 | |

**Key insights:**
- ResNet consistently outperforms baseline (+0.5 to +1.9 dB)
- ResNet 8x quality similar to baseline 4x at 2x more compression
- JPEG-2000 vastly outperforms autoencoders (operates on 8-bit, AE on float32)

### Best Checkpoints (All 6)

| Model | Checkpoint | PSNR | SSIM |
|-------|------------|------|------|
| **ResNet 4x** | `notebooks/checkpoints/resnet_c64_b64_cr4x_20260129_141535/best.pth` | 24.95 dB | 0.9074 |
| ResNet 8x | `notebooks/checkpoints/resnet_c32_b64_cr8x_20260128_213848/best.pth` | 23.13 dB | 0.8568 |
| ResNet 16x | `notebooks/checkpoints/resnet_c16_b64_cr16x_20260128_003926/best.pth` | 20.52 dB | 0.7536 |
| Baseline 4x | `notebooks/checkpoints/baseline_c64_b64_cr4x_20260128_181726/best.pth` | 24.41 dB | 0.8714 |
| Baseline 8x | `notebooks/checkpoints/baseline_c32_b64_cr8x_20260128_192221/best.pth` | 21.27 dB | 0.7182 |
| Baseline 16x | `notebooks/checkpoints/baseline_c16_b64_cr16x_20260127_231730/best.pth` | 18.81 dB | 0.6095 |

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
| **Shared sample evaluation** | Enables paired statistical tests | 200 samples for all 9 models |
| **Per-sample metric storage** | JSON format with explicit indices | Enables t-tests, detailed analysis |

### Technical Notes

- **Data pipeline:** Complete - SARDataModule delivers (32, 1, 256, 256) batches to GPU
- **Dataset:** 696,277 patches across 44 .npy files (182GB), lazy loaded via mmap
- **Preprocessing params:** vmin=14.7688, vmax=24.5407
- **Hardware:** RTX 3070 with 8GB VRAM, batch_size=16 for ResNet with AMP
- **Best model:** ResNet 4x (24.95 dB, 0.9074 SSIM) for quality-critical applications
- **Evaluation:** 200 test samples, all models evaluated on identical data
- **Building blocks ready:** PreActResidualBlock, PreActResidualBlockDown, PreActResidualBlockUp, CBAM
- **GeoTIFF I/O:** read_geotiff, write_geotiff, write_cog with metadata preservation
- **Tiling:** Cosine-squared blending with offset padding, <1e-7 reconstruction error
- **SARCompressor:** Full pipeline with batched GPU inference, AMP support, progress callbacks
- **CLI:** sarcodec compress/decompress with rich progress bars, exit codes

### Blockers

- None

### TODOs (Deferred Items)

- [ ] Longer training run for ResNet (models still improving)
- [ ] Full dataset training (currently using 10% subset)
- [ ] Quantization analysis for reduced model size

---

## Session Continuity

### Last Session

- **Date:** 2026-01-29
- **Activity:** Executed Plan 06-02 (Systematic Evaluation Sweep)
- **Outcome:**
  - Evaluated all 6 autoencoder models + 3 JPEG-2000 baselines
  - Collected per-sample metrics for statistical analysis
  - Saved results to JSON/CSV format
  - Plan 06-02 complete

### Next Session

- **Priority:** Execute Plan 06-03 (Technical Report Generation)
- **Artifacts ready:**
  - `reports/data/all_results.json` - Aggregated metrics
  - `reports/data/per_sample_metrics.json` - Per-sample data
  - `reports/tables/results_summary.csv` - Summary table
- **Expected outputs:**
  - Rate-distortion plots
  - Statistical significance tests
  - Technical report in Markdown format

---

## Quick Reference

**Key Files:**
- Project definition: `.planning/PROJECT.md`
- Requirements: `.planning/REQUIREMENTS.md`
- Research: `.planning/research/SUMMARY.md`
- Roadmap: `.planning/ROADMAP.md`
- Phase 6 Summaries: `.planning/phases/06-final-experiments/06-*-SUMMARY.md`

**Evaluation Results:**
- `reports/data/all_results.json` - All 9 models aggregated
- `reports/data/per_sample_metrics.json` - Per-sample for statistics
- `reports/tables/results_summary.csv` - Summary table

**Codebase Entry Points:**
- Preprocessing: `src/data/preprocessing.py`
- Dataset classes: `src/data/dataset.py`
- DataModule: `src/data/datamodule.py`
- Models: `src/models/` (SARAutoencoder, ResNetAutoencoder)
- Building blocks: `src/models/blocks.py`
- Training: `src/training/trainer.py`
- Evaluation metrics: `src/evaluation/metrics.py`
- Codec baselines: `src/evaluation/codec_baselines.py`
- **Evaluation sweep: `scripts/run_evaluation_sweep.py`**
- SARCompressor: `src/inference/compressor.py`
- CLI: `scripts/sarcodec.py`

**Best Checkpoints:**
- **ResNet 4x:** `notebooks/checkpoints/resnet_c64_b64_cr4x_20260129_141535/best.pth`
- **ResNet 8x:** `notebooks/checkpoints/resnet_c32_b64_cr8x_20260128_213848/best.pth`

---

*State updated: 2026-01-29 (Plan 06-02 complete — ready for report generation)*
