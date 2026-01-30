# Project State

## Project Reference

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression

**Core Value:** Achieve maximum compression ratio while preserving SAR image quality sufficient for downstream analysis.

**Current Focus:** Phase 6.1 (Fair Bitrate Comparison) - Plan 02 complete

---

## Current Position

**Phase:** 6.1 of 7 (Fair Bitrate Comparison) - IN PROGRESS
**Plan:** 02 of 03 (bitrate-matched evaluation complete)
**Status:** Plan 02 complete - ready for Plan 03 (report update)

**Progress:**
```
Phase 1: Data Pipeline      [##########] 100%
Phase 2: Baseline Model     [##########] 100%
Phase 3: SAR Evaluation     [##########] 100%
Phase 4: Architecture       [##########] 100%
Phase 5: Full Inference     [##########] 100%
Phase 6: Final Experiments  [##########] 100%
Phase 6.1: Fair Bitrate     [######----] 67%     <- IN PROGRESS
Phase 7: Deployment         [----------] 0%
```

**Phase 6.1 Progress:**
- [x] Plan 01: Entropy-based bitrate calculation module
- [x] Plan 02: JPEG-2000 R-D curve and bitrate-matched comparison
- [ ] Plan 03: Report update with new conclusions

**Phase 6 Progress (Complete):**
- [x] Plan 01: Checkpoint verification and training setup
- [x] Plan 02: Comprehensive evaluation (9 models)
- [x] Plan 03: Technical report generation

---

## Performance Metrics (Updated from Bitrate-Matched Evaluation)

### Fair Comparison at Matched Bitrates

| Model | BPP | AE PSNR | JP2 PSNR | Diff | SSIM Diff |
|-------|-----|---------|----------|------|-----------|
| ResNet 4x | 1.53 | 25.56 | 28.77 | **-3.22 dB** | -0.035 |
| ResNet 8x | 0.89 | 23.78 | 25.19 | **-1.42 dB** | -0.001 |
| ResNet 16x | 0.44 | 21.20 | 22.53 | **-1.33 dB** | +0.040 |
| Baseline 4x | 1.77 | 24.78 | 30.20 | -5.42 dB | -0.097 |
| Baseline 8x | 0.88 | 21.67 | 25.17 | -3.50 dB | -0.160 |
| Baseline 16x | 0.44 | 19.43 | 22.52 | -3.10 dB | -0.114 |

**Key insights from fair comparison:**
- ResNet models are competitive with JPEG-2000 at matched bitrates
- **ResNet 16x only 1.33 dB below JPEG-2000** at same 0.44 BPP
- Gap narrows at lower bitrates (higher compression)
- Baseline models are 3-5 dB worse than JPEG-2000
- At 16x compression, ResNet SSIM actually beats JPEG-2000 (+0.040)

### Entropy-Based BPP vs Geometric BPP

| Model | Geometric BPP | Actual BPP | Reduction |
|-------|---------------|------------|-----------|
| 4x models | 2.0 | 1.53-1.77 | 12-23% |
| 8x models | 1.0 | 0.88-0.89 | 11-12% |
| 16x models | 0.5 | 0.44 | 12% |

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
| **Data precision mismatch** | JPEG-2000 on 8-bit, autoencoders on float32 | Unfair comparison - document clearly |
| **Wilcoxon for non-normal** | Shapiro-Wilk test selects appropriate test | More robust statistical inference |
| **Bonferroni correction** | Multiple comparisons need adjustment | alpha=0.00278 for 18 tests |
| **Per-channel quantization** | Different channels have different distributions | Preserves more precision |
| **Entropy base=2** | scipy.stats.entropy with base=2 for bits | Standard learned compression practice |
| **64-bit overhead per channel** | 32-bit float x 2 (min/max) | Include storage cost for quantization params |
| **Quality from quantized latent** | Entropy assumes quantized storage; quality must match | Fair comparison metric |
| **Log-spaced JPEG-2000 quality** | Even BPP coverage for smooth R-D interpolation | 23 points, 0.03-6.4 BPP range |
| **Linear R-D interpolation** | Simple, robust for dense curve | scipy.interpolate.interp1d |

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
- **Bitrate-matched evaluation:** 23-point JPEG-2000 R-D curve, entropy-based autoencoder BPP

### Blockers

- None

### TODOs (Deferred Items)

- [ ] Longer training run for ResNet (models still improving)
- [ ] Full dataset training (currently using 10% subset)
- [ ] Quantization analysis for reduced model size

---

## Session Continuity

### Last Session

- **Date:** 2026-01-30
- **Activity:** Executed Plan 06.1-02 (Bitrate-Matched Evaluation)
- **Outcome:**
  - Created scripts/run_bitrate_matched_evaluation.py (732 lines)
  - Generated 23-point JPEG-2000 R-D curve (0.03-6.4 BPP)
  - Computed entropy-based BPP for all 6 autoencoders
  - Created bitrate-matched comparison showing ResNet within 1.3-3.2 dB of JPEG-2000
  - Generated R-D figures and summary CSV

### Next Session

- **Priority:** Continue Phase 6.1 (Plan 03 - Report Update)
- **Phase 6.1-02 outputs ready:**
  - `reports/bitrate_matched/data/autoencoder_bpp.json` - Actual BPP values
  - `reports/bitrate_matched/data/jpeg2000_rd_curve.json` - R-D reference data
  - `reports/bitrate_matched/data/bitrate_matched_results.json` - Comparison results
  - `reports/bitrate_matched/figures/*.png` - R-D curve figures
  - `reports/bitrate_matched/tables/bitrate_matched_summary.csv` - Summary table
- **Expected Phase 6.1-03 work:**
  - Update final_comparison.md with bitrate-matched conclusions
  - Add new R-D figures to report
  - Update executive summary with fair comparison results

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
- `reports/tables/statistical_tests.csv` - Statistical significance tests

**Bitrate-Matched Results:**
- `reports/bitrate_matched/data/autoencoder_bpp.json` - Entropy-based BPP
- `reports/bitrate_matched/data/jpeg2000_rd_curve.json` - 23-point R-D curve
- `reports/bitrate_matched/data/bitrate_matched_results.json` - Fair comparison
- `reports/bitrate_matched/figures/rd_curve_bitrate_matched_*.png` - R-D figures
- `reports/bitrate_matched/tables/bitrate_matched_summary.csv` - Comparison table

**Final Report:**
- `reports/final_comparison.ipynb` - Reproducible analysis notebook
- `reports/final_comparison.md` - Markdown report with executive summary
- `reports/figures/rate_distortion_*.png` - Rate-distortion curves
- `reports/figures/visual_comparison_*.png` - Visual galleries

**Codebase Entry Points:**
- Preprocessing: `src/data/preprocessing.py`
- Dataset classes: `src/data/dataset.py`
- DataModule: `src/data/datamodule.py`
- Models: `src/models/` (SARAutoencoder, ResNetAutoencoder)
- Building blocks: `src/models/blocks.py`
- Training: `src/training/trainer.py`
- Evaluation metrics: `src/evaluation/metrics.py`
- Codec baselines: `src/evaluation/codec_baselines.py`
- Bitrate estimation: `src/evaluation/bitrate.py`
- Evaluation sweep: `scripts/run_evaluation_sweep.py`
- **Bitrate-matched evaluation: `scripts/run_bitrate_matched_evaluation.py`**
- SARCompressor: `src/inference/compressor.py`
- CLI: `scripts/sarcodec.py`

**Best Checkpoints:**
- **ResNet 4x:** `notebooks/checkpoints/resnet_c64_b64_cr4x_20260129_141535/best.pth`
- **ResNet 8x:** `notebooks/checkpoints/resnet_c32_b64_cr8x_20260128_213848/best.pth`

---

*State updated: 2026-01-30 (Phase 6.1 Plan 02 complete - bitrate-matched evaluation)*
