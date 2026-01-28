# Phase 6: Final Experiments - Context

**Gathered:** 2026-01-28 (updated)
**Status:** Ready for planning

<domain>
## Phase Boundary

Execute experiment matrix and produce comprehensive comparison study with rate-distortion analysis comparing autoencoders against JPEG-2000.

**Starting point:** Phase 4 complete — ResNet b=64 selected (21.13 dB @ 16x).

</domain>

<decisions>
## Implementation Decisions

### Experiment Scope

- **Architectures:** ResNet + Baseline (2 architectures)
- **Ratios:** 4x, 8x, 16x initially (6 models total)
- **Skip 12x:** Redundant, not needed
- **Future additions:** Space for Attention/Residual when trained; 32x replaces 4x when time allows
- **Existing models:** Evaluate first, retrain only if doesn't beat JPEG-2000
- **Replication:** Single training run per config (no statistical replication)
- **Priority order:** 16x first (primary target), then 8x, then 4x
- **Reporting:** Include all results regardless of performance (shows architectural impact)

### Test Data

- **Sources:** Both existing test split (consistency) + new unseen Sentinel-1 scenes (generalization)
- **Geographic diversity:** New scenes from different regions than training data
- **Content stratification:** Report metrics separately for each terrain type (urban, water, forest, etc.)

### Comparison Methodology

- **Codec baseline:** JPEG-2000 only (JPEG excluded as known inferior)
- **Comparison basis:** Both same-ratio AND same-BPP views
- **BPP matching:** Bracket approach — test JPEG-2000 at higher/lower BPP, interpolate
- **Metrics:** Full SAR suite (PSNR, SSIM, MS-SSIM, ENL ratio, EPI)
- **Statistics:** Formal hypothesis tests (paired t-tests/Wilcoxon) plus mean ± std
- **Evaluation scale:** Patch-level (detailed stats) + full-image (via SARCompressor)
- **Compute benchmark:** Full timing comparison — training time, inference speed, memory usage vs JPEG-2000

### Failure Analysis

- **Depth:** Detailed investigation when codec wins
- **Content:** Root cause analysis, specific examples, identify WHY autoencoder underperforms

### Results Presentation

- **Format:** Jupyter notebook (reproducibility) + Markdown report (readability)
- **Output location:** `reports/` top-level directory
- **Progress tracking:** Incremental updates as each model completes
- **Conclusion style:** Data presentation with brief conclusion (let data speak, but summarize)
- **Charts:** Static matplotlib (PNG/SVG), paper-ready
- **Tables:** CSV for data processing, LaTeX for paper; separate per ratio
- **Visual examples:** 5 per ratio (15 total), 3-column (Original | Autoencoder | JPEG-2000)
- **Error maps:** Color-coded heatmaps showing per-pixel error magnitude

### Training Strategy

- **Data percentage:** 10% (same as Phase 4)
- **Epochs:** 35 epochs target
- **Hyperparameters:** Start consistent across ratios, tune only if model struggles
- **LR schedule:** ReduceLROnPlateau (reduce when validation loss plateaus)
- **Batch size:** Maximum per model (largest that fits on 8GB VRAM)
- **Checkpointing:** Save every epoch
- **Execution:** Scripts for training, notebooks for analysis
- **Checkpoint naming:** Keep current pattern (e.g., `resnet_c16_b64_cr16x_YYYYMMDD`)

### Claude's Discretion

- Loss function weighting (MSE/SSIM balance)
- Base channels per ratio (b=64 for all vs scale with difficulty)
- Data augmentation approach
- Retraining approach if models don't beat JPEG-2000 (epochs vs data vs both)
- Resume vs restart on training interruption
- Number and selection of new unseen test scenes

</decisions>

<specifics>
## Specific Ideas

- Technical report structured to support future academic paper conversion
- Failure analysis should identify WHY autoencoder underperforms (not just that it does)
- Error heatmaps enable visual identification of problem regions
- Stratified results help identify where each approach excels

</specifics>

<deferred>
## Deferred Ideas

- Attention/Residual architectures — train when time allows, add to comparison
- 32x compression ratio — replaces 4x when time constraints allow
- Full dataset training (100%) — only if results warrant
- Statistical replication (multiple runs) — only for winning configs if needed
- Interactive plotly visualizations — nice-to-have, not essential

</deferred>

---

*Phase: 06-final-experiments*
*Context updated: 2026-01-28*
