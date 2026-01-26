# Phase 6: Final Experiments - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Execute the complete experiment matrix (3 architectures × 3 compression ratios = 9 models) and produce a comprehensive comparison study with rate-distortion analysis comparing autoencoders against JPEG-2000.

**Prerequisite:** Phase 4 must be completed first — Residual v2 and Attention v2 need proper training before Phase 6 begins.

</domain>

<decisions>
## Implementation Decisions

### Experiment Scope

- **Phase 4 first:** Complete deferred training (Residual v2, Attention v2) before starting Phase 6
- **Full matrix:** Train all 9 configurations (Plain, Residual, Attention at 8x, 16x, 32x)
- **Quick search strategy:** Train each config for ~10 epochs initially, identify promising configs
- **Winner criterion:** Configs that beat JPEG-2000 on PSNR get full training
- **Final training data:** 50% of dataset for winning configs (up from 20%)
- **No winners handling:** Investigate why before proceeding (don't blindly continue)
- **Sequential execution:** Single RTX 3070, one model at a time
- **Logging:** Keep all checkpoints and logs from quick search and finals
- **Naming:** Ratio + numbered format (e.g., 16x_01, 8x_02)
- **Phase 4 approach:** Fresh training from scratch with improved hyperparams

### Comparison Methodology

- **BPP matching:** Bracket approach — test JPEG-2000 at higher/lower BPP, interpolate for fair comparison
- **Metrics:** Full SAR suite (PSNR, SSIM, MS-SSIM, ENL ratio, EPI)
- **Codec baseline:** JPEG-2000 only (JPEG excluded as known inferior)
- **Statistics:** Formal hypothesis tests (paired t-tests/Wilcoxon) plus mean ± std
- **Rate-distortion curves:** Best performing config per architecture family
- **Test data:** New unseen Sentinel-1 scenes (not existing test split)
- **Evaluation scale:** Both patch-level (detailed stats) and full-image (via SARCompressor)
- **Failure analysis:** Detailed documentation of cases where codec wins, with analysis
- **SAR artifacts:** Quantitative measurement (ENL/EPI) plus visual examples
- **Compute benchmark:** Full timing and memory comparison for autoencoder vs codec
- **Recommendations:** Include practical guidance on when to use each approach

### Results Presentation

- **Document format:** Both Jupyter notebook (reproducibility) and Markdown report (readability)
- **Analysis depth:** Technical report with potential for future academic paper
- **Output location:** New `reports/` top-level directory
- **Table exports:** CSV for data processing, LaTeX for paper
- **Charts:** Static matplotlib (PNG/SVG), paper-ready
- **Training curves:** Include only for winning models
- **Table organization:** Separate tables per compression ratio (8x, 16x, 32x)
- **Summary style:** Academic-style abstract paragraph
- **Appendix:** Full per-image results included
- **Architecture diagrams:** Detailed block diagrams for each variant
- **Hyperparameters:** Document key params only (LR, batch size, epochs)
- **Citations:** Minimal — only directly used methods/libraries

### Visual Examples

- **Count:** 5 examples per compression ratio (15 total)
- **Layout:** 3-column side-by-side (Original | Autoencoder | JPEG-2000)
- **Curation:** Best cases, worst cases, and specific regional examples (urban, water, forest, edges, homogeneous)
- **Metric display:** Separate table (keep images clean without overlays)
- **Resolution:** Web-friendly with links to full resolution
- **Format:** PNG (lossless for comparison integrity)
- **Error maps:** Color-coded heatmaps showing per-pixel error magnitude
- **Animation:** Optional supplement — GIF toggling original/reconstructed (not in main report)

### Claude's Discretion

- Early stopping criterion during quick search (monitor convergence per-model)
- Number of new unseen scenes to source for testing
- Specific scene selection criteria

</decisions>

<specifics>
## Specific Ideas

- Technical report should be structured to support future conversion to academic paper
- Failure analysis should identify WHY autoencoder underperforms (not just that it does)
- Recommendations should be practical: "Use autoencoder when X, use JPEG-2000 when Y"
- Error heatmaps enable visual identification of problem regions

</specifics>

<deferred>
## Deferred Ideas

- Full dataset training (100%) — resource intensive, only if results warrant
- Interactive plotly visualizations — nice-to-have, not essential
- Extensive literature review — save for actual paper submission

</deferred>

---

*Phase: 06-final-experiments*
*Context gathered: 2026-01-26*
