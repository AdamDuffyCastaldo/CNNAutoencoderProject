# Phase 3: SAR Evaluation Framework - Context

**Gathered:** 2026-01-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement SAR-specific quality metrics (ENL ratio, EPI) and evaluation tools that enable informed architecture decisions. Establish traditional codec baselines (JPEG-2000, JPEG) for meaningful comparison at multiple compression ratios.

</domain>

<decisions>
## Implementation Decisions

### Metric Output Format
- Primary output: JSON files (machine-readable)
- Two-tier granularity: summary file + detailed per-patch file (separate files)
- Rate-distortion data: both raw data (CSV/JSON) AND generated Matplotlib plots
- Output location: dedicated `evaluations/` directory (e.g., `evaluations/resnet_lite_v2_c16/`)
- File naming: model-based (e.g., `resnet_lite_v2_c16_eval.json`)
- Support both single-model and multi-model comparison modes
- Quality metrics only (no timing/throughput in evaluation output)
- CLI script that wraps notebook-friendly API (both interfaces available)

### ENL Region Detection
- Both automatic detection and manual override available
- Region size: configurable, Claude picks sensible default
- Compute ENL in both linear intensity AND dB domains, report both
- Sample ALL detected homogeneous regions, report statistics across them
- Save region masks as visualization for inspection

### Comparison Visualization
- Comprehensive output: side-by-side + difference map + zoomed crops
- Difference maps: diverging colormap (blue-white-red for under/over prediction)
- Metrics overlay: include PSNR, SSIM in figure titles
- Sample count: configurable (default 5)
- Zoomed crops: both auto-select interesting regions AND fixed locations available
- Include histogram comparison (original vs reconstructed intensity distributions)

### Codec Baseline Scope
- Include JPEG-2000 and JPEG as baselines
- Evaluate at 8x, 16x, 32x compression ratios (matching autoencoder variants)
- Use quality parameter sweep to match target compression ratios
- Use same test set as autoencoder evaluation (direct comparison)
- Cache encoded files to avoid re-encoding on repeat evaluations

### Claude's Discretion
- Statistical reporting: Claude picks appropriate approach (point estimates vs confidence intervals)
- JSON metadata: Claude includes reproducibility info if useful
- Homogeneity criterion for ENL detection (CV vs variance threshold)
- Zoom factor for detail crops
- Library choice for codec implementation (OpenCV vs Pillow)

</decisions>

<specifics>
## Specific Ideas

- Diverging colormap for error visualization shows under/over prediction clearly
- Region masks saved for ENL inspection — allows debugging detection algorithm
- Histogram comparison important for SAR (intensity distribution preservation matters)
- Cache encoded codec files for faster iteration during development

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-sar-evaluation*
*Context gathered: 2026-01-24*
