# Phase 2: Baseline Model - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement and train a plain 4-layer CNN autoencoder that compresses 256x256 SAR patches to a 16x16xC latent space and reconstructs them with convergent training. This is the baseline architecture — residual blocks, attention mechanisms, and advanced metrics belong in later phases.

</domain>

<decisions>
## Implementation Decisions

### Loss function balance
- Default weights: 0.5 MSE + 0.5 SSIM (equal balance)
- Weights configurable via config file (not hardcoded)
- SSIM window size: standard 11x11
- Start with MSE + SSIM only for baseline training
- Architecture supports future loss terms (MS-SSIM, BCE) via single configurable combined loss class
- When BCE is added later: standard pixel-wise formulation treating each pixel as independent probability

### Training feedback
- TensorBoard scalar logging: every epoch (loss, PSNR, SSIM)
- TensorBoard image logging: every epoch (reconstruction samples)
- Reconstruction visualization: triple view (original | reconstructed | difference map)
- Weight histograms: every 10 epochs
- Console output: tqdm progress bar + epoch summary
- Progress bar shows: loss + PSNR + SSIM during batch processing
- Display ETA (estimated time remaining)
- Log GPU memory usage in epoch summary
- Save training logs to file (in addition to TensorBoard)

### Claude's Discretion
- Number of sample images to log per epoch (fixed vs random)
- Exact progress bar formatting
- Log file location and format
- Early stopping patience and checkpoint strategy (not discussed)

</decisions>

<specifics>
## Specific Ideas

- User wants to explore multiple loss functions (MS-SSIM, BCE) in future experiments, so the combined loss class should be extensible
- GPU memory logging is important given RTX 3070 8GB VRAM constraint
- Full metrics visibility preferred (loss + PSNR + SSIM in progress bar)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-baseline-model*
*Context gathered: 2026-01-21*
