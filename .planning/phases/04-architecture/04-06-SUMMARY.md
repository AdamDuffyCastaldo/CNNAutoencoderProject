---
phase: 04-architecture
plan: 06
subsystem: evaluation
tags: [comparison, evaluation, phase-wrap-up]

# Dependency graph
requires:
  - phase: 04-04
    provides: Trained variants (partial)
  - phase: 04-05
    provides: Attention training notebook
provides:
  - Architecture comparison notebook
  - Phase 4 wrap-up documentation
  - Best model recommendation for Phase 5
affects: [05-inference]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Unified model evaluation on consistent validation subset
    - Side-by-side visual comparison

key-files:
  created:
    - notebooks/compare_architectures.ipynb
    - notebooks/compare_architectures_results.json
  modified:
    - .planning/STATE.md

key-decisions:
  - "Proceed to Phase 5 with ResNet-Lite v2 (best available at 21.2 dB)"
  - "Defer Residual/Attention training - infrastructure ready for future"
  - "Phase 4 marked as partial completion"

patterns-established:
  - "Architecture comparison notebook pattern for future variants"

# Metrics
duration: partial
completed: 2026-01-26
status: partial_completion
---

# Phase 4 Plan 06: Architecture Comparison Summary

**Phase 4 wrapped up with partial completion; ResNet-Lite v2 selected for Phase 5**

## Status: Partial Completion

Phase 4 is wrapped up with the following status:
- Building blocks (04-01): COMPLETE
- ResidualAutoencoder (04-02): COMPLETE
- AttentionAutoencoder (04-03): COMPLETE
- Training improvements (04-04): PARTIAL - infrastructure ready, full training deferred
- Training notebook (04-05): COMPLETE
- Comparison (04-06): PARTIAL - comparing available models

## Comparison Results

| Model | Params | PSNR | SSIM | Status |
|-------|--------|------|------|--------|
| Baseline | 2.3M | 20.47 dB | 0.646 | Complete |
| **ResNet-Lite v2** | 5.6M | **21.20 dB** | **0.726** | **Complete (Best)** |
| Residual v1 | 23.8M | 19.78 dB | - | Suboptimal (LR issue) |
| Attention v1 | 24.0M | - | - | Quick test only |

## Phase 4 Success Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| ResidualBlock preserves dimensions | PASS | Implemented and tested |
| CBAM applies attention correctly | PASS | Implemented and tested |
| Residual >= +1.5 dB over baseline | DEFER | Training deferred |
| Attention >= +0.5 dB over Residual | DEFER | Training deferred |
| ENL ratio 0.8-1.2 | PASS | Met for complete models |

**Result:** 3/5 criteria met, 2 deferred

## Recommendation

**Proceed to Phase 5 with ResNet-Lite v2:**
- Best available model at 21.20 dB PSNR
- 5.6M parameters (reasonable size)
- +0.73 dB improvement over baseline
- ENL ratio and EPI within acceptable ranges

**Return to Phase 4 later:**
- Training infrastructure ready (warmup, AdamW, gradient clipping)
- Notebooks configured with QUICK_SEARCH toggle
- Can retrain Residual/Attention with improved hyperparameters

## Files Created

- `notebooks/compare_architectures.ipynb` - Unified comparison notebook
- `notebooks/compare_architectures_results.json` - Comparison results (created on notebook run)

## Deferred Work

- [ ] Retrain Residual v2 with LR=5e-5, warmup=3, AdamW
- [ ] Train Attention v2 with same config
- [ ] Full training runs (20% data, 30 epochs)
- [ ] Re-run comparison with fully trained variants

---
*Phase: 04-architecture*
*Status: Partial completion*
*Completed: 2026-01-26*
