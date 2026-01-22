# Phase 2 Plan 01: Building Blocks and Loss Functions Summary

**One-liner:** ConvBlock/DeconvBlock for 2x spatial scaling and CombinedLoss with 0.5/0.5 MSE/SSIM weights using pytorch-msssim

---

## Metadata

```yaml
phase: 02-baseline-model
plan: 01
subsystem: model-blocks, loss-functions
tags: [conv-blocks, ssim, mse, pytorch-msssim]

dependency-graph:
  requires: []
  provides: [ConvBlock, DeconvBlock, SSIMLoss, CombinedLoss]
  affects: [02-02, 02-03, 02-04]

tech-stack:
  added: [pytorch-msssim]
  patterns: [encoder-decoder-blocks, combined-loss]

key-files:
  created: []
  modified:
    - src/models/blocks.py
    - src/losses/ssim.py
    - src/losses/combined.py

decisions:
  - id: ssim-library
    choice: pytorch-msssim over hand-rolled SSIM
    reason: GPU-optimized, well-tested, handles edge cases

metrics:
  duration: ~3 minutes
  completed: 2026-01-21
```

---

## What Was Done

### Task 1: Migrate ConvBlock and DeconvBlock

**Files modified:** `src/models/blocks.py`

Implemented from notebook Cell 4 and Cell 10:

- **ConvBlock:** Conv2d -> BatchNorm -> LeakyReLU(0.2)
  - Halves spatial dimensions with kernel=5, stride=2, padding=2
  - bias=False when using BatchNorm (BN has its own bias)

- **DeconvBlock:** ConvTranspose2d -> BatchNorm -> ReLU
  - Doubles spatial dimensions with output_padding=1
  - Same bias logic as ConvBlock

**Verification:**
```
Input:  (2, 1, 256, 256)
After c1(1->64):   (2, 64, 128, 128)
After c2(64->128): (2, 128, 64, 64)
After d1(128->64): (2, 64, 128, 128)
After d2(64->1):   (2, 1, 256, 256)
```

### Task 2: Migrate SSIMLoss using pytorch-msssim

**Files modified:** `src/losses/ssim.py`

Simplified implementation using pytorch-msssim library:
- SSIM module with data_range=1.0, size_average=True, nonnegative_ssim=True
- Returns 1 - SSIM as loss (0 for identical images)
- Default window_size=11 (standard SSIM)

**Verification:**
- Identical images: loss = 0.000000
- Different images: loss > 0

### Task 3: Migrate CombinedLoss

**Files modified:** `src/losses/combined.py`

- Default weights: 0.5 MSE + 0.5 SSIM (per CONTEXT.md, not 1.0/0.1 from notebook)
- Returns (loss_tensor, metrics_dict)
- metrics contains: loss, mse, ssim, psnr
- Loss tensor remains differentiable for backprop

**Verification:**
- Identical images: loss = 0.000000, PSNR = 100.0 dB
- Small noise (5%): loss ~0.008, PSNR ~26 dB, SSIM ~0.98

---

## Decisions Made

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| SSIM implementation | Hand-rolled vs pytorch-msssim | pytorch-msssim | GPU-optimized, handles edge cases, well-tested |
| Default loss weights | 1.0/0.1 (notebook) vs 0.5/0.5 (CONTEXT.md) | 0.5/0.5 | Balanced weighting per project context |

---

## Deviations from Plan

None - plan executed exactly as written.

---

## Technical Notes

### ConvBlock/DeconvBlock Shape Math

For stride=2, kernel=5, padding=2:
- ConvBlock: `output = (input + 2*padding - kernel) / stride + 1 = (input + 4 - 5) / 2 + 1 = input/2`
- DeconvBlock: `output = (input - 1) * stride + kernel - 2*padding + output_padding = (input-1)*2 + 5 - 4 + 1 = input*2`

### SSIM Behavior

With pytorch-msssim:
- Identical images: SSIM = 1.0, Loss = 0.0
- Random different images: SSIM ~= 0, Loss ~= 1.0
- nonnegative_ssim=True ensures stable gradients

---

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| 936e5db | feat | Implement ConvBlock and DeconvBlock |
| 308820e | feat | Implement SSIMLoss using pytorch-msssim |
| 73701bc | feat | Implement CombinedLoss with 0.5/0.5 weights |

---

## Next Phase Readiness

**Ready for:** Plan 02-02 (Baseline Autoencoder Architecture)

**Dependencies satisfied:**
- ConvBlock available for encoder
- DeconvBlock available for decoder
- CombinedLoss available for training

**No blockers identified.**
