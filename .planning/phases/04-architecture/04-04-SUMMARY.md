---
phase: 04-architecture
plan: 04
subsystem: training
tags: [training, hyperparameters, warmup, adamw, residual, attention]

# Dependency graph
requires:
  - phase: 04-02
    provides: ResidualAutoencoder (Variant B)
  - phase: 04-03
    provides: AttentionAutoencoder (Variant C)
provides:
  - Training stability improvements (warmup, AdamW, gradient clipping)
  - Residual v1 checkpoint (suboptimal - LR too conservative)
  - Training notebooks with quick search mode
affects: [04-06-comparison]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - LR warmup (linear from LR/10 to LR over N epochs)
    - AdamW optimizer with weight_decay
    - Quick search mode (5% data, 20 epochs) for hyperparameter validation

key-files:
  modified:
    - src/training/trainer.py
    - notebooks/train_residual.ipynb
    - notebooks/train_attention.ipynb

key-decisions:
  - "LR warmup 3 epochs - stabilizes early training"
  - "AdamW with weight_decay=1e-5 - better regularization"
  - "Gradient clipping 0.5 - more aggressive for deep networks"
  - "Quick search mode - 5% data for fast hyperparameter validation"
  - "Full training deferred - proceed to Phase 5, return later for improvements"

patterns-established:
  - "QUICK_SEARCH toggle in training notebooks (5% vs 20% data)"
  - "Warmup epochs parameter in Trainer"

# Metrics
duration: partial
completed: 2026-01-26
status: deferred
---

# Phase 4 Plan 04: Training Improvements Summary

**Training infrastructure improved; full Variant B/C training deferred to return later**

## Status: Partially Complete (Deferred)

Training stability improvements were implemented, but full training of Residual (Variant B) and Attention (Variant C) architectures is deferred. The user wants to proceed to Phase 5 and return to improve training later.

## Accomplishments

### Training Infrastructure Improvements

Added to `src/training/trainer.py`:
- **LR Warmup**: Linear warmup from LR/10 to LR over configurable epochs
- **AdamW Support**: Optional AdamW optimizer with weight_decay parameter
- **Configurable Optimizer**: `optimizer` parameter accepts 'adam' or 'adamw'

### Notebook Improvements

Updated `notebooks/train_residual.ipynb` and `notebooks/train_attention.ipynb`:
- QUICK_SEARCH toggle (5% data, 20 epochs vs 20% data, 30 epochs)
- Increased LR from 1e-5 to 5e-5
- Added 3-epoch warmup
- Switched to AdamW with weight_decay=1e-5
- Reduced gradient clipping from 1.0 to 0.5
- Fixed comparison table formatting
- Fixed SAR evaluation cells to use val_loader

## Training Results

| Model | Params | PSNR | SSIM | Notes |
|-------|--------|------|------|-------|
| Baseline | 2.3M | 20.47 dB | 0.646 | Reference |
| ResNet-Lite v2 | 5.6M | 21.20 dB | 0.726 | Best available |
| Residual v1 | 23.8M | 19.78 dB | - | LR=1e-5 too conservative |
| Attention v1 | 24.0M | - | - | Quick test only (50 batches) |

**Note:** Residual v1 underperformed baseline due to overly conservative learning rate (1e-5 instead of typical 1e-4 to 5e-5). Training infrastructure now improved.

## Deferred Work

The following requires extended training time and is deferred:
- [ ] Retrain Residual v2 with LR=5e-5, warmup=3, AdamW
- [ ] Train Attention v2 with same improved config
- [ ] Full training (20% data, 30 epochs, ~10+ hours each)

## Files Modified

- `src/training/trainer.py` - warmup_epochs, optimizer choice, _update_warmup_lr()
- `notebooks/train_residual.ipynb` - config updates, QUICK_SEARCH toggle
- `notebooks/train_attention.ipynb` - config updates, QUICK_SEARCH toggle, BASE_CHANNELS=48

## Decisions Made

1. **LR warmup 3 epochs** - Prevents early instability in deep networks
2. **AdamW over Adam** - Better weight decay handling
3. **Gradient clipping 0.5** - More aggressive for deeper architectures
4. **Quick search mode** - Fast hyperparameter validation (5% data, 20 epochs)
5. **BASE_CHANNELS=48 for Attention** - Reduces params to ~13M for faster iteration
6. **Defer full training** - Proceed to Phase 5, return for training improvements later

## Checkpoints Available

| Checkpoint | Path | Status |
|------------|------|--------|
| Baseline | `notebooks/checkpoints/baseline_c16_fast/best.pth` | Complete |
| ResNet-Lite v2 | `notebooks/checkpoints/resnet_lite_v2_c16/best.pth` | Complete, Best |
| Residual v1 | `notebooks/checkpoints/residual_v1_c16/best.pth` | Suboptimal |
| Attention v1 | `notebooks/checkpoints/attention_v1_c16/quick_test.pth` | Quick test only |

## Next Steps

1. **Phase 5 (Full Image Inference)** - Proceed with ResNet-Lite v2 as best available model
2. **Return to Phase 4** - When time permits, run improved training configs
3. **Phase 6 experiments** - Can include retrained variants when available

---
*Phase: 04-architecture*
*Completed: 2026-01-26 (partial - training deferred)*
