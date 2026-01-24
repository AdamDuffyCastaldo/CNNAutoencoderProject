---
phase: 04-architecture
plan: 01
subsystem: models
tags: [blocks, residual, pre-activation, cbam, attention, resnet-v2]

# Dependency graph
requires:
  - phase: 02-baseline-model
    provides: Basic blocks.py with ConvBlock, DeconvBlock, ResidualBlock
provides:
  - PreActResidualBlock with BN->ReLU->Conv ordering (ResNet v2)
  - PreActResidualBlockDown for stride=2 downsampling
  - PreActResidualBlockUp for bilinear upsample with residual
  - ChannelAttention, SpatialAttention, CBAM attention modules
affects: [04-02, 04-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [pre-activation residual, CBAM channel-then-spatial attention]

key-files:
  created: []
  modified:
    - src/models/blocks.py

key-decisions:
  - "Pre-activation (BN->ReLU->Conv) instead of post-activation for cleaner gradient flow"
  - "1x1 conv projection for skip connection when dimensions change"
  - "CBAM uses 1x1 convs for MLP (more efficient than Linear)"
  - "Handle edge case: max(channels//reduction, 1) for small channel counts"
  - "Bilinear upsample + 1x1 conv for PreActResidualBlockUp (cleaner than transposed conv)"

patterns-established:
  - "Pre-activation: no ReLU after skip addition"
  - "Kaiming init for conv weights, weight=1/bias=0 for BatchNorm"
  - "CBAM sequential: channel attention first, then spatial attention"

# Metrics
duration: 8min
completed: 2026-01-24
---

# Phase 4 Plan 01: Building Blocks Summary

**Pre-activation residual blocks (ResNet v2) and CBAM attention modules for enhanced autoencoder architectures**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-01-24 19:43 UTC
- **Completed:** 2026-01-24 19:51 UTC
- **Tasks:** 2/2
- **Files modified:** 1

## Accomplishments

- Implemented PreActResidualBlock with BN->ReLU->Conv ordering (ResNet v2 style)
- Added PreActResidualBlockDown convenience wrapper for stride=2 downsampling
- Implemented PreActResidualBlockUp using bilinear upsample + residual convolutions
- Completed ChannelAttention stub with max+avg pooling and shared MLP
- Completed SpatialAttention stub with channel-wise max/mean concatenation
- Completed CBAM stub combining channel-then-spatial attention
- Updated test_blocks() with comprehensive tests for all new blocks

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement Pre-Activation Residual Block** - `f3f828d`
   - PreActResidualBlock with BN->ReLU->Conv ordering
   - Projection shortcut for dimension changes
   - Kaiming initialization, no ReLU after skip addition
   - PreActResidualBlockDown wrapper
   - PreActResidualBlockUp with bilinear upsample

2. **Task 2: Implement CBAM Attention Module** - `fa9856e`
   - ChannelAttention: SE-style with max+avg pooling, shared 1x1 conv MLP
   - SpatialAttention: concat channel max/mean -> conv -> sigmoid
   - CBAM: sequential channel-then-spatial attention
   - Edge case handling for small channels
   - Updated test_blocks() with all new block tests

## Files Modified

- `src/models/blocks.py` - Added ~180 lines for new blocks and updated tests

## Key Implementation Details

### PreActResidualBlock (ResNet v2)

```
x -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+x)
     ^                                           ^
     Pre-activation                        No ReLU after
```

- Cleaner gradient flow through identity path
- Projection shortcut only when stride != 1 or channels change
- Kaiming normal initialization for all conv weights

### CBAM Attention

- **ChannelAttention**: `sigmoid(MLP(avgpool) + MLP(maxpool))` -> (B, C, 1, 1)
- **SpatialAttention**: `sigmoid(conv(concat(max, mean)))` -> (B, 1, H, W)
- **CBAM**: `x * channel_attn(x) * spatial_attn(refined_x)`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Success Criteria Verification

1. [x] PreActResidualBlock implements BN->ReLU->Conv ordering (pre-activation style)
2. [x] PreActResidualBlock with stride=2 correctly downsamples spatial dimensions by 2x
3. [x] PreActResidualBlockUp correctly upsamples spatial dimensions by 2x
4. [x] ChannelAttention and SpatialAttention produce attention weights in [0, 1]
5. [x] CBAM combines both attentions and preserves input tensor shape
6. [x] All existing blocks (ConvBlock, DeconvBlock, ResidualBlock) still function correctly

## Next Plan Readiness

**Ready for 04-02 (Variant Autoencoders):**
- PreActResidualBlock, PreActResidualBlockDown, PreActResidualBlockUp available
- CBAM attention module ready for integration
- Existing ResNet-Lite blocks unchanged (backward compatible)
- All blocks tested and verified

---
*Phase: 04-architecture*
*Completed: 2026-01-24*
