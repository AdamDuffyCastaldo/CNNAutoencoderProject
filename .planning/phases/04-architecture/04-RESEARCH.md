# Phase 4: Architecture Enhancement - Research

**Researched:** 2026-01-24
**Domain:** Pre-activation Residual Blocks + CBAM Attention for SAR Autoencoder
**Confidence:** HIGH

## Summary

This research investigates the implementation of pre-activation residual blocks (ResNet v2 style) and CBAM (Convolutional Block Attention Module) for enhancing the SAR autoencoder architecture. The existing codebase already has post-activation residual blocks in `src/models/blocks.py` and stub implementations for CBAM, providing a foundation to build upon.

The standard approach for pre-activation residual blocks uses BN -> ReLU -> Conv ordering (from He et al. 2016), which provides better gradient flow for deeper networks. CBAM follows the original paper's design with reduction ratio 16 for channel attention and 7x7 kernel for spatial attention, applied sequentially (channel-first, then spatial).

**Primary recommendation:** Implement pre-activation residual blocks as new classes (not modifying existing), implement CBAM following the paper's architecture exactly, then create variant models that compose these building blocks. The existing `ResNetAutoencoder` provides a template but uses post-activation blocks, so new encoder/decoder classes are needed.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Deep learning framework | Already in use, native support for all operations |
| torch.nn | built-in | Module definitions | Conv2d, BatchNorm2d, ReLU, Sigmoid all native |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytorch-msssim | installed | SSIM loss computation | Already integrated in `src/losses/ssim.py` |
| tensorboard | installed | Training visualization | Already integrated in `src/training/trainer.py` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom CBAM | timm library CBAM | timm adds dependency; custom gives full control for SAR-specific tuning |
| Pre-activation blocks | Post-activation (existing) | Post-activation already in codebase but doesn't match user's decision for v2-style |

**No new packages required** - all components can be built with existing PyTorch primitives.

## Architecture Patterns

### Recommended Project Structure

Build on existing structure in `src/models/`:

```
src/models/
├── blocks.py              # Add PreActResidualBlock, CBAM, ChannelAttention, SpatialAttention
├── encoder.py             # Keep baseline encoder
├── decoder.py             # Keep baseline decoder
├── autoencoder.py         # Keep baseline autoencoder
├── resnet_autoencoder.py  # Existing post-activation ResNet (reference)
├── residual_autoencoder.py   # NEW: Pre-activation residual encoder/decoder
└── attention_autoencoder.py  # NEW: Residual + CBAM encoder/decoder
```

### Pattern 1: Pre-Activation Residual Block (ResNet v2)

**What:** BN -> ReLU -> Conv -> BN -> ReLU -> Conv with identity skip connection
**When to use:** All residual blocks in the enhanced architecture
**Why:** Better gradient flow for deeper networks; the skip connection directly adds the input without any transformation applied to the sum.

```python
# Source: PyTorch Lightning ResNet Tutorial
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html

class PreActResidualBlock(nn.Module):
    """Pre-activation residual block (ResNet v2 style).

    Order: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    Skip connection is identity (or 1x1 projection for channel mismatch).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Main path: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Projection shortcut for channel/spatial mismatch
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv projection (no activation - this is the identity path)
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)

        # Skip connection
        if self.shortcut is not None:
            identity = self.shortcut(x)
        else:
            identity = x

        # No activation after addition (key difference from v1)
        return out + identity
```

### Pattern 2: CBAM Module

**What:** Sequential channel-then-spatial attention
**When to use:** After every residual block (per user decision)
**Why:** Channel attention filters uninformative channels, spatial attention focuses on relevant regions.

```python
# Source: CBAM Paper (ECCV 2018) + External-Attention-pytorch
# https://github.com/xmu-xiaoma666/External-Attention-pytorch

class ChannelAttention(nn.Module):
    """Channel attention module from CBAM.

    Uses both max-pool and avg-pool, shared MLP, sigmoid activation.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP using 1x1 convolutions (equivalent to FC on 1x1 spatial)
        # No BatchNorm inside - follows original paper
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out = self.mlp(self.maxpool(x))
        avg_out = self.mlp(self.avgpool(x))
        return self.sigmoid(max_out + avg_out)


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM.

    Concatenates channel-wise max and avg, applies conv, sigmoid.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, avg_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies channel attention then spatial attention sequentially.
    Output is input * channel_weights * spatial_weights.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention: x * Mc(x)
        x = x * self.channel_attention(x)
        # Spatial attention: x * Ms(x)
        x = x * self.spatial_attention(x)
        return x
```

### Pattern 3: Residual Block with CBAM

**What:** Compose residual block followed by CBAM
**When to use:** Variant C (Res+CBAM) architecture

```python
class ResidualBlockWithCBAM(nn.Module):
    """Pre-activation residual block followed by CBAM attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.residual = PreActResidualBlock(in_channels, out_channels, stride)
        self.cbam = CBAM(out_channels, reduction, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual(x)
        x = self.cbam(x)
        return x
```

### Anti-Patterns to Avoid

- **Activation after skip addition:** In pre-activation blocks, do NOT apply ReLU after the sum. The identity path should be clean.
- **CBAM inside residual block:** CBAM goes AFTER the residual block, not inside it. This provides cleaner separation and better gradient flow.
- **BatchNorm in CBAM MLP:** The original paper does NOT use BatchNorm inside the channel attention MLP. Only Conv -> ReLU -> Conv.
- **Parallel attention instead of sequential:** Channel-first then spatial is empirically validated to outperform parallel or inverted schemes.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSIM computation | Custom SSIM | pytorch-msssim (installed) | Numerical stability, GPU-optimized |
| Gradient clipping | Manual norm computation | `torch.nn.utils.clip_grad_norm_` | Already in trainer.py |
| Learning rate scheduling | Custom decay | `torch.optim.lr_scheduler.ReduceLROnPlateau` | Already in trainer.py |
| Mixed precision | Manual casting | `torch.amp.autocast` + `GradScaler` | Already in trainer.py |

**Key insight:** The existing `src/training/trainer.py` already handles all training infrastructure. No modifications needed for training - only model architecture changes.

## Common Pitfalls

### Pitfall 1: Channel Mismatch Without Projection

**What goes wrong:** Skip connection fails when input and output channels differ
**Why it happens:** Assuming all blocks maintain same channel count
**How to avoid:** Always check if `in_channels != out_channels` and add 1x1 projection shortcut
**Warning signs:** RuntimeError about tensor shape mismatch during addition

### Pitfall 2: Activation After Skip Connection (Post-Activation Habit)

**What goes wrong:** Applying ReLU after the residual addition defeats the purpose of pre-activation
**Why it happens:** Copying patterns from original ResNet v1
**How to avoid:** Return `out + identity` directly with no activation
**Warning signs:** Slightly worse gradient flow in very deep networks

### Pitfall 3: CBAM Reduction Ratio Too Aggressive

**What goes wrong:** Channels reduced to 0 if reduction > channels
**Why it happens:** Using reduction=16 on layers with < 16 channels
**How to avoid:** `reduced = max(channels // reduction, 1)` or ensure minimum 16 channels
**Warning signs:** Zero-division error or degenerate attention maps

### Pitfall 4: Forgetting to Handle Decoder Upsampling

**What goes wrong:** Transposed convolution output size mismatch
**Why it happens:** Stride=2 doesn't exactly double spatial size without output_padding
**How to avoid:** Use `output_padding=1` for ConvTranspose2d with stride=2
**Warning signs:** Encoder/decoder spatial dimension mismatch

### Pitfall 5: Training Instability with Deeper Networks

**What goes wrong:** Loss explodes or NaN in early epochs
**Why it happens:** Large gradients before network stabilizes
**How to avoid:**
- Learning rate warmup (5-10 epochs starting from lr/10)
- Kaiming initialization (already standard)
- Gradient clipping (already in trainer at max_norm=1.0)
**Warning signs:** NaN loss, validation loss not decreasing

### Pitfall 6: Memory Issues with CBAM on Every Block

**What goes wrong:** GPU OOM with CBAM applied to all blocks
**Why it happens:** Attention maps add memory overhead, especially at high resolution
**How to avoid:** Monitor GPU memory; if needed, apply CBAM only at lower-resolution stages
**Warning signs:** CUDA OOM error, need to reduce batch size significantly

## Code Examples

### Complete Encoder Stage with Pre-Activation Residual Blocks

```python
# Source: Based on user decisions in CONTEXT.md
# 2 residual blocks per stage, 4 stages total

class ResidualEncoderStage(nn.Module):
    """One encoder stage: 2 residual blocks + downsampling."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super().__init__()

        # First block handles channel change and downsampling
        blocks = [PreActResidualBlock(in_channels, out_channels, stride=2)]

        # Remaining blocks maintain channels
        for _ in range(num_blocks - 1):
            blocks.append(PreActResidualBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
```

### Complete Decoder Stage with Upsampling

```python
class ResidualDecoderStage(nn.Module):
    """One decoder stage: upsample + 2 residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super().__init__()

        # Upsample first (using bilinear + 1x1 conv for stability)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Residual blocks at new resolution
        blocks = []
        for _ in range(num_blocks):
            blocks.append(PreActResidualBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.blocks(x)
```

### Variant B: Residual-Only Autoencoder Structure

```python
class ResidualEncoder(nn.Module):
    """Pre-activation residual encoder: 256x256x1 -> 16x16xC"""

    def __init__(self, latent_channels: int = 64, base_channels: int = 64):
        super().__init__()

        # Initial conv (no downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 4 stages with 2 blocks each, doubling channels
        # 256 -> 128 -> 64 -> 32 -> 16
        self.stage1 = ResidualEncoderStage(base_channels, base_channels * 2)      # 128x128
        self.stage2 = ResidualEncoderStage(base_channels * 2, base_channels * 4)  # 64x64
        self.stage3 = ResidualEncoderStage(base_channels * 4, base_channels * 8)  # 32x32
        self.stage4 = ResidualEncoderStage(base_channels * 8, latent_channels)    # 16x16
```

### Combined Loss with Configurable Weights

```python
# Already exists in src/losses/combined.py
# Usage for 0.7 MSE + 0.3 SSIM:
loss_fn = CombinedLoss(mse_weight=0.7, ssim_weight=0.3)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Post-activation ResNet (v1) | Pre-activation ResNet (v2) | 2016 | Better gradient flow for 100+ layer networks |
| SE-Net (channel only) | CBAM (channel + spatial) | 2018 | Spatial attention provides additional benefit |
| Fixed learning rate | Warmup + scheduling | Standard practice | Stability with deeper networks |

**Deprecated/outdated:**
- Using BatchNorm inside CBAM MLP: Original paper doesn't use it
- Parallel channel+spatial attention: Sequential (channel-first) performs better

## Open Questions

Things that couldn't be fully resolved:

1. **Learning rate for deeper network**
   - What we know: Baseline used 1e-3 with Adam; deeper networks often need warmup
   - What's unclear: Optimal warmup duration and final LR for this specific architecture
   - Recommendation: Start with 1e-4 with 5-epoch linear warmup to 1e-3; monitor for instability

2. **Batch size impact with CBAM**
   - What we know: CBAM adds memory overhead; baseline used batch_size=32
   - What's unclear: Whether batch_size=32 will still fit with CBAM on all blocks
   - Recommendation: Test memory usage; may need to reduce to 16 or use gradient accumulation

3. **ENL ratio behavior with attention**
   - What we know: Target is 0.7-1.3; attention may affect speckle differently than baseline
   - What's unclear: How CBAM specifically affects SAR speckle preservation
   - Recommendation: Monitor ENL closely during training; may need to adjust loss weights

## Sources

### Primary (HIGH confidence)
- [PyTorch Lightning ResNet Tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html) - Pre-activation block implementation
- [External-Attention-pytorch CBAM](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py) - CBAM implementation
- [Official CBAM Paper](https://arxiv.org/abs/1807.06521) - Architecture decisions: reduction=16, kernel=7
- [He et al. 2016 - Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) - Pre-activation rationale

### Secondary (MEDIUM confidence)
- [Shadecoder CBAM Guide 2025](https://www.shadecoder.com/topics/convolutional-block-attention-module-a-comprehensive-guide-for-2025) - Best practices, common pitfalls
- [PyTorch torch.nn.init documentation](https://docs.pytorch.org/docs/stable/nn.init.html) - Kaiming initialization
- [pytorch-warmup library](https://github.com/Tony-Y/pytorch_warmup) - Learning rate warmup patterns

### Tertiary (LOW confidence)
- [SAR Autoencoder Variance Analysis 2025](https://www.mdpi.com/2227-7390/13/3/457) - SAR-specific autoencoder considerations
- [ACTD-Net for SAR](https://www.mdpi.com/2304-6732/13/1/46) - Attention mechanisms for SAR imagery

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch native, no new dependencies
- Architecture: HIGH - Well-documented patterns from papers and official implementations
- Pitfalls: HIGH - Common issues well-documented in community
- Training considerations: MEDIUM - SAR-specific tuning may need experimentation

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (30 days - stable domain, established patterns)
