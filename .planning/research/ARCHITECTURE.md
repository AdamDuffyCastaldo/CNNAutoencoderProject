# Architecture Patterns for SAR Image Compression Autoencoders

**Domain:** Learned SAR image compression
**Researched:** 2026-01-21
**Confidence:** MEDIUM (based on existing project knowledge docs + training data; WebSearch unavailable for verification)

## Executive Summary

For compressing 256x256 single-channel SAR patches to 16x-64x compression ratios on an RTX 3070 (8GB), the recommended architecture progression is:

1. **Baseline:** 4-layer strided convolutional encoder-decoder (existing skeleton)
2. **Enhancement 1:** Add residual blocks at each resolution level
3. **Enhancement 2:** Add CBAM attention at bottleneck
4. **Optional:** Self-attention at bottleneck for global context (memory permitting)

This document details the rationale, component designs, memory estimates, and build order.

---

## Recommended Architecture

### Overview Diagram

```
Input: 256x256x1 (normalized SAR patch in [0,1])
         |
    [ENCODER]
         |
    +----v----+
    | Conv 5x5|  stride=2, 1->64 channels
    | BN+LReLU|
    +---------+  -> 128x128x64
         |
    +----v----+
    |ResBlock |  64->64 channels (optional)
    +---------+
         |
    +----v----+
    | Conv 5x5|  stride=2, 64->128 channels
    | BN+LReLU|
    +---------+  -> 64x64x128
         |
    +----v----+
    |ResBlock |  128->128 channels (optional)
    +---------+
         |
    +----v----+
    | Conv 5x5|  stride=2, 128->256 channels
    | BN+LReLU|
    +---------+  -> 32x32x256
         |
    +----v----+
    |ResBlock |  256->256 channels (optional)
    +---------+
         |
    +----v----+
    | Conv 5x5|  stride=2, 256->C_latent
    | (no act)|
    +---------+  -> 16x16xC_latent
         |
    +----v----+
    |  CBAM   |  Channel + Spatial attention (optional)
    +---------+
         |
    [BOTTLENECK: 16x16xC_latent]
         |
    [DECODER - Mirror of Encoder]
         |
    +----v----+
    |Deconv5x5|  stride=2, C_latent->256
    | BN+ReLU |
    +---------+  -> 32x32x256
         |
    +----v----+
    |ResBlock |  (optional)
    +---------+
         |
    [...continue mirroring...]
         |
    +----v----+
    |Deconv5x5|  stride=2, 64->1
    | Sigmoid |
    +---------+  -> 256x256x1

Output: 256x256x1 (reconstructed patch in [0,1])
```

### Compression Ratio vs Latent Channels

| Latent Channels | Latent Size | Compression Ratio | Use Case |
|-----------------|-------------|-------------------|----------|
| 64 | 16x16x64 = 16,384 | 4x | High quality, baseline |
| 32 | 16x16x32 = 8,192 | 8x | Good quality |
| 16 | 16x16x16 = 4,096 | 16x | Balanced |
| 8 | 16x16x8 = 2,048 | 32x | High compression |
| 4 | 16x16x4 = 1,024 | 64x | Extreme compression |

**Calculation:** Input = 256x256x1 = 65,536 values. CR = 65,536 / latent_size.

---

## Component Design Patterns

### Pattern 1: Basic Convolutional Block (ConvBlock)

The fundamental encoder building block.

```
ConvBlock: Conv2d -> BatchNorm -> LeakyReLU(0.2)

Parameters:
- kernel_size: 5x5 (provides receptive field ~61px after 4 layers)
- stride: 2 (halves spatial dimensions)
- padding: 2 (maintains dimension/stride ratio)
- bias: False when using BatchNorm (BN has its own bias)
```

**Why 5x5 kernels:**
- Larger than 3x3 provides better receptive field for SAR structures
- SAR edges and features span multiple pixels due to speckle
- Receptive field after 4 layers: ~61 pixels (covers meaningful structures)

**Why LeakyReLU(0.2):**
- Prevents dead neurons (important for high dynamic range SAR data)
- Allows small negative gradients through

### Pattern 2: Transposed Convolutional Block (DeconvBlock)

The fundamental decoder building block.

```
DeconvBlock: ConvTranspose2d -> BatchNorm -> ReLU

Parameters:
- kernel_size: 5x5 (matches encoder)
- stride: 2 (doubles spatial dimensions)
- padding: 2
- output_padding: 1 (critical for exact 2x upsampling)
```

**Why output_padding=1:**
Formula: output = (input - 1) * stride + kernel - 2*padding + output_padding
For 16->32: (16-1)*2 + 5 - 4 + 1 = 32. Without output_padding, we get 31.

**Why ReLU (not LeakyReLU):**
- Decoder receives varied latent values, ReLU works well
- Slight simplification from encoder

### Pattern 3: Residual Block (ResidualBlock)

Skip connections for gradient flow and detail preservation.

```
ResidualBlock:
    x ----+---> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN --+--> ReLU -> out
          |                                              |
          +---------------(identity)---------------------+

When input/output channels differ:
    x ----+---> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN --+--> ReLU -> out
          |                                              |
          +---> Conv1x1 -> BN ---------------------------+
               (projection shortcut)
```

**Why residual blocks for SAR compression:**
1. **Detail preservation:** SAR edges and texture must survive the bottleneck
2. **Gradient flow:** 256x compression is aggressive; residuals help gradients reach early layers
3. **Identity baseline:** Network starts from "keep input as-is" and learns refinements

**Placement recommendations:**
- **Minimum:** 1-2 blocks at bottleneck (16x16)
- **Better:** 1 block per resolution level
- **Best:** 2 blocks per resolution level (more compute but better quality)

### Pattern 4: Channel Attention (Squeeze-and-Excitation)

Learn which feature channels are important.

```
ChannelAttention:
    Input (HxWxC)
         |
    GlobalAvgPool -> 1x1xC
         |
    FC: C -> C/r -> ReLU -> FC: C/r -> C -> Sigmoid
         |
    Weights: 1x1xC
         |
    Input * Weights -> Output (HxWxC)

Parameters:
- reduction (r): 16 typical, use 8 for small channel counts (<64)
```

**Why for SAR:**
- Different channels detect edges, textures, noise patterns
- Network learns to emphasize edge-detecting channels
- Suppresses noise-capturing channels

### Pattern 5: Spatial Attention

Learn which spatial locations are important.

```
SpatialAttention:
    Input (HxWxC)
         |
    MaxPool across C -> HxWx1
    AvgPool across C -> HxWx1
         |
    Concat -> HxWx2
         |
    Conv 7x7 -> HxWx1 -> Sigmoid
         |
    Attention map: HxWx1
         |
    Input * Attention -> Output (HxWxC)

Parameters:
- kernel_size: 7 (captures local context for attention decision)
```

**Why for SAR:**
- Edges and structures need careful reconstruction (high attention)
- Homogeneous regions can be approximated (lower attention)
- Speckle regions are statistically predictable (lower attention)

### Pattern 6: CBAM (Combined Block Attention Module)

Sequential channel + spatial attention.

```
CBAM:
    Input -> ChannelAttention -> SpatialAttention -> Output

Order matters: Channel first (what features), then spatial (where to look)
```

**Recommendation for SAR:** CBAM at bottleneck is the sweet spot.
- Significant quality improvement
- Minimal additional parameters
- Helps the most compressed representation focus resources

### Pattern 7: Self-Attention (Optional, Advanced)

Global context at bottleneck.

```
SelfAttention at 16x16:
    Attention matrix: 256 x 256 = 65,536 elements
    Memory: ~0.5MB (manageable)

    Query: Conv1x1, C -> C/8
    Key:   Conv1x1, C -> C/8
    Value: Conv1x1, C -> C

    Attention = softmax(Q @ K.T / sqrt(d))
    Output = gamma * (Attention @ V) + Input

    gamma starts at 0, learns to increase
```

**When to use:**
- Only at bottleneck (16x16 is small enough)
- When global structure matters (large SAR features spanning the patch)
- After residual + CBAM experiments show room for improvement

**Memory note:** Self-attention at larger resolutions (32x32+) is expensive. Avoid.

---

## Architecture Variants for Comparison

### Variant A: Plain (Baseline)

```
Encoder: 4x ConvBlock (stride=2)
Decoder: 4x DeconvBlock (stride=2)
```

**Parameters:** ~3-5M (depends on base_channels)
**Memory (batch 8):** ~2GB
**Expected:** Baseline quality, fastest training

### Variant B: Residual

```
Encoder: ConvBlock -> ResBlock -> ConvBlock -> ResBlock -> ...
Decoder: DeconvBlock -> ResBlock -> DeconvBlock -> ResBlock -> ...
```

**Parameters:** ~5-8M
**Memory (batch 8):** ~3GB
**Expected:** +2-3 dB PSNR over baseline

### Variant C: Residual + CBAM at Bottleneck

```
Encoder: [Variant B encoder] -> CBAM
Decoder: [Variant B decoder]
```

**Parameters:** ~6-9M
**Memory (batch 8):** ~3.5GB
**Expected:** +1-2 dB PSNR over Variant B

### Variant D: Full Attention (Every Level)

```
Encoder: ConvBlock -> ResBlock+CBAM -> ConvBlock -> ResBlock+CBAM -> ...
Decoder: Mirror with CBAM at each level
```

**Parameters:** ~8-12M
**Memory (batch 8):** ~5GB
**Expected:** Best quality, slowest training
**Risk:** Diminishing returns, may overfit with limited data

---

## SAR-Specific Architecture Considerations

### Speckle and the Multiplicative Noise Model

SAR images have multiplicative speckle noise: `I_observed = I_true * Speckle`

**Implication for architecture:**
- Working in dB domain converts to additive noise (handled in preprocessing)
- Network sees approximately Gaussian noise patterns
- Standard CNN architectures work well

### Edge Preservation Priority

SAR analysts rely heavily on edges for:
- Feature extraction
- Change detection
- Classification

**Architecture implications:**
1. Use MSE + SSIM loss (SSIM emphasizes structure)
2. Add edge-preserving loss term (optional)
3. Attention mechanisms help focus on edges
4. Residual connections preserve high-frequency detail

### Single-Channel Input

Unlike RGB images, SAR patches are single-channel.

**Implications:**
- First conv layer: 1 input channel (not 3)
- Channel attention at first layer may be less useful (only 1 channel)
- Start channel attention after first convolution (64+ channels)

### Dynamic Range Handling

Even after dB conversion and normalization, SAR has varied statistics.

**Implications:**
- BatchNorm is important for training stability
- Instance Norm or Layer Norm may help if batch statistics vary significantly
- Monitor for batch size sensitivity

---

## Memory and Compute Estimates

### Parameter Count Estimates

```python
# Rough formula for encoder (decoder similar):
# ConvBlock: kernel^2 * in_ch * out_ch + out_ch (BN params)
# ResBlock: 2 * kernel^2 * ch * ch + 2 * ch (BN params)

# Example: base_channels=64, latent=64
# Layer 1: 5*5*1*64 + 64 = 1,664
# Layer 2: 5*5*64*128 + 128 = 204,928
# Layer 3: 5*5*128*256 + 256 = 819,456
# Layer 4: 5*5*256*64 + 64 = 409,664
# Encoder total: ~1.4M

# With ResBlocks (3x3, at each level):
# ResBlock 64:  2*3*3*64*64 + 128 = 73,856
# ResBlock 128: 2*3*3*128*128 + 256 = 295,168
# ResBlock 256: 2*3*3*256*256 + 512 = 1,180,160
# ResBlock 64:  2*3*3*64*64 + 128 = 73,856
# Res total: ~1.6M

# Full model (encoder + decoder + res blocks): ~5-8M parameters
```

### Memory Estimates (Training)

| Configuration | Batch 16 | Batch 8 | Batch 4 |
|---------------|----------|---------|---------|
| Plain 64ch | 4.5 GB | 2.5 GB | 1.5 GB |
| Residual 64ch | 5.5 GB | 3.0 GB | 2.0 GB |
| Res+CBAM 64ch | 6.0 GB | 3.5 GB | 2.0 GB |
| Full Attention | 7.5 GB | 5.0 GB | 3.0 GB |

**Recommendation for 8GB GPU:**
- Batch size 8 for Variant C (Residual + CBAM)
- Batch size 4-8 for experiments with full attention
- Use gradient accumulation if batch size 4 is insufficient

### Training Time Estimates (per epoch, ~5000 patches)

| Variant | Time/Epoch (estimated) |
|---------|------------------------|
| Plain | ~2 min |
| Residual | ~3 min |
| Res+CBAM | ~4 min |
| Full Attention | ~6 min |

---

## Build Order and Dependencies

### Phase 1: Implement Basic Blocks

**Files:** `src/models/blocks.py`

1. `ConvBlock` - basic encoder layer
2. `DeconvBlock` - basic decoder layer
3. Test shapes and gradient flow

**Dependencies:** None
**Output:** Working building blocks

### Phase 2: Implement Plain Encoder-Decoder

**Files:** `src/models/encoder.py`, `src/models/decoder.py`, `src/models/autoencoder.py`

1. `SAREncoder` - 4-layer strided conv encoder
2. `SARDecoder` - 4-layer transposed conv decoder
3. `SARAutoencoder` - combined model with utilities
4. Test end-to-end shape and compression ratio

**Dependencies:** Phase 1 blocks
**Output:** Working baseline autoencoder

### Phase 3: Implement Residual Blocks

**Files:** `src/models/blocks.py`

1. `ResidualBlock` - basic same-resolution residual
2. `ResidualBlockWithDownsample` - residual with stride
3. `ResidualBlockWithUpsample` - residual with transposed conv
4. Test skip connections and gradient flow

**Dependencies:** Phase 1 blocks
**Output:** Residual building blocks

### Phase 4: Integrate Residuals into Encoder-Decoder

**Files:** Update encoder.py, decoder.py

1. Add optional `use_residual` parameter
2. Insert residual blocks after each conv layer
3. Test shape preservation and increased parameter count

**Dependencies:** Phases 2-3
**Output:** Residual encoder-decoder variant

### Phase 5: Implement Attention Mechanisms

**Files:** `src/models/blocks.py`

1. `ChannelAttention` (SE block)
2. `SpatialAttention`
3. `CBAM` (combined)
4. Test attention weight shapes and gradients

**Dependencies:** None (can parallel with Phase 3)
**Output:** Attention modules

### Phase 6: Integrate Attention into Architecture

**Files:** Update autoencoder.py or create attention variants

1. Add CBAM at bottleneck (simplest)
2. Optionally add CBAM at each level
3. Test full model with attention

**Dependencies:** Phases 4-5
**Output:** Full architecture variants ready for training

### Phase 7 (Optional): Self-Attention

**Files:** `src/models/blocks.py`

1. `SelfAttention` module
2. Integrate at bottleneck only
3. Memory profiling to ensure fits in 8GB

**Dependencies:** Phase 6
**Output:** Advanced attention variant

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Attention at High Resolution

**Problem:** Self-attention at 64x64 or 128x128 creates huge attention matrices.
**Solution:** Only use self-attention at bottleneck (16x16). Use CBAM at higher resolutions if needed.

### Anti-Pattern 2: Too Many Consecutive Residual Blocks

**Problem:** Many residual blocks at same resolution without downsampling uses excessive memory.
**Solution:** 1-2 residual blocks per resolution level maximum.

### Anti-Pattern 3: Skipping BatchNorm

**Problem:** Without BatchNorm, training is unstable, especially at high compression ratios.
**Solution:** Always use BatchNorm (or GroupNorm for very small batches).

### Anti-Pattern 4: ReLU Everywhere

**Problem:** ReLU can cause dead neurons, especially with SAR's varied value distributions.
**Solution:** Use LeakyReLU(0.2) in encoder. ReLU is fine in decoder.

### Anti-Pattern 5: No Projection in Residual Shortcuts

**Problem:** When channels/dimensions change, identity shortcut fails.
**Solution:** Always use 1x1 projection conv when in/out channels differ.

### Anti-Pattern 6: Activation on Final Encoder Layer

**Problem:** Activation limits the range of latent values (LeakyReLU clamps negatives to small values).
**Solution:** No activation on final encoder conv. Let latent be unbounded, add activation at decoder start if needed.

---

## Experimental Recommendations

### Baseline Experiments (Required)

| Experiment | Latent Ch | Architecture | Purpose |
|------------|-----------|--------------|---------|
| B1 | 64 | Plain | Baseline at 4x compression |
| B2 | 32 | Plain | Baseline at 8x compression |
| B3 | 16 | Plain | Baseline at 16x compression |
| B4 | 8 | Plain | Baseline at 32x compression |

### Architecture Experiments (After Baseline)

| Experiment | Latent Ch | Architecture | Purpose |
|------------|-----------|--------------|---------|
| A1 | 16 | Residual | Residual impact at target CR |
| A2 | 16 | Res + CBAM bottleneck | Attention impact |
| A3 | 16 | Res + CBAM all levels | Full attention impact |

### Final Tuning (After Finding Sweet Spot)

| Experiment | Latent Ch | Architecture | Purpose |
|------------|-----------|--------------|---------|
| T1 | [best] | [best arch] | Longer training (100 epochs) |
| T2 | [best] | [best arch] | Different loss weights |

---

## Summary Recommendations

### For Your SAR Compression Project

1. **Start with:** Plain 4-layer encoder-decoder (Variant A)
   - Establishes baseline
   - Fast iteration for debugging pipeline

2. **Next step:** Add residual blocks (Variant B)
   - Expected +2-3 dB PSNR improvement
   - Worth the parameter increase

3. **Recommended:** Add CBAM at bottleneck (Variant C)
   - Best quality/cost tradeoff
   - Fits comfortably in 8GB VRAM

4. **Optional:** Full attention if quality insufficient
   - Experiment carefully with memory
   - Likely diminishing returns

### Key Architecture Decisions

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Kernel size | 5x5 | Better receptive field for SAR structures |
| Stride | 2 at each layer | 4 layers reaches 16x16 from 256x256 |
| Activation | LeakyReLU(0.2) encoder, ReLU decoder | Prevent dead neurons |
| Normalization | BatchNorm | Training stability |
| Residual blocks | 1 per resolution level | Detail preservation |
| Attention | CBAM at bottleneck | Best quality/cost ratio |
| Output activation | Sigmoid | Bounds output to [0,1] matching normalized input |

### Build Order Summary

1. ConvBlock, DeconvBlock -> 2. Encoder, Decoder -> 3. ResidualBlock -> 4. Integrate residuals -> 5. Attention blocks -> 6. Integrate attention

---

## Sources and Confidence Assessment

| Finding | Source | Confidence |
|---------|--------|------------|
| 4-layer encoder-decoder for 256->16 | Existing project skeleton, standard practice | HIGH |
| 5x5 kernels for SAR | Existing project docs, domain knowledge | HIGH |
| Residual blocks improve reconstruction | Knowledge doc 02_RESIDUAL_BLOCKS.md | HIGH |
| CBAM for attention | Knowledge doc 03_ATTENTION_MECHANISMS.md | HIGH |
| Memory estimates | Training data + standard PyTorch behavior | MEDIUM |
| Expected PSNR improvements | Training data, no SAR-specific verification | LOW |
| Self-attention at 16x16 only | Standard practice, memory reasoning | MEDIUM |

**Note:** WebSearch was unavailable. Findings rely on existing comprehensive project knowledge documents and training data. Expected quality improvements (+X dB PSNR) are estimates that should be validated experimentally.

---

*Architecture research completed: 2026-01-21*
