# Deep Dive: Attention Mechanisms for Image Processing

This document provides a comprehensive explanation of attention mechanisms, why they improve neural networks, and how to effectively use them in image compression autoencoders.

---

## Table of Contents

1. [The Intuition Behind Attention](#the-intuition-behind-attention)
2. [Channel Attention (SE Blocks)](#channel-attention-se-blocks)
3. [Spatial Attention](#spatial-attention)
4. [CBAM: Combining Channel and Spatial](#cbam-combining-channel-and-spatial)
5. [Self-Attention (Transformers)](#self-attention-transformers)
6. [Attention in Autoencoders](#attention-in-autoencoders)
7. [Implementation Details](#implementation-details)
8. [When to Use What](#when-to-use-what)

---

## The Intuition Behind Attention

### The Problem: Equal Treatment

Standard convolutions treat all channels and spatial locations equally:

```
Conv filter applied uniformly:
[1 2 1]
[2 4 2] × every location in the image, with same weights
[1 2 1]
```

But not all parts of an image are equally important:
- Edges and structures matter more than flat regions
- Some feature channels detect useful patterns, others detect noise
- For reconstruction, some areas need more precision than others

### The Solution: Learned Importance Weights

Attention mechanisms learn **what to focus on**:

```
Input feature map → Attention module → Importance weights
                                             ↓
Input feature map × Importance weights → Refined output
```

The network learns to:
- Emphasize informative channels (channel attention)
- Focus on important spatial locations (spatial attention)
- Both simultaneously (combined attention)

### Analogy: Human Visual Attention

When you look at a complex scene:
- You don't process every pixel equally
- Your eyes focus on salient regions (faces, text, movement)
- Your brain filters out irrelevant background

Attention mechanisms give neural networks similar capabilities.

---

## Channel Attention (SE Blocks)

### Squeeze-and-Excitation (SE)

The SE module (2017) learns to recalibrate channel importance.

**Key insight:** Different channels detect different features. Some features are more relevant for the current input than others.

### Architecture

```
Input: H × W × C
         │
         ↓
┌─── Global Average Pool ───┐
│    (Squeeze: H×W×C → 1×1×C)│
│           │                │
│           ↓                │
│    FC: C → C/r            │
│    (Reduce dimensions)     │
│           │                │
│           ↓                │
│         ReLU               │
│           │                │
│           ↓                │
│    FC: C/r → C            │
│    (Restore dimensions)    │
│           │                │
│           ↓                │
│       Sigmoid              │
│    (Excitation: weights    │
│     between 0 and 1)       │
└───────────┬────────────────┘
            │
            ↓
      Channel weights: 1×1×C
            │
            ↓
Input × Channel weights → Output: H×W×C
(Scale each channel by its weight)
```

### Step-by-Step Example

**Input:** 64×64×128 feature map (128 channels)

**Step 1: Squeeze (Global Average Pooling)**
```
Each of 128 channels → single value (average of all 64×64 pixels)
Output: 1×1×128 (one value per channel)
```

This captures the "global summary" of each channel.

**Step 2: Excitation (FC layers)**
```
128 → 8 (reduction ratio r=16)
ReLU
8 → 128 (restore dimensions)
Sigmoid → values between 0 and 1
Output: 128 weights
```

The two FC layers learn the channel interdependencies:
- First FC compresses information (forces the network to summarize)
- Second FC produces per-channel importance scores

**Step 3: Scale**
```
Original 64×64×128 × weights (1×1×128, broadcast)
= 64×64×128 (each channel scaled by its weight)
```

Channels with weight ~1.0 are preserved.
Channels with weight ~0.0 are suppressed.

### PyTorch Implementation

```python
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        # Squeeze: H×W×C → 1×1×C → C
        y = self.avg_pool(x).view(b, c)

        # Excitation: C → C/r → C
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y.expand_as(x)
```

### Variants

**Max pooling variant:**
```python
# Use max pooling instead of (or alongside) average pooling
max_out = self.max_pool(x).view(b, c)
avg_out = self.avg_pool(x).view(b, c)
y = self.fc(max_out + avg_out)  # Combine both
```

Max pooling captures the "peak activation" while average pooling captures "typical activation."

**ECA (Efficient Channel Attention):**
```python
# 1D convolution instead of FC layers (more efficient)
class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # B×C×1×1
        y = y.squeeze(-1).transpose(-1, -2)  # B×1×C
        y = self.conv(y)  # B×1×C
        y = y.transpose(-1, -2).unsqueeze(-1)  # B×C×1×1
        y = self.sigmoid(y)
        return x * y
```

### What SE Blocks Learn

After training, you can visualize channel weights:

```
Channel  0: edge detector       → weight 0.85 (important)
Channel  1: horizontal texture  → weight 0.72
Channel  2: noise pattern       → weight 0.15 (suppressed)
Channel  3: smooth gradient     → weight 0.45
...
Channel 63: diagonal edges      → weight 0.91 (very important)
```

The network learns which feature detectors are useful for the task.

---

## Spatial Attention

### The Concept

While channel attention asks "which features?", spatial attention asks "which locations?"

**Intuition:** For SAR image compression:
- Edge regions need careful reconstruction (high attention)
- Uniform regions can be approximated (lower attention)
- Noise regions might need special handling

### Architecture

```
Input: H × W × C
         │
    ┌────┴────┐
    │         │
    ↓         ↓
 MaxPool   AvgPool
(across C) (across C)
    │         │
    ↓         ↓
 H×W×1     H×W×1
    │         │
    └────┬────┘
         │
         ↓
    Concatenate
       H×W×2
         │
         ↓
    Conv 7×7
       H×W×1
         │
         ↓
      Sigmoid
    (spatial weights)
         │
         ↓
Input × spatial weights → Output: H×W×C
```

### Step-by-Step Example

**Input:** 64×64×128 feature map

**Step 1: Pool across channels**
```
MaxPool: For each spatial position, take max across 128 channels
         → 64×64×1 (highlights where any channel is strongly activated)

AvgPool: For each spatial position, take average across 128 channels
         → 64×64×1 (shows overall activation level)
```

**Step 2: Concatenate**
```
Stack: 64×64×2 (two channels: max and avg)
```

**Step 3: Convolution**
```
7×7 Conv (large receptive field to consider neighborhood)
64×64×2 → 64×64×1

Sigmoid: Each value between 0 and 1
→ 64×64 attention map
```

**Step 4: Scale**
```
Original 64×64×128 × attention map (64×64×1, broadcast)
= 64×64×128 (each spatial location scaled)
```

### PyTorch Implementation

```python
class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B×1×H×W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # B×1×H×W

        # Concatenate
        y = torch.cat([avg_out, max_out], dim=1)  # B×2×H×W

        # Conv + sigmoid
        y = self.conv(y)  # B×1×H×W
        y = self.sigmoid(y)

        return x * y
```

### Visualizing Spatial Attention

The attention map shows what the network focuses on:

```
Original Image:        Attention Map:
┌─────────────────┐    ┌─────────────────┐
│     sky         │    │ 0.2  0.2  0.2   │  ← Low attention (uniform)
│─────────────────│    │─────────────────│
│   building      │    │ 0.9  0.8  0.9   │  ← High attention (edges/structure)
│   with edges    │    │ 0.8  0.7  0.8   │
│─────────────────│    │─────────────────│
│    ground       │    │ 0.4  0.4  0.4   │  ← Medium attention (texture)
└─────────────────┘    └─────────────────┘
```

---

## CBAM: Combining Channel and Spatial

### The Architecture

CBAM (Convolutional Block Attention Module) applies channel and spatial attention sequentially:

```
Input
  │
  ↓
Channel Attention → Intermediate
  │
  ↓
Spatial Attention → Output
```

**Why sequential (not parallel)?**
- Channel attention first decides "which features to look at"
- Spatial attention then decides "where to look" for those features
- This order works better empirically

### PyTorch Implementation

```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
```

### CBAM in Residual Blocks

CBAM is often added at the end of residual blocks:

```
x ──────────────────────────────────────────────────┐
│                                                   │
└→ Conv → BN → ReLU → Conv → BN → CBAM → (+) → ReLU
              F(x)             attention
```

```python
class ResidualBlockWithCBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels, reduction)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)  # Apply attention

        return F.relu(out + identity)
```

---

## Self-Attention (Transformers)

### The Concept

SE and spatial attention are "local" — they don't capture relationships between distant pixels. **Self-attention** captures global relationships.

**Question:** How does pixel (10, 10) relate to pixel (200, 200)?

Self-attention computes:
- Every position attends to every other position
- Learns which positions are related

### Query-Key-Value Framework

Self-attention uses three projections of the input:

```
Input X: N positions × D dimensions

Query (Q): "What am I looking for?" — Q = X × W_Q
Key (K): "What do I contain?" — K = X × W_K
Value (V): "What information do I provide?" — V = X × W_V

Attention: softmax(Q × K^T / √d) × V
```

**Intuition:**
- Q asks "what patterns should I pay attention to?"
- K answers "here's what patterns I have"
- V provides "here's my actual content"
- Attention weights decide how much each position contributes to each other

### Computational Cost

For an image with H×W pixels:

```
Attention matrix: (H×W) × (H×W)
For 64×64 image: 4096 × 4096 = 16.7 million elements!
For 256×256 image: 65536 × 65536 = 4.3 billion elements!
```

This is why self-attention is expensive for images.

### Efficient Variants

**1. Multi-Head Attention:**
Split into multiple "heads" that attend to different patterns:
```python
# 8 heads, each with d/8 dimensions
heads = [attention(Q_i, K_i, V_i) for i in range(8)]
output = concat(heads) × W_O
```

**2. Local Self-Attention:**
Only attend to nearby pixels (e.g., 7×7 window):
```
Attention cost: (H×W) × (7×7) = 49 × (H×W)
Much cheaper than full attention
```

**3. Axial Attention:**
Factorize into row attention + column attention:
```
Row attention: H × (W × W)
Column attention: W × (H × H)
Total: H×W² + W×H² instead of (H×W)²
```

**4. Linear Attention:**
Approximate attention to reduce to O(N) instead of O(N²):
```python
# Linear attention
attention = (elu(Q) + 1) @ ((elu(K) + 1).T @ V)
```

### PyTorch Implementation (Basic)

```python
class SelfAttention(nn.Module):
    """Self-attention for feature maps."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Project to Q, K, V
        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # B×N×C'
        k = self.key(x).view(B, -1, N)  # B×C'×N
        v = self.value(x).view(B, -1, N)  # B×C×N

        # Attention
        attention = F.softmax(torch.bmm(q, k), dim=-1)  # B×N×N
        out = torch.bmm(v, attention.permute(0, 2, 1))  # B×C×N
        out = out.view(B, C, H, W)

        # Residual connection with learnable weight
        return self.gamma * out + x
```

### When to Use Self-Attention

| Scenario | Recommendation |
|----------|----------------|
| Small feature maps (16×16, 32×32) | Can use full self-attention |
| Large feature maps (128×128+) | Use local or linear variants |
| Capturing global structure | Self-attention helps |
| Computational budget limited | Skip self-attention, use CBAM |

**For your SAR autoencoder:** Self-attention at the bottleneck (16×16) is feasible and can help capture global image structure.

---

## Attention in Autoencoders

### Where to Place Attention

**Option 1: Bottleneck only**
```
Encoder → Bottleneck (with attention) → Decoder
```
- Cheapest option
- Helps the most compressed representation focus on important features
- Good starting point

**Option 2: After each encoder block**
```
Conv → Attention → Conv → Attention → ... → Bottleneck
```
- Encoder learns to focus progressively
- More expensive
- Can improve quality

**Option 3: After each decoder block**
```
Bottleneck → ... → Conv → Attention → Conv → Attention → Output
```
- Decoder focuses on reconstructing important regions
- Helps with fine details
- Useful for high-quality reconstruction

**Option 4: Everywhere**
```
Attention after every block in encoder and decoder
```
- Maximum capacity
- Most expensive
- May be overkill for simple tasks

### Attention for SAR Compression

SAR images have specific characteristics:

**Edges and structures:**
- Important for downstream analysis
- Should have high attention

**Speckle noise:**
- Statistically predictable
- Can be reconstructed approximately
- Should have lower attention

**Homogeneous regions:**
- Easy to compress
- Don't need special attention

CBAM can learn these patterns:
- Channel attention: Emphasize edge-detecting channels
- Spatial attention: Focus on structurally important regions

### Example Architecture

```python
class AttentionAutoencoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()

        # Encoder with CBAM at bottleneck
        self.encoder = nn.Sequential(
            ConvBlock(1, 64, stride=2),      # 256 → 128
            ConvBlock(64, 128, stride=2),    # 128 → 64
            ConvBlock(128, 256, stride=2),   # 64 → 32
            ConvBlock(256, latent_channels, stride=2),  # 32 → 16
            CBAM(latent_channels),           # Attention at bottleneck!
        )

        # Decoder
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, 256),  # 16 → 32
            DeconvBlock(256, 128),              # 32 → 64
            DeconvBlock(128, 64),               # 64 → 128
            DeconvBlock(64, 1, activation='sigmoid'),  # 128 → 256
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
```

---

## Implementation Details

### Hyperparameters

**Channel attention reduction ratio:**
```python
reduction = 16  # Common default
# Higher = fewer parameters, less capacity
# Lower = more parameters, more capacity
```

For small channel counts (< 64), use smaller reduction (4 or 8).

**Spatial attention kernel size:**
```python
kernel_size = 7  # Common default
# Larger = bigger receptive field, more computation
# Smaller = local attention only
```

### Memory Considerations

Attention modules add parameters and memory:

```
SE block (channels=128, reduction=16):
Parameters: 128×8 + 8×128 = 2048
Memory: Negligible (just 1×1×128 intermediate)

Spatial attention (64×64 feature map):
Memory: 64×64×2 = 8192 floats intermediate
Parameters: 7×7×2×1 = 98

Self-attention (16×16 feature map):
Attention matrix: 256×256 = 65536 floats
Memory scales with (H×W)²
```

**For 8GB GPU:** CBAM is always fine. Self-attention should be limited to small feature maps (≤32×32).

### Gradient Flow

Attention modules should not block gradients:

```python
# Good: Multiplicative attention (gradient flows through both paths)
output = input * attention_weights

# Also good: Additive with residual
output = input + attention_weights * input
       = input * (1 + attention_weights)

# Bad: Discrete attention (non-differentiable)
output = input * (attention_weights > 0.5)  # Don't do this!
```

### Initialization

Attention modules should start close to identity:

```python
# For SE blocks: Initialize last FC layer to small values
nn.init.zeros_(self.fc[-2].weight)  # Output near 0 → sigmoid ≈ 0.5

# For learned gamma in self-attention
self.gamma = nn.Parameter(torch.zeros(1))  # Start at 0, learn to increase
```

This ensures attention doesn't disrupt training early on.

---

## When to Use What

### Decision Guide

| Need | Solution | Overhead |
|------|----------|----------|
| "Which features matter?" | Channel attention (SE) | Low |
| "Which locations matter?" | Spatial attention | Low |
| "Both features and locations" | CBAM | Medium |
| "Global relationships" | Self-attention | High |
| "Minimal overhead" | No attention | None |

### For Your SAR Autoencoder

**Recommended starting point:**
1. CBAM at bottleneck only
2. Train and evaluate
3. If quality insufficient, add CBAM to decoder
4. If still insufficient, try self-attention at bottleneck

**Experiment plan:**
| Variant | Attention | Expected Quality | Expected Cost |
|---------|-----------|------------------|---------------|
| Baseline | None | Baseline | Baseline |
| V1 | CBAM @ bottleneck | +2-5% SSIM | +5% time |
| V2 | CBAM @ all levels | +5-10% SSIM | +15% time |
| V3 | Self-attention @ bottleneck | +3-7% SSIM | +10% time |

### Summary

- **Channel attention (SE):** Learn which features to emphasize
- **Spatial attention:** Learn where to focus
- **CBAM:** Both channel and spatial, sequentially
- **Self-attention:** Capture global relationships (expensive)

For SAR compression, CBAM is likely the sweet spot: significant quality improvement with reasonable overhead.
