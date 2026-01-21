# SAR Autoencoder Project - Complete Research Compilation

**Project:** CNN Autoencoder for Sentinel-1 SAR Image Compression
**Compiled:** 2026-01-21
**Purpose:** Complete reference document for understanding all key decisions and concepts

---

# Part 1: Project Research

Research conducted to inform architecture decisions, feature priorities, and implementation approach.

---

## 1.1 Technology Stack Analysis

### Executive Summary

The existing stack is well-suited for this project. **Keep PyTorch 2.0+ as the core** with targeted additions for SSIM computation and SAR data handling. No major stack changes needed.

### Recommended Stack

| Technology | Version | Purpose | Decision |
|------------|---------|---------|----------|
| **PyTorch** | >=2.0.0 | Model development, training | KEEP |
| **torchvision** | >=0.15.0 | Image transforms, utilities | KEEP |
| **pytorch-msssim** | >=1.0.0 | SSIM and MS-SSIM loss | KEEP |
| **rasterio** | >=1.3.0 | GeoTIFF loading | KEEP |
| **TensorBoard** | >=2.12.0 | Training visualization | KEEP |
| **scikit-image** | >=0.20.0 | Quality metrics | KEEP |

### Libraries to Avoid

| Library | Why Avoid |
|---------|-----------|
| PyTorch Lightning | Unnecessary refactoring for project size |
| CompressAI | Entropy coding out of scope for v1 |
| Custom CUDA kernels | Standard ops are sufficient |
| Weights & Biases | TensorBoard already adequate |

### Hardware Optimization (RTX 3070, 8GB)

- Batch size 8-16 fits comfortably
- Keep tensor dimensions as multiples of 8 for Tensor Core utilization
- Mixed precision (BFloat16) available if VRAM becomes constrained

---

## 1.2 Feature Priorities

### Table Stakes (Must Have)

| Feature | Complexity | Rationale |
|---------|------------|-----------|
| dB transform (linear to log) | Low | SAR dynamic range incompatible with neural networks without this |
| Invalid value handling | Low | log(0) = -inf crashes training |
| Dynamic range clipping | Low | Outliers destabilize training |
| Normalization to [0,1] | Low | NN weight initialization assumes bounded inputs |
| Patch extraction (256x256) | Low | Full SAR images exceed GPU memory |
| Quality filtering | Low | Remove no-data and corrupted patches |
| CNN encoder/decoder | Medium | Core functionality |
| Configurable latent dimensions | Low | Control compression ratio |
| MSE + SSIM combined loss | Low | Balanced pixel accuracy and structure |
| PSNR, SSIM metrics | Low | Standard quality measurement |
| Checkpointing | Low | Resume training, save best models |

### Should Have (Quality Gate)

| Feature | Complexity | Rationale |
|---------|------------|-----------|
| Residual/skip connections | Medium | +2-3 dB PSNR improvement |
| ENL ratio metric | Medium | SAR-specific speckle validation |
| EPI (Edge Preservation Index) | Medium | SAR-specific edge validation |
| Patch-based inference with tiling | Medium | Process full satellite images |
| Patch blending (cosine ramp) | Medium | Avoid visible tile seams |

### Nice to Have (Differentiators)

| Feature | Complexity | Rationale |
|---------|------------|-----------|
| CBAM attention at bottleneck | Medium | Edge quality improvement |
| Multiple compression ratio variants | Low | Comparison study |
| Rate-distortion curves | Medium | Professional evaluation |

### Anti-Features (Defer to Post-MVP)

| Feature | Why Defer |
|---------|-----------|
| End-to-end entropy coding | Massive complexity, marginal gains for research |
| Variable-rate single model | High complexity |
| Multi-polarization (VV+VH) | Doubles complexity, prove concept with single channel first |
| Phase preservation | Rarely needed, huge complexity |
| Real-time streaming | Different architecture requirements |

---

## 1.3 Architecture Decisions

### Recommended Architecture Progression

1. **Baseline (Variant A):** Plain 4-layer strided convolutions
   - Parameters: ~3-5M
   - Memory (batch 8): ~2GB
   - Purpose: Establish baseline, fast debugging

2. **Enhanced (Variant B):** Add residual blocks at each resolution level
   - Parameters: ~5-8M
   - Memory (batch 8): ~3GB
   - Expected: +2-3 dB PSNR over baseline

3. **Recommended (Variant C):** Residual + CBAM attention at bottleneck
   - Parameters: ~6-9M
   - Memory (batch 8): ~3.5GB
   - Expected: +1-2 dB PSNR over Variant B
   - Best quality/cost tradeoff for 8GB VRAM

### Architecture Diagram

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
    |ResBlock |  (optional)
    +---------+
         |
    +----v----+
    | Conv 5x5|  stride=2, 64->128 channels
    +---------+  -> 64x64x128
         |
    [...continue to 16x16xC_latent...]
         |
    +----v----+
    |  CBAM   |  (optional attention)
    +---------+
         |
    [BOTTLENECK: 16x16xC_latent]
         |
    [DECODER - Mirror of Encoder]
         |
Output: 256x256x1 (reconstructed)
```

### Key Architecture Parameters

| Parameter | Recommendation | Rationale |
|-----------|----------------|-----------|
| Kernel size | 5x5 | Better receptive field for SAR structures |
| Stride | 2 at each layer | 4 layers: 256x256 -> 16x16 |
| Encoder activation | LeakyReLU(0.2) | Prevents dead neurons |
| Decoder activation | ReLU | Standard choice |
| Output activation | Sigmoid | Bounds output to [0,1] |
| Normalization | BatchNorm | Training stability |
| Final encoder layer | No activation | Allow unbounded latent values |

### Compression Ratio Targets

| Latent Channels | Latent Size | Compression Ratio | Use Case |
|-----------------|-------------|-------------------|----------|
| 64 | 16x16x64 | 4x | High quality baseline |
| 32 | 16x16x32 | 8x | Good quality |
| **16** | **16x16x16** | **16x** | **Balanced (start here)** |
| 8 | 16x16x8 | 32x | High compression |
| 4 | 16x16x4 | 64x | Extreme (risky for SAR) |

---

## 1.4 Critical Pitfalls

### Severity: CRITICAL (Must Address First)

| # | Pitfall | Impact | Prevention |
|---|---------|--------|------------|
| 1 | Training on linear intensity values | Model learns nothing useful | Always convert to dB: `dB = 10 * log10(intensity + 1e-10)` |
| 2 | Inconsistent preprocessing parameters | Silent metric corruption | Compute vmin/vmax from training set ONLY; save with checkpoint |
| 3 | Missing SAR-specific metrics | Over-smoothing goes undetected | Always report ENL ratio and EPI |
| 4 | MSE-only loss | Model becomes denoiser | Use balanced loss: 0.5*MSE + 0.5*SSIM |
| 5 | Not handling invalid values | Training crashes with NaN | Apply noise floor: `max(intensity, 1e-10)` |

### Severity: MAJOR (Address in Early Phases)

| # | Pitfall | Impact | Prevention |
|---|---------|--------|------------|
| 6 | Latent space too small | Cannot represent SAR detail | Start with 16x compression, not 32x or 64x |
| 7 | BatchNorm inference issues | Inconsistent deployment | Call model.eval(); test single-sample inference |
| 8 | Output activation mismatch | Clipping or unbounded outputs | Sigmoid for [0,1] normalized input |
| 9 | Memory overflow on full images | Cannot process real data | Implement tiled inference with overlap |
| 10 | Checkpoint incompatibility | Lost training progress | Save config dict and preprocessing params |

### Quick Checklist Before Training

- [ ] Data converted to dB (not linear intensity)
- [ ] Invalid values handled (noise floor applied)
- [ ] Preprocessing params computed from training set only
- [ ] Preprocessing params saved for inference
- [ ] Output activation matches data normalization
- [ ] ENL ratio and EPI included in evaluation metrics
- [ ] Loss function balances MSE and SSIM
- [ ] Learning rate is conservative (1e-4)
- [ ] Checkpoint saves config, params, and model state

---

# Part 2: Knowledge Base

Deep explanations of key concepts and techniques used in this project. This section provides the theoretical foundation needed to understand why each design decision was made.

---

## 2.1 Convolutional Neural Networks

### Why Convolutions for Images?

**The Problem with Fully Connected Layers:**

Imagine processing a 256×256 grayscale image with a fully connected (dense) layer:

```
Input: 256 × 256 = 65,536 neurons
Output: 65,536 neurons (same size)
Weights: 65,536 × 65,536 = 4,294,967,296 parameters (4.3 billion!)
```

This is:
- **Computationally impossible** — too many parameters to train
- **Wasteful** — doesn't exploit image structure
- **Overfitting prone** — far more parameters than training examples

**What Makes Images Special:**

Images have three key properties that convolutions exploit:

| Property | Description | How Convolution Helps |
|----------|-------------|----------------------|
| Local Patterns | A pixel's meaning depends mostly on its neighbors | Small kernel (3×3) only looks at neighbors |
| Translation Invariance | A cat in the top-left is the same as one in bottom-right | Same kernel applied everywhere (weight sharing) |
| Hierarchical Structure | Low-level edges → mid-level shapes → high-level objects | Stack layers: early = edges, later = complex patterns |

**Parameter Comparison:**
```
Fully connected: 4.3 billion parameters
Convolution (3×3, 64 channels): 9 × 64 = 576 parameters

That's a 7-million-fold reduction!
```

### How Convolution Works

**The Intuition: Sliding Window Feature Detection**

Imagine you have a small "template" (kernel) that detects a specific pattern. You slide this template across the entire image, computing how well the local region matches at each position.

**Step-by-Step Example:**

Input image patch (5×5) with a vertical edge:
```
[10  10  10  50  50]
[10  10  10  50  50]
[10  10  10  50  50]
[10  10  10  50  50]
[10  10  10  50  50]
```

Kernel (3×3) for vertical edge detection:
```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Convolution at position (1,1) - uniform region:**
```
Extract 3×3 region:        Element-wise multiply:        Sum:
[10  10  10]               [-10   0  10]
[10  10  10]       ×       [-10   0  10]         =      0
[10  10  10]               [-10   0  10]
```
Result: 0 (no edge here, values are uniform)

**Convolution at position (1,2) - on the edge:**
```
Extract 3×3 region:        Element-wise multiply:        Sum:
[10  10  50]               [-10   0  50]
[10  10  50]       ×       [-10   0  50]         =      120
[10  10  50]               [-10   0  50]
```
Result: 120 (strong positive response — edge detected!)

**The Mathematics:**

For a 2D convolution (technically cross-correlation in deep learning):

```
Output[i,j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m, n]
```

Where:
- `i, j` = position in output
- `m, n` = position within kernel
- Σ = sum over all kernel positions

### Key Parameters

**Kernel Size:**

| Size | Parameters | Use Case |
|------|------------|----------|
| 3×3 | 9 | Most common, efficient. Two 3×3 = same receptive field as one 5×5 but fewer params |
| 5×5 | 25 | Larger receptive field for SAR structures |
| 1×1 | 1 | Only mixes channels, no spatial mixing (used to change channel count cheaply) |
| 7×7 | 49 | Sometimes in first layer to quickly capture low-frequency patterns |

**Stride:**

Stride = how far the kernel moves between positions.

| Stride | Effect | Output Size |
|--------|--------|-------------|
| 1 | No downsampling | Same as input (with proper padding) |
| 2 | 2× downsampling | Half the input size |

**Why use stride for downsampling instead of pooling?**
- Learnable downsampling (network chooses what to keep)
- Fewer operations
- Better gradient flow (no pooling non-linearity)

**Padding:**

Padding adds border pixels to control output size.

```
No padding (valid):     Same padding:
Input: 8×8              Input: 8×8
Kernel: 3×3             Kernel: 3×3, padding=1
Output: 6×6             Output: 8×8 (same size!)
```

**Padding formula for "same" output:** `padding = (kernel_size - 1) / 2`

**Output Size Formula:**
```
Output_size = floor((Input_size + 2×padding - kernel_size) / stride) + 1

Examples:
Input=256, kernel=3, padding=1, stride=1 → Output=256
Input=256, kernel=3, padding=1, stride=2 → Output=128
Input=256, kernel=5, padding=2, stride=2 → Output=128
```

**Receptive Field:**

The receptive field is how much of the input each output pixel "sees":

```
Layer 1 (3×3): Each output sees 3×3 input region
Layer 2 (3×3): Each output sees 5×5 input region
Layer 3 (3×3): Each output sees 7×7 input region

With stride-2 downsampling:
Layer 1 (3×3, stride 2): 3×3 receptive field
Layer 2 (3×3, stride 2): 7×7 receptive field
Layer 3 (3×3, stride 2): 15×15 receptive field
Layer 4 (3×3, stride 2): 31×31 receptive field
```

For your 256×256 input with 4 stride-2 layers, the bottleneck "sees" essentially the whole image.

### Multiple Channels

Real images have multiple channels. The kernel becomes 3D:

```
Input: H × W × C_in
Kernel: K × K × C_in (one kernel)
Output: H × W × 1 (one channel)
```

To produce multiple output channels, use multiple kernels:
```
Input: H × W × C_in
Kernels: C_out kernels, each K × K × C_in
Output: H × W × C_out
Total parameters: K × K × C_in × C_out
```

**Example:**
```
Input: 256 × 256 × 64
Kernel: 3 × 3, 128 output channels
Parameters: 3 × 3 × 64 × 128 = 73,728
Output: 256 × 256 × 128
```

### Transposed Convolutions (Upsampling)

The encoder reduces spatial size (256 → 128 → 64 → 32 → 16).
The decoder must increase it back (16 → 32 → 64 → 128 → 256).

**How Transposed Convolution Works:**

Think of it as "reverse" convolution:
- Regular conv: many input pixels → one output pixel
- Transposed conv: one input pixel → many output pixels

For each input pixel, "stamp" the kernel onto the output, then sum overlapping regions.

**Checkerboard Artifacts:**

A common problem when kernel_size is not divisible by stride:

```
Bad: kernel=3, stride=2 (3/2 = 1.5, not integer)
    Some output positions receive more contributions than others
    → Creates checkerboard pattern

Good: kernel=4, stride=2 (4/2 = 2, integer)
Good: kernel=2, stride=2 (2/2 = 1, integer)
```

**Practical recommendation for stride=2 upsampling:**
```python
nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
```

**Output Size Formula for Transposed Conv:**
```
Output_size = (Input_size - 1) × stride - 2×padding + kernel_size + output_padding
```

### Practical Conv Layer Design for SAR Autoencoder

**Encoder layers:**
```python
# Layer 1: 256×256×1 → 128×128×64
nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
nn.BatchNorm2d(64)
nn.LeakyReLU(0.2)

# Layer 2: 128×128×64 → 64×64×128
nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
nn.BatchNorm2d(128)
nn.LeakyReLU(0.2)

# Continue pattern to reach 16×16×C_latent
```

**Decoder layers:**
```python
# Layer 1: 16×16×C_latent → 32×32×256
nn.ConvTranspose2d(C_latent, 256, kernel_size=4, stride=2, padding=1)
nn.BatchNorm2d(256)
nn.ReLU()

# Continue pattern...

# Final layer: → 256×256×1 with Sigmoid
nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
nn.Sigmoid()  # Output bounded to [0, 1]
```

**Key choices:**
- kernel_size=5 for encoder (better receptive field for SAR)
- kernel_size=4 for transposed conv (avoids checkerboard)
- stride=2 for spatial reduction/expansion
- LeakyReLU(0.2) for encoder (prevents dead neurons)
- ReLU for decoder (standard choice)
- Sigmoid for final output (constrains to [0, 1] matching normalized input)

---

## 2.2 Residual Blocks

### The Deep Learning Problem

**Deeper Networks Should Be Better:**

Intuitively, a deeper network should learn everything a shallower network can, plus more:
```
Shallow network (6 layers): Can learn functions of complexity X
Deep network (20 layers): Should learn functions of complexity ≥ X
```

At worst, extra layers could just learn identity mappings (pass input unchanged).

**But That's Not What Happened (Before 2015):**

```
Experiment on ImageNet (circa 2014):
18-layer network: 27.9% error
34-layer network: 28.5% error  ← Deeper is WORSE!
```

This wasn't overfitting (training error was also higher). The deeper network was fundamentally harder to optimize.

### The Vanishing/Exploding Gradient Problem

During backpropagation, gradients flow backward through layers:
```
Loss → Layer N → Layer N-1 → ... → Layer 2 → Layer 1
```

At each layer, gradients are multiplied by the layer's weights. With many layers:

**Vanishing gradients:** If weights typically < 1
```
Gradient at layer 1 ≈ gradient at layer N × (0.9)^N
For N=50: 0.9^50 ≈ 0.005 (gradient almost disappears!)
```

**Exploding gradients:** If weights typically > 1
```
Gradient at layer 1 ≈ gradient at layer N × (1.1)^N
For N=50: 1.1^50 ≈ 117 (gradient explodes!)
```

**Result:** Early layers barely learn (vanishing) or training becomes unstable (exploding).

### The Residual Learning Insight

**The Key Observation:**

If deep networks are hard to train, but shallow networks work, can we make deep networks behave more like shallow ones?

**Insight:** Instead of learning the full mapping H(x), learn the *residual* F(x) = H(x) - x

```
Traditional: Learn H(x) directly
Residual: Learn F(x) = H(x) - x, then H(x) = F(x) + x
```

**Why This Helps:**

1. **Identity is easy to learn:** If the optimal transformation is identity (no change), the network just needs to learn F(x) = 0 (all weights → 0). Learning "do nothing" is much easier than learning a complex transformation that happens to equal the input.

2. **Gradients flow directly:** The skip connection provides a "highway" for gradients:
```
Without skip: gradient must flow through all transformations
With skip: gradient can flow directly through the addition
```

### Basic Residual Block Structure

```
         ┌─────────────────────────────────────┐
         │                                     │
Input x ─┤                                     │
         │                                     ↓
         └──→ [Conv → BN → ReLU → Conv → BN] ──(+)──→ ReLU ──→ Output
                        F(x)                    │
                                                │
                                         F(x) + x
```

**PyTorch Implementation:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # Save input for skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # Skip connection!
        out = F.relu(out)

        return out
```

### The Mathematics of Gradient Flow

**Without Skip Connections:**

For a network of L layers, gradient at layer 1:
```
∂L/∂x = ∂L/∂f_L × ∂f_L/∂f_{L-1} × ... × ∂f_2/∂f_1 × ∂f_1/∂x
```

This is a **product of L terms**. If any term is small, the product vanishes.

**With Skip Connections:**

With residual blocks, each block computes y = F(x) + x:
```
∂y/∂x = ∂F(x)/∂x + 1
```

For L residual blocks:
```
∂L/∂x = ∂L/∂y_L × (∂F_L/∂y_{L-1} + 1) × ... × (∂F_1/∂x + 1)
```

Each term is **(something + 1)**, not just **(something)**.

**Key insight:** Even if all ∂F/∂y terms are 0, the gradient is still:
```
∂L/∂x = ∂L/∂y_L × 1 × 1 × ... × 1 = ∂L/∂y_L
```

The gradient flows directly from output to input!

### Residual Blocks with Dimension Changes

Basic residual blocks require input and output to have the same dimensions:
```
out = F(x) + x  # Only works if F(x) and x have same shape!
```

But encoder/decoder need to change dimensions:
- Encoder: spatial ↓, channels ↑
- Decoder: spatial ↑, channels ↓

**Solution: Projection Shortcut**

When dimensions change, transform the skip connection too:

```python
class ResidualBlockDown(nn.Module):
    """Residual block that halves spatial size and changes channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip path (projection to match dimensions)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.skip(x)  # Project to match dimensions

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return F.relu(out + identity)
```

**When to Use Projections:**

| Situation | Skip Connection |
|-----------|----------------|
| Same spatial, same channels | Identity (just x) |
| Same spatial, different channels | 1×1 conv |
| Different spatial, same channels | Strided/transposed conv |
| Different spatial, different channels | Strided/transposed 1×1 conv |

### Residual Block Variants

**Pre-Activation ResNet (ResNet v2):**

Original (post-activation):
```
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
```

Pre-activation (often better):
```
x → BN → ReLU → Conv → BN → ReLU → Conv → (+x)
```

Why pre-activation is often better:
- Cleaner gradient flow (ReLU doesn't gate the skip path)
- BN acts as pre-processing for each weight layer
- Easier to train very deep networks

**Bottleneck Block (for deeper networks):**

Reduce computation with 1×1 convolutions:
```
x → Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → (+x) → ReLU
   (reduce)                (process)              (expand)
   256→64                   64→64                  64→256
```

Parameter comparison:
```
Standard (two 3×3 with 256 channels): 1.2M parameters
Bottleneck (1×1 → 3×3 → 1×1 with 64 intermediate): 69K parameters
```
17× fewer parameters for similar capacity!

### Why Use Residual Blocks in Autoencoders?

1. **Detail Preservation:** The encoder must compress drastically (256×256 → 16×16). Residual connections help preserve fine details that might otherwise be lost.

2. **Gradient Flow Through Bottleneck:** The bottleneck is the information choke point. Gradients must flow through it during training. Residual connections ensure the bottleneck doesn't block gradients.

3. **Identity as Baseline:** At each stage, the network starts from "keep the input as-is" and learns refinements. This is especially useful in the decoder where the goal is reconstruction.

**Expected Improvement:** +2-3 dB PSNR over plain convolutions at same compression ratio.

---

## 2.3 Attention Mechanisms

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

### Channel Attention (Squeeze-and-Excitation Blocks)

The SE module (2017) learns to recalibrate channel importance.

**Key insight:** Different channels detect different features. Some features are more relevant for the current input than others.

**Architecture:**
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

**Step-by-Step Example:**

Input: 64×64×128 feature map (128 channels)

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
The two FC layers learn the channel interdependencies.

**Step 3: Scale**
```
Original 64×64×128 × weights (1×1×128, broadcast)
= 64×64×128 (each channel scaled by its weight)
```
Channels with weight ~1.0 are preserved; channels with weight ~0.0 are suppressed.

**PyTorch Implementation:**
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

### Spatial Attention

While channel attention asks "which features?", spatial attention asks "which locations?"

**Intuition for SAR compression:**
- Edge regions need careful reconstruction (high attention)
- Uniform regions can be approximated (lower attention)
- Noise regions might need special handling

**Architecture:**
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

**PyTorch Implementation:**
```python
class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super().__init__()
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

**Visualizing Spatial Attention:**
```
Original Image:        Attention Map:
┌─────────────────┐    ┌─────────────────┐
│     sky         │    │ 0.2  0.2  0.2   │  ← Low attention (uniform)
│─────────────────│    │─────────────────│
│   building      │    │ 0.9  0.8  0.9   │  ← High attention (edges)
│   with edges    │    │ 0.8  0.7  0.8   │
│─────────────────│    │─────────────────│
│    ground       │    │ 0.4  0.4  0.4   │  ← Medium attention (texture)
└─────────────────┘    └─────────────────┘
```

### CBAM: Combining Channel and Spatial

CBAM (Convolutional Block Attention Module) applies channel and spatial attention sequentially:

```
Input → Channel Attention → Spatial Attention → Output
```

**Why sequential (not parallel)?**
- Channel attention first decides "which features to look at"
- Spatial attention then decides "where to look" for those features
- This order works better empirically

**PyTorch Implementation:**
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

### Where to Place Attention in Autoencoders

| Option | Description | Cost | Quality |
|--------|-------------|------|---------|
| Bottleneck only | Attention at 16×16 latent | Low | Good starting point |
| After each encoder block | Progressive focus | Medium | Better quality |
| After each decoder block | Focus on reconstruction | Medium | Better details |
| Everywhere | Maximum capacity | High | May be overkill |

**Recommendation for SAR:** CBAM at bottleneck only. Adds ~5% training time but can improve PSNR by 1-2 dB.

### Self-Attention (Advanced)

SE and spatial attention are "local" — they don't capture relationships between distant pixels. **Self-attention** captures global relationships.

**Computational Cost (why we avoid it at large resolutions):**
```
For an image with H×W pixels:
Attention matrix: (H×W) × (H×W)
For 64×64 image: 4096 × 4096 = 16.7 million elements!
For 256×256 image: 65536 × 65536 = 4.3 billion elements!
```

**For your SAR autoencoder:** Self-attention at the bottleneck (16×16 = 256 elements, so 256×256 = 65K elements) is feasible but CBAM is usually sufficient.

---

## 2.4 Loss Functions

### What Loss Functions Do

The loss function defines **what "good" reconstruction means**. During training:
```
1. Network produces reconstruction: x̂ = Decoder(Encoder(x))
2. Loss measures quality: L = Loss(x̂, x)
3. Gradients update network: minimize L over training data
```

Different loss functions lead to different network behaviors:
- MSE: Minimize pixel-by-pixel errors → can be blurry
- SSIM: Preserve structure → better edges, may have artifacts
- Combined: Balance between pixel accuracy and structure

### Mean Squared Error (MSE)

**Definition:**
```
MSE = (1/N) × Σᵢ (xᵢ - x̂ᵢ)²

Where:
- N = total number of pixels (H × W for grayscale)
- xᵢ = original pixel value
- x̂ᵢ = reconstructed pixel value
```

**Properties:**

| Aspect | MSE Behavior |
|--------|--------------|
| Computation | Simple, fast |
| Gradient | Proportional to error at each pixel |
| Pixel treatment | All pixels equal |
| Typical result | Blurry reconstructions |

**Why MSE Causes Blur:**

Consider reconstructing an edge:
```
Original:     Possible reconstructions with same MSE:
[0 0 1 1]     [0 0.1 0.9 1]  (sharp but shifted)
              [0 0 1 1]      (perfect)
              [0 0.25 0.75 1] (blurry)
```

If the network is uncertain about exact edge location:
- Sharp edge at wrong position → large MSE
- Blurry edge spanning both positions → moderate MSE

The network learns to "hedge its bets" by producing blur!

**Gradient of MSE:**
```
∂MSE/∂x̂ᵢ = (2/N) × (x̂ᵢ - xᵢ)
```

The gradient is proportional to the error at each pixel. This seems reasonable, but:
- All pixels with the same error get the same gradient
- Edges and flat regions treated identically

**PyTorch:**
```python
mse_loss = nn.MSELoss()
loss = mse_loss(reconstruction, original)
```

### Structural Similarity (SSIM)

SSIM was designed to match human perception. Two images with the same MSE can look very different:
```
Image A: Uniform noise added to every pixel → MSE = 0.01
Image B: Structure distorted (edges shifted) → MSE = 0.01

Humans: Image B looks much worse!
MSE: They're the same!
```

**The Three Components:**

SSIM compares images based on three factors computed over local windows:

**1. Luminance (l):** Are the mean intensities similar?
```
l(x,y) = (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁)
```

**2. Contrast (c):** Are the standard deviations similar?
```
c(x,y) = (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂)
```

**3. Structure (s):** Is the pattern/correlation similar?
```
s(x,y) = (σₓᵧ + C₃) / (σₓσᵧ + C₃)
```

**Combined SSIM:**
```
SSIM(x,y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
```

**Stability Constants (prevent division by zero):**
```
C₁ = (0.01 × L)² = 0.0001  (for [0,1] images)
C₂ = (0.03 × L)² = 0.0009
```

**Properties:**

| Aspect | Value |
|--------|-------|
| Range | [-1, 1] |
| Identical images | SSIM = 1 |
| No structural similarity | SSIM = 0 |
| As loss | Use `Loss = 1 - SSIM` |

**PyTorch (using pytorch-msssim):**
```python
from pytorch_msssim import ssim

ssim_value = ssim(reconstruction, original, data_range=1.0)
ssim_loss = 1 - ssim_value
```

### Multi-Scale SSIM (MS-SSIM)

Computes SSIM at multiple resolutions (original, 2× downsampled, 4× downsampled, etc.).

**Advantages:**
- Captures both fine and coarse structural similarity
- Often better than single-scale SSIM
- More robust to scale variations

```python
from pytorch_msssim import ms_ssim

ms_ssim_value = ms_ssim(reconstruction, original, data_range=1.0)
```

### Combined Loss (Recommended)

**Why Combine?**

| Aspect | MSE | SSIM |
|--------|-----|------|
| Pixel accuracy | Good | Moderate |
| Edge preservation | Poor | Good |
| Gradient stability | Excellent | Can be unstable |
| Computation | Fast | Slower |

**Combined Loss:**
```
Loss = α × MSE + β × (1 - SSIM)
```

**Common weight choices:**

| Weights | Use Case |
|---------|----------|
| α=0.5, β=0.5 | Balanced (start here) |
| α=0.16, β=0.84 | SSIM-heavy (sharper results) |
| α=1.0, β=0.1 | MSE-heavy with SSIM regularization |

**PyTorch Implementation:**
```python
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.5, ssim_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, reconstruction, original):
        mse_loss = F.mse_loss(reconstruction, original)
        ssim_loss = 1 - ssim(reconstruction, original, data_range=1.0)

        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss

        return total_loss, {'mse': mse_loss.item(), 'ssim': 1 - ssim_loss.item()}
```

**Tuning Guide:**

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Blurry output | MSE too dominant | Increase SSIM weight |
| Noisy output | SSIM too dominant | Increase MSE weight |
| Lost edges | Need edge supervision | Consider gradient loss |
| Training unstable | SSIM gradient issues | Reduce SSIM weight or use MS-SSIM |

### SAR-Specific Considerations

**Speckle and Loss Functions:**

SAR images contain speckle noise. In log/dB domain:
- Speckle becomes additive (easier for network)
- MSE treats speckle fluctuations as errors
- SSIM preserves speckle as "texture"

**Recommendation:**
- Train in dB domain (not linear)
- Use balanced MSE + SSIM (0.5/0.5)
- Monitor ENL ratio to catch over-smoothing
- If ENL ratio > 1.2 consistently, reduce MSE weight

---

## 2.5 SAR Preprocessing

### Understanding SAR Images

**How SAR Works:**

Synthetic Aperture Radar is an **active** imaging system:
1. **Transmit:** Satellite sends microwave pulses toward Earth
2. **Interact:** Pulses bounce off terrain, vegetation, buildings, water
3. **Receive:** Satellite receives reflected signals
4. **Process:** Complex signal processing creates an image

Unlike optical cameras that measure sunlight reflection, SAR measures its own transmitted signal's return.

**What SAR Measures:**

Each pixel contains **intensity** (amplitude squared):
- Bright pixels = strong reflection (buildings, rough surfaces)
- Dark pixels = weak reflection (smooth water, shadows)

**Typical Value Ranges:**
```
Smooth water: ~0.0001 to 0.001
Vegetation: ~0.01 to 0.1
Urban: ~0.1 to 10
Very bright (metal): can exceed 100

Dynamic range: 40-60 dB (10,000× to 1,000,000× variation)
```

### Why SAR Needs Special Preprocessing

**Problem 1: Extreme Dynamic Range**

Neural networks work best with inputs in a reasonable range (e.g., [0, 1]).

SAR values span 4-6 orders of magnitude:
```
Min: ~0.00001
Max: ~100
Ratio: 10,000,000×
```

If fed directly to a network:
- Bright pixels dominate the loss
- Dark areas are essentially 0 (gradients vanish)
- Network can't learn useful representations

**Problem 2: Speckle Noise**

SAR images contain **speckle** — granular noise from coherent imaging:
```
Observed = True_Signal × Speckle_Noise

Where Speckle ~ Gamma distribution (in intensity)
```

Key properties:
- **Multiplicative:** Noise scales with signal (bright areas have more absolute noise)
- **Not Gaussian:** Heavy-tailed distribution
- **Looks like "salt and pepper":** But it's signal-dependent

In linear domain, speckle makes training difficult because:
- Noise magnitude varies across the image
- Network might try to memorize noise patterns
- Loss function weighted toward high-intensity areas

**Problem 3: Invalid Values**

SAR images can contain:
- **Zeros:** No-data regions, shadow areas, calibration failures
- **Negative values:** Shouldn't exist but sometimes appear (processing artifacts)
- **Infinities/NaNs:** Numerical errors, corrupted pixels

These must be handled before training because `log(0) = -inf` and `log(negative) = NaN`.

### The dB Transformation

**The Mathematics:**
```
I_dB = 10 × log₁₀(I_linear + noise_floor)

I_linear = 10^(I_dB / 10)
```

**Why Logarithm Helps:**

| Problem | Linear Domain | dB Domain |
|---------|---------------|-----------|
| Dynamic range | 10,000,000× | 70 dB range |
| Speckle | Multiplicative | Additive |
| Distribution | Highly skewed | More Gaussian |
| Neural network compatibility | Poor | Good |

**Multiplicative becomes additive:**
```
Linear: I_observed = I_true × Speckle
Log: log(I_observed) = log(I_true) + log(Speckle)
```

In log domain, speckle is additive noise with approximately constant variance!

**Visual Example:**
```
Linear domain:
[0.001, 0.01, 0.1, 1.0, 10, 100] → [mostly dark, one bright]

dB domain:
[-30, -20, -10, 0, 10, 20] → [evenly spaced visually]
```

### Complete Preprocessing Pipeline

**Step 1: Handle Invalid Values**
```python
def handle_invalid(image, noise_floor=1e-10):
    """Replace invalid values with noise floor."""
    image = image.copy()
    invalid = (image <= 0) | np.isnan(image) | np.isinf(image)
    image[invalid] = noise_floor
    return image
```

**Step 2: Convert to dB**
```python
def linear_to_db(intensity, floor=1e-10):
    """Convert linear intensity to decibels."""
    intensity = np.maximum(intensity, floor)
    return 10 * np.log10(intensity)
```

**Step 3: Clip Outliers**

Even in dB, outliers exist:
```
Typical range: -25 dB to +5 dB (30 dB total)
Outliers: -50 dB (deep shadows) to +30 dB (corner reflectors)
```

Clipping methods:

| Method | When to Use |
|--------|-------------|
| Percentile (1st-99th) | Unknown data distribution |
| Sigma (mean ± 3σ) | Data is approximately Gaussian |
| Fixed (-25 to +5 dB) | Well-understood data, consistency |

```python
def percentile_clip(image_db, low=1, high=99):
    vmin = np.percentile(image_db, low)
    vmax = np.percentile(image_db, high)
    return np.clip(image_db, vmin, vmax), vmin, vmax
```

**Step 4: Normalize to [0, 1]**
```python
def normalize(image_db, vmin, vmax):
    return (image_db - vmin) / (vmax - vmin)

def denormalize(image_norm, vmin, vmax):
    return image_norm * (vmax - vmin) + vmin
```

**Step 5: Extract Patches**

Full SAR images are huge (25,000 × 16,000 pixels = 1.6 GB). Extract smaller patches:

```python
def extract_patches(image, patch_size=256, stride=128):
    H, W = image.shape
    patches = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)
```

**Step 6: Filter Quality**

Remove patches that are mostly invalid or blank:
```python
def filter_patches(patches, min_std=0.02, max_invalid_ratio=0.01):
    valid = []
    for patch in patches:
        if np.std(patch) < min_std:  # Too uniform
            continue
        edge_ratio = np.mean((patch <= 0.001) | (patch >= 0.999))
        if edge_ratio > max_invalid_ratio:  # Too many edge values
            continue
        valid.append(patch)
    return np.array(valid)
```

### Critical Rules

1. **Compute normalization params from training set ONLY**
2. **Save params (vmin, vmax, noise_floor) with checkpoint**
3. **Apply same params to validation/test data**
4. **Apply same params during inference**

```python
# Training: Fit and save
vmin, vmax = compute_bounds(training_images)
save_params({'vmin': vmin, 'vmax': vmax})

# Inference: Load and apply
params = load_params()
image_norm = normalize(image_db, params['vmin'], params['vmax'])
```

---

## 2.6 SAR Quality Metrics

### Why Multiple Metrics?

Different metrics capture different aspects of quality:

| Metric | What It Measures | Limitation |
|--------|------------------|------------|
| PSNR | Overall pixel fidelity | Doesn't match perception |
| SSIM | Structural similarity | Can miss fine details |
| ENL | Speckle properties | Only for homogeneous regions |
| EPI | Edge preservation | Focused on boundaries only |

A reconstruction might score well on one metric but poorly on another.

### PSNR (Peak Signal-to-Noise Ratio)

**Definition:**
```
PSNR = 10 × log₁₀(MAX² / MSE)

Where MAX = maximum possible value (1.0 for normalized images)
```

**Interpretation:**

| PSNR | Quality |
|------|---------|
| >40 dB | Excellent, near-lossless |
| >35 dB | Very good |
| >30 dB | Good, acceptable for most uses |
| >25 dB | Visible degradation |
| <25 dB | Poor |

**PyTorch:**
```python
def psnr(original, reconstruction, max_val=1.0):
    mse = torch.mean((original - reconstruction) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse)
```

### SSIM (Structural Similarity Index)

Already covered in Loss Functions section. As a metric:

| SSIM | Quality |
|------|---------|
| >0.95 | Excellent |
| >0.90 | Good |
| >0.80 | Acceptable |
| <0.70 | Noticeable structural differences |

### ENL (Equivalent Number of Looks)

**What it measures:** Speckle smoothness in homogeneous regions.

**Definition:**
```
ENL = μ² / σ²

Where μ = mean intensity, σ² = variance
For fully developed speckle: ENL ≈ 1
More averaging → higher ENL
```

**ENL Ratio (for comparing original vs reconstruction):**
```
ENL_ratio = ENL(reconstruction) / ENL(original)
```

| ENL Ratio | Meaning |
|-----------|---------|
| 0.9 - 1.1 | Excellent - speckle preserved |
| 0.8 - 1.2 | Acceptable |
| >1.2 | Over-smoothed (lost texture) |
| <0.8 | Added noise (noisier than original) |

**Finding Homogeneous Regions:**
```python
def find_homogeneous_regions(image, window_size=15, cv_threshold=0.3):
    """Find regions with low coefficient of variation."""
    from scipy.ndimage import uniform_filter

    local_mean = uniform_filter(image, size=window_size)
    local_sq_mean = uniform_filter(image**2, size=window_size)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))

    cv = local_std / (local_mean + 1e-10)
    return cv < cv_threshold
```

**Computing ENL:**
```python
def compute_enl(image, mask):
    """Compute ENL for masked region."""
    if mask.sum() < 100:
        return 1.0  # Not enough samples

    values = image[mask]
    mean = np.mean(values)
    var = np.var(values)

    if var < 1e-10:
        return float('inf')

    return (mean ** 2) / var
```

### EPI (Edge Preservation Index)

**What it measures:** How well edges are preserved after compression.

**Definition:**
```
EPI = Correlation of gradient magnitudes

EPI = Σ|∇R × ∇I| / sqrt(Σ|∇R|² × Σ|∇I|²)

Where ∇ = gradient (Sobel), R = reconstruction, I = original
```

**Implementation:**
```python
from scipy.ndimage import sobel

def compute_epi(original, reconstruction):
    """Compute Edge Preservation Index."""
    # Gradient magnitudes
    def gradient_magnitude(image):
        gx = sobel(image, axis=1)
        gy = sobel(image, axis=0)
        return np.sqrt(gx**2 + gy**2)

    grad_orig = gradient_magnitude(original).flatten()
    grad_recon = gradient_magnitude(reconstruction).flatten()

    # Correlation
    numerator = np.sum(grad_orig * grad_recon)
    denominator = np.sqrt(np.sum(grad_orig**2) * np.sum(grad_recon**2))

    return numerator / denominator if denominator > 0 else 0
```

| EPI | Quality |
|-----|---------|
| >0.95 | Excellent edge preservation |
| >0.90 | Good |
| >0.85 | Acceptable |
| <0.80 | Significant edge degradation |

### Complete Evaluation Framework

```python
class SARQualityEvaluator:
    """Comprehensive SAR quality metrics."""

    def evaluate(self, original, reconstruction):
        # Find homogeneous regions for ENL
        homogeneous_mask = self.find_homogeneous(original)

        # Compute all metrics
        return {
            'psnr': self.compute_psnr(original, reconstruction),
            'ssim': self.compute_ssim(original, reconstruction),
            'enl_ratio': self.compute_enl_ratio(original, reconstruction, homogeneous_mask),
            'epi': self.compute_epi(original, reconstruction)
        }
```

### Quality Targets for SAR Compression

| Metric | Good | Excellent |
|--------|------|-----------|
| PSNR | >30 dB | >35 dB |
| SSIM | >0.85 | >0.95 |
| ENL ratio | 0.8-1.2 | 0.9-1.1 |
| EPI | >0.85 | >0.95 |

---

## 2.7 Compression Tradeoffs

### The Fundamental Tradeoff

Compression is fundamentally about removing information:

```
Original: 256×256×1 = 65,536 values
Latent: H×W×C values

The smaller the latent, the more information must be discarded.
```

**Rate-Distortion Theory:**
```
Rate ↓ → Distortion ↑ (more compression = worse quality)
Rate ↑ → Distortion ↓ (less compression = better quality)
```

For any given compression rate, there's a minimum achievable distortion. Neural networks can approach this theoretical bound but not exceed it.

### What Gets Lost During Compression?

When compressing images, networks typically discard:

**First to go (easy to lose):**
- High-frequency noise
- Fine textures
- Subtle intensity variations

**Last to go (hard to compress):**
- Major structures and edges
- Object boundaries
- Large-scale patterns

**For SAR specifically:**
- Speckle can be compressed (it's random noise)
- Edges and structures must be preserved
- This is why SAR compresses better than you might expect!

### The Quality Cliff

Quality doesn't degrade linearly with compression:

```
CR:     4×    8×    16×   32×   64×   128×
PSNR:   45    40    35    30    25    20    (dB)
        ↑     ↑     ↑     ↑     ↑     ↑
        Near  Very  Good  Accept Visible  Poor
        less  good        able   artifacts
```

There's often a "knee" in the curve where quality drops rapidly.

### Latent Space Design

The latent space has three key dimensions:

**1. Spatial dimensions (H_lat × W_lat):**
```
256×256 input → ?×? latent

4× spatial reduction: 64×64 latent (aggressive)
16× spatial reduction: 16×16 latent (typical, 4 stride-2 layers)
256× spatial reduction: 1×1 latent (extreme - usually too aggressive)
```

**2. Channel dimensions (C_lat):**
```
More channels = more capacity at each spatial location
Fewer channels = higher compression
```

**3. Bit precision (for deployment):**
```
Float32: 32 bits per value (training default)
Float16: 16 bits per value (minimal quality loss, 2× compression)
Int8: 8 bits per value (some quality loss, 4× compression)
```

### Compression Ratio Calculation

```python
def compression_ratio(input_shape, latent_shape):
    """
    input_shape: (H, W, C) or (H, W) for grayscale
    latent_shape: (H_lat, W_lat, C_lat)
    """
    input_size = np.prod(input_shape)
    latent_size = np.prod(latent_shape)
    return input_size / latent_size
```

**Examples:**
```
Input: 256×256×1 = 65,536 values

Latent: 16×16×64 → CR = 65536 / 16384 = 4×
Latent: 16×16×32 → CR = 65536 / 8192 = 8×
Latent: 16×16×16 → CR = 65536 / 4096 = 16×
Latent: 16×16×8  → CR = 65536 / 2048 = 32×
Latent: 8×8×16   → CR = 65536 / 1024 = 64×
```

### Spatial vs Channel Tradeoff

Two ways to achieve the same compression ratio:

**Option A: Large spatial, few channels**
```
16×16×16 = 4096 values
More spatial resolution, less feature diversity
Good for: Images with fine spatial details
```

**Option B: Small spatial, many channels**
```
8×8×64 = 4096 values
Less spatial resolution, more feature diversity
Good for: Images with complex textures/patterns
```

**For SAR:**
- Edges need spatial resolution
- Speckle is random (doesn't need spatial detail)
- Balance: 16×16 spatial with moderate channels (16-64) often works well

### Choosing Your Operating Point

**Step 1: Define quality threshold**
```
"I need PSNR > 30 dB for my application"
```

**Step 2: Run experiments across compression ratios**
```python
for cr in [4, 8, 16, 32, 64]:
    train_and_evaluate(compression_ratio=cr)
```

**Step 3: Find the highest CR meeting your threshold**
```
Results:
CR 64×: PSNR 25 dB (below threshold)
CR 32×: PSNR 29 dB (below threshold)
CR 16×: PSNR 33 dB (meets threshold) ← Choose this!
CR 8×: PSNR 37 dB (exceeds threshold)
```

**Step 4: Refine with architecture changes**
```
Can we achieve CR 32× with better architecture?
- Add residual blocks: CR 32× → PSNR 31 dB (meets threshold!)
```

### Recommended Starting Points for SAR

| Use Case | Suggested CR | Latent Config | Expected Quality |
|----------|--------------|---------------|------------------|
| High quality archive | 8× | 16×16×32 | PSNR > 35 dB |
| Bandwidth constrained | 16× | 16×16×16 | PSNR > 30 dB |
| Extreme constraint | 32× | 16×16×8 | PSNR > 27 dB |
| Preview/thumbnail | 64× | 8×8×16 | PSNR > 24 dB |

### Memory Budget (RTX 3070, 8GB)

```
256×256 input, 4-layer encoder/decoder, 64 base channels:
- ~5M parameters
- Batch 16: ~4 GB
- Batch 8: ~2.5 GB
- Batch 4: ~1.5 GB
```

You have room to experiment with batch size 8-16.

### Training Stability at High Compression

Higher compression can make training less stable:

| CR | Training Behavior |
|----|-------------------|
| 8× | Easy to train, converges smoothly |
| 16× | May need learning rate tuning |
| 32× | May need warmup, careful initialization |
| 64× | May collapse, need extensive tuning |

**Tips for high compression:**
1. Use learning rate warmup
2. Start with lower compression, fine-tune to higher
3. Use residual connections
4. Monitor for mode collapse (network outputs constant)

---

# Part 3: Summary Tables

## Recommended Experiment Matrix

| Variant | 8x (32ch) | 16x (16ch) | 32x (8ch) |
|---------|-----------|------------|-----------|
| Plain (A) | Must | Must | Must |
| Residual (B) | Must | Must | Must |
| Res+CBAM (C) | Should | Should | Should |

**Total: 6-9 trained models**

## Quality Targets

| Metric | Good | Excellent |
|--------|------|-----------|
| PSNR | >30 dB | >35 dB |
| SSIM | >0.85 | >0.95 |
| ENL ratio | 0.8-1.2 | 0.9-1.1 |
| EPI | >0.85 | >0.95 |

## Phase Summary

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| 1 | Data Pipeline | Working preprocessing + patches |
| 2 | Baseline Model | Trained plain autoencoder |
| 3 | SAR Evaluation | ENL/EPI metrics working |
| 4 | Architecture | Residual + CBAM variants |
| 5 | Full Inference | Seamless tiled processing |
| 6 | Experiments | Complete comparison study |

---

*This compilation was generated from the `.planning/research/` and `.planning/knowledge/` documents.*
