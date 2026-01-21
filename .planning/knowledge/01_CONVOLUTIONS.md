# Deep Dive: Convolutional Neural Networks

This document provides an in-depth explanation of how convolutions work, why they're used for images, and how to design convolutional layers for image compression.

---

## Table of Contents

1. [Why Convolutions for Images?](#why-convolutions-for-images)
2. [How Convolution Works](#how-convolution-works)
3. [The Mathematics](#the-mathematics)
4. [Kernel/Filter Design](#kernelfilter-design)
5. [Stride and Padding](#stride-and-padding)
6. [Multiple Channels](#multiple-channels)
7. [Transposed Convolutions](#transposed-convolutions)
8. [Practical Considerations](#practical-considerations)

---

## Why Convolutions for Images?

### The Problem with Fully Connected Layers

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

### What Makes Images Special

Images have three key properties that convolutions exploit:

**1. Local Patterns**
A pixel's meaning depends mostly on its neighbors. An edge at position (100, 100) looks the same as an edge at (50, 200). We don't need to learn separate detectors for every location.

**2. Translation Invariance**
A cat in the top-left corner is the same cat as one in the bottom-right. Features should be detected regardless of position.

**3. Hierarchical Structure**
- Low level: edges, gradients, textures
- Mid level: shapes, parts, patterns
- High level: objects, scenes, semantic content

### How Convolutions Solve This

Convolutions exploit these properties:

| Property | How Convolution Helps |
|----------|----------------------|
| Local patterns | Small kernel (3×3) only looks at neighbors |
| Translation invariance | Same kernel applied everywhere (weight sharing) |
| Hierarchical structure | Stack layers: early = edges, later = complex patterns |

**Parameter count for convolution:**
```
Kernel: 3 × 3 = 9 parameters (same 9 weights used everywhere!)
With 64 output channels: 9 × 64 = 576 parameters
```

Compare: 576 parameters vs 4.3 billion. That's a 7-million-fold reduction.

---

## How Convolution Works

### The Intuition: Sliding Window Feature Detection

Imagine you have a small "template" (kernel) that detects a specific pattern, like a vertical edge:

```
Kernel (3×3):        What it detects:
[-1  0  1]           Bright on right, dark on left
[-1  0  1]           = vertical edge
[-1  0  1]
```

You slide this template across the entire image. At each position, you compute how well the local region matches the template (via dot product). High values = strong match.

### Step-by-Step Example

**Input image patch (5×5):**
```
[10  10  10  50  50]
[10  10  10  50  50]
[10  10  10  50  50]
[10  10  10  50  50]
[10  10  10  50  50]
```
This shows a vertical edge in the middle (10s on left, 50s on right).

**Kernel (3×3) for vertical edge detection:**
```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Convolution at position (1,1):**
```
Extract 3×3 region:        Element-wise multiply:        Sum:
[10  10  10]               [-10   0  10]
[10  10  10]       ×       [-10   0  10]         =      0
[10  10  10]               [-10   0  10]
```
Result: 0 (no edge here, values are uniform)

**Convolution at position (1,2) — on the edge:**
```
Extract 3×3 region:        Element-wise multiply:        Sum:
[10  10  50]               [-10   0  50]
[10  10  50]       ×       [-10   0  50]         =      120
[10  10  50]               [-10   0  50]
```
Result: 120 (strong positive response — edge detected!)

**Output feature map:**
After sliding across all positions, you get a "feature map" showing where vertical edges are:
```
[0   0  120  0]
[0   0  120  0]
[0   0  120  0]
```

### Learned vs Hand-Designed Kernels

Traditional image processing uses hand-designed kernels:
- Sobel edge detector
- Gaussian blur
- Sharpening

Neural networks **learn** the kernels from data:
- Random initialization
- Backpropagation adjusts weights
- Network discovers useful patterns automatically

This is powerful because:
- Discovers patterns humans might miss
- Adapts to specific data (SAR vs natural images)
- Can learn very complex, non-intuitive detectors

---

## The Mathematics

### Formal Definition

For a 2D convolution (actually cross-correlation in deep learning):

```
Output[i,j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m, n]
```

Where:
- `i, j` = position in output
- `m, n` = position within kernel
- Σ = sum over all kernel positions

### Matrix Form

For implementation efficiency, convolution is often converted to matrix multiplication:

1. **im2col**: Reshape input patches into columns
2. **Matrix multiply**: Kernel weights × patch columns
3. **Reshape**: Convert back to spatial layout

This leverages highly optimized BLAS/cuBLAS libraries.

### Gradient Computation

For backpropagation, we need gradients:

**Gradient w.r.t. weights:**
```
∂L/∂Kernel[m,n] = Σ_i Σ_j ∂L/∂Output[i,j] × Input[i+m, j+n]
```

**Gradient w.r.t. input:**
```
∂L/∂Input[i,j] = Σ_m Σ_n ∂L/∂Output[i-m, j-n] × Kernel[m, n]
```

This is why transposed convolution is also called "backward convolution" — it's the gradient operation.

---

## Kernel/Filter Design

### Kernel Size

**3×3 (most common):**
- Smallest size that captures spatial patterns
- Efficient computation
- Two 3×3 layers have same receptive field as one 5×5 but fewer parameters

```
3×3 kernel: 9 parameters
5×5 kernel: 25 parameters
Two 3×3 layers: 18 parameters, 5×5 receptive field
```

**1×1 (pointwise convolution):**
- No spatial mixing
- Only mixes channels
- Used to increase/decrease channel count cheaply

```
Input: H × W × 64
1×1 Conv with 32 filters
Output: H × W × 32
```

**Larger kernels (5×5, 7×7):**
- Larger receptive field
- More parameters
- Used in first layer sometimes (to quickly capture low-freq patterns)
- Mostly replaced by stacked 3×3 in modern architectures

### Receptive Field

The **receptive field** is how much of the input each output pixel "sees":

```
Layer 1 (3×3): Each output sees 3×3 input region
Layer 2 (3×3): Each output sees 5×5 input region
Layer 3 (3×3): Each output sees 7×7 input region
...
Layer N (3×3): Each output sees (2N+1) × (2N+1) input region
```

With stride-2 downsampling:
```
Layer 1 (3×3, stride 2): 3×3 receptive field
Layer 2 (3×3, stride 2): 7×7 receptive field
Layer 3 (3×3, stride 2): 15×15 receptive field
Layer 4 (3×3, stride 2): 31×31 receptive field
```

For your 256×256 input with 4 stride-2 layers, the bottleneck "sees" essentially the whole image.

### Number of Filters (Output Channels)

Each filter produces one output channel. More filters = more feature types detected.

**Common patterns:**
```
Early layers: fewer filters (32, 64) — detecting simple patterns
Later layers: more filters (128, 256, 512) — detecting complex patterns
```

**For autoencoders:**
```
Encoder: channels increase (1 → 64 → 128 → 256 → latent)
Decoder: channels decrease (latent → 256 → 128 → 64 → 1)
```

The encoder is "spreading" spatial information into channels.
The decoder is "gathering" channel information back to spatial.

---

## Stride and Padding

### Stride

**Stride** = how far the kernel moves between positions.

**Stride 1 (default):**
```
Input: 8×8
Kernel: 3×3, stride 1
Output: 6×6 (slightly smaller due to borders)
```

**Stride 2 (downsampling):**
```
Input: 8×8
Kernel: 3×3, stride 2
Output: 3×3 (half the size, roughly)
```

**Why use stride for downsampling?**

Alternative: convolution + pooling
```
Conv(stride=1) → 8×8 → MaxPool(2×2) → 4×4
```

Better: strided convolution
```
Conv(stride=2) → 4×4
```

Strided conv advantages:
- Learnable downsampling (network chooses what to keep)
- Fewer operations
- Better gradient flow (no pooling non-linearity)

### Padding

**Padding** adds border pixels to control output size.

**No padding (valid):**
```
Input: 8×8
Kernel: 3×3
Output: 6×6 (shrinks by kernel_size - 1)
```

**Same padding (most common):**
```
Input: 8×8
Kernel: 3×3, padding=1
Output: 8×8 (same size!)
```

Padding formula for "same" output:
```
padding = (kernel_size - 1) / 2
```

For 3×3: padding = 1
For 5×5: padding = 2
For 7×7: padding = 3

**Zero padding vs other methods:**
- **Zero padding**: Border filled with 0s (standard)
- **Reflection padding**: Border mirrors the edge pixels (good for images)
- **Replication padding**: Border repeats edge pixels

For SAR images, **reflection padding** might be slightly better (avoids artificial edges at borders).

### Output Size Formula

```
Output_size = floor((Input_size + 2×padding - kernel_size) / stride) + 1
```

Examples:
```
Input=256, kernel=3, padding=1, stride=1 → Output=256
Input=256, kernel=3, padding=1, stride=2 → Output=128
Input=256, kernel=4, padding=1, stride=2 → Output=128
```

---

## Multiple Channels

### Multi-Channel Input

Real images have multiple channels (RGB has 3, SAR might have 1 or 2).

The kernel becomes 3D:
```
Input: H × W × C_in
Kernel: K × K × C_in (one kernel)
Output: H × W × 1 (one channel)
```

The convolution sums over all input channels:
```
Output[i,j] = Σ_c Σ_m Σ_n Input[i+m, j+n, c] × Kernel[m, n, c]
```

### Multiple Output Channels

To produce multiple output channels, use multiple kernels:
```
Input: H × W × C_in
Kernels: C_out kernels, each K × K × C_in
Output: H × W × C_out
```

Total parameters:
```
params = K × K × C_in × C_out
```

Example:
```
Input: 256 × 256 × 64
Kernel: 3 × 3, 128 output channels
Parameters: 3 × 3 × 64 × 128 = 73,728
Output: 256 × 256 × 128
```

### Depthwise Separable Convolutions

For efficiency, split into:

**1. Depthwise**: One kernel per input channel (no mixing)
```
Input: H × W × C
C kernels of size K × K × 1
Output: H × W × C
Parameters: K × K × C
```

**2. Pointwise**: Mix channels with 1×1 conv
```
Input: H × W × C
Kernels: C_out kernels of size 1 × 1 × C
Output: H × W × C_out
Parameters: C × C_out
```

**Total: K×K×C + C×C_out** (much less than K×K×C×C_out)

Example:
```
Standard: 3×3×64×128 = 73,728 params
Depthwise separable: 3×3×64 + 64×128 = 576 + 8,192 = 8,768 params
```

8× fewer parameters! Used in MobileNet, EfficientNet.

---

## Transposed Convolutions

### The Problem: Upsampling

The encoder reduces spatial size (256 → 128 → 64 → 32 → 16).
The decoder must increase it back (16 → 32 → 64 → 128 → 256).

### Options for Upsampling

**1. Nearest neighbor + Conv:**
```python
x = F.interpolate(x, scale_factor=2, mode='nearest')
x = conv(x)
```
Simple, no learnable upsampling, can be blurry.

**2. Bilinear + Conv:**
```python
x = F.interpolate(x, scale_factor=2, mode='bilinear')
x = conv(x)
```
Smoother than nearest, still no learnable upsampling.

**3. Transposed Convolution (learnable):**
```python
x = conv_transpose(x)  # ConvTranspose2d
```
Learns how to upsample. Most common in autoencoders.

### How Transposed Convolution Works

Think of it as "reverse" convolution:
- Regular conv: many input pixels → one output pixel
- Transposed conv: one input pixel → many output pixels

**Step by step:**

Input (2×2):
```
[a  b]
[c  d]
```

Kernel (3×3):
```
[1  2  3]
[4  5  6]
[7  8  9]
```

For each input pixel, "stamp" the kernel onto the output:

Position a stamps:
```
[a×1  a×2  a×3   0    0 ]
[a×4  a×5  a×6   0    0 ]
[a×7  a×8  a×9   0    0 ]
[ 0    0    0    0    0 ]
[ 0    0    0    0    0 ]
```

Position b stamps (shifted right by stride):
```
[ 0    0   b×1  b×2  b×3]
[ 0    0   b×4  b×5  b×6]
[ 0    0   b×7  b×8  b×9]
[ 0    0    0    0    0 ]
[ 0    0    0    0    0 ]
```

Sum all stamps → output (larger than input!)

### Checkerboard Artifacts

A common problem with transposed convolutions:

When kernel_size is not divisible by stride, some output positions receive more contributions than others, creating a checkerboard pattern.

**Bad:** kernel=3, stride=2 (3/2 = 1.5, not integer)
**Good:** kernel=4, stride=2 (4/2 = 2, integer)
**Good:** kernel=2, stride=2 (2/2 = 1, integer)

**Practical recommendation:**
```python
# For stride=2 upsampling, use kernel=4
nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
```

### Output Size Formula

```
Output_size = (Input_size - 1) × stride - 2×padding + kernel_size + output_padding
```

For doubling with kernel=4, stride=2, padding=1:
```
Output = (Input - 1) × 2 - 2 + 4 = 2×Input - 2 - 2 + 4 = 2×Input
```


---

## Practical Considerations

### Initialization

How you initialize weights affects training:

**Xavier/Glorot** (good for tanh, sigmoid):
```python
nn.init.xavier_uniform_(layer.weight)
```

**He/Kaiming** (good for ReLU, LeakyReLU):
```python
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
```

For your SAR autoencoder with LeakyReLU, use Kaiming initialization.

### Bias Terms

Each conv layer can have a bias term:
```
Output = Conv(Input) + bias
```

When using BatchNorm after conv, bias is redundant (BatchNorm has its own bias):
```python
nn.Conv2d(64, 128, 3, padding=1, bias=False)  # No bias
nn.BatchNorm2d(128)  # BatchNorm has bias
```

### Memory Considerations

For your 8GB GPU, memory usage during training:

```
Forward pass: Input + intermediate activations + output
Backward pass: Gradients for each layer
Optimizer: Momentum/Adam states (2× parameters for Adam)
```

Rough estimate for one forward pass:
```
256×256×64 × 4 bytes = 16 MB per layer
With 10 layers: ~160 MB activations
Gradients double this: ~320 MB
```

Batch size multiplies this:
```
Batch 16: ~5 GB for activations + gradients
Batch 8: ~2.5 GB
Batch 4: ~1.2 GB
```

With 8GB VRAM, batch size 8-16 should be feasible depending on model size.

### Dilated Convolutions

**Dilation** inserts gaps in the kernel, increasing receptive field without more parameters:

```
Standard 3×3:       Dilated 3×3 (dilation=2):
[x x x]             [x . x . x]
[x x x]             [. . . . .]
[x x x]             [x . x . x]
                    [. . . . .]
                    [x . x . x]

Receptive field:    Receptive field:
3×3                 5×5
```

Useful for:
- Capturing large-scale patterns without downsampling
- Semantic segmentation
- Potentially useful in SAR for capturing large structures

---

## Summary: Conv Layer Design for SAR Autoencoder

**Encoder layers:**
```python
# Layer 1: 256×256×1 → 128×128×64
nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
nn.BatchNorm2d(64)
nn.LeakyReLU(0.2)

# Layer 2: 128×128×64 → 64×64×128
nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
nn.BatchNorm2d(128)
nn.LeakyReLU(0.2)

# ... continue pattern
```

**Decoder layers:**
```python
# Layer 1: 16×16×256 → 32×32×128
nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
nn.BatchNorm2d(128)
nn.ReLU()

# ... continue pattern

# Final layer: → 256×256×1 with Sigmoid
nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
nn.Sigmoid()  # Output in [0, 1]
```

**Key choices:**
- kernel_size=3 for regular conv, kernel_size=4 for transposed
- stride=2 for spatial reduction/expansion
- padding=1 to maintain size relationships
- LeakyReLU for encoder (prevents dead neurons)
- ReLU for decoder (standard choice)
- Sigmoid for final output (constrains to [0, 1])
