# Deep Dive: Loss Functions for Image Reconstruction

This document provides a comprehensive explanation of loss functions for training image autoencoders, with focus on how different losses affect reconstruction quality.

---

## Table of Contents

1. [What Loss Functions Do](#what-loss-functions-do)
2. [Mean Squared Error (MSE)](#mean-squared-error-mse)
3. [Structural Similarity (SSIM)](#structural-similarity-ssim)
4. [Combining MSE and SSIM](#combining-mse-and-ssim)
5. [Perceptual Losses](#perceptual-losses)
6. [Gradient-Based Losses](#gradient-based-losses)
7. [SAR-Specific Considerations](#sar-specific-considerations)
8. [Practical Guidelines](#practical-guidelines)

---

## What Loss Functions Do

### The Role of Loss

The loss function defines **what "good" reconstruction means**. During training:

```
1. Network produces reconstruction: x̂ = Decoder(Encoder(x))
2. Loss measures quality: L = Loss(x̂, x)
3. Gradients update network: minimize L over training data
```

Different loss functions lead to different network behaviors:
- MSE: Minimize pixel-by-pixel errors → can be blurry
- SSIM: Preserve structure → better edges, may have artifacts
- Perceptual: Match high-level features → realistic but may hallucinate

### The Gradient Matters

The network only sees the **gradient** of the loss:

```
∂L/∂x̂ = "How should I change my output to reduce the loss?"
```

A loss function with gradients pointing to:
- Bright pixels → Network learns to make those pixels brighter
- Edge regions → Network focuses on reconstructing edges
- Everything equally → Network spreads effort uniformly

Understanding gradients helps predict network behavior.

---

## Mean Squared Error (MSE)

### Definition

MSE measures the average squared difference between pixels:

```
MSE = (1/N) × Σᵢ (xᵢ - x̂ᵢ)²

Where:
- N = total number of pixels (H × W for grayscale)
- xᵢ = original pixel value
- x̂ᵢ = reconstructed pixel value
```

### Properties

**Advantages:**
- Simple and fast to compute
- Differentiable everywhere
- Well-understood mathematically
- Directly minimizes pixel-level error

**Disadvantages:**
- Treats all pixels equally (edge pixel = flat region pixel)
- Penalizes large errors heavily, small errors lightly (squared)
- Often leads to blurry reconstructions
- Doesn't match human perception

### Why MSE Causes Blur

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

### Mathematical Analysis

**Gradient of MSE:**
```
∂MSE/∂x̂ᵢ = (2/N) × (x̂ᵢ - xᵢ)
```

The gradient is proportional to the error at each pixel:
- Large error → Large gradient (fix this pixel!)
- Small error → Small gradient (this pixel is fine)

This seems reasonable, but it means:
- All pixels with the same error get the same gradient
- Edges and flat regions treated identically

### PyTorch Implementation

```python
import torch.nn as nn
import torch.nn.functional as F

# Using built-in
mse_loss = nn.MSELoss()
loss = mse_loss(reconstruction, original)

# Or manually
loss = F.mse_loss(reconstruction, original)

# Or even more manual
loss = ((reconstruction - original) ** 2).mean()
```

### Variants

**Mean Absolute Error (MAE/L1):**
```python
mae_loss = nn.L1Loss()
# MAE = (1/N) × Σᵢ |xᵢ - x̂ᵢ|
```

Properties:
- Linear penalty (not squared)
- More robust to outliers
- Can produce sharper results
- Gradient is constant (±1), doesn't depend on error magnitude

**Smooth L1 (Huber Loss):**
```python
smooth_l1 = nn.SmoothL1Loss()
# L1 for large errors, L2 for small errors
```

Best of both worlds for some applications.

---

## Structural Similarity (SSIM)

### The Problem SSIM Solves

MSE doesn't match human perception. Two images with the same MSE can look very different:

```
Image A: Uniform noise added to every pixel → MSE = 0.01
Image B: Structure distorted (edges shifted) → MSE = 0.01

Humans: Image B looks much worse!
MSE: They're the same!
```

SSIM (Structural Similarity Index) was designed to match human perception.

### The Three Components

SSIM compares images based on three factors:

**1. Luminance (l):** Are the mean intensities similar?
```
μₓ = mean of original patch
μᵧ = mean of reconstruction patch

l(x,y) = (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁)
```

**2. Contrast (c):** Are the standard deviations similar?
```
σₓ = std dev of original patch
σᵧ = std dev of reconstruction patch

c(x,y) = (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂)
```

**3. Structure (s):** Is the pattern/correlation similar?
```
σₓᵧ = covariance between original and reconstruction

s(x,y) = (σₓᵧ + C₃) / (σₓσᵧ + C₃)
```

**Combined SSIM:**
```
SSIM(x,y) = l(x,y) × c(x,y) × s(x,y)

Simplified (C₃ = C₂/2):
SSIM(x,y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ²+ C₁)(σₓ² + σᵧ² + C₂))
```

### Stability Constants

The constants C₁ and C₂ prevent division by zero:

```
C₁ = (K₁ × L)² where K₁ = 0.01, L = dynamic range (1.0 for [0,1] images)
C₂ = (K₂ × L)² where K₂ = 0.03

For [0,1] images:
C₁ = 0.0001
C₂ = 0.0009
```

### Local Window Computation

SSIM is computed over local windows (not globally):

```
For each position in the image:
1. Extract 11×11 patch (or window_size × window_size)
2. Weight patch with Gaussian (σ = 1.5)
3. Compute local SSIM
4. Average all local SSIMs → final SSIM
```

The Gaussian weighting emphasizes the center of each patch.

### Properties

**Range:** SSIM ∈ [-1, 1]
- SSIM = 1: Identical images
- SSIM = 0: No structural similarity
- SSIM < 0: Negative correlation (rare)

**As a loss:** Use `Loss = 1 - SSIM` (minimize to maximize SSIM)

**Advantages:**
- Matches human perception better than MSE
- Preserves structural information
- Good for edges and textures

**Disadvantages:**
- More expensive to compute
- Gradient can be unstable near SSIM = 1
- Might not penalize color/intensity shifts enough

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def gaussian_window(window_size, sigma, channels):
    """Create 2D Gaussian window."""
    # 1D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    # 2D Gaussian
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)

    # Expand to (channels, 1, window_size, window_size)
    window = gauss_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    return window


class SSIMLoss(nn.Module):
    """SSIM loss for training."""

    def __init__(self, window_size=11, sigma=1.5, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels

        # Pre-compute Gaussian window
        self.register_buffer('window', gaussian_window(window_size, sigma, channels))

        # SSIM constants
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, img1, img2):
        """Compute SSIM loss = 1 - SSIM."""
        # Local means (using Gaussian-weighted average)
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Local variances and covariance
        sigma1_sq = F.conv2d(img1 ** 2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        # Return 1 - mean(SSIM) as loss
        return 1 - ssim_map.mean()
```

Or use a library:
```python
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# As a metric (higher is better)
ssim_value = ssim(reconstruction, original, data_range=1.0)

# As a loss (lower is better)
ssim_loss = 1 - ssim(reconstruction, original, data_range=1.0)
```

### Multi-Scale SSIM (MS-SSIM)

Computes SSIM at multiple resolutions:

```
MS-SSIM = Π_j [l(x,y)]^αⱼ × [c(x,y)]^βⱼ × [s(x,y)]^γⱼ

Where j indexes different scales (original, 2× downsampled, 4× downsampled, ...)
```

**Advantages:**
- Captures both fine and coarse structural similarity
- Often better than single-scale SSIM
- More robust to scale variations

```python
from pytorch_msssim import ms_ssim

ms_ssim_value = ms_ssim(reconstruction, original, data_range=1.0)
```

---

## Combining MSE and SSIM

### Why Combine?

MSE and SSIM have complementary strengths:

| Aspect | MSE | SSIM |
|--------|-----|------|
| Pixel accuracy | Good | Moderate |
| Edge preservation | Poor | Good |
| Gradient stability | Excellent | Can be unstable |
| Computation | Fast | Slower |

Combining them can get the best of both.

### Simple Weighted Combination

```
Loss = α × MSE + β × (1 - SSIM)
```

Where α and β control the balance.

**Common choices:**
- α = 1.0, β = 1.0: Equal weight
- α = 0.16, β = 0.84: SSIM-heavy (common in image restoration)
- α = 1.0, β = 0.1: MSE-heavy with SSIM regularization

### Implementation

```python
class CombinedLoss(nn.Module):
    """Combined MSE + SSIM loss."""

    def __init__(self, mse_weight=0.16, ssim_weight=0.84):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, reconstruction, original):
        mse_loss = self.mse(reconstruction, original)
        ssim_loss = self.ssim(reconstruction, original)  # Already 1 - SSIM

        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss

        # Return loss and components for logging
        return total_loss, {
            'mse': mse_loss.item(),
            'ssim': 1 - ssim_loss.item(),  # Convert back to SSIM for logging
            'loss': total_loss.item()
        }
```

### Tuning the Weights

**Start with balanced weights:**
```python
mse_weight = 1.0
ssim_weight = 1.0
```

**Adjust based on results:**
- Blurry outputs → Increase ssim_weight
- Artifacts/noise → Increase mse_weight
- Color shifts → Add more MSE
- Lost structure → Add more SSIM

**Log both components during training to diagnose issues.**

### Dynamic Weighting

Some approaches adjust weights during training:

```python
class DynamicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable log weights
        self.log_mse_weight = nn.Parameter(torch.zeros(1))
        self.log_ssim_weight = nn.Parameter(torch.zeros(1))

    def forward(self, reconstruction, original):
        mse_loss = F.mse_loss(reconstruction, original)
        ssim_loss = 1 - ssim(reconstruction, original, data_range=1.0)

        # Use exp to ensure positive weights
        mse_weight = torch.exp(-self.log_mse_weight)
        ssim_weight = torch.exp(-self.log_ssim_weight)

        # Regularize to prevent collapse
        loss = mse_weight * mse_loss + ssim_weight * ssim_loss
        loss = loss + self.log_mse_weight + self.log_ssim_weight

        return loss
```

---

## Perceptual Losses

### The Concept

Instead of comparing pixels, compare high-level features:

```
Original → Pretrained Network → Features (original)
Reconstruction → Pretrained Network → Features (reconstruction)

Perceptual Loss = ||Features(original) - Features(reconstruction)||²
```

The pretrained network (usually VGG16 trained on ImageNet) extracts semantic features.

### Why Perceptual Loss?

Pixel-level losses can't capture:
- "This looks like a face"
- "This texture is realistic"
- "The overall structure is preserved"

Perceptual losses compare "what the image represents" rather than exact pixel values.

### Implementation

```python
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3']):
        super().__init__()
        # Load pretrained VGG
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Layer indices for VGG16
        self.layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
            'relu5_3': 30
        }

        # Build feature extractors
        self.feature_extractors = nn.ModuleList()
        prev_idx = 0
        for layer in layers:
            idx = self.layer_indices[layer]
            self.feature_extractors.append(vgg[prev_idx:idx])
            prev_idx = idx

    def forward(self, reconstruction, original):
        # VGG expects 3-channel input, normalized
        if reconstruction.shape[1] == 1:
            reconstruction = reconstruction.repeat(1, 3, 1, 1)
            original = original.repeat(1, 3, 1, 1)

        loss = 0
        x, y = reconstruction, original

        for extractor in self.feature_extractors:
            x = extractor(x)
            y = extractor(y)
            loss += F.mse_loss(x, y)

        return loss
```

### Caveats for SAR

**Problem:** VGG was trained on natural images, not SAR.

SAR images have:
- Single channel (grayscale)
- Speckle noise (not in natural images)
- Different texture statistics
- Different semantic content

**Options:**
1. Use perceptual loss anyway (might still help with structure)
2. Train a domain-specific feature extractor
3. Skip perceptual loss for SAR

**Recommendation:** Start without perceptual loss. Add it later if results are unsatisfactory and you have time to experiment.

---

## Gradient-Based Losses

### Edge Preservation

Edges are crucial for image quality. Gradient-based losses explicitly preserve edges:

```python
def gradient_loss(reconstruction, original):
    """Penalize differences in image gradients (edges)."""
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # Compute gradients
    grad_x_orig = F.conv2d(original, sobel_x, padding=1)
    grad_y_orig = F.conv2d(original, sobel_y, padding=1)
    grad_x_recon = F.conv2d(reconstruction, sobel_x, padding=1)
    grad_y_recon = F.conv2d(reconstruction, sobel_y, padding=1)

    # L1 loss on gradients
    loss = F.l1_loss(grad_x_recon, grad_x_orig) + F.l1_loss(grad_y_recon, grad_y_orig)

    return loss
```

### Total Variation Loss

Encourages smooth reconstructions (reduces noise but can over-smooth):

```python
def total_variation_loss(image):
    """Penalize pixel differences between neighbors."""
    diff_x = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    diff_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    return diff_x.mean() + diff_y.mean()
```

**Use sparingly:** Can over-smooth and destroy texture.

### Laplacian Loss

Preserves second-order structure (edges and corners):

```python
def laplacian_loss(reconstruction, original):
    """Penalize differences in Laplacian (second derivatives)."""
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32
    ).view(1, 1, 3, 3)

    lap_orig = F.conv2d(original, laplacian_kernel, padding=1)
    lap_recon = F.conv2d(reconstruction, laplacian_kernel, padding=1)

    return F.l1_loss(lap_recon, lap_orig)
```

---

## SAR-Specific Considerations

### The Speckle Problem

SAR images contain **speckle noise** — multiplicative noise from coherent imaging:

```
Observed = True_Signal × Speckle

In linear domain: I = σ × s, where s ~ Gamma distribution
In log/dB domain: log(I) = log(σ) + log(s), where log(s) ~ approximately Gaussian
```

**Implications for loss functions:**
- MSE in linear domain: Penalizes speckle fluctuations heavily
- MSE in log domain: Speckle becomes additive, more uniform penalty
- SSIM: Works well in log domain (speckle is "texture" that can be preserved statistically)

### Log Domain vs Linear Domain

**Training in log (dB) domain is recommended:**

| Aspect | Linear Domain | Log/dB Domain |
|--------|---------------|---------------|
| Speckle | Multiplicative | Additive |
| Dynamic range | Very large | Compressed |
| Loss behavior | Dominated by bright pixels | More uniform |
| Reconstruction | Harder | Easier |

**Conversion:**
```python
# Linear to dB
db = 10 * np.log10(linear + epsilon)

# dB to linear
linear = 10 ** (db / 10)
```

### ENL-Aware Loss

ENL (Equivalent Number of Looks) measures speckle reduction:

```
ENL = μ² / σ² (for homogeneous regions)

Higher ENL = less speckle = smoother
```

An ENL-aware loss could:
- Encourage appropriate ENL in homogeneous regions
- Allow natural speckle statistics elsewhere

```python
def enl_loss(reconstruction, original, homogeneous_mask):
    """Penalize ENL differences in homogeneous regions."""
    # Compute local statistics
    recon_masked = reconstruction * homogeneous_mask
    orig_masked = original * homogeneous_mask

    # Local mean and variance (using average pooling)
    kernel_size = 15
    mu_recon = F.avg_pool2d(recon_masked, kernel_size, stride=1, padding=kernel_size//2)
    mu_orig = F.avg_pool2d(orig_masked, kernel_size, stride=1, padding=kernel_size//2)

    var_recon = F.avg_pool2d(recon_masked ** 2, kernel_size, stride=1, padding=kernel_size//2) - mu_recon ** 2
    var_orig = F.avg_pool2d(orig_masked ** 2, kernel_size, stride=1, padding=kernel_size//2) - mu_orig ** 2

    # ENL
    enl_recon = (mu_recon ** 2) / (var_recon + 1e-8)
    enl_orig = (mu_orig ** 2) / (var_orig + 1e-8)

    return F.mse_loss(enl_recon, enl_orig)
```

**Note:** This is an advanced technique. Start with MSE + SSIM first.

---

## Practical Guidelines

### Recommended Starting Configuration

```python
class SARReconstructionLoss(nn.Module):
    """Combined loss for SAR autoencoder."""

    def __init__(self, mse_weight=0.5, ssim_weight=0.5, edge_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

        self.ssim_loss = SSIMLoss()

    def forward(self, reconstruction, original):
        # Basic losses
        mse = F.mse_loss(reconstruction, original)
        ssim = self.ssim_loss(reconstruction, original)

        # Optional: edge loss
        edge = gradient_loss(reconstruction, original) if self.edge_weight > 0 else 0

        # Combine
        total = self.mse_weight * mse + self.ssim_weight * ssim + self.edge_weight * edge

        return total, {
            'mse': mse.item(),
            'ssim': 1 - ssim.item(),
            'total': total.item()
        }
```

### Experimentation Order

1. **MSE only:** Baseline, simple to debug
2. **MSE + SSIM (0.5, 0.5):** Balanced combination
3. **SSIM-heavy (0.16, 0.84):** If MSE+SSIM is too blurry
4. **Add edge loss:** If edges are still soft
5. **MS-SSIM:** If single-scale SSIM isn't enough

### Monitoring During Training

Log all loss components:

```python
# During training loop
loss, metrics = criterion(reconstruction, original)

# Log to TensorBoard
writer.add_scalar('Loss/total', metrics['total'], step)
writer.add_scalar('Loss/mse', metrics['mse'], step)
writer.add_scalar('Metrics/ssim', metrics['ssim'], step)
writer.add_scalar('Metrics/psnr', 10 * np.log10(1 / metrics['mse']), step)
```

### Common Issues and Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Blurry output | MSE too dominant | Increase SSIM weight |
| Noisy output | SSIM too dominant | Increase MSE weight |
| Lost edges | Need edge supervision | Add gradient loss |
| Training unstable | SSIM gradient issues | Reduce SSIM weight or use MS-SSIM |
| Color shift | Global offset | Ensure MSE is included |

---

## Summary

**Key Takeaways:**

1. **MSE** minimizes pixel error but causes blur
2. **SSIM** preserves structure but can have gradient issues
3. **Combine MSE + SSIM** for best results (start with 0.5/0.5)
4. **For SAR**, work in log/dB domain
5. **Monitor all components** during training to diagnose issues
6. **Start simple**, add complexity only if needed
