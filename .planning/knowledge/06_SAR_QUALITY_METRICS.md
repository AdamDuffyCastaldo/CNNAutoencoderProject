# Deep Dive: Quality Metrics for SAR Image Compression

This document provides a comprehensive explanation of quality metrics for evaluating SAR image reconstruction, including both standard image metrics and SAR-specific measures.

---

## Table of Contents

1. [Why Multiple Metrics?](#why-multiple-metrics)
2. [Pixel-Level Metrics](#pixel-level-metrics)
3. [Structural Metrics](#structural-metrics)
4. [SAR-Specific Metrics](#sar-specific-metrics)
5. [Statistical Metrics](#statistical-metrics)
6. [Compression Metrics](#compression-metrics)
7. [Practical Evaluation Framework](#practical-evaluation-framework)

---

## Why Multiple Metrics?

### No Single Perfect Metric

Different metrics capture different aspects of quality:

| Metric | What It Measures | Limitation |
|--------|-----------------|------------|
| MSE | Pixel accuracy | Doesn't match perception |
| PSNR | Signal fidelity | Insensitive to structure |
| SSIM | Structural similarity | Can miss fine details |
| ENL | Speckle properties | Only for homogeneous regions |
| EPI | Edge preservation | Focused on boundaries only |

A reconstruction might score well on one metric but poorly on another.

### For SAR Compression

You need metrics that capture:
1. **Overall fidelity:** How close is the reconstruction?
2. **Structure preservation:** Are edges and features intact?
3. **SAR statistics:** Is the speckle behavior correct?
4. **Application utility:** Can downstream analysis still work?

### Metric Suite Recommendation

```
Primary metrics:
- PSNR (overall quality)
- SSIM (structure)
- ENL ratio (speckle)
- EPI (edges)

Secondary metrics:
- MAE (absolute errors)
- Correlation coefficient (linear relationship)
- Histogram similarity (distribution preservation)
```

---

## Pixel-Level Metrics

### Mean Squared Error (MSE)

**Definition:**
```
MSE = (1/N) × Σᵢ (xᵢ - x̂ᵢ)²
```

**Interpretation:**
- MSE = 0: Perfect reconstruction
- Higher MSE: More pixel-level errors

**Implementation:**
```python
def mse(original, reconstruction):
    """Compute Mean Squared Error."""
    return np.mean((original - reconstruction) ** 2)
```

**Limitations:**
- Doesn't distinguish structured errors from random errors
- Sensitive to outliers (squared term)
- Doesn't match human perception

### Mean Absolute Error (MAE)

**Definition:**
```
MAE = (1/N) × Σᵢ |xᵢ - x̂ᵢ|
```

**Interpretation:**
- More robust to outliers than MSE
- Linear penalty (not squared)

**Implementation:**
```python
def mae(original, reconstruction):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(original - reconstruction))
```

### Peak Signal-to-Noise Ratio (PSNR)

**Definition:**
```
PSNR = 10 × log₁₀(MAX² / MSE)

Where MAX = maximum possible value (1.0 for normalized images)
```

**Interpretation:**
- Measured in decibels (dB)
- Higher is better
- Typical values: 20-40 dB for compression
- PSNR > 30 dB: Generally considered good
- PSNR > 40 dB: Excellent, near-lossless

**Implementation:**
```python
def psnr(original, reconstruction, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio."""
    mse_val = mse(original, reconstruction)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse_val)
```

**For SAR:**
Compute PSNR in the normalized domain (where you trained):
```python
# If training in [0, 1] normalized dB domain
psnr_db = psnr(original_norm, reconstruction_norm, max_val=1.0)
```

### Relative Error Metrics

**Normalized Root MSE (NRMSE):**
```python
def nrmse(original, reconstruction):
    """Normalized RMSE (by original range)."""
    rmse = np.sqrt(mse(original, reconstruction))
    range_val = original.max() - original.min()
    return rmse / range_val if range_val > 0 else 0
```

**Percentage Error:**
```python
def percentage_error(original, reconstruction):
    """Mean absolute percentage error."""
    # Avoid division by zero
    mask = np.abs(original) > 1e-10
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((original[mask] - reconstruction[mask]) / original[mask])) * 100
```

---

## Structural Metrics

### Structural Similarity Index (SSIM)

**Definition:**
```
SSIM(x,y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
```

**Interpretation:**
- SSIM ∈ [-1, 1]
- SSIM = 1: Identical images
- SSIM > 0.9: Excellent structural similarity
- SSIM > 0.8: Good
- SSIM < 0.7: Noticeable structural differences

**Implementation:**
```python
from skimage.metrics import structural_similarity

def ssim(original, reconstruction, win_size=11, data_range=1.0):
    """Compute Structural Similarity Index."""
    return structural_similarity(
        original, reconstruction,
        win_size=win_size,
        data_range=data_range
    )
```

### Multi-Scale SSIM (MS-SSIM)

Evaluates at multiple resolutions:

```python
from pytorch_msssim import ms_ssim
import torch

def ms_ssim_score(original, reconstruction):
    """Multi-scale SSIM."""
    # Convert to torch tensors
    orig_t = torch.from_numpy(original).unsqueeze(0).unsqueeze(0).float()
    recon_t = torch.from_numpy(reconstruction).unsqueeze(0).unsqueeze(0).float()

    return ms_ssim(orig_t, recon_t, data_range=1.0).item()
```

**When to use:**
- MS-SSIM is more robust to scale variations
- Better for images viewed at different sizes
- Often correlates better with human perception

### Universal Quality Index (UQI)

Predecessor to SSIM, simpler formula:

```python
def uqi(original, reconstruction):
    """Universal Quality Index."""
    # Means
    mu_x = np.mean(original)
    mu_y = np.mean(reconstruction)

    # Variances and covariance
    sigma_x = np.std(original)
    sigma_y = np.std(reconstruction)
    sigma_xy = np.mean((original - mu_x) * (reconstruction - mu_y))

    # UQI
    numerator = 4 * sigma_xy * mu_x * mu_y
    denominator = (sigma_x**2 + sigma_y**2) * (mu_x**2 + mu_y**2)

    return numerator / denominator if denominator > 0 else 0
```

---

## SAR-Specific Metrics

### Equivalent Number of Looks (ENL)

**What it measures:** Speckle smoothness in homogeneous regions.

**Definition:**
```
ENL = μ² / σ²

Where μ = mean intensity, σ² = variance
For fully developed speckle: ENL ≈ 1
More looks (averaging) → higher ENL
```

**For evaluating reconstruction:**
Compare ENL of original vs reconstruction in homogeneous regions.

```python
def compute_enl(image, homogeneous_mask=None):
    """Compute Equivalent Number of Looks."""
    if homogeneous_mask is not None:
        values = image[homogeneous_mask]
    else:
        values = image.flatten()

    mean = np.mean(values)
    var = np.var(values)

    if var < 1e-10:
        return float('inf')  # Perfectly smooth

    return (mean ** 2) / var


def enl_ratio(original, reconstruction, homogeneous_mask=None):
    """Ratio of ENL: reconstruction vs original."""
    enl_orig = compute_enl(original, homogeneous_mask)
    enl_recon = compute_enl(reconstruction, homogeneous_mask)

    return enl_recon / enl_orig if enl_orig > 0 else float('inf')
```

**Interpretation:**
- ENL ratio ≈ 1: Speckle properties preserved
- ENL ratio > 1: Over-smoothed (lost speckle texture)
- ENL ratio < 1: Added noise (reconstruction is noisier)

**Finding homogeneous regions:**
```python
def find_homogeneous_regions(image, window_size=15, cv_threshold=0.3):
    """Find regions with low coefficient of variation."""
    from scipy.ndimage import uniform_filter

    # Local mean and std
    local_mean = uniform_filter(image, size=window_size)
    local_sq_mean = uniform_filter(image**2, size=window_size)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))

    # Coefficient of variation
    cv = local_std / (local_mean + 1e-10)

    # Homogeneous where CV is low
    return cv < cv_threshold
```

### Edge Preservation Index (EPI)

**What it measures:** How well edges are preserved after compression.

**Definition:**
```
EPI = Σ|∇R × ∇I| / (Σ|∇R|² × Σ|∇I|²)^0.5

Where ∇ = gradient (Sobel), R = reconstruction, I = original
```

This is essentially the correlation of gradient magnitudes.

```python
from scipy.ndimage import sobel

def compute_gradients(image):
    """Compute gradient magnitude using Sobel."""
    gx = sobel(image, axis=1)  # Horizontal gradient
    gy = sobel(image, axis=0)  # Vertical gradient
    return np.sqrt(gx**2 + gy**2)


def edge_preservation_index(original, reconstruction):
    """Compute Edge Preservation Index."""
    grad_orig = compute_gradients(original)
    grad_recon = compute_gradients(reconstruction)

    # Flatten for correlation
    go = grad_orig.flatten()
    gr = grad_recon.flatten()

    # Correlation
    numerator = np.sum(go * gr)
    denominator = np.sqrt(np.sum(go**2) * np.sum(gr**2))

    return numerator / denominator if denominator > 0 else 0
```

**Interpretation:**
- EPI ≈ 1: Edges well preserved
- EPI < 0.9: Some edge degradation
- EPI < 0.8: Significant edge loss (blurring)

### Radiometric Resolution Preservation

**What it measures:** Whether the reconstruction maintains correct intensity relationships.

```python
def radiometric_accuracy(original, reconstruction):
    """Assess radiometric fidelity."""

    # Linear regression: recon = a * orig + b
    # Ideal: a = 1, b = 0
    from scipy.stats import linregress

    orig_flat = original.flatten()
    recon_flat = reconstruction.flatten()

    slope, intercept, r_value, _, _ = linregress(orig_flat, recon_flat)

    return {
        'slope': slope,  # Should be ~1
        'intercept': intercept,  # Should be ~0
        'r_squared': r_value**2,  # Should be ~1
        'bias': np.mean(recon_flat - orig_flat)  # Should be ~0
    }
```

### Mean Ratio (for multiplicative noise)

**What it measures:** Ratio statistics in linear domain.

For SAR, the ratio I_reconstruction / I_original should have:
- Mean ≈ 1
- Variance related to speckle properties

```python
def mean_ratio_metrics(original, reconstruction, epsilon=1e-10):
    """Compute ratio statistics (linear domain)."""
    ratio = reconstruction / (original + epsilon)

    return {
        'mean_ratio': np.mean(ratio),  # Should be ~1
        'std_ratio': np.std(ratio),
        'median_ratio': np.median(ratio)
    }
```

---

## Statistical Metrics

### Correlation Coefficient

**Definition:**
```
r = Σ[(x - μₓ)(y - μᵧ)] / √[Σ(x - μₓ)² × Σ(y - μᵧ)²]
```

**Implementation:**
```python
def correlation(original, reconstruction):
    """Pearson correlation coefficient."""
    orig = original.flatten()
    recon = reconstruction.flatten()
    return np.corrcoef(orig, recon)[0, 1]
```

**Interpretation:**
- r = 1: Perfect positive correlation
- r > 0.99: Excellent
- r > 0.95: Very good
- r < 0.9: Significant deviation

### Histogram Similarity

**What it measures:** Whether the intensity distribution is preserved.

```python
def histogram_similarity(original, reconstruction, bins=256):
    """Compare histograms using multiple metrics."""

    hist_orig, bin_edges = np.histogram(original.flatten(), bins=bins, density=True)
    hist_recon, _ = np.histogram(reconstruction.flatten(), bins=bin_edges, density=True)

    # Normalize
    hist_orig = hist_orig / (hist_orig.sum() + 1e-10)
    hist_recon = hist_recon / (hist_recon.sum() + 1e-10)

    # Bhattacharyya coefficient
    bhatt = np.sum(np.sqrt(hist_orig * hist_recon))

    # Chi-square distance
    chi_sq = np.sum((hist_orig - hist_recon)**2 / (hist_orig + hist_recon + 1e-10))

    # KL divergence
    kl_div = np.sum(hist_orig * np.log((hist_orig + 1e-10) / (hist_recon + 1e-10)))

    return {
        'bhattacharyya': bhatt,  # 1 = identical, 0 = no overlap
        'chi_square': chi_sq,    # 0 = identical
        'kl_divergence': kl_div  # 0 = identical
    }
```

### Local Statistics Preservation

**What it measures:** Whether local mean/variance patterns are preserved.

```python
def local_stats_error(original, reconstruction, window_size=15):
    """Compare local statistics."""
    from scipy.ndimage import uniform_filter

    # Local means
    mean_orig = uniform_filter(original, size=window_size)
    mean_recon = uniform_filter(reconstruction, size=window_size)

    # Local variances
    var_orig = uniform_filter(original**2, size=window_size) - mean_orig**2
    var_recon = uniform_filter(reconstruction**2, size=window_size) - mean_recon**2

    return {
        'mean_mse': np.mean((mean_orig - mean_recon)**2),
        'var_mse': np.mean((var_orig - var_recon)**2),
        'mean_correlation': np.corrcoef(mean_orig.flatten(), mean_recon.flatten())[0, 1],
        'var_correlation': np.corrcoef(var_orig.flatten(), var_recon.flatten())[0, 1]
    }
```

---

## Compression Metrics

### Compression Ratio

**Definition:**
```
CR = Original_size / Compressed_size
```

**For autoencoders:**
```python
def compression_ratio(input_shape, latent_shape, input_bits=32, latent_bits=32):
    """Compute compression ratio from shapes."""
    input_size = np.prod(input_shape) * input_bits
    latent_size = np.prod(latent_shape) * latent_bits

    return input_size / latent_size
```

**Example:**
```
Input: 256 × 256 × 1 = 65,536 values
Latent: 16 × 16 × 16 = 4,096 values
CR = 65,536 / 4,096 = 16×
```

### Bits Per Pixel (BPP)

**Definition:**
```
BPP = Total_bits / Number_of_pixels
```

```python
def bits_per_pixel(latent_shape, input_spatial_shape, latent_bits=32):
    """Compute bits per pixel."""
    total_bits = np.prod(latent_shape) * latent_bits
    total_pixels = np.prod(input_spatial_shape)

    return total_bits / total_pixels
```

**Example:**
```
Latent: 16 × 16 × 16 × 32 bits = 131,072 bits
Input pixels: 256 × 256 = 65,536 pixels
BPP = 131,072 / 65,536 = 2.0 bpp
```

For reference:
- Uncompressed float32: 32 bpp
- JPEG (typical): 0.5-2 bpp
- Aggressive compression: < 0.5 bpp

### Rate-Distortion Analysis

Plot quality vs compression:

```python
def rate_distortion_analysis(model_variants, test_data):
    """Evaluate multiple models at different compression levels."""
    results = []

    for variant in model_variants:
        model = variant['model']
        cr = variant['compression_ratio']

        # Evaluate on test data
        psnr_vals = []
        ssim_vals = []

        for batch in test_data:
            recon = model(batch)
            psnr_vals.append(psnr(batch, recon))
            ssim_vals.append(ssim(batch, recon))

        results.append({
            'compression_ratio': cr,
            'bpp': 32 / cr,  # Assuming float32 latent
            'psnr': np.mean(psnr_vals),
            'ssim': np.mean(ssim_vals)
        })

    return pd.DataFrame(results)
```

---

## Practical Evaluation Framework

### Complete Metrics Class

```python
import numpy as np
from scipy.ndimage import sobel, uniform_filter
from skimage.metrics import structural_similarity
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SARMetrics:
    """Container for SAR image quality metrics."""
    mse: float
    mae: float
    psnr: float
    ssim: float
    enl_ratio: float
    epi: float
    correlation: float
    mean_ratio: float

    def to_dict(self) -> Dict:
        return {
            'mse': self.mse,
            'mae': self.mae,
            'psnr': self.psnr,
            'ssim': self.ssim,
            'enl_ratio': self.enl_ratio,
            'epi': self.epi,
            'correlation': self.correlation,
            'mean_ratio': self.mean_ratio
        }

    def __str__(self):
        return (f"PSNR: {self.psnr:.2f} dB | SSIM: {self.ssim:.4f} | "
                f"ENL ratio: {self.enl_ratio:.3f} | EPI: {self.epi:.4f}")


class SARQualityEvaluator:
    """Comprehensive SAR image quality evaluation."""

    def __init__(self, data_range=1.0, ssim_win_size=11):
        self.data_range = data_range
        self.ssim_win_size = ssim_win_size

    def evaluate(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        homogeneous_mask: Optional[np.ndarray] = None
    ) -> SARMetrics:
        """Compute all metrics."""

        # Basic metrics
        mse_val = np.mean((original - reconstruction) ** 2)
        mae_val = np.mean(np.abs(original - reconstruction))

        # PSNR
        if mse_val == 0:
            psnr_val = float('inf')
        else:
            psnr_val = 10 * np.log10(self.data_range ** 2 / mse_val)

        # SSIM
        ssim_val = structural_similarity(
            original, reconstruction,
            win_size=self.ssim_win_size,
            data_range=self.data_range
        )

        # ENL ratio
        if homogeneous_mask is None:
            homogeneous_mask = self._find_homogeneous(original)

        enl_orig = self._compute_enl(original, homogeneous_mask)
        enl_recon = self._compute_enl(reconstruction, homogeneous_mask)
        enl_ratio = enl_recon / enl_orig if enl_orig > 0 else 1.0

        # Edge Preservation Index
        epi_val = self._compute_epi(original, reconstruction)

        # Correlation
        corr_val = np.corrcoef(original.flatten(), reconstruction.flatten())[0, 1]

        # Mean ratio (in original domain, assumes values > 0)
        epsilon = 1e-10
        ratio = reconstruction / (original + epsilon)
        mean_ratio = np.mean(ratio)

        return SARMetrics(
            mse=mse_val,
            mae=mae_val,
            psnr=psnr_val,
            ssim=ssim_val,
            enl_ratio=enl_ratio,
            epi=epi_val,
            correlation=corr_val,
            mean_ratio=mean_ratio
        )

    def evaluate_batch(
        self,
        original_batch: np.ndarray,
        reconstruction_batch: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate a batch and return averaged metrics."""
        all_metrics = []

        for orig, recon in zip(original_batch, reconstruction_batch):
            # Remove channel dimension if present
            if orig.ndim == 3:
                orig = orig.squeeze()
                recon = recon.squeeze()

            metrics = self.evaluate(orig, recon)
            all_metrics.append(metrics.to_dict())

        # Average
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def _find_homogeneous(self, image, window_size=15, cv_threshold=0.3):
        """Find homogeneous regions."""
        local_mean = uniform_filter(image, size=window_size)
        local_sq_mean = uniform_filter(image**2, size=window_size)
        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))

        cv = local_std / (local_mean + 1e-10)
        return cv < cv_threshold

    def _compute_enl(self, image, mask):
        """Compute ENL for masked region."""
        if mask.sum() < 100:  # Need enough samples
            return 1.0

        values = image[mask]
        mean = np.mean(values)
        var = np.var(values)

        if var < 1e-10:
            return float('inf')

        return (mean ** 2) / var

    def _compute_epi(self, original, reconstruction):
        """Compute Edge Preservation Index."""
        grad_orig = self._gradient_magnitude(original)
        grad_recon = self._gradient_magnitude(reconstruction)

        go = grad_orig.flatten()
        gr = grad_recon.flatten()

        numerator = np.sum(go * gr)
        denominator = np.sqrt(np.sum(go**2) * np.sum(gr**2))

        return numerator / denominator if denominator > 0 else 0

    def _gradient_magnitude(self, image):
        """Compute gradient magnitude."""
        gx = sobel(image, axis=1)
        gy = sobel(image, axis=0)
        return np.sqrt(gx**2 + gy**2)
```

### Usage Example

```python
# Initialize evaluator
evaluator = SARQualityEvaluator(data_range=1.0)

# Evaluate single pair
metrics = evaluator.evaluate(original_image, reconstruction)
print(metrics)
# Output: PSNR: 32.45 dB | SSIM: 0.9234 | ENL ratio: 1.023 | EPI: 0.9156

# Evaluate batch
batch_metrics = evaluator.evaluate_batch(original_batch, reconstruction_batch)
print(f"Batch PSNR: {batch_metrics['psnr']:.2f} dB")

# Track during training
for epoch in range(num_epochs):
    # ... training code ...

    # Validation metrics
    val_metrics = evaluator.evaluate_batch(val_originals, val_reconstructions)

    # Log to TensorBoard
    for name, value in val_metrics.items():
        writer.add_scalar(f'Validation/{name}', value, epoch)
```

---

## Summary

**Key Metrics for SAR Compression:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| PSNR | Overall fidelity | > 30 dB (good), > 35 dB (excellent) |
| SSIM | Structural similarity | > 0.9 (good), > 0.95 (excellent) |
| ENL ratio | Speckle preservation | ≈ 1.0 (0.9-1.1 acceptable) |
| EPI | Edge preservation | > 0.9 (good), > 0.95 (excellent) |
| Correlation | Linear relationship | > 0.99 (excellent) |

**Evaluation Strategy:**
1. Use PSNR/SSIM for primary quality assessment
2. Use ENL ratio to check speckle handling
3. Use EPI to verify edge preservation
4. Plot rate-distortion curves to compare architectures
5. Visualize reconstructions for qualitative assessment
