# Deep Dive: SAR Image Preprocessing

This document provides a comprehensive explanation of preprocessing techniques for Synthetic Aperture Radar (SAR) imagery, why they're necessary, and how to implement them for compression autoencoders.

---

## Table of Contents

1. [Understanding SAR Images](#understanding-sar-images)
2. [Why SAR Needs Special Preprocessing](#why-sar-needs-special-preprocessing)
3. [The dB Transformation](#the-db-transformation)
4. [Handling Invalid Values](#handling-invalid-values)
5. [Dynamic Range Compression](#dynamic-range-compression)
6. [Normalization Strategies](#normalization-strategies)
7. [Patch Extraction](#patch-extraction)
8. [Complete Preprocessing Pipeline](#complete-preprocessing-pipeline)

---

## Understanding SAR Images

### How SAR Works

Synthetic Aperture Radar is an **active** imaging system:

1. **Transmit:** Satellite sends microwave pulses toward Earth
2. **Interact:** Pulses bounce off terrain, vegetation, buildings, water
3. **Receive:** Satellite receives reflected signals
4. **Process:** Complex signal processing creates an image

Unlike optical cameras that measure sunlight reflection, SAR measures its own transmitted signal's return.

### What SAR Measures

Each pixel contains:

**Amplitude (Intensity):**
- Proportional to how much energy was reflected back
- Bright pixels = strong reflection (buildings, rough surfaces)
- Dark pixels = weak reflection (smooth water, shadows)

**Phase:**
- Related to distance and surface properties
- Used in advanced applications (InSAR, coherence)
- Often discarded for basic imaging

For compression, we typically work with **intensity** (amplitude squared):
```
I = |complex_signal|²
```

### Sentinel-1 Specifics

Sentinel-1 provides:
- **Resolution:** 10m × 10m (IW mode)
- **Polarizations:** VV and VH (usually)
  - VV: Vertical transmit, vertical receive
  - VH: Vertical transmit, horizontal receive
- **Format:** GeoTIFF in SAFE archive structure
- **Units:** Usually sigma-nought (σ°) in decibels or linear

### Typical Value Ranges

**Linear intensity (σ°):**
```
Smooth water: ~0.0001 to 0.001
Vegetation: ~0.01 to 0.1
Urban: ~0.1 to 10
Very bright (metal): can exceed 100
```

**Dynamic range:** Often 40-60 dB (10,000× to 1,000,000× variation)

This extreme dynamic range is why preprocessing is essential.

---

## Why SAR Needs Special Preprocessing

### Problem 1: Extreme Dynamic Range

Neural networks work best with inputs in a reasonable range (e.g., [0, 1] or [-1, 1]).

SAR values span 4-6 orders of magnitude:
```
Min: ~0.00001
Max: ~100
Ratio: 10,000,000×
```

If you feed this directly to a network:
- Bright pixels dominate the loss
- Dark areas are essentially 0 (gradients vanish)
- Network can't learn useful representations

### Problem 2: Speckle Noise

SAR images contain **speckle** — granular noise from coherent imaging:

```
Observed = True_Signal × Speckle_Noise

Where Speckle ~ Gamma distribution (in intensity)
```

Key properties:
- **Multiplicative:** Noise scales with signal (bright areas have more absolute noise)
- **Fully developed speckle:** CV (coefficient of variation) ≈ 1 for single-look data
- **Not Gaussian:** Heavy-tailed distribution

In linear domain, speckle makes training difficult because:
- Noise magnitude varies across the image
- Network might try to memorize noise patterns
- Loss function weighted toward high-intensity areas

### Problem 3: Invalid Values

SAR images can contain:
- **Zeros:** No-data regions, shadow areas, calibration failures
- **Negative values:** Shouldn't exist but sometimes appear (processing artifacts)
- **Infinities/NaNs:** Numerical errors, corrupted pixels

These must be handled before training.

### The Solution: Log/dB Transform

Converting to decibels (dB) solves several problems:

```
dB = 10 × log₁₀(linear_intensity)
```

| Problem | Linear Domain | dB Domain |
|---------|---------------|-----------|
| Dynamic range | 10,000,000× | 70 dB range |
| Speckle | Multiplicative | Additive |
| Value distribution | Highly skewed | More Gaussian |
| Neural network compatibility | Poor | Good |

---

## The dB Transformation

### The Mathematics

**Linear intensity to dB:**
```
I_dB = 10 × log₁₀(I_linear)
```

**dB to linear intensity:**
```
I_linear = 10^(I_dB / 10)
```

### Why Logarithm Helps

**Multiplicative becomes additive:**
```
Linear: I_observed = I_true × Speckle
Log: log(I_observed) = log(I_true) + log(Speckle)
```

In log domain, speckle is additive noise with approximately constant variance.

**Dynamic range compression:**
```
Linear: 0.0001 to 100 → range of 1,000,000
dB: -40 to +20 → range of 60
```

**Distribution normalization:**
```
Linear: Heavily right-skewed (few very bright pixels)
dB: Approximately symmetric (closer to Gaussian)
```

### Implementation

```python
import numpy as np

def linear_to_db(intensity, floor=1e-10):
    """Convert linear intensity to decibels."""
    # Prevent log(0) by applying floor
    intensity = np.maximum(intensity, floor)
    return 10 * np.log10(intensity)

def db_to_linear(db):
    """Convert decibels to linear intensity."""
    return 10 ** (db / 10)
```

### Visual Example

```
Linear domain:
[0.001, 0.01, 0.1, 1.0, 10, 100] → [mostly dark, one bright]

dB domain:
[-30, -20, -10, 0, 10, 20] → [evenly spaced visually]
```

---

## Handling Invalid Values

### Types of Invalid Values

**1. Zeros and very small values:**
```python
# Can't take log of zero!
np.log10(0)  # -inf
np.log10(1e-20)  # -200 (extremely negative)
```

**2. Negative values:**
```python
# Log of negative is undefined (or complex)
np.log10(-0.5)  # nan (or complex in some implementations)
```

**3. NaN and Inf:**
```python
# Propagate through calculations, corrupt training
np.mean([1, 2, np.nan])  # nan
```

### Handling Strategy

**Step 1: Identify invalid pixels**
```python
def find_invalid(image):
    """Identify invalid pixels."""
    invalid = np.zeros_like(image, dtype=bool)
    invalid |= (image <= 0)  # Zero or negative
    invalid |= np.isnan(image)  # NaN
    invalid |= np.isinf(image)  # Inf
    return invalid
```

**Step 2: Replace with noise floor**
```python
def handle_invalid_values(image, noise_floor=1e-10):
    """Replace invalid values with noise floor."""
    image = image.copy()
    invalid = find_invalid(image)
    image[invalid] = noise_floor
    return image, invalid
```

**Why noise floor (not zero)?**
- log(noise_floor) gives a finite value
- Represents "no signal detected" physically
- Typical SAR noise floor: 1e-10 to 1e-6 depending on system

### Dealing with No-Data Regions

Large no-data regions (shadows, water sometimes) might need special handling:

**Option 1: Mask and ignore in loss**
```python
def masked_mse_loss(reconstruction, original, mask):
    """MSE only over valid pixels."""
    diff = (reconstruction - original) ** 2
    return (diff * mask).sum() / mask.sum()
```

**Option 2: Fill with neighborhood statistics**
```python
from scipy.ndimage import generic_filter

def fill_nodata(image, mask, window=5):
    """Fill no-data with local median."""
    def local_median(values):
        valid = values[~np.isnan(values)]
        return np.median(valid) if len(valid) > 0 else np.nan

    image[~mask] = np.nan
    filled = generic_filter(image, local_median, size=window)
    return filled
```

**Option 3: Exclude patches with too many invalid pixels**
```python
def is_valid_patch(patch, max_invalid_ratio=0.01):
    """Check if patch has acceptable amount of valid data."""
    invalid = find_invalid(patch)
    return invalid.sum() / invalid.size < max_invalid_ratio
```

---

## Dynamic Range Compression

### The Problem

Even in dB, SAR images can have outliers:

```
Typical range: -25 dB to +5 dB (30 dB total)
Outliers: -50 dB (deep shadows) to +30 dB (corner reflectors)
```

These outliers can:
- Dominate the loss function
- Cause numerical instability
- Waste network capacity on rare values

### Clipping Strategies

**Percentile clipping:**
```python
def percentile_clip(image_db, low_percentile=1, high_percentile=99):
    """Clip to percentile range."""
    vmin = np.percentile(image_db, low_percentile)
    vmax = np.percentile(image_db, high_percentile)
    return np.clip(image_db, vmin, vmax), vmin, vmax
```

**Sigma clipping (assumes approximately Gaussian):**
```python
def sigma_clip(image_db, n_sigma=3):
    """Clip to mean ± n_sigma * std."""
    mean = np.mean(image_db)
    std = np.std(image_db)
    vmin = mean - n_sigma * std
    vmax = mean + n_sigma * std
    return np.clip(image_db, vmin, vmax), vmin, vmax
```

**Fixed clipping (domain knowledge):**
```python
def fixed_clip(image_db, vmin=-25, vmax=5):
    """Clip to known SAR range."""
    return np.clip(image_db, vmin, vmax), vmin, vmax
```

### Which to Use?

| Method | When to Use |
|--------|-------------|
| Percentile | Unknown data distribution, want automatic bounds |
| Sigma | Data is approximately Gaussian (often true in dB) |
| Fixed | Well-understood data, consistency across images |

**Recommendation for Sentinel-1:**
- Start with fixed clipping: vmin=-25 dB, vmax=+5 dB
- These values cover most terrain types
- Adjust based on your specific data

### Computing Clip Bounds from Training Data

**Important:** Compute bounds from training set, apply same bounds to validation/test:

```python
def compute_dataset_bounds(images_db, method='percentile', **kwargs):
    """Compute clip bounds from entire dataset."""
    all_values = np.concatenate([img.flatten() for img in images_db])

    if method == 'percentile':
        low_p = kwargs.get('low_percentile', 1)
        high_p = kwargs.get('high_percentile', 99)
        vmin = np.percentile(all_values, low_p)
        vmax = np.percentile(all_values, high_p)
    elif method == 'sigma':
        n_sigma = kwargs.get('n_sigma', 3)
        mean = np.mean(all_values)
        std = np.std(all_values)
        vmin = mean - n_sigma * std
        vmax = mean + n_sigma * std
    else:
        raise ValueError(f"Unknown method: {method}")

    return vmin, vmax
```

---

## Normalization Strategies

### Why Normalize?

After clipping, values are in a fixed range (e.g., -25 to +5 dB).
Neural networks prefer [0, 1] or [-1, 1]:

```
Typical initialization: weights ~ N(0, 0.01)
Expected activations: small values around 0

Input of -25 dB → activations explode
Input of 0 to 1 → activations well-behaved
```

### Min-Max Normalization

```python
def minmax_normalize(image_db, vmin, vmax):
    """Normalize to [0, 1] range."""
    return (image_db - vmin) / (vmax - vmin)

def minmax_denormalize(image_norm, vmin, vmax):
    """Convert back from [0, 1] to original range."""
    return image_norm * (vmax - vmin) + vmin
```

**Properties:**
- Output guaranteed in [0, 1]
- Preserves relative differences
- Simple and interpretable

### Z-Score Normalization

```python
def zscore_normalize(image_db, mean, std):
    """Normalize to zero mean, unit variance."""
    return (image_db - mean) / std

def zscore_denormalize(image_norm, mean, std):
    """Convert back from z-score."""
    return image_norm * std + mean
```

**Properties:**
- Output centered at 0
- May extend beyond [-1, 1] (no hard bounds)
- Works well with batch normalization

### Which to Use?

| Method | Output Range | Best When |
|--------|--------------|-----------|
| Min-max | [0, 1] | Using sigmoid output, need bounded values |
| Z-score | Unbounded | Using batch norm, data approximately Gaussian |

**For autoencoders with sigmoid output:** Min-max normalization to [0, 1]

### Dataset-Wide vs Per-Image

**Dataset-wide (recommended):**
```python
# Compute once from training set
vmin, vmax = compute_dataset_bounds(training_images)

# Apply to all images
normalized = minmax_normalize(image_db, vmin, vmax)
```

**Per-image:**
```python
# Each image normalized independently
vmin, vmax = image_db.min(), image_db.max()
normalized = minmax_normalize(image_db, vmin, vmax)
```

**Why dataset-wide is better:**
- Consistent meaning of pixel values across images
- Network learns absolute values, not relative
- Needed for comparing reconstructions across images

---

## Patch Extraction

### Why Use Patches?

Full SAR images are huge:
- Sentinel-1 IW: ~25,000 × 16,000 pixels
- Memory: 25000 × 16000 × 4 bytes = 1.6 GB per image
- Doesn't fit in GPU memory for training

Solution: Extract smaller patches for training.

### Patch Size Selection

| Patch Size | Pros | Cons |
|------------|------|------|
| 64×64 | Fast, fits easily | Limited context, may miss structures |
| 128×128 | Good balance | Moderate memory |
| 256×256 | Rich context | Higher memory |
| 512×512 | Very rich context | May not fit in memory |

**For SAR with 10m resolution:**
- 256×256 patches = 2.56 km × 2.56 km ground coverage
- Captures most terrain features
- Fits in 8GB GPU memory with reasonable batch size

### Extraction Strategies

**Grid extraction (non-overlapping):**
```python
def extract_patches_grid(image, patch_size):
    """Extract non-overlapping patches."""
    H, W = image.shape
    patches = []

    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return np.array(patches)
```

**Overlapping extraction (more data):**
```python
def extract_patches_overlap(image, patch_size, stride):
    """Extract overlapping patches."""
    H, W = image.shape
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return np.array(patches)
```

**Random extraction (data augmentation):**
```python
def extract_patches_random(image, patch_size, n_patches):
    """Extract random patches."""
    H, W = image.shape
    patches = []

    for _ in range(n_patches):
        i = np.random.randint(0, H - patch_size + 1)
        j = np.random.randint(0, W - patch_size + 1)
        patch = image[i:i+patch_size, j:j+patch_size]
        patches.append(patch)

    return np.array(patches)
```

### Quality Filtering

Remove patches that are:
- Mostly no-data
- Completely homogeneous (no information)
- Corrupted

```python
def filter_patches(patches, min_std=0.01, max_invalid_ratio=0.01):
    """Filter out low-quality patches."""
    valid_patches = []

    for patch in patches:
        # Check for invalid values
        invalid_ratio = np.sum(patch <= 0) / patch.size
        if invalid_ratio > max_invalid_ratio:
            continue

        # Check for sufficient variation (not blank)
        if np.std(patch) < min_std:
            continue

        valid_patches.append(patch)

    return np.array(valid_patches)
```

---

## Complete Preprocessing Pipeline

### Full Pipeline Implementation

```python
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import rasterio

class SARPreprocessor:
    """Complete SAR preprocessing pipeline."""

    def __init__(
        self,
        noise_floor: float = 1e-10,
        clip_method: str = 'percentile',
        clip_low: float = 1,
        clip_high: float = 99,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ):
        self.noise_floor = noise_floor
        self.clip_method = clip_method
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.vmin = vmin
        self.vmax = vmax

        # Will be set after fitting
        self.fitted = False

    def fit(self, images: list):
        """Compute normalization parameters from training data."""
        all_values_db = []

        for img in images:
            # Handle invalid values
            img = self._handle_invalid(img)

            # Convert to dB
            img_db = self._to_db(img)

            all_values_db.append(img_db.flatten())

        all_values_db = np.concatenate(all_values_db)

        # Compute clip bounds
        if self.clip_method == 'percentile':
            self.vmin = np.percentile(all_values_db, self.clip_low)
            self.vmax = np.percentile(all_values_db, self.clip_high)
        elif self.clip_method == 'sigma':
            mean = np.mean(all_values_db)
            std = np.std(all_values_db)
            self.vmin = mean - 3 * std
            self.vmax = mean + 3 * std
        elif self.clip_method == 'fixed':
            # Use provided vmin/vmax or defaults
            self.vmin = self.vmin if self.vmin is not None else -25
            self.vmax = self.vmax if self.vmax is not None else 5

        self.fitted = True
        print(f"Fitted: vmin={self.vmin:.2f} dB, vmax={self.vmax:.2f} dB")

        return self

    def transform(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Transform single image."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # Step 1: Handle invalid values
        image = self._handle_invalid(image)

        # Step 2: Convert to dB
        image_db = self._to_db(image)

        # Step 3: Clip to range
        image_clipped = np.clip(image_db, self.vmin, self.vmax)

        # Step 4: Normalize to [0, 1]
        image_norm = (image_clipped - self.vmin) / (self.vmax - self.vmin)

        # Store parameters for inverse
        params = {
            'vmin': self.vmin,
            'vmax': self.vmax,
            'noise_floor': self.noise_floor
        }

        return image_norm, params

    def inverse_transform(self, image_norm: np.ndarray, params: Dict) -> np.ndarray:
        """Convert normalized image back to linear intensity."""
        vmin = params['vmin']
        vmax = params['vmax']

        # Denormalize
        image_db = image_norm * (vmax - vmin) + vmin

        # Convert to linear
        image_linear = 10 ** (image_db / 10)

        return image_linear

    def _handle_invalid(self, image: np.ndarray) -> np.ndarray:
        """Replace invalid values with noise floor."""
        image = image.copy()
        invalid = (image <= 0) | np.isnan(image) | np.isinf(image)
        image[invalid] = self.noise_floor
        return image

    def _to_db(self, image: np.ndarray) -> np.ndarray:
        """Convert to decibels."""
        return 10 * np.log10(image)


def load_sar_image(filepath: str) -> np.ndarray:
    """Load SAR image from GeoTIFF."""
    with rasterio.open(filepath) as src:
        image = src.read(1).astype(np.float32)
    return image


def preprocess_and_extract_patches(
    image_paths: list,
    patch_size: int = 256,
    stride: int = 128,
    preprocessor: Optional[SARPreprocessor] = None
) -> Tuple[np.ndarray, SARPreprocessor]:
    """Full pipeline: load, preprocess, extract patches."""

    # Load images
    images = [load_sar_image(p) for p in image_paths]

    # Fit preprocessor if not provided
    if preprocessor is None:
        preprocessor = SARPreprocessor(clip_method='percentile')
        preprocessor.fit(images)

    # Process and extract patches
    all_patches = []

    for image in images:
        # Preprocess
        image_norm, _ = preprocessor.transform(image)

        # Extract patches
        patches = extract_patches_overlap(image_norm, patch_size, stride)

        # Filter quality
        patches = filter_patches(patches)

        all_patches.append(patches)

    all_patches = np.concatenate(all_patches, axis=0)
    print(f"Extracted {len(all_patches)} patches")

    return all_patches, preprocessor


def extract_patches_overlap(image, patch_size, stride):
    """Extract overlapping patches."""
    H, W = image.shape
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return np.array(patches)


def filter_patches(patches, min_std=0.02, max_invalid_ratio=0.01):
    """Filter low-quality patches."""
    valid = []
    for patch in patches:
        # Check variation
        if np.std(patch) < min_std:
            continue
        # Check for edge values (likely invalid after normalization)
        edge_ratio = np.mean((patch <= 0.001) | (patch >= 0.999))
        if edge_ratio > max_invalid_ratio:
            continue
        valid.append(patch)
    return np.array(valid) if valid else np.array([]).reshape(0, *patches.shape[1:])
```

### Usage Example

```python
# Training: Fit and transform
from pathlib import Path

# Get image paths
image_dir = Path("data/sentinel1/")
image_paths = list(image_dir.glob("*.tif"))

# Preprocess and extract patches
patches, preprocessor = preprocess_and_extract_patches(
    image_paths,
    patch_size=256,
    stride=128
)

# Save patches
np.save("data/patches/patches.npy", patches)

# Save preprocessor params
import json
params = {
    'vmin': preprocessor.vmin,
    'vmax': preprocessor.vmax,
    'noise_floor': preprocessor.noise_floor
}
with open("data/patches/preprocess_params.json", 'w') as f:
    json.dump(params, f)


# Inference: Load params and inverse transform
with open("data/patches/preprocess_params.json", 'r') as f:
    params = json.load(f)

# After reconstruction
reconstructed_norm = model(input_batch)  # Network output in [0, 1]

# Convert back to linear intensity
reconstructed_linear = preprocessor.inverse_transform(
    reconstructed_norm.cpu().numpy(),
    params
)
```

---

## Summary

**Preprocessing Pipeline for SAR:**

1. **Handle invalid values:** Replace zeros, negatives, NaN/Inf with noise floor
2. **Convert to dB:** `10 * log10(intensity)` to compress dynamic range
3. **Clip outliers:** Percentile or sigma clipping (typically -25 to +5 dB)
4. **Normalize:** Scale to [0, 1] for neural network
5. **Extract patches:** 256×256 with overlap for training data
6. **Filter patches:** Remove low-quality (blank, corrupted) patches

**Key Points:**
- Always use dataset-wide normalization parameters
- Save parameters for inverse transform during inference
- Work in dB domain for training (makes speckle additive)
- Convert back to linear for final output if needed
