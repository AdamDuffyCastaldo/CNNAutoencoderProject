"""
SAR Image Preprocessing

This module implements the preprocessing pipeline for SAR images:
1. Handle invalid values (zeros, NaN, negative)
2. Convert to dB scale
3. Clip outliers
4. Normalize to [0, 1]

Key Principles:
- Work in dB domain (makes speckle additive, compresses dynamic range)
- Use DATASET-WIDE statistics (not per-image) for normalization
- Preserve preprocessing parameters for inverse transform

References:
    - Day 1, Session 4 of the learning guide
"""

import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path


def handle_invalid_values(
    image: np.ndarray,
    noise_floor: float = 1e-10
) -> np.ndarray:
    """
    Handle invalid values in SAR image.
    
    SAR images can have:
    - Zeros (nodata, failed calibration)
    - Negative values (shouldn't exist but sometimes do)
    - NaN/Inf (processing errors)
    
    Args:
        image: Input SAR intensity image
        noise_floor: Value to replace invalid pixels with
    
    Returns:
        Cleaned image with no invalid values
    """
    # TODO: Implement invalid value handling
    #
    # image = np.copy(image)
    # image = np.where(image <= 0, noise_floor, image)
    # image = np.where(np.isnan(image), noise_floor, image)
    # image = np.where(np.isinf(image), noise_floor, image)
    # return image
    
    raise NotImplementedError("TODO: Implement invalid value handling")




def from_db(db: np.ndarray) -> np.ndarray:
    """
    Convert dB back to linear intensity.
    
    Formula: intensity = 10^(dB/10)
    
    Args:
        db: Values in dB scale
    
    Returns:
        Linear intensity values
    """
    # TODO: Implement inverse dB conversion
    raise NotImplementedError("TODO: Implement inverse dB conversion")


def compute_clip_bounds(
    images: np.ndarray,
    method: str = 'percentile',
    **kwargs
) -> Tuple[float, float]:
    """
    Compute clip bounds from training data.
    
    Methods:
    - 'percentile': Use data-driven percentiles (low_pct, high_pct)
    - 'fixed': Use domain knowledge (vmin, vmax)
    - 'sigma': Use mean ± k×std
    
    Args:
        images: Array of images (in dB)
        method: Clipping method
        **kwargs: Method-specific parameters
    
    Returns:
        (vmin, vmax) clip bounds in dB
    
    Example:
        >>> bounds = compute_clip_bounds(images, method='percentile', 
        ...                              low_pct=1, high_pct=99)
    """
    # TODO: Implement clip bounds computation
    #
    # if method == 'percentile':
    #     low_pct = kwargs.get('low_pct', 1)
    #     high_pct = kwargs.get('high_pct', 99)
    #     vmin = np.percentile(images, low_pct)
    #     vmax = np.percentile(images, high_pct)
    # elif method == 'fixed':
    #     vmin = kwargs.get('vmin', -25)
    #     vmax = kwargs.get('vmax', 5)
    # elif method == 'sigma':
    #     k = kwargs.get('k', 3)
    #     vmin = np.mean(images) - k * np.std(images)
    #     vmax = np.mean(images) + k * np.std(images)
    # return vmin, vmax
    
    raise NotImplementedError("TODO: Implement clip bounds computation")


def preprocess_sar_complete(image, vmin=None, vmax=None, clip_percentiles=(1, 99)):
    """
    Complete preprocessing pipeline for SAR images.
        
    Returns:
        normalized: Image normalized to [0, 1]
        params: Dict with parameters needed for inverse transform
    """
    # Step 1: Handle invalid values
    invalid_mask = (
        (image <= 0) | 
        np.isnan(image) | 
        np.isinf(image)
    )
    image_clean = np.copy(image)
    noise_floor = 1e-10
    image_clean = np.where(image <= 0, noise_floor, image_clean)
    image_clean = np.where(np.isnan(image_clean), noise_floor, image_clean)
    image_clean = np.where(np.isinf(image_clean), noise_floor, image_clean)

    
    
    # Step 2: Convert to dB
    image_db = 10 * np.log10(image_clean)
    image_db = np.maximum(image_db, noise_floor)
    
    
    # Step 3: Determine clip bounds
    if vmin is None or vmax is None:
        valid_db = image_db
        if vmin is None:
            vmin = np.percentile(valid_db, clip_percentiles[0])
        if vmax is None:
            vmax = np.percentile(valid_db, clip_percentiles[1])
    
    # Step 4: Clip
    image_clipped = np.clip(image_db, vmin, vmax)
    
    # Step 5: Normalize to [0, 1]
    normalized = (image_clipped - vmin) / (vmax - vmin)
    
    # Store parameters for inverse transform
    params = {
        'vmin': vmin,
        'vmax': vmax,
        'invalid_mask': invalid_mask
    }
    
    return normalized, params


def inverse_preprocess(normalized, params):
    """
    Inverse preprocessing: convert [0, 1] normalized image back to linear intensity.
    
    Args:
        normalized: Normalized image in [0, 1]
        params: Parameters from preprocess_sar_complete
        
    Returns:
        image_linear: Reconstructed linear intensity
    """
    vmin = params['vmin']
    vmax = params['vmax']
    
    # Step 1: Denormalize to dB
    image_db = normalized * (vmax - vmin) + vmin
    
    # Step 2: Convert to linear
    image_linear = 10 ** (image_db / 10)
    
    return image_linear


def extract_patches(image, patch_size=256, stride=128, min_valid=0.9):
    """
    Extract patches from normalized image.
    
    Args:
        image: Normalized image [0, 1]
        patch_size: Size of square patches
        stride: Step between patches (stride < patch_size = overlap)
        min_valid: Minimum fraction of non-saturated pixels
        
    Returns:
        patches: Array of shape (N, patch_size, patch_size)
        positions: List of (y, x) positions
    """
    patches = []
    positions = []
    
    h, w = image.shape
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            
            # Check for valid pixels (not at clip boundaries)
            valid_frac = np.mean((patch > 0.01) & (patch < 0.99))
            
            if valid_frac >= min_valid:
                patches.append(patch)
                positions.append((i, j))
    
    return np.array(patches), positions


def analyze_sar_statistics(image):
    """
    Compute comprehensive statistics for SAR image.
    
    Returns a dictionary containing:
    - Linear domain statistics
    - dB domain statistics  
    - Percentiles for clipping decisions
    - Estimated number of looks from CV
    """
    # Handle invalid values
    image_valid = np.where(image > 0, image, np.nan)
    image_clean = image_valid[~np.isnan(image_valid)]
    
    # =========================================
    # LINEAR DOMAIN STATISTICS
    # =========================================
    linear_stats = {
        'min': float(np.min(image_clean)),
        'max': float(np.max(image_clean)),
        'mean': float(np.mean(image_clean)),
        'std': float(np.std(image_clean)),
        'median': float(np.median(image_clean)),
    }
    
    # Dynamic range in dB
    linear_stats['dynamic_range_dB'] = 10 * np.log10(
        linear_stats['max'] / linear_stats['min']
    )
    
    # =========================================
    # CONVERT TO dB
    # =========================================
    image_db = 10 * np.log10(image_clean)
    
    db_stats = {
        'min': float(np.min(image_db)),
        'max': float(np.max(image_db)),
        'mean': float(np.mean(image_db)),
        'std': float(np.std(image_db)),
        'median': float(np.median(image_db)),
    }
    
    # =========================================
    # PERCENTILES (for clipping decisions)
    # =========================================
    percentiles = {
        'p0.5': float(np.percentile(image_db, 0.5)),
        'p1': float(np.percentile(image_db, 1)),
        'p5': float(np.percentile(image_db, 5)),
        'p50': float(np.percentile(image_db, 50)),
        'p95': float(np.percentile(image_db, 95)),
        'p99': float(np.percentile(image_db, 99)),
        'p99.5': float(np.percentile(image_db, 99.5)),
    }
    
    # =========================================
    # ESTIMATE NUMBER OF LOOKS
    # For gamma-distributed intensity: CV = 1/sqrt(L)
    # So L = 1/CV²
    # =========================================
    cv = linear_stats['std'] / linear_stats['mean']
    estimated_looks = 1 / (cv ** 2)
    
    # =========================================
    # COUNT INVALID PIXELS
    # =========================================
    n_total = image.size
    n_invalid = np.sum(image <= 0)
    
    return {
        'linear': linear_stats,
        'dB': db_stats,
        'percentiles': percentiles,
        'estimated_looks': estimated_looks,
        'coefficient_of_variation': cv,
        'invalid_pixel_fraction': n_invalid / n_total,
    }


def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")
    
    # Create synthetic SAR-like data
    np.random.seed(42)
    # Exponential distribution mimics SAR intensity
    test_image = np.random.exponential(0.1, (512, 512)).astype(np.float32)
    
    # Test preprocessing
    normalized, params = preprocess_sar_complete(test_image)
    
    assert normalized.min() >= 0, "Normalized min should be >= 0"
    assert normalized.max() <= 1, "Normalized max should be <= 1"
    print(f"✓ Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Test inverse
    reconstructed = inverse_preprocess(normalized, params)
    
    # Check roundtrip (for non-clipped values)
    image_db = 10 * np.log10(np.maximum(test_image, 1e-10))
    non_clipped = (image_db >= params['vmin']) & (image_db <= params['vmax'])
    
    error = np.abs(test_image[non_clipped] - reconstructed[non_clipped])
    relative_error = error / test_image[non_clipped]
    
    print(f"✓ Roundtrip mean relative error: {relative_error.mean()*100:.4f}%")
    
    # Test patch extraction
    patches, positions = extract_patches(normalized, patch_size=256, stride=128)
    print(f"✓ Extracted {len(patches)} patches")
    
    print("All preprocessing tests passed!")


if __name__ == "__main__":
    test_preprocessing()
