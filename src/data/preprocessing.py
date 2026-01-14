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


def to_db(
    intensity: np.ndarray,
    floor_db: float = -50
) -> np.ndarray:
    """
    Convert intensity to decibels.
    
    Formula: dB = 10 × log₁₀(intensity)
    
    Args:
        intensity: Linear intensity values
        floor_db: Minimum dB value (prevents -inf)
    
    Returns:
        Image in dB scale
    """
    # TODO: Implement dB conversion
    #
    # db = 10 * np.log10(intensity)
    # db = np.maximum(db, floor_db)
    # return db
    
    raise NotImplementedError("TODO: Implement dB conversion")


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


def preprocess_sar_image(
    image: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip_percentiles: Tuple[float, float] = (1, 99)
) -> Tuple[np.ndarray, Dict]:
    """
    Complete preprocessing pipeline for SAR image.
    
    Steps:
    1. Handle invalid values
    2. Convert to dB
    3. Determine clip bounds (if not provided)
    4. Clip outliers
    5. Normalize to [0, 1]
    
    Args:
        image: Raw SAR intensity image
        vmin: Lower clip bound in dB (computed if None)
        vmax: Upper clip bound in dB (computed if None)
        clip_percentiles: Percentiles for automatic clip bounds
    
    Returns:
        normalized: Image in [0, 1] range
        params: Dict with preprocessing parameters for inverse transform
    
    Example:
        >>> normalized, params = preprocess_sar_image(raw_image)
        >>> # Save params for later inverse transform
        >>> np.save('preprocess_params.npy', params)
    """
    # TODO: Implement complete preprocessing pipeline
    #
    # Step 1: Handle invalid values
    # Step 2: Convert to dB
    # Step 3: Determine clip bounds
    # Step 4: Clip
    # Step 5: Normalize to [0, 1]
    # Return normalized image and params dict
    
    raise NotImplementedError("TODO: Implement preprocessing pipeline")


def inverse_preprocess(
    normalized: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    Inverse preprocessing: convert [0,1] back to linear intensity.
    
    Args:
        normalized: Normalized image in [0, 1]
        params: Preprocessing parameters from preprocess_sar_image
    
    Returns:
        Linear intensity image
    
    Example:
        >>> reconstructed = inverse_preprocess(model_output, params)
    """
    # TODO: Implement inverse preprocessing
    #
    # vmin = params['vmin']
    # vmax = params['vmax']
    # 
    # # Denormalize to dB
    # image_db = normalized * (vmax - vmin) + vmin
    # 
    # # Convert to linear
    # image_linear = 10 ** (image_db / 10)
    # 
    # return image_linear
    
    raise NotImplementedError("TODO: Implement inverse preprocessing")


def extract_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    min_valid_fraction: float = 0.9
) -> Tuple[np.ndarray, list]:
    """
    Extract overlapping patches from image.
    
    Args:
        image: Input image (already preprocessed to [0,1])
        patch_size: Size of square patches
        stride: Step between patches (< patch_size for overlap)
        min_valid_fraction: Minimum fraction of valid pixels
    
    Returns:
        patches: Array of shape (N, patch_size, patch_size)
        positions: List of (row, col) positions for each patch
    
    Notes:
        - Patches with too many boundary values are excluded
        - stride = patch_size: no overlap (max unique patches)
        - stride = patch_size // 2: 50% overlap (4× more patches)
    """
    # TODO: Implement patch extraction
    #
    # patches = []
    # positions = []
    # h, w = image.shape
    # 
    # for i in range(0, h - patch_size + 1, stride):
    #     for j in range(0, w - patch_size + 1, stride):
    #         patch = image[i:i+patch_size, j:j+patch_size]
    #         
    #         # Check validity
    #         valid_frac = np.mean((patch > 0.01) & (patch < 0.99))
    #         if valid_frac >= min_valid_fraction:
    #             patches.append(patch)
    #             positions.append((i, j))
    # 
    # return np.array(patches), positions
    
    raise NotImplementedError("TODO: Implement patch extraction")


def compute_statistics(image: np.ndarray) -> Dict:
    """
    Compute comprehensive statistics for SAR image.
    
    Useful for data exploration and preprocessing decisions.
    
    Args:
        image: SAR image (linear intensity)
    
    Returns:
        Dict containing:
        - Linear domain: min, max, mean, std, median
        - dB domain: min, max, mean, std, median
        - Percentiles: p1, p5, p50, p95, p99
        - Estimated looks (from CV)
        - Dynamic range in dB
    """
    # TODO: Implement statistics computation
    raise NotImplementedError("TODO: Implement statistics computation")


def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")
    
    # Create synthetic SAR-like data
    np.random.seed(42)
    # Exponential distribution mimics SAR intensity
    test_image = np.random.exponential(0.1, (512, 512)).astype(np.float32)
    
    # Test preprocessing
    normalized, params = preprocess_sar_image(test_image)
    
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
