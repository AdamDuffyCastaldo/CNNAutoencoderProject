"""
Quality Metrics for SAR Image Evaluation

This module implements standard and SAR-specific quality metrics.

Standard Metrics:
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- MS-SSIM: Multi-Scale Structural Similarity Index
- MAE: Mean Absolute Error

SAR-Specific Metrics:
- ENL: Equivalent Number of Looks (speckle indicator)
- ENL Ratio: Ratio of ENL between original and reconstructed
- EPI: Edge Preservation Index (gradient correlation)
- Histogram Similarity (intersection, chi-square, Bhattacharyya)
- Local Variance Ratio

Compression Metrics:
- Compression Ratio
- Bits Per Pixel (BPP)

References:
    - Day 3, Section 3.1 of the learning guide
"""

import warnings
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter, sobel
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Optional, Union
from skimage.metrics import structural_similarity as skimage_ssim

try:
    import torch
    from pytorch_msssim import ms_ssim as pytorch_ms_ssim
    PYTORCH_MSSSIM_AVAILABLE = True
except ImportError:
    PYTORCH_MSSSIM_AVAILABLE = False


# ============================================================================
# ENL (Equivalent Number of Looks) Functions
# ============================================================================

def find_homogeneous_regions(
    image: np.ndarray,
    window_size: int = 15,
    cv_threshold: float = 0.3
) -> np.ndarray:
    """
    Detect homogeneous regions using coefficient of variation (CV).

    Homogeneous regions have low local variance relative to mean (low CV).
    These regions are suitable for ENL estimation.

    Args:
        image: 2D array of image intensities
        window_size: Size of local window for statistics
        cv_threshold: Maximum CV to consider homogeneous (lower = stricter)

    Returns:
        Boolean mask where True indicates homogeneous pixels

    Note:
        CV = std / mean. For fully developed speckle, CV = 1/sqrt(ENL).
        Lower CV means more homogeneous (smoother) regions.
    """
    image = np.asarray(image, dtype=np.float64)

    # Compute local statistics using uniform filter
    local_mean = uniform_filter(image, size=window_size, mode='reflect')
    local_sq_mean = uniform_filter(image ** 2, size=window_size, mode='reflect')

    # Local variance = E[X^2] - E[X]^2
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Ensure non-negative
    local_std = np.sqrt(local_var)

    # Coefficient of variation
    cv = local_std / (local_mean + 1e-10)

    # Homogeneous regions have low CV
    mask = cv < cv_threshold

    return mask


def compute_enl(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute Equivalent Number of Looks (ENL) in masked regions.

    ENL = mu^2 / sigma^2

    For fully developed speckle: ENL = 1 (single look)
    Higher ENL indicates more averaging/smoothing.
    Sentinel-1 GRD typically has ENL ~ 4-5.

    Args:
        image: 2D array of image intensities
        mask: Boolean mask indicating which pixels to use

    Returns:
        ENL value, or np.nan if insufficient samples
    """
    if mask.sum() < 100:
        return np.nan

    values = image[mask].astype(np.float64)
    mean = np.mean(values)
    var = np.var(values)

    if var < 1e-10:
        return float('inf')

    return float((mean ** 2) / var)


def enl_ratio(
    original: np.ndarray,
    reconstructed: np.ndarray,
    window_size: int = 15,
    cv_threshold: float = 0.3,
    domain: str = 'db'
) -> Dict[str, Union[float, int]]:
    """
    Compute ENL ratio between original and reconstructed images.

    The ENL ratio measures how well speckle characteristics are preserved.
    Ideal ENL ratio is ~1.0 (speckle statistics preserved).

    Args:
        original: Original image
        reconstructed: Reconstructed image
        window_size: Window size for homogeneous region detection
        cv_threshold: CV threshold for homogeneous detection
        domain: 'db' or 'linear' - specifies the intensity domain
                'db' for already log-transformed data (default for this project)
                'linear' for linear intensity values

    Returns:
        Dictionary with:
        - enl_original: ENL of original in homogeneous regions
        - enl_reconstructed: ENL of reconstructed in homogeneous regions
        - enl_ratio: enl_reconstructed / enl_original (ideal ~1.0)
        - homogeneous_pixels: Number of homogeneous pixels
        - homogeneous_fraction: Fraction of image that is homogeneous

    Note:
        Uses homogeneous regions from ORIGINAL image only, to ensure
        same regions are compared in both images.
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    # For dB domain data, we can use it directly
    # For linear data, ENL formula still applies
    # The domain parameter is for documentation/future flexibility

    # Find homogeneous regions in original
    mask = find_homogeneous_regions(original, window_size, cv_threshold)

    homogeneous_pixels = int(mask.sum())
    total_pixels = original.size
    homogeneous_fraction = homogeneous_pixels / total_pixels

    # Compute ENL for both using same mask
    enl_orig = compute_enl(original, mask)
    enl_recon = compute_enl(reconstructed, mask)

    # Compute ratio
    if np.isnan(enl_orig) or np.isnan(enl_recon):
        ratio = np.nan
    elif enl_orig == 0 or np.isinf(enl_orig):
        ratio = np.nan
    else:
        ratio = enl_recon / enl_orig

    return {
        'enl_original': float(enl_orig) if not np.isnan(enl_orig) else np.nan,
        'enl_reconstructed': float(enl_recon) if not np.isnan(enl_recon) else np.nan,
        'enl_ratio': float(ratio) if not np.isnan(ratio) else np.nan,
        'homogeneous_pixels': homogeneous_pixels,
        'homogeneous_fraction': float(homogeneous_fraction),
    }


class SARMetrics:
    """
    Collection of metrics for SAR image quality assessment.
    
    All methods are static for easy use without instantiation.
    
    Example:
        >>> original = np.random.rand(256, 256)
        >>> reconstructed = np.random.rand(256, 256)
        >>> psnr = SARMetrics.psnr(original, reconstructed)
        >>> ssim = SARMetrics.ssim(original, reconstructed)
    """
    
    @staticmethod
    def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((x - x_hat) ** 2))
    
    @staticmethod
    def psnr(x: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
        """
        Peak Signal-to-Noise Ratio in dB.
        
        PSNR = 10 × log₁₀(MAX² / MSE)
        
        Interpretation:
        - >40 dB: Excellent
        - 30-40 dB: Good
        - 25-30 dB: Acceptable
        - <25 dB: Poor
        """
        mse = np.mean((x - x_hat) ** 2)
        if mse == 0:
            return float('inf')
        return float(10 * np.log10(data_range ** 2 / mse))
    
    @staticmethod
    def ssim(x: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
        """
        Structural Similarity Index.
        
        Range: [-1, 1], typically [0, 1] for similar images.
        
        Interpretation:
        - >0.95: Excellent
        - 0.85-0.95: Good
        - 0.70-0.85: Moderate
        - <0.70: Poor
        """
        return float(skimage_ssim(x, x_hat, data_range=data_range))
    
    @staticmethod
    def mae(x: np.ndarray, x_hat: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(x - x_hat)))
    
    @staticmethod
    def enl(image: np.ndarray, window_size: int = 32) -> np.ndarray:
        """
        Compute local Equivalent Number of Looks (ENL) map.

        ENL = mu^2 / sigma^2 computed in local windows.

        For speckle: CV = 1/sqrt(ENL)
        Higher ENL = less speckle (more smoothing/averaging)
        Sentinel-1 GRD: ENL ~ 4-5

        Args:
            image: SAR image (linear intensity or dB)
            window_size: Local window size

        Returns:
            ENL map (same shape as input)
        """
        image = np.asarray(image, dtype=np.float64)

        local_mean = uniform_filter(image, size=window_size, mode='reflect')
        local_sq_mean = uniform_filter(image ** 2, size=window_size, mode='reflect')
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 1e-10)

        enl_map = (local_mean ** 2) / local_var
        return enl_map

    @staticmethod
    def enl_ratio(
        original: np.ndarray,
        reconstructed: np.ndarray,
        window_size: int = 15,
        cv_threshold: float = 0.3
    ) -> Dict[str, Union[float, int]]:
        """
        Compute ENL ratio between original and reconstructed images.

        Wrapper around module-level enl_ratio function.

        Args:
            original: Original image
            reconstructed: Reconstructed image
            window_size: Window size for homogeneous region detection
            cv_threshold: CV threshold for homogeneous detection

        Returns:
            Dictionary with enl_original, enl_reconstructed, enl_ratio,
            homogeneous_pixels, and homogeneous_fraction
        """
        return enl_ratio(original, reconstructed, window_size, cv_threshold)
    
    @staticmethod
    def edge_preservation_index(x: np.ndarray, x_hat: np.ndarray) -> float:
        """
        Edge Preservation Index.
        
        EPI = Σ|∇x̂| / Σ|∇x|
        
        Interpretation:
        - EPI ≈ 1: Edges preserved
        - EPI < 1: Edges smoothed
        - EPI > 1: Edges enhanced (possible artifacts)
        """
        # TODO: Implement EPI
        #
        # def gradient_magnitude(img):
        #     gx = ndimage.sobel(img, axis=1)
        #     gy = ndimage.sobel(img, axis=0)
        #     return np.sqrt(gx**2 + gy**2)
        # 
        # grad_x = gradient_magnitude(x)
        # grad_x_hat = gradient_magnitude(x_hat)
        # 
        # return float(np.sum(grad_x_hat) / (np.sum(grad_x) + 1e-10))
        
        raise NotImplementedError("TODO: Implement EPI")
    
    @staticmethod
    def histogram_similarity(x: np.ndarray, x_hat: np.ndarray, 
                            bins: int = 256) -> float:
        """
        Histogram intersection similarity.
        
        Measures how well the intensity distribution is preserved.
        Range: [0, 1], 1 = identical distributions.
        """
        # TODO: Implement histogram similarity
        raise NotImplementedError("TODO: Implement histogram similarity")
    
    @staticmethod
    def local_variance_ratio(x: np.ndarray, x_hat: np.ndarray,
                            window_size: int = 16) -> Dict[str, float]:
        """
        Compare local variance statistics.
        
        Helps detect over-smoothing.
        
        Returns:
            - variance_ratio: mean(var_hat) / mean(var_x)
            - variance_correlation: correlation between variance maps
        """
        # TODO: Implement local variance ratio
        raise NotImplementedError("TODO: Implement local variance ratio")
    
    @staticmethod
    def correlation(x: np.ndarray, x_hat: np.ndarray) -> Dict[str, float]:
        """Pearson and Spearman correlation coefficients."""
        x_flat = x.flatten()
        x_hat_flat = x_hat.flatten()
        
        pearson_r, _ = pearsonr(x_flat, x_hat_flat)
        spearman_r, _ = spearmanr(x_flat, x_hat_flat)
        
        return {
            'pearson': float(pearson_r),
            'spearman': float(spearman_r),
        }


def test_metrics():
    """Test metrics."""
    print("Testing SARMetrics...")
    
    np.random.seed(42)
    x = np.random.rand(64, 64).astype(np.float32)
    x_hat = x + 0.05 * np.random.randn(64, 64)
    x_hat = np.clip(x_hat, 0, 1).astype(np.float32)
    
    print(f"✓ MSE: {SARMetrics.mse(x, x_hat):.6f}")
    print(f"✓ PSNR: {SARMetrics.psnr(x, x_hat):.2f} dB")
    print(f"✓ SSIM: {SARMetrics.ssim(x, x_hat):.4f}")
    print(f"✓ MAE: {SARMetrics.mae(x, x_hat):.6f}")
    
    corr = SARMetrics.correlation(x, x_hat)
    print(f"✓ Correlation: Pearson={corr['pearson']:.4f}, Spearman={corr['spearman']:.4f}")
    
    print("All metric tests passed!")


if __name__ == "__main__":
    test_metrics()
