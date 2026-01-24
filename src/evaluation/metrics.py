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


# ============================================================================
# Edge Preservation Index (EPI)
# ============================================================================

def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude using Sobel operators.

    Args:
        image: 2D array of image intensities

    Returns:
        Gradient magnitude array (same shape as input)
    """
    gx = sobel(image, axis=1)  # Horizontal gradient
    gy = sobel(image, axis=0)  # Vertical gradient
    return np.sqrt(gx ** 2 + gy ** 2)


def edge_preservation_index(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Edge Preservation Index as gradient correlation.

    EPI measures how well edges are preserved in the reconstruction.
    Uses correlation of gradient magnitudes, which is more robust than
    the simple ratio formula.

    Args:
        original: Original image
        reconstructed: Reconstructed image

    Returns:
        EPI value in [0, 1], where 1 = perfect edge preservation

    Note:
        This correlation-based formula differs from the ratio-based formula
        (sum(grad_recon) / sum(grad_orig)). Correlation is more robust
        to global intensity changes and provides bounded output.
    """
    grad_orig = compute_gradient_magnitude(original.astype(np.float64))
    grad_recon = compute_gradient_magnitude(reconstructed.astype(np.float64))

    go = grad_orig.flatten()
    gr = grad_recon.flatten()

    numerator = np.sum(go * gr)
    denominator = np.sqrt(np.sum(go ** 2) * np.sum(gr ** 2))

    if denominator < 1e-10:
        return 0.0

    epi = numerator / denominator
    return float(np.clip(epi, 0.0, 1.0))


# ============================================================================
# Histogram Similarity Metrics
# ============================================================================

def histogram_similarity(
    original: np.ndarray,
    reconstructed: np.ndarray,
    bins: int = 256,
    value_range: Tuple[float, float] = (0.0, 1.0)
) -> Dict[str, float]:
    """
    Compute multiple histogram similarity metrics.

    Measures how well the intensity distribution is preserved.

    Args:
        original: Original image
        reconstructed: Reconstructed image
        bins: Number of histogram bins
        value_range: Value range for histograms (min, max)

    Returns:
        Dictionary with:
        - intersection: Sum of element-wise minimum (1 = identical)
        - chi_square: Chi-square distance (0 = identical)
        - bhattacharyya: Bhattacharyya coefficient (1 = identical)
    """
    # Compute normalized histograms
    hist_orig, _ = np.histogram(original.flatten(), bins=bins,
                                 range=value_range, density=False)
    hist_recon, _ = np.histogram(reconstructed.flatten(), bins=bins,
                                  range=value_range, density=False)

    # Normalize to sum to 1 (probability distributions)
    hist_orig = hist_orig.astype(np.float64)
    hist_recon = hist_recon.astype(np.float64)

    hist_orig = hist_orig / (hist_orig.sum() + 1e-10)
    hist_recon = hist_recon / (hist_recon.sum() + 1e-10)

    # Intersection: sum of element-wise minimum
    # Range: [0, 1], 1 = identical
    intersection = float(np.sum(np.minimum(hist_orig, hist_recon)))

    # Chi-square: sum((h1-h2)^2 / (h1+h2))
    # Range: [0, inf), 0 = identical
    denominator = hist_orig + hist_recon
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_square_terms = np.where(
            denominator > 1e-10,
            ((hist_orig - hist_recon) ** 2) / denominator,
            0.0
        )
    chi_square = float(np.sum(chi_square_terms))

    # Bhattacharyya coefficient: sum(sqrt(h1 * h2))
    # Range: [0, 1], 1 = identical
    bhattacharyya = float(np.sum(np.sqrt(hist_orig * hist_recon)))

    return {
        'intersection': intersection,
        'chi_square': chi_square,
        'bhattacharyya': bhattacharyya,
    }


# ============================================================================
# MS-SSIM (Multi-Scale Structural Similarity)
# ============================================================================

def compute_ms_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 1.0
) -> float:
    """
    Compute Multi-Scale Structural Similarity Index.

    MS-SSIM evaluates image quality at multiple scales, providing
    better correlation with human perception than single-scale SSIM.

    Args:
        original: Original image (2D numpy array)
        reconstructed: Reconstructed image (2D numpy array)
        data_range: Data range of the input images

    Returns:
        MS-SSIM value in [0, 1], or NaN if computation fails

    Note:
        - Requires pytorch-msssim package
        - Images must be at least 160x160 for default 5-scale MS-SSIM
        - Smaller images will trigger a warning and return NaN
    """
    if not PYTORCH_MSSSIM_AVAILABLE:
        warnings.warn("pytorch_msssim not available. Install with: pip install pytorch-msssim")
        return np.nan

    # Check minimum size requirement
    min_size = 160  # For 5 scales: 160 -> 80 -> 40 -> 20 -> 10
    if original.shape[0] < min_size or original.shape[1] < min_size:
        warnings.warn(f"Image size {original.shape} too small for MS-SSIM (min {min_size}x{min_size})")
        return np.nan

    try:
        # Convert to torch tensors with shape (1, 1, H, W)
        orig_t = torch.from_numpy(original.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        recon_t = torch.from_numpy(reconstructed.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        # Compute MS-SSIM
        ms_ssim_val = pytorch_ms_ssim(orig_t, recon_t, data_range=data_range, size_average=True)

        return float(ms_ssim_val.item())
    except Exception as e:
        warnings.warn(f"MS-SSIM computation failed: {e}")
        return np.nan


# ============================================================================
# Local Variance Metrics
# ============================================================================

def local_variance_ratio(
    original: np.ndarray,
    reconstructed: np.ndarray,
    window_size: int = 16
) -> Dict[str, float]:
    """
    Compare local variance statistics between images.

    Local variance ratio helps detect over-smoothing in reconstruction.
    A ratio < 1 indicates smoothing (reduced variance), > 1 indicates
    added noise or texture.

    Args:
        original: Original image
        reconstructed: Reconstructed image
        window_size: Window size for local variance computation

    Returns:
        Dictionary with:
        - variance_ratio: mean(var_recon) / mean(var_orig)
        - variance_correlation: Pearson correlation of variance maps
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    # Compute local variance maps
    def local_variance(image: np.ndarray) -> np.ndarray:
        local_mean = uniform_filter(image, size=window_size, mode='reflect')
        local_sq_mean = uniform_filter(image ** 2, size=window_size, mode='reflect')
        local_var = local_sq_mean - local_mean ** 2
        return np.maximum(local_var, 0)

    var_orig = local_variance(original)
    var_recon = local_variance(reconstructed)

    # Variance ratio
    mean_var_orig = np.mean(var_orig)
    mean_var_recon = np.mean(var_recon)

    if mean_var_orig < 1e-10:
        variance_ratio = np.nan
    else:
        variance_ratio = mean_var_recon / mean_var_orig

    # Variance correlation
    vo_flat = var_orig.flatten()
    vr_flat = var_recon.flatten()

    # Use Pearson correlation
    try:
        variance_corr, _ = pearsonr(vo_flat, vr_flat)
    except Exception:
        variance_corr = np.nan

    return {
        'variance_ratio': float(variance_ratio) if not np.isnan(variance_ratio) else np.nan,
        'variance_correlation': float(variance_corr) if not np.isnan(variance_corr) else np.nan,
    }


# ============================================================================
# Compression Metrics
# ============================================================================

def compute_compression_ratio(
    original_size: Union[int, float],
    compressed_size: Union[int, float]
) -> float:
    """
    Compute compression ratio.

    Compression ratio = original_size / compressed_size

    Args:
        original_size: Size of original data (bytes, bits, or any unit)
        compressed_size: Size of compressed data (same unit as original)

    Returns:
        Compression ratio (e.g., 16.0 means 16:1 compression)

    Example:
        # For autoencoder: 256x256x1 float32 -> 16x16x16 float32
        original = 256 * 256 * 1 * 4  # bytes
        compressed = 16 * 16 * 16 * 4  # bytes
        ratio = compute_compression_ratio(original, compressed)  # 16.0
    """
    if compressed_size <= 0:
        return float('inf')
    return float(original_size / compressed_size)


def compute_bpp(
    original_shape: Tuple[int, ...],
    compressed_bits: Union[int, float]
) -> float:
    """
    Compute Bits Per Pixel (BPP).

    BPP = total_compressed_bits / number_of_pixels

    Args:
        original_shape: Shape of original image (H, W) or (H, W, C)
        compressed_bits: Total bits in compressed representation

    Returns:
        Bits per pixel

    Example:
        # For autoencoder: 256x256 -> 16x16x16 float32 latent
        shape = (256, 256)
        latent_bits = 16 * 16 * 16 * 32  # float32 = 32 bits
        bpp = compute_bpp(shape, latent_bits)  # 2.0 bpp
    """
    num_pixels = original_shape[0] * original_shape[1]
    if num_pixels <= 0:
        return float('inf')
    return float(compressed_bits / num_pixels)


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 1.0
) -> Dict[str, Union[float, Dict]]:
    """
    Compute all available quality metrics in one call.

    This is a convenience function that computes all metrics and returns
    them in a single dictionary.

    Args:
        original: Original image (2D numpy array)
        reconstructed: Reconstructed image (2D numpy array)
        data_range: Data range of images (default 1.0 for normalized)

    Returns:
        Dictionary with all metrics:
        - mse: Mean Squared Error
        - psnr: Peak Signal-to-Noise Ratio (dB)
        - ssim: Structural Similarity Index
        - ms_ssim: Multi-Scale SSIM (or NaN if unavailable)
        - mae: Mean Absolute Error
        - epi: Edge Preservation Index
        - enl_ratio: Dict with ENL statistics
        - histogram: Dict with histogram similarity metrics
        - local_variance: Dict with variance ratio and correlation
        - correlation: Dict with Pearson and Spearman coefficients
    """
    metrics = {}

    # Basic metrics
    metrics['mse'] = SARMetrics.mse(original, reconstructed)
    metrics['psnr'] = SARMetrics.psnr(original, reconstructed, data_range)
    metrics['ssim'] = SARMetrics.ssim(original, reconstructed, data_range)
    metrics['mae'] = SARMetrics.mae(original, reconstructed)

    # MS-SSIM (with error handling)
    try:
        metrics['ms_ssim'] = compute_ms_ssim(original, reconstructed, data_range)
    except Exception:
        metrics['ms_ssim'] = np.nan

    # SAR-specific metrics
    metrics['epi'] = edge_preservation_index(original, reconstructed)
    metrics['enl_ratio'] = enl_ratio(original, reconstructed)
    metrics['histogram'] = histogram_similarity(original, reconstructed)
    metrics['local_variance'] = local_variance_ratio(original, reconstructed)
    metrics['correlation'] = SARMetrics.correlation(original, reconstructed)

    return metrics


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Classes
    'SARMetrics',

    # ENL functions
    'find_homogeneous_regions',
    'compute_enl',
    'enl_ratio',

    # Edge preservation
    'edge_preservation_index',
    'compute_gradient_magnitude',

    # Histogram similarity
    'histogram_similarity',

    # MS-SSIM
    'compute_ms_ssim',
    'PYTORCH_MSSSIM_AVAILABLE',

    # Local variance
    'local_variance_ratio',

    # Compression metrics
    'compute_compression_ratio',
    'compute_bpp',

    # Convenience functions
    'compute_all_metrics',
]


# ============================================================================
# SARMetrics Class
# ============================================================================

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
        Edge Preservation Index using gradient correlation.

        EPI measures how well edges are preserved in reconstruction.
        Uses correlation of gradient magnitudes (more robust than ratio).

        Args:
            x: Original image
            x_hat: Reconstructed image

        Returns:
            EPI value in [0, 1], where 1 = perfect edge preservation

        Interpretation:
        - EPI > 0.95: Excellent edge preservation
        - EPI 0.85-0.95: Good
        - EPI 0.70-0.85: Moderate (some edge degradation)
        - EPI < 0.70: Poor (significant edge loss)
        """
        return edge_preservation_index(x, x_hat)

    @staticmethod
    def histogram_similarity(
        x: np.ndarray,
        x_hat: np.ndarray,
        bins: int = 256
    ) -> Dict[str, float]:
        """
        Compute histogram similarity metrics.

        Measures how well the intensity distribution is preserved.

        Args:
            x: Original image
            x_hat: Reconstructed image
            bins: Number of histogram bins

        Returns:
            Dictionary with intersection, chi_square, and bhattacharyya

        Interpretation:
        - intersection: [0, 1], 1 = identical distributions
        - chi_square: [0, inf), 0 = identical distributions
        - bhattacharyya: [0, 1], 1 = identical distributions
        """
        return histogram_similarity(x, x_hat, bins=bins)

    @staticmethod
    def local_variance_ratio(
        x: np.ndarray,
        x_hat: np.ndarray,
        window_size: int = 16
    ) -> Dict[str, float]:
        """
        Compare local variance statistics between images.

        Helps detect over-smoothing in reconstruction.

        Args:
            x: Original image
            x_hat: Reconstructed image
            window_size: Window size for local variance computation

        Returns:
            Dictionary with variance_ratio and variance_correlation

        Interpretation:
        - variance_ratio < 1: Over-smoothing (reduced texture)
        - variance_ratio ~ 1: Variance preserved
        - variance_ratio > 1: Added noise/texture
        - variance_correlation ~ 1: Similar variance patterns
        """
        return local_variance_ratio(x, x_hat, window_size=window_size)

    @staticmethod
    def ms_ssim(x: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
        """
        Multi-Scale Structural Similarity Index.

        MS-SSIM evaluates image quality at multiple scales.

        Args:
            x: Original image
            x_hat: Reconstructed image
            data_range: Data range of images

        Returns:
            MS-SSIM value in [0, 1], or NaN if computation fails

        Note:
            Images must be at least 160x160 for default 5-scale MS-SSIM.
        """
        return compute_ms_ssim(x, x_hat, data_range=data_range)
    
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
    """Test all metrics comprehensively."""
    print("=" * 60)
    print("Testing SAR Quality Metrics")
    print("=" * 60)

    # Use larger images for MS-SSIM (requires >= 160x160)
    np.random.seed(42)
    x = np.random.rand(256, 256).astype(np.float32)
    x_hat = x + 0.03 * np.random.randn(256, 256)
    x_hat = np.clip(x_hat, 0, 1).astype(np.float32)

    print("\n--- Basic Metrics ---")
    print(f"MSE: {SARMetrics.mse(x, x_hat):.6f}")
    print(f"PSNR: {SARMetrics.psnr(x, x_hat):.2f} dB")
    print(f"SSIM: {SARMetrics.ssim(x, x_hat):.4f}")
    print(f"MAE: {SARMetrics.mae(x, x_hat):.6f}")

    print("\n--- Correlation ---")
    corr = SARMetrics.correlation(x, x_hat)
    print(f"Pearson: {corr['pearson']:.4f}")
    print(f"Spearman: {corr['spearman']:.4f}")

    print("\n--- MS-SSIM ---")
    ms = compute_ms_ssim(x, x_hat)
    print(f"MS-SSIM: {ms:.4f}" if not np.isnan(ms) else "MS-SSIM: N/A (pytorch-msssim not available)")

    print("\n--- Edge Preservation Index ---")
    epi = SARMetrics.edge_preservation_index(x, x_hat)
    print(f"EPI (correlation): {epi:.4f}")

    print("\n--- ENL Ratio ---")
    enl_result = SARMetrics.enl_ratio(x, x_hat)
    print(f"ENL original: {enl_result['enl_original']:.2f}")
    print(f"ENL reconstructed: {enl_result['enl_reconstructed']:.2f}")
    print(f"ENL ratio: {enl_result['enl_ratio']:.4f}")
    print(f"Homogeneous fraction: {enl_result['homogeneous_fraction']:.2%}")

    print("\n--- Histogram Similarity ---")
    hist = SARMetrics.histogram_similarity(x, x_hat)
    print(f"Intersection: {hist['intersection']:.4f}")
    print(f"Chi-square: {hist['chi_square']:.4f}")
    print(f"Bhattacharyya: {hist['bhattacharyya']:.4f}")

    print("\n--- Local Variance Ratio ---")
    lvr = SARMetrics.local_variance_ratio(x, x_hat)
    print(f"Variance ratio: {lvr['variance_ratio']:.4f}")
    print(f"Variance correlation: {lvr['variance_correlation']:.4f}")

    print("\n--- Compression Metrics ---")
    # Simulate 16x compression: 256x256x1 -> 16x16x16
    orig_size = 256 * 256 * 1 * 4  # bytes (float32)
    comp_size = 16 * 16 * 16 * 4   # bytes (float32)
    cr = compute_compression_ratio(orig_size, comp_size)
    print(f"Compression ratio (16x example): {cr:.2f}:1")

    latent_bits = 16 * 16 * 16 * 32  # bits
    bpp = compute_bpp((256, 256), latent_bits)
    print(f"BPP (16x example): {bpp:.2f}")

    print("\n--- compute_all_metrics() ---")
    all_metrics = compute_all_metrics(x, x_hat)
    print(f"Keys: {list(all_metrics.keys())}")
    print(f"PSNR: {all_metrics['psnr']:.2f} dB")
    print(f"EPI: {all_metrics['epi']:.4f}")
    print(f"ENL ratio: {all_metrics['enl_ratio']['enl_ratio']:.4f}")

    print("\n" + "=" * 60)
    print("All metric tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
