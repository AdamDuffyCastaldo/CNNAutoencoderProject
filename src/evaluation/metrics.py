"""
Quality Metrics for SAR Image Evaluation

This module implements standard and SAR-specific quality metrics.

Standard Metrics:
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- MAE: Mean Absolute Error

SAR-Specific Metrics:
- ENL: Equivalent Number of Looks (speckle indicator)
- EPI: Edge Preservation Index
- Histogram Similarity
- Local Variance Ratio

References:
    - Day 3, Section 3.1 of the learning guide
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from typing import Dict
from skimage.metrics import structural_similarity as skimage_ssim


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
        Equivalent Number of Looks.
        
        ENL = μ² / σ² computed in local windows.
        
        For speckle: CV = 1/√ENL
        Higher ENL = less speckle (more smoothing)
        Sentinel-1 GRD: ENL ≈ 4-5
        
        Args:
            image: SAR image (linear intensity)
            window_size: Local window size
        
        Returns:
            ENL map
        """
        # TODO: Implement ENL computation
        #
        # from scipy.ndimage import uniform_filter
        # local_mean = uniform_filter(image, size=window_size)
        # local_sq_mean = uniform_filter(image ** 2, size=window_size)
        # local_var = local_sq_mean - local_mean ** 2
        # local_var = np.maximum(local_var, 1e-10)
        # enl = local_mean ** 2 / local_var
        # return enl
        
        raise NotImplementedError("TODO: Implement ENL")
    
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
