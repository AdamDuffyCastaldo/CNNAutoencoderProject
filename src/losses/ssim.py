"""
Structural Similarity Index (SSIM) Loss

SSIM measures image quality based on:
- Luminance (mean intensity)
- Contrast (standard deviation)
- Structure (normalized cross-correlation)

References:
    - Day 1, Session 3 of the learning guide
    - Wang et al. "Image Quality Assessment: From Error Visibility 
      to Structural Similarity" (2004)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    
    Returns 1 - SSIM as a loss (0 when images are identical).
    
    SSIM formula:
    SSIM = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    
    Where:
    - μx, μy: Local means
    - σx², σy²: Local variances
    - σxy: Local covariance
    - C1, C2: Stability constants
    
    Args:
        window_size: Size of Gaussian window (default 11)
        sigma: Gaussian sigma (default 1.5)
        data_range: Range of input data (1.0 for [0,1] normalized)
        channel: Number of input channels
    
    Example:
        >>> ssim_loss = SSIMLoss()
        >>> x = torch.rand(4, 1, 64, 64)
        >>> x_hat = torch.rand(4, 1, 64, 64)
        >>> loss = ssim_loss(x_hat, x)  # Returns 1 - SSIM
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channel: int = 1
    ):
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channel = channel
        
        # SSIM constants
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        
        # TODO: Create Gaussian window
        #
        # self.register_buffer('window', self._create_window())
        
        raise NotImplementedError("TODO: Implement SSIMLoss initialization")
    
    def _create_window(self) -> torch.Tensor:
        """
        Create Gaussian window for SSIM computation.
        
        Returns:
            Tensor of shape (channel, 1, window_size, window_size)
        """
        # TODO: Create 2D Gaussian window
        #
        # 1. Create 1D Gaussian
        # coords = torch.arange(self.window_size, dtype=torch.float32)
        # coords -= self.window_size // 2
        # g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        # g /= g.sum()
        #
        # 2. Create 2D Gaussian (outer product)
        # window = g.unsqueeze(1) @ g.unsqueeze(0)
        #
        # 3. Expand for conv2d
        # window = window.unsqueeze(0).unsqueeze(0)
        # window = window.expand(self.channel, 1, 
        #                        self.window_size, self.window_size)
        # return window.contiguous()
        
        raise NotImplementedError("TODO: Create Gaussian window")
    
    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).
        
        Args:
            x_hat: Reconstructed image (B, C, H, W)
            x: Original image (B, C, H, W)
        
        Returns:
            1 - SSIM (scalar tensor, 0 when identical)
        """
        # TODO: Implement SSIM computation
        #
        # 1. Compute local means using conv2d with Gaussian window
        # mu_x = F.conv2d(x, self.window, padding=self.window_size//2, groups=self.channel)
        # mu_y = F.conv2d(x_hat, self.window, ...)
        #
        # 2. Compute local variances: var = E[X²] - E[X]²
        # sigma_x_sq = F.conv2d(x**2, ...) - mu_x**2
        # sigma_y_sq = F.conv2d(x_hat**2, ...) - mu_y**2
        # sigma_xy = F.conv2d(x * x_hat, ...) - mu_x * mu_y
        #
        # 3. Compute SSIM
        # numerator = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        # denominator = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        # ssim_map = numerator / (denominator + 1e-8)
        #
        # 4. Return 1 - mean SSIM
        # return 1 - ssim_map.mean()
        
        raise NotImplementedError("TODO: Implement SSIM forward pass")


def test_ssim():
    """Test SSIM loss."""
    print("Testing SSIMLoss...")
    
    ssim_loss = SSIMLoss()
    
    # Perfect match should give loss ≈ 0
    x = torch.rand(2, 1, 64, 64)
    loss_perfect = ssim_loss(x, x)
    print(f"✓ Perfect match loss: {loss_perfect.item():.6f} (should be ~0)")
    
    # Different images should give higher loss
    x_different = torch.rand(2, 1, 64, 64)
    loss_different = ssim_loss(x_different, x)
    print(f"✓ Different images loss: {loss_different.item():.4f}")
    
    assert loss_perfect < loss_different, "Perfect match should have lower loss"
    
    print("SSIM tests passed!")


if __name__ == "__main__":
    test_ssim()
