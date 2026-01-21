"""
Structural Similarity Index (SSIM) Loss

SSIM measures image quality based on:
- Luminance (mean intensity)
- Contrast (standard deviation)
- Structure (normalized cross-correlation)

Uses pytorch-msssim library for robust, GPU-optimized SSIM computation.

References:
    - Day 1, Session 3 of the learning guide
    - Wang et al. "Image Quality Assessment: From Error Visibility
      to Structural Similarity" (2004)
"""

import torch
import torch.nn as nn
from pytorch_msssim import SSIM


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.

    Returns 1 - SSIM as a loss (0 when images are identical).

    Uses pytorch-msssim library internally for robust SSIM computation.

    Args:
        window_size: Size of Gaussian window (default 11)
        data_range: Range of input data (1.0 for [0,1] normalized)
        channel: Number of input channels (default 1 for SAR)

    Example:
        >>> ssim_loss = SSIMLoss()
        >>> x = torch.rand(4, 1, 64, 64)
        >>> x_hat = torch.rand(4, 1, 64, 64)
        >>> loss = ssim_loss(x_hat, x)  # Returns 1 - SSIM
    """

    def __init__(
        self,
        window_size: int = 11,
        data_range: float = 1.0,
        channel: int = 1
    ):
        super().__init__()

        self.window_size = window_size
        self.data_range = data_range
        self.channel = channel

        # Use pytorch-msssim SSIM module
        self.ssim_module = SSIM(
            data_range=data_range,
            size_average=True,
            channel=channel,
            win_size=window_size,
            nonnegative_ssim=True
        )

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            x_hat: Reconstructed image (B, C, H, W)
            x: Original image (B, C, H, W)

        Returns:
            1 - SSIM (scalar tensor, 0 when identical)
        """
        ssim_value = self.ssim_module(x_hat, x)
        return 1 - ssim_value


def test_ssim():
    """Test SSIM loss."""
    print("Testing SSIMLoss...")

    ssim_loss = SSIMLoss()

    # Perfect match should give loss near 0
    x = torch.rand(2, 1, 64, 64)
    loss_perfect = ssim_loss(x, x)
    print(f"Perfect match loss: {loss_perfect.item():.6f} (should be ~0)")

    # Different images should give higher loss
    x_different = torch.rand(2, 1, 64, 64)
    loss_different = ssim_loss(x_different, x)
    print(f"Different images loss: {loss_different.item():.4f}")

    assert loss_perfect < loss_different, "Perfect match should have lower loss"

    print("SSIM tests passed!")


if __name__ == "__main__":
    test_ssim()
