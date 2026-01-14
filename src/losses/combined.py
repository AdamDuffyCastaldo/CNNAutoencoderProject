"""
Combined Loss Functions

This module implements loss functions that combine multiple objectives.

The primary loss is: L = λ_MSE × MSE + λ_SSIM × (1 - SSIM)

References:
    - Day 1, Session 3 of the learning guide
    - Day 2, Section 2.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .mse import MSELoss
from .ssim import SSIMLoss


class CombinedLoss(nn.Module):
    """
    Combined MSE + SSIM loss.
    
    Loss = mse_weight × MSE + ssim_weight × (1 - SSIM)
    
    Balances:
    - MSE: Pixel-wise accuracy (tends to blur)
    - SSIM: Structural preservation (encourages sharpness)
    
    Args:
        mse_weight: Weight for MSE term
        ssim_weight: Weight for SSIM term
        window_size: SSIM window size
    
    Returns:
        loss: Combined loss value (scalar tensor)
        metrics: Dict with individual components
    
    Example:
        >>> loss_fn = CombinedLoss(mse_weight=1.0, ssim_weight=0.1)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> x_hat = torch.randn(4, 1, 256, 256)
        >>> loss, metrics = loss_fn(x_hat, x)
        >>> print(f"Loss: {metrics['loss']:.4f}, PSNR: {metrics['psnr']:.2f}")
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.1,
        window_size: int = 11
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        
        # TODO: Initialize loss components
        #
        # self.mse_loss = MSELoss()
        # self.ssim_loss = SSIMLoss(window_size=window_size)
        
        raise NotImplementedError("TODO: Implement CombinedLoss initialization")
    
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.
        
        Args:
            x_hat: Reconstructed image (B, C, H, W)
            x: Original image (B, C, H, W)
        
        Returns:
            loss: Combined loss value
            metrics: Dict with 'loss', 'mse', 'ssim', 'psnr'
        """
        # TODO: Implement combined loss
        #
        # mse = self.mse_loss(x_hat, x)
        # ssim_loss = self.ssim_loss(x_hat, x)  # Returns 1 - SSIM
        # 
        # loss = self.mse_weight * mse + self.ssim_weight * ssim_loss
        # 
        # # Compute metrics
        # with torch.no_grad():
        #     psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        #     ssim = 1 - ssim_loss
        # 
        # metrics = {
        #     'loss': loss.item(),
        #     'mse': mse.item(),
        #     'ssim': ssim.item(),
        #     'psnr': psnr.item(),
        # }
        # 
        # return loss, metrics
        
        raise NotImplementedError("TODO: Implement combined loss forward")


class EdgePreservingLoss(nn.Module):
    """
    Loss that emphasizes edge preservation.
    
    Loss = λ_MSE × MSE + λ_SSIM × (1-SSIM) + λ_edge × EdgeLoss
    
    Where EdgeLoss = MSE(∇x_hat, ∇x) using Sobel gradients.
    
    Args:
        mse_weight: Weight for MSE term
        ssim_weight: Weight for SSIM term
        edge_weight: Weight for edge term
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.1,
        edge_weight: float = 0.1
    ):
        super().__init__()
        
        # TODO: Implement edge-preserving loss
        #
        # Initialize MSE and SSIM losses
        # Create Sobel filters for edge detection:
        # 
        # sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        #                        dtype=torch.float32).view(1, 1, 3, 3)
        # sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        #                        dtype=torch.float32).view(1, 1, 3, 3)
        # self.register_buffer('sobel_x', sobel_x)
        # self.register_buffer('sobel_y', sobel_y)
        
        raise NotImplementedError("TODO: Implement EdgePreservingLoss")
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        # TODO: Implement edge computation
        raise NotImplementedError("TODO: Implement edge computation")
    
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute edge-preserving loss."""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


def test_losses():
    """Test loss functions."""
    print("Testing loss functions...")
    
    # Create test data
    x = torch.rand(4, 1, 64, 64)
    x_perfect = x.clone()
    x_noisy = x + 0.1 * torch.randn_like(x)
    x_noisy = x_noisy.clamp(0, 1)
    
    # Test CombinedLoss
    loss_fn = CombinedLoss(mse_weight=1.0, ssim_weight=0.1)
    
    loss_perfect, metrics_perfect = loss_fn(x_perfect, x)
    loss_noisy, metrics_noisy = loss_fn(x_noisy, x)
    
    assert loss_perfect < loss_noisy, "Perfect should have lower loss"
    print(f"✓ Perfect: loss={metrics_perfect['loss']:.4f}, psnr={metrics_perfect['psnr']:.1f}")
    print(f"✓ Noisy: loss={metrics_noisy['loss']:.4f}, psnr={metrics_noisy['psnr']:.1f}")
    
    print("All loss tests passed!")


if __name__ == "__main__":
    test_losses()
