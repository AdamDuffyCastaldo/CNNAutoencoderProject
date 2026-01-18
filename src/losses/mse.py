"""
Mean Squared Error Loss

Simple MSE loss wrapper with optional reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Standard MSE loss with optional reduction."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x, reduction=self.reduction)
