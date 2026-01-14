"""
SAR Encoder Implementation

This module implements the encoder portion of the SAR autoencoder.

Architecture:
    Input: 256×256×1 (single polarization SAR patch)
    Output: 16×16×C (latent representation, C = latent_channels)

Design Decisions:
    - 4 strided convolution layers (stride=2) for 16× spatial reduction
    - 5×5 kernels for adequate receptive field
    - LeakyReLU activation to prevent dead neurons
    - BatchNorm for training stability
    - Channel progression: 1 → 64 → 128 → 256 → latent_channels

References:
    - Day 2, Session 1 of the learning guide
    - Ballé et al. "End-to-end Optimized Image Compression" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .blocks import ConvBlock


class SAREncoder(nn.Module):
    """
    Encoder for SAR images.
    
    Compresses 256×256×1 input to 16×16×C latent representation.
    
    Args:
        in_channels: Number of input channels (1 for single polarization)
        latent_channels: Number of channels in latent representation
        base_channels: Number of channels after first layer
        use_bn: Whether to use batch normalization
    
    Example:
        >>> encoder = SAREncoder(latent_channels=64)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> z = encoder(x)
        >>> print(z.shape)  # torch.Size([4, 64, 16, 16])
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 64,
        base_channels: int = 64,
        use_bn: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        
        # TODO: Implement encoder layers
        # 
        # Architecture:
        # Layer 1: in_channels → base_channels, 256→128
        # Layer 2: base_channels → base_channels*2, 128→64
        # Layer 3: base_channels*2 → base_channels*4, 64→32
        # Layer 4: base_channels*4 → latent_channels, 32→16 (no activation)
        #
        # Use ConvBlock from blocks.py for layers 1-3
        # Use plain Conv2d for layer 4 (no BN or activation on output)
        
        raise NotImplementedError("TODO: Implement encoder layers")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization.
        
        He init is optimal for ReLU-family activations.
        """
        # TODO: Implement weight initialization
        # 
        # For each Conv2d layer:
        #   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        #   if m.bias is not None: nn.init.zeros_(m.bias)
        # 
        # For each BatchNorm2d layer:
        #   nn.init.ones_(m.weight)
        #   nn.init.zeros_(m.bias)
        
        raise NotImplementedError("TODO: Implement weight initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, 1, 256, 256)
        
        Returns:
            Latent tensor of shape (batch, latent_channels, 16, 16)
        """
        # TODO: Implement forward pass
        # 
        # x = self.layer1(x)  # (B, 64, 128, 128)
        # x = self.layer2(x)  # (B, 128, 64, 64)
        # x = self.layer3(x)  # (B, 256, 32, 32)
        # x = self.layer4(x)  # (B, latent_channels, 16, 16)
        # return x
        
        raise NotImplementedError("TODO: Implement forward pass")
    
    def get_receptive_field(self) -> int:
        """
        Calculate theoretical receptive field of encoder.
        
        For kernel=5, stride=2 at each layer:
        RF_l = RF_{l-1} + (kernel-1) × stride_product
        
        Returns:
            Receptive field size in pixels
        """
        # TODO: Calculate receptive field
        # Layer 1: RF = 5
        # Layer 2: RF = 5 + (5-1)*2 = 13
        # Layer 3: RF = 13 + (5-1)*4 = 29
        # Layer 4: RF = 29 + (5-1)*8 = 61
        
        raise NotImplementedError("TODO: Implement receptive field calculation")


def test_encoder():
    """Test encoder implementation."""
    print("Testing SAREncoder...")
    
    encoder = SAREncoder(latent_channels=64)
    x = torch.randn(2, 1, 256, 256)
    z = encoder(x)
    
    assert z.shape == (2, 64, 16, 16), f"Wrong shape: {z.shape}"
    print(f"✓ Output shape correct: {z.shape}")
    
    # Test gradient flow
    loss = z.mean()
    loss.backward()
    print("✓ Gradients flow correctly")
    
    # Parameter count
    params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ Parameters: {params:,}")
    
    print("All encoder tests passed!")


if __name__ == "__main__":
    test_encoder()
