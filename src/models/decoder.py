"""
SAR Decoder Implementation

This module implements the decoder portion of the SAR autoencoder.

Architecture:
    Input: 16×16×C (latent representation)
    Output: 256×256×1 (reconstructed SAR patch)

Design Decisions:
    - 4 transposed convolution layers (stride=2) for 16× upsampling
    - 5×5 kernels to match encoder
    - ReLU activation (decoder receives varied inputs)
    - BatchNorm for training stability
    - Sigmoid output to bound to [0, 1]
    - output_padding=1 to achieve exact 2× upsampling

References:
    - Day 2, Session 1 of the learning guide
    - Dumoulin & Visin "A guide to convolution arithmetic" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .blocks import DeconvBlock


class SARDecoder(nn.Module):
    """
    Decoder for SAR images.
    
    Reconstructs 256×256×1 output from 16×16×C latent representation.
    
    Args:
        out_channels: Number of output channels (1 for single polarization)
        latent_channels: Number of channels in latent representation
        base_channels: Number of channels before last layer
        use_bn: Whether to use batch normalization
    
    Example:
        >>> decoder = SARDecoder(latent_channels=64)
        >>> z = torch.randn(4, 64, 16, 16)
        >>> x_hat = decoder(z)
        >>> print(x_hat.shape)  # torch.Size([4, 1, 256, 256])
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 64,
        base_channels: int = 64,
        use_bn: bool = True
    ):
        super().__init__()

        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        # Layer 1: latent_channels -> base_channels*4, 16->32
        self.layer1 = DeconvBlock(
            latent_channels, base_channels * 4,
            kernel_size=5, stride=2, padding=2, output_padding=1, use_bn=use_bn
        )

        # Layer 2: base_channels*4 -> base_channels*2, 32->64
        self.layer2 = DeconvBlock(
            base_channels * 4, base_channels * 2,
            kernel_size=5, stride=2, padding=2, output_padding=1, use_bn=use_bn
        )

        # Layer 3: base_channels*2 -> base_channels, 64->128
        self.layer3 = DeconvBlock(
            base_channels * 2, base_channels,
            kernel_size=5, stride=2, padding=2, output_padding=1, use_bn=use_bn
        )

        # Layer 4: base_channels -> out_channels, 128->256
        # No batchnorm, sigmoid applied in forward
        self.layer4 = nn.ConvTranspose2d(
            base_channels, out_channels,
            kernel_size=5, stride=2, padding=2, output_padding=1
        )

        # Initialize weights after all layers are defined
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent tensor of shape (batch, latent_channels, 16, 16)

        Returns:
            Reconstructed tensor of shape (batch, 1, 256, 256)
        """
        x = self.layer1(z)  # (B, 256, 32, 32)
        x = self.layer2(x)  # (B, 128, 64, 64)
        x = self.layer3(x)  # (B, 64, 128, 128)
        x = self.layer4(x)  # (B, 1, 256, 256)
        x = torch.sigmoid(x)  # FR2.8: Sigmoid output for [0,1] bounded output
        return x


def test_decoder():
    """Test decoder implementation."""
    print("Testing SARDecoder...")
    
    decoder = SARDecoder(latent_channels=64)
    z = torch.randn(2, 64, 16, 16)
    x_hat = decoder(z)
    
    assert x_hat.shape == (2, 1, 256, 256), f"Wrong shape: {x_hat.shape}"
    print(f"✓ Output shape correct: {x_hat.shape}")
    
    assert x_hat.min() >= 0 and x_hat.max() <= 1, "Output not in [0, 1]"
    print("✓ Output in [0, 1] range")
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"✓ Parameters: {params:,}")
    
    print("All decoder tests passed!")


if __name__ == "__main__":
    test_decoder()
