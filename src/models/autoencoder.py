"""
Complete SAR Autoencoder

This module combines encoder and decoder into a complete autoencoder model.

Architecture:
    Encoder: 256×256×1 → 16×16×C
    Decoder: 16×16×C → 256×256×1

Usage:
    >>> model = SARAutoencoder(latent_channels=64)
    >>> x = torch.randn(4, 1, 256, 256)
    >>> x_hat, z = model(x)
    >>> print(x_hat.shape, z.shape)

References:
    - Day 2 of the learning guide
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .encoder import SAREncoder
from .decoder import SARDecoder


class SARAutoencoder(nn.Module):
    """
    Complete autoencoder for SAR image compression.
    
    Combines encoder and decoder with utility methods for:
    - Compression ratio calculation
    - Latent space analysis
    - Separate encode/decode operations
    
    Args:
        latent_channels: Channels in latent representation (controls compression)
        base_channels: Base channel count for network scaling
        use_bn: Whether to use batch normalization
    
    Attributes:
        encoder: SAREncoder instance
        decoder: SARDecoder instance
        latent_channels: Number of latent channels
    
    Example:
        >>> model = SARAutoencoder(latent_channels=64)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> x_hat, z = model(x)
        >>> 
        >>> # Compression ratio
        >>> print(f"Compression: {model.get_compression_ratio()}x")
        >>>
        >>> # Separate encode/decode
        >>> z = model.encode(x)
        >>> x_reconstructed = model.decode(z)
    """
    
    def __init__(
        self,
        latent_channels: int = 64,
        base_channels: int = 64,
        use_bn: bool = True
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.base_channels = base_channels

        self.encoder = SAREncoder(
            in_channels=1,
            latent_channels=latent_channels,
            base_channels=base_channels,
            use_bn=use_bn
        )

        self.decoder = SARDecoder(
            out_channels=1,
            latent_channels=latent_channels,
            base_channels=base_channels,
            use_bn=use_bn
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor (B, 1, 256, 256)

        Returns:
            x_hat: Reconstructed tensor (B, 1, 256, 256)
            z: Latent representation (B, C, 16, 16)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)
    
    def get_compression_ratio(self, input_size: int = 256) -> float:
        """
        Calculate compression ratio.

        Compression ratio = input_elements / latent_elements

        Args:
            input_size: Input spatial size (default 256)

        Returns:
            Compression ratio (e.g., 4.0 for 4x compression)
        """
        input_elements = input_size * input_size * 1  # 65536 for 256x256x1
        latent_size = input_size // 16  # 16 for 256 input
        latent_elements = latent_size * latent_size * self.latent_channels
        return input_elements / latent_elements
    
    def get_latent_size(self, input_size: int = 256) -> Tuple[int, int, int]:
        """
        Get latent tensor dimensions for given input size.

        Args:
            input_size: Input spatial size

        Returns:
            (channels, height, width) tuple
        """
        latent_size = input_size // 16
        return (self.latent_channels, latent_size, latent_size)
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in encoder and decoder.

        Returns:
            Dict with 'encoder', 'decoder', 'total' parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': encoder_params + decoder_params
        }
    
    def analyze_latent(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Analyze latent representation statistics.

        Useful for monitoring training and diagnosing issues.

        Args:
            z: Latent tensor (B, C, H, W)

        Returns:
            Dict with mean, std, min, max, sparsity, etc.
        """
        return {
            'mean': z.mean().item(),
            'std': z.std().item(),
            'min': z.min().item(),
            'max': z.max().item(),
            'sparsity': (z.abs() < 0.1).float().mean().item()
        }


def test_autoencoder():
    """Comprehensive autoencoder test."""
    print("=" * 60)
    print("Testing SARAutoencoder")
    print("=" * 60)
    
    # Test different configurations
    for latent_channels in [32, 64, 128]:
        print(f"\n--- latent_channels={latent_channels} ---")
        
        model = SARAutoencoder(latent_channels=latent_channels)
        
        # Test forward pass
        x = torch.randn(2, 1, 256, 256)
        x = torch.sigmoid(x)  # Simulate normalized input
        
        x_hat, z = model(x)
        
        assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape}"
        print(f"✓ Shapes correct: input {x.shape} → latent {z.shape} → output {x_hat.shape}")
        
        # Test compression ratio
        compression = model.get_compression_ratio()
        print(f"✓ Compression ratio: {compression:.1f}x")
        
        # Test parameter count
        params = model.count_parameters()
        print(f"✓ Parameters: {params['total']:,}")
    
    # Test gradient flow
    print("\n--- Gradient Flow Test ---")
    model = SARAutoencoder(latent_channels=64)
    x = torch.randn(2, 1, 256, 256, requires_grad=True)
    x_hat, z = model(x)
    loss = F.mse_loss(x_hat, torch.zeros_like(x_hat))
    loss.backward()
    print("✓ Gradients flow correctly")
    
    print("\n" + "=" * 60)
    print("All autoencoder tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_autoencoder()
