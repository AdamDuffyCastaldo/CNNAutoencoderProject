"""
ResNet-style Autoencoder for SAR Image Compression

Uses residual blocks with skip connections for better gradient flow
and improved reconstruction quality.

Architecture:
- Encoder: 4 stages of ResidualBlockWithDownsample (256->128->64->32->16)
- Each stage has additional ResidualBlock for more capacity
- Decoder: 4 stages of ResidualBlockWithUpsample (16->32->64->128->256)

Compared to baseline:
- Same bottleneck size (16x16xlatent_channels = 16x compression)
- Better gradient flow via skip connections within blocks
- More parameters but better quality

References:
    - He et al. "Deep Residual Learning for Image Recognition" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .blocks import ResidualBlock, ResidualBlockWithDownsample, ResidualBlockWithUpsample


class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder with residual blocks.

    Architecture:
        Input (1, 256, 256)
        -> Conv 7x7 stride 1 (base_channels, 256, 256)
        -> ResBlock (base_channels, 256, 256)
        -> ResBlockDown (base_channels*2, 128, 128)
        -> ResBlock (base_channels*2, 128, 128)
        -> ResBlockDown (base_channels*4, 64, 64)
        -> ResBlock (base_channels*4, 64, 64)
        -> ResBlockDown (base_channels*8, 32, 32)
        -> ResBlock (base_channels*8, 32, 32)
        -> ResBlockDown (latent_channels, 16, 16)

    Args:
        in_channels: Input image channels (default 1 for SAR)
        base_channels: Base channel count (default 64)
        latent_channels: Latent space channels (default 16 for 16x compression)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 16
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        # Initial convolution (no downsampling)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # Stage 1: 256 -> 128
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlockWithDownsample(base_channels, base_channels * 2)
        )

        # Stage 2: 128 -> 64
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels * 2),
            ResidualBlockWithDownsample(base_channels * 2, base_channels * 4)
        )

        # Stage 3: 64 -> 32
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlockWithDownsample(base_channels * 4, base_channels * 8)
        )

        # Stage 4: 32 -> 16 (to latent)
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlockWithDownsample(base_channels * 8, latent_channels)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input tensor (B, 1, 256, 256)

        Returns:
            Latent tensor (B, latent_channels, 16, 16)
        """
        x = self.conv_in(x)   # (B, 64, 256, 256)
        x = self.stage1(x)    # (B, 128, 128, 128)
        x = self.stage2(x)    # (B, 256, 64, 64)
        x = self.stage3(x)    # (B, 512, 32, 32)
        x = self.stage4(x)    # (B, latent, 16, 16)
        return x


class ResNetDecoder(nn.Module):
    """
    ResNet-style decoder with residual upsampling blocks.

    Architecture mirrors the encoder:
        Latent (latent_channels, 16, 16)
        -> ResBlockUp (base_channels*8, 32, 32)
        -> ResBlock (base_channels*8, 32, 32)
        -> ResBlockUp (base_channels*4, 64, 64)
        -> ResBlock (base_channels*4, 64, 64)
        -> ResBlockUp (base_channels*2, 128, 128)
        -> ResBlock (base_channels*2, 128, 128)
        -> ResBlockUp (base_channels, 256, 256)
        -> ResBlock (base_channels, 256, 256)
        -> Conv 7x7 (1, 256, 256)
        -> Sigmoid

    Args:
        out_channels: Output image channels (default 1 for SAR)
        base_channels: Base channel count (default 64)
        latent_channels: Latent space channels (default 16)
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 16
    ):
        super().__init__()

        self.out_channels = out_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        # Stage 1: 16 -> 32
        self.stage1 = nn.Sequential(
            ResidualBlockWithUpsample(latent_channels, base_channels * 8),
            ResidualBlock(base_channels * 8)
        )

        # Stage 2: 32 -> 64
        self.stage2 = nn.Sequential(
            ResidualBlockWithUpsample(base_channels * 8, base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        # Stage 3: 64 -> 128
        self.stage3 = nn.Sequential(
            ResidualBlockWithUpsample(base_channels * 4, base_channels * 2),
            ResidualBlock(base_channels * 2)
        )

        # Stage 4: 128 -> 256
        self.stage4 = nn.Sequential(
            ResidualBlockWithUpsample(base_channels * 2, base_channels),
            ResidualBlock(base_channels)
        )

        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()  # Output bounded to [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: Latent tensor (B, latent_channels, 16, 16)

        Returns:
            Reconstructed image (B, 1, 256, 256)
        """
        x = self.stage1(z)    # (B, 512, 32, 32)
        x = self.stage2(x)    # (B, 256, 64, 64)
        x = self.stage3(x)    # (B, 128, 128, 128)
        x = self.stage4(x)    # (B, 64, 256, 256)
        x = self.conv_out(x)  # (B, 1, 256, 256)
        return x


class ResNetAutoencoder(nn.Module):
    """
    Complete ResNet-style autoencoder for SAR image compression.

    Combines ResNetEncoder and ResNetDecoder with residual blocks
    for improved gradient flow and reconstruction quality.

    Args:
        in_channels: Input channels (default 1)
        base_channels: Base feature channels (default 64)
        latent_channels: Latent space channels (default 16 for 16x compression)

    Example:
        >>> model = ResNetAutoencoder(latent_channels=16)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> x_hat, z = model(x)
        >>> print(x_hat.shape, z.shape)
        torch.Size([4, 1, 256, 256]) torch.Size([4, 16, 16, 16])
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 16
    ):
        super().__init__()

        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.decoder = ResNetDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.latent_channels = latent_channels
        self.base_channels = base_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input image (B, 1, 256, 256)

        Returns:
            Tuple of (reconstructed_image, latent_representation)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.decoder(z)

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio.

        For 256x256 input and 16x16xC latent:
        ratio = (256*256) / (16*16*C) = 65536 / (256*C)
        """
        input_size = 256 * 256
        latent_size = 16 * 16 * self.latent_channels
        return input_size / latent_size

    def get_latent_size(self) -> Tuple[int, int, int]:
        """Return latent tensor dimensions (C, H, W)."""
        return (self.latent_channels, 16, 16)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': encoder_params + decoder_params
        }


def test_resnet_autoencoder():
    """Test the ResNet autoencoder."""
    print("Testing ResNetAutoencoder...")

    model = ResNetAutoencoder(latent_channels=16, base_channels=64)
    x = torch.randn(2, 1, 256, 256)

    # Test forward
    x_hat, z = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Latent: {z.shape}")
    print(f"  Output: {x_hat.shape}")

    # Test properties
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,} (encoder: {params['encoder']:,}, decoder: {params['decoder']:,})")
    print(f"  Compression ratio: {model.get_compression_ratio():.1f}x")

    # Verify shapes
    assert x_hat.shape == x.shape, f"Output shape mismatch: {x_hat.shape}"
    assert z.shape == (2, 16, 16, 16), f"Latent shape mismatch: {z.shape}"

    print("  All tests passed!")


if __name__ == "__main__":
    test_resnet_autoencoder()
