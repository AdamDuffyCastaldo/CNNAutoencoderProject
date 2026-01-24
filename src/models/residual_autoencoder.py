"""
Pre-Activation Residual Autoencoder for SAR Image Compression (Variant B)

Uses pre-activation residual blocks (ResNet v2 style) for improved gradient flow.
Key difference from ResNetAutoencoder: BN->ReLU->Conv ordering with NO activation
after skip connection addition.

Architecture:
- Encoder: Stem + 4 stages of 2x PreActResidualBlock each (256->128->64->32->16)
- Decoder: 4 stages of 2x PreActResidualBlockUp each + output (16->32->64->128->256)
- 8 residual blocks in encoder, 8 in decoder

Compared to ResNetAutoencoder:
- Pre-activation ordering (BN->ReLU->Conv vs Conv->BN->ReLU)
- 2 blocks per stage (vs 1 ResidualBlock + 1 ResidualBlockWithDown/Up)
- No ReLU after skip addition (cleaner gradient flow)
- Expected: +0.3-1.0 dB PSNR improvement

References:
    - He et al. "Identity Mappings in Deep Residual Networks" (2016)
    - Phase 4 CONTEXT.md architecture decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .blocks import PreActResidualBlock


class PreActResidualEncoder(nn.Module):
    """
    Pre-activation residual encoder with 4 stages.

    Architecture:
        Input (1, 256, 256)
        -> Stem: Conv 7x7 stride 1 -> BN -> ReLU (base_channels, 256, 256)
        -> Stage 1: 2x PreActResidualBlock (256->128, base*2 channels)
        -> Stage 2: 2x PreActResidualBlock (128->64, base*4 channels)
        -> Stage 3: 2x PreActResidualBlock (64->32, base*8 channels)
        -> Stage 4: 2x PreActResidualBlock (32->16, latent_channels)
        Output (latent_channels, 16, 16)

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

        # Stem: initial convolution (no downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # Stage 1: 256 -> 128, channels: base -> base*2
        self.stage1 = nn.Sequential(
            PreActResidualBlock(base_channels, base_channels * 2, stride=2),
            PreActResidualBlock(base_channels * 2, base_channels * 2, stride=1)
        )

        # Stage 2: 128 -> 64, channels: base*2 -> base*4
        self.stage2 = nn.Sequential(
            PreActResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            PreActResidualBlock(base_channels * 4, base_channels * 4, stride=1)
        )

        # Stage 3: 64 -> 32, channels: base*4 -> base*8
        self.stage3 = nn.Sequential(
            PreActResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            PreActResidualBlock(base_channels * 8, base_channels * 8, stride=1)
        )

        # Stage 4: 32 -> 16, channels: base*8 -> latent_channels
        self.stage4 = nn.Sequential(
            PreActResidualBlock(base_channels * 8, latent_channels, stride=2),
            PreActResidualBlock(latent_channels, latent_channels, stride=1)
        )

        # Initialize stem weights (blocks init themselves)
        self._init_weights()

    def _init_weights(self):
        """Initialize stem weights using Kaiming initialization."""
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
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
        x = self.stem(x)      # (B, 64, 256, 256)
        x = self.stage1(x)    # (B, 128, 128, 128)
        x = self.stage2(x)    # (B, 256, 64, 64)
        x = self.stage3(x)    # (B, 512, 32, 32)
        x = self.stage4(x)    # (B, latent, 16, 16)
        return x


class PreActResidualDecoder(nn.Module):
    """
    Pre-activation residual decoder with 4 stages.

    Architecture mirrors the encoder:
        Latent (latent_channels, 16, 16)
        -> Stage 1: Upsample 16->32, 2x PreActResidualBlock (base*8 channels)
        -> Stage 2: Upsample 32->64, 2x PreActResidualBlock (base*4 channels)
        -> Stage 3: Upsample 64->128, 2x PreActResidualBlock (base*2 channels)
        -> Stage 4: Upsample 128->256, 2x PreActResidualBlock (base channels)
        -> Output: Conv 7x7 -> Sigmoid
        Output (1, 256, 256)

    Uses bilinear upsample + 1x1 conv for stable upsampling.

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

        # Stage 1: 16 -> 32, channels: latent -> base*8
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(latent_channels, base_channels * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels * 8)
        )
        self.stage1 = nn.Sequential(
            PreActResidualBlock(base_channels * 8, base_channels * 8, stride=1),
            PreActResidualBlock(base_channels * 8, base_channels * 8, stride=1)
        )

        # Stage 2: 32 -> 64, channels: base*8 -> base*4
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels * 4)
        )
        self.stage2 = nn.Sequential(
            PreActResidualBlock(base_channels * 4, base_channels * 4, stride=1),
            PreActResidualBlock(base_channels * 4, base_channels * 4, stride=1)
        )

        # Stage 3: 64 -> 128, channels: base*4 -> base*2
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels * 2)
        )
        self.stage3 = nn.Sequential(
            PreActResidualBlock(base_channels * 2, base_channels * 2, stride=1),
            PreActResidualBlock(base_channels * 2, base_channels * 2, stride=1)
        )

        # Stage 4: 128 -> 256, channels: base*2 -> base
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels)
        )
        self.stage4 = nn.Sequential(
            PreActResidualBlock(base_channels, base_channels, stride=1),
            PreActResidualBlock(base_channels, base_channels, stride=1)
        )

        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()  # Output bounded to [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize upsample and output weights using Kaiming initialization."""
        for module in [self.upsample1, self.upsample2, self.upsample3, self.upsample4, self.conv_out]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
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
        x = self.upsample1(z)   # (B, 512, 32, 32)
        x = self.stage1(x)      # (B, 512, 32, 32)

        x = self.upsample2(x)   # (B, 256, 64, 64)
        x = self.stage2(x)      # (B, 256, 64, 64)

        x = self.upsample3(x)   # (B, 128, 128, 128)
        x = self.stage3(x)      # (B, 128, 128, 128)

        x = self.upsample4(x)   # (B, 64, 256, 256)
        x = self.stage4(x)      # (B, 64, 256, 256)

        x = self.conv_out(x)    # (B, 1, 256, 256)
        return x


class ResidualAutoencoder(nn.Module):
    """
    Complete Pre-Activation Residual Autoencoder for SAR image compression (Variant B).

    Combines PreActResidualEncoder and PreActResidualDecoder with pre-activation
    residual blocks for improved gradient flow and reconstruction quality.

    Args:
        in_channels: Input channels (default 1)
        base_channels: Base feature channels (default 64)
        latent_channels: Latent space channels (default 16 for 16x compression)

    Example:
        >>> model = ResidualAutoencoder(latent_channels=16)
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

        self.encoder = PreActResidualEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.decoder = PreActResidualDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        self.in_channels = in_channels
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


def test_residual_autoencoder():
    """Test the Pre-Activation Residual Autoencoder."""
    print("Testing ResidualAutoencoder (Variant B - Pre-Activation)...")
    print("=" * 60)

    # Test encoder/decoder individually
    print("\n1. Testing PreActResidualEncoder...")
    encoder = PreActResidualEncoder(in_channels=1, base_channels=64, latent_channels=16)
    x = torch.randn(2, 1, 256, 256)
    z = encoder(x)
    assert z.shape == (2, 16, 16, 16), f"Encoder output shape wrong: {z.shape}"
    print(f"   Input: {x.shape} -> Latent: {z.shape}")
    print("   [OK] Encoder shape correct")

    print("\n2. Testing PreActResidualDecoder...")
    decoder = PreActResidualDecoder(out_channels=1, base_channels=64, latent_channels=16)
    x_hat = decoder(z)
    assert x_hat.shape == (2, 1, 256, 256), f"Decoder output shape wrong: {x_hat.shape}"
    assert x_hat.min() >= 0 and x_hat.max() <= 1, f"Output not bounded [0,1]: [{x_hat.min():.4f}, {x_hat.max():.4f}]"
    print(f"   Latent: {z.shape} -> Output: {x_hat.shape}")
    print(f"   Output range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")
    print("   [OK] Decoder shape and bounds correct")

    # Test full autoencoder
    print("\n3. Testing ResidualAutoencoder...")
    model = ResidualAutoencoder(latent_channels=16, base_channels=64)

    # Multiple batch sizes
    for batch_size in [1, 2, 4]:
        x = torch.randn(batch_size, 1, 256, 256)
        x_hat, z = model(x)
        assert x_hat.shape == x.shape, f"Output shape mismatch for batch={batch_size}"
        assert z.shape == (batch_size, 16, 16, 16), f"Latent shape wrong for batch={batch_size}"
    print("   [OK] Multiple batch sizes work")

    # Test properties
    params = model.count_parameters()
    print(f"\n4. Model Properties:")
    print(f"   Parameters: {params['total']:,}")
    print(f"     - Encoder: {params['encoder']:,}")
    print(f"     - Decoder: {params['decoder']:,}")
    print(f"   Compression ratio: {model.get_compression_ratio():.1f}x")
    print(f"   Latent size: {model.get_latent_size()}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    model.train()
    x = torch.randn(2, 1, 256, 256, requires_grad=True)
    x_hat, z = model(x)
    loss = torch.nn.functional.mse_loss(x_hat, torch.zeros_like(x_hat))
    loss.backward()
    assert x.grad is not None, "No gradient computed for input"
    print("   [OK] Gradients flow correctly")

    # Test encode/decode methods
    print("\n6. Testing encode/decode methods...")
    x = torch.randn(2, 1, 256, 256)
    z = model.encode(x)
    x_hat = model.decode(z)
    assert z.shape == (2, 16, 16, 16), "encode() shape wrong"
    assert x_hat.shape == (2, 1, 256, 256), "decode() shape wrong"
    print("   [OK] encode() and decode() work correctly")

    # GPU test if available
    if torch.cuda.is_available():
        print("\n7. Testing on GPU...")
        model_gpu = model.cuda()
        x_gpu = torch.randn(2, 1, 256, 256).cuda()
        x_hat_gpu, z_gpu = model_gpu(x_gpu)
        print(f"   GPU memory used: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
        print("   [OK] GPU forward pass works")

        # Larger batch test
        torch.cuda.empty_cache()
        x_gpu = torch.randn(32, 1, 256, 256).cuda()
        x_hat_gpu, z_gpu = model_gpu(x_gpu)
        print(f"   GPU memory (batch=32): {torch.cuda.memory_allocated()/1024**3:.3f} GB")
        print("   [OK] Batch size 32 fits in GPU memory")
    else:
        print("\n7. GPU not available, skipping GPU tests")

    print("\n" + "=" * 60)
    print("All ResidualAutoencoder tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_residual_autoencoder()
