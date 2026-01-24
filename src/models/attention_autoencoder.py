"""
Attention Autoencoder for SAR Image Compression (Variant C)

Pre-Activation Residual Blocks + CBAM Attention after every block.

This is Variant C in the architecture enhancement plan:
- Uses PreActResidualBlock (BN->ReLU->Conv ordering)
- Applies CBAM attention after EVERY residual block (16 CBAM modules total)
- Target: +0.5 dB PSNR improvement over Variant B (residual-only)

Architecture:
- Encoder: 4 stages, 2 blocks per stage (8 residual + 8 CBAM)
- Decoder: 4 stages, 2 blocks per stage (8 residual + 8 CBAM)
- Total: 16 residual blocks + 16 CBAM modules

References:
    - He et al. "Identity Mappings in Deep Residual Networks" (2016)
    - Woo et al. "CBAM: Convolutional Block Attention Module" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .blocks import PreActResidualBlock, CBAM


class ResidualBlockWithCBAM(nn.Module):
    """
    Pre-activation residual block followed by CBAM attention.

    Combines PreActResidualBlock + CBAM sequentially for enhanced
    feature extraction with both channel and spatial attention.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride (1 preserves spatial, 2 downsamples)
        reduction: CBAM channel attention reduction ratio (default 16)
        kernel_size: CBAM spatial attention kernel size (default 7)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()

        self.residual = PreActResidualBlock(in_channels, out_channels, stride)
        self.cbam = CBAM(out_channels, reduction, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: residual block then CBAM attention."""
        x = self.residual(x)
        x = self.cbam(x)
        return x


class AttentionEncoder(nn.Module):
    """
    Encoder with Pre-Activation Residual Blocks + CBAM attention.

    Architecture:
        Input (1, 256, 256)
        -> Stem: Conv 7x7 stride 1 -> BN -> ReLU (base_channels, 256, 256)
        -> Stage 1: ResBlockCBAM(base->base*2, stride=2) + ResBlockCBAM(base*2, stride=1) -> (128, 128)
        -> Stage 2: ResBlockCBAM(base*2->base*4, stride=2) + ResBlockCBAM(base*4, stride=1) -> (256, 64)
        -> Stage 3: ResBlockCBAM(base*4->base*8, stride=2) + ResBlockCBAM(base*8, stride=1) -> (512, 32)
        -> Stage 4: ResBlockCBAM(base*8->latent, stride=2) + ResBlockCBAM(latent, stride=1) -> (latent, 16)

    Total: 8 residual blocks + 8 CBAM modules

    Args:
        in_channels: Input image channels (default 1 for SAR)
        base_channels: Base channel count (default 64)
        latent_channels: Latent space channels (default 16 for 16x compression)
        reduction: CBAM reduction ratio (default 16)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 16,
        reduction: int = 16
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        # Stem: 7x7 conv without downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # Stage 1: 256 -> 128
        self.stage1 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels, base_channels * 2, stride=2, reduction=reduction),
            ResidualBlockWithCBAM(base_channels * 2, base_channels * 2, stride=1, reduction=reduction)
        )

        # Stage 2: 128 -> 64
        self.stage2 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels * 2, base_channels * 4, stride=2, reduction=reduction),
            ResidualBlockWithCBAM(base_channels * 4, base_channels * 4, stride=1, reduction=reduction)
        )

        # Stage 3: 64 -> 32
        self.stage3 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels * 4, base_channels * 8, stride=2, reduction=reduction),
            ResidualBlockWithCBAM(base_channels * 8, base_channels * 8, stride=1, reduction=reduction)
        )

        # Stage 4: 32 -> 16 (to latent)
        self.stage4 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels * 8, latent_channels, stride=2, reduction=reduction),
            ResidualBlockWithCBAM(latent_channels, latent_channels, stride=1, reduction=reduction)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for conv weights."""
        for m in self.modules():
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


class AttentionDecoder(nn.Module):
    """
    Decoder with Pre-Activation Residual Blocks + CBAM attention.

    Uses bilinear upsample + 1x1 conv at start of each stage,
    followed by two ResidualBlockWithCBAM.

    Architecture:
        Latent (latent_channels, 16, 16)
        -> Stage 1: Upsample + 1x1 -> ResBlockCBAM -> ResBlockCBAM -> (base*8, 32, 32)
        -> Stage 2: Upsample + 1x1 -> ResBlockCBAM -> ResBlockCBAM -> (base*4, 64, 64)
        -> Stage 3: Upsample + 1x1 -> ResBlockCBAM -> ResBlockCBAM -> (base*2, 128, 128)
        -> Stage 4: Upsample + 1x1 -> ResBlockCBAM -> ResBlockCBAM -> (base, 256, 256)
        -> Output: Conv 7x7 -> Sigmoid -> (1, 256, 256)

    Total: 8 residual blocks + 8 CBAM modules

    Args:
        out_channels: Output image channels (default 1 for SAR)
        base_channels: Base channel count (default 64)
        latent_channels: Latent space channels (default 16)
        reduction: CBAM reduction ratio (default 16)
    """

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 16,
        reduction: int = 16
    ):
        super().__init__()

        self.out_channels = out_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        # Stage 1: 16 -> 32
        self.upsample1 = nn.Conv2d(latent_channels, base_channels * 8, kernel_size=1, bias=False)
        self.stage1 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels * 8, base_channels * 8, stride=1, reduction=reduction),
            ResidualBlockWithCBAM(base_channels * 8, base_channels * 8, stride=1, reduction=reduction)
        )

        # Stage 2: 32 -> 64
        self.upsample2 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1, bias=False)
        self.stage2 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels * 4, base_channels * 4, stride=1, reduction=reduction),
            ResidualBlockWithCBAM(base_channels * 4, base_channels * 4, stride=1, reduction=reduction)
        )

        # Stage 3: 64 -> 128
        self.upsample3 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1, bias=False)
        self.stage3 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels * 2, base_channels * 2, stride=1, reduction=reduction),
            ResidualBlockWithCBAM(base_channels * 2, base_channels * 2, stride=1, reduction=reduction)
        )

        # Stage 4: 128 -> 256
        self.upsample4 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1, bias=False)
        self.stage4 = nn.Sequential(
            ResidualBlockWithCBAM(base_channels, base_channels, stride=1, reduction=reduction),
            ResidualBlockWithCBAM(base_channels, base_channels, stride=1, reduction=reduction)
        )

        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()  # Output bounded to [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for conv weights."""
        for m in self.modules():
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
        # Stage 1: 16 -> 32
        x = F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upsample1(x)
        x = self.stage1(x)    # (B, 512, 32, 32)

        # Stage 2: 32 -> 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upsample2(x)
        x = self.stage2(x)    # (B, 256, 64, 64)

        # Stage 3: 64 -> 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upsample3(x)
        x = self.stage3(x)    # (B, 128, 128, 128)

        # Stage 4: 128 -> 256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upsample4(x)
        x = self.stage4(x)    # (B, 64, 256, 256)

        # Output
        x = self.conv_out(x)  # (B, 1, 256, 256)
        return x


class AttentionAutoencoder(nn.Module):
    """
    Complete Attention Autoencoder for SAR image compression (Variant C).

    Combines AttentionEncoder and AttentionDecoder, both using
    Pre-Activation Residual Blocks with CBAM attention after every block.

    Args:
        in_channels: Input channels (default 1)
        base_channels: Base feature channels (default 64)
        latent_channels: Latent space channels (default 16 for 16x compression)
        reduction: CBAM reduction ratio (default 16)

    Example:
        >>> model = AttentionAutoencoder(latent_channels=16)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> x_hat, z = model(x)
        >>> print(x_hat.shape, z.shape)
        torch.Size([4, 1, 256, 256]) torch.Size([4, 16, 16, 16])
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 16,
        reduction: int = 16
    ):
        super().__init__()

        self.encoder = AttentionEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            reduction=reduction
        )

        self.decoder = AttentionDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            reduction=reduction
        )

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels
        self.reduction = reduction

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


def test_attention_autoencoder():
    """Test the Attention autoencoder."""
    print("Testing AttentionAutoencoder...")
    print("=" * 50)

    # Test encoder/decoder separately
    print("\n1. Testing Encoder/Decoder components...")

    encoder = AttentionEncoder(in_channels=1, base_channels=64, latent_channels=16)
    x = torch.randn(2, 1, 256, 256)
    z = encoder(x)
    assert z.shape == (2, 16, 16, 16), f"Encoder output shape wrong: {z.shape}"
    print(f"   Encoder: {x.shape} -> {z.shape}")

    decoder = AttentionDecoder(out_channels=1, base_channels=64, latent_channels=16)
    x_hat = decoder(z)
    assert x_hat.shape == (2, 1, 256, 256), f"Decoder output shape wrong: {x_hat.shape}"
    print(f"   Decoder: {z.shape} -> {x_hat.shape}")

    # Verify output bounded
    assert x_hat.min() >= 0 and x_hat.max() <= 1, "Output not bounded [0,1]"
    print("   Output bounded [0, 1]: OK")

    # Count CBAM modules
    cbam_count = sum(1 for m in encoder.modules() if m.__class__.__name__ == 'CBAM')
    cbam_count += sum(1 for m in decoder.modules() if m.__class__.__name__ == 'CBAM')
    print(f"   Total CBAM modules: {cbam_count} (expected 16)")
    assert cbam_count == 16, f"Expected 16 CBAM modules, got {cbam_count}"

    # Test full autoencoder
    print("\n2. Testing full AttentionAutoencoder...")

    model = AttentionAutoencoder(latent_channels=16, base_channels=64)
    x = torch.randn(2, 1, 256, 256)
    x_hat, z = model(x)

    assert x_hat.shape == x.shape, f"Output shape mismatch: {x_hat.shape}"
    assert z.shape == (2, 16, 16, 16), f"Latent shape wrong: {z.shape}"
    print(f"   Forward: {x.shape} -> {z.shape} -> {x_hat.shape}")

    # Test properties
    params = model.count_parameters()
    print(f"   Parameters: {params['total']:,} (encoder: {params['encoder']:,}, decoder: {params['decoder']:,})")
    print(f"   Compression ratio: {model.get_compression_ratio():.1f}x")
    print(f"   Latent size: {model.get_latent_size()}")

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    model.train()
    x = torch.randn(2, 1, 256, 256, requires_grad=True)
    x_hat, z = model(x)
    loss = torch.nn.functional.mse_loss(x_hat, torch.zeros_like(x_hat))
    loss.backward()
    print("   Gradient flow: OK")

    # Test on GPU if available
    print("\n4. Testing GPU compatibility...")
    if torch.cuda.is_available():
        model_gpu = model.cuda()
        x_gpu = torch.randn(2, 1, 256, 256).cuda()
        x_hat_gpu, z_gpu = model_gpu(x_gpu)
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU test (batch=2): OK (memory: {mem_gb:.3f} GB)")

        # Test with larger batch to estimate training capacity
        torch.cuda.empty_cache()
        try:
            x_batch = torch.randn(16, 1, 256, 256).cuda()
            x_hat_batch, _ = model_gpu(x_batch)
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU test (batch=16): OK (memory: {mem_gb:.3f} GB)")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("   GPU test (batch=16): OOM - may need smaller batch for training")
                torch.cuda.empty_cache()
            else:
                raise

        # Test batch=32 (training batch size)
        torch.cuda.empty_cache()
        try:
            x_batch = torch.randn(32, 1, 256, 256).cuda()
            x_hat_batch, _ = model_gpu(x_batch)
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU test (batch=32): OK (memory: {mem_gb:.3f} GB)")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("   GPU test (batch=32): OOM - recommend batch_size=16-24 for training")
                torch.cuda.empty_cache()
            else:
                raise
    else:
        print("   No GPU available, skipping GPU tests")

    print("\n" + "=" * 50)
    print("All AttentionAutoencoder tests passed!")


if __name__ == "__main__":
    test_attention_autoencoder()
