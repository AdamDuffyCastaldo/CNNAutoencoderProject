"""
Reusable Building Blocks for SAR Autoencoder

This module contains common building blocks:
- ConvBlock: Convolution → BatchNorm → LeakyReLU
- DeconvBlock: TransposedConv → BatchNorm → ReLU
- ResidualBlock: Skip-connected convolutional block
- ResidualBlockWithDownsample: Residual block with strided downsampling
- ResidualBlockWithUpsample: Residual block with upsampling

References:
    - Day 2 and Day 4 of the learning guide
    - He et al. "Deep Residual Learning" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv → BatchNorm → LeakyReLU
    
    This is the fundamental building block of the encoder.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default 5)
        stride: Convolution stride (default 2 for downsampling)
        padding: Padding size (default 2 for kernel=5)
        use_bn: Whether to use batch normalization
        negative_slope: LeakyReLU negative slope (default 0.2)
    
    Example:
        >>> block = ConvBlock(64, 128, stride=2)
        >>> x = torch.randn(4, 64, 128, 128)
        >>> y = block(x)
        >>> print(y.shape)  # torch.Size([4, 128, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
        use_bn: bool = True,
        negative_slope: float = 0.2
    ):
        super().__init__()

        # Conv2d with bias=False when using BatchNorm (BN has its own bias)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv -> bn -> activation"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DeconvBlock(nn.Module):
    """
    Basic deconvolution block: ConvTranspose → BatchNorm → ReLU
    
    This is the fundamental building block of the decoder.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (default 5)
        stride: Stride for upsampling (default 2)
        padding: Padding size (default 2)
        output_padding: Additional output size adjustment (default 1)
        use_bn: Whether to use batch normalization
    
    Note:
        output_padding=1 is needed for exact 2× upsampling with kernel=5, stride=2
        Formula: output = (input - 1) × stride + kernel - 2×padding + output_padding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
        output_padding: int = 1,
        use_bn: bool = True
    ):
        super().__init__()

        # ConvTranspose2d with output_padding=1 for exact 2x upsampling
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: deconv -> bn -> activation"""
        x = self.deconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Basic residual block with skip connection.

    Structure: x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU

    The skip connection allows gradient to flow directly, enabling
    training of deeper networks.

    Args:
        channels: Number of input/output channels (must match for skip)
        kernel_size: Convolution kernel size (default 3)

    References:
        - Day 4, Section 4.1-4.2 of learning guide
        - He et al. "Deep Residual Learning" (2016)
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual  # Skip connection
        out = F.relu(out)

        return out


class ResidualBlockWithDownsample(nn.Module):
    """
    Residual block with spatial downsampling.

    When changing spatial dimensions, the skip connection needs a
    1×1 strided convolution to match dimensions.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Downsampling stride (default 2)
        kernel_size: Main conv kernel size (default 3)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        kernel_size: int = 3
    ):
        super().__init__()

        padding = kernel_size // 2

        # Main path: first conv with stride for downsampling
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip path: 1×1 conv with stride to match dimensions
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with downsampling."""
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual  # Skip connection
        out = F.relu(out)

        return out


class ResidualBlockWithUpsample(nn.Module):
    """
    Residual block with spatial upsampling.

    Uses transposed convolution for learnable upsampling.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        scale_factor: Upsampling factor (default 2)
        kernel_size: Main conv kernel size (default 3)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 3
    ):
        super().__init__()

        padding = kernel_size // 2

        # Main path: transposed conv for upsampling, then regular conv
        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=scale_factor, padding=padding, output_padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip path: transposed 1×1 conv to match dimensions
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 1, stride=scale_factor, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with upsampling."""
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual  # Skip connection
        out = F.relu(out)

        return out


# ============================================================================
# PRE-ACTIVATION RESIDUAL BLOCKS (ResNet v2)
# ============================================================================

class PreActResidualBlock(nn.Module):
    """
    Pre-activation residual block (ResNet v2 style).

    Structure: x -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+x)

    Key difference from post-activation: NO ReLU after skip addition.
    This provides cleaner gradient flow through identity path.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride (1 preserves spatial, 2 downsamples)

    References:
        - He et al. "Identity Mappings in Deep Residual Networks" (2016)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Pre-activation path: BN -> ReLU -> Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )

        # Projection shortcut when dimensions change
        self.need_projection = (stride != 1) or (in_channels != out_channels)
        if self.need_projection:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for conv weights, standard BN init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation."""
        # Identity path
        identity = x
        if self.need_projection:
            identity = self.projection(x)

        # Main path: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # NO activation after addition (key for pre-activation)
        return out + identity


class PreActResidualBlockDown(nn.Module):
    """
    Pre-activation residual block for downsampling.

    Convenience wrapper that applies stride=2 for 2x spatial reduction.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = PreActResidualBlock(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PreActResidualBlockUp(nn.Module):
    """
    Pre-activation residual block for upsampling.

    Uses bilinear upsample + 1x1 conv for channel change,
    followed by two pre-activation convolutions.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale_factor: Upsampling factor (default 2)
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()

        self.scale_factor = scale_factor

        # Channel projection for skip connection (applied after upsample)
        self.projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        # Pre-activation residual convolutions
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for conv weights, standard BN init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with upsampling."""
        # Upsample then project channels
        upsampled = F.interpolate(
            x, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False
        )
        identity = self.projection(upsampled)

        # Pre-activation path: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        out = self.bn1(identity)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # NO activation after addition
        return out + identity


# ============================================================================
# ATTENTION BLOCKS (Day 4, Advanced)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention (CBAM variant).

    Uses both max-pool and avg-pool with shared MLP, per CBAM paper.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for bottleneck (default 16)

    Returns:
        Attention weights of shape (B, C, 1, 1) in range [0, 1]

    References:
        - Day 4, Section 4.4 of learning guide
        - Hu et al. "Squeeze-and-Excitation Networks" (2018)
        - Woo et al. "CBAM: Convolutional Block Attention Module" (2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # Handle edge case: ensure at least 1 channel in reduced layer
        reduced = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP using 1x1 convolutions (more efficient than Linear)
        # NO BatchNorm inside MLP per CBAM paper
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention: sigmoid(mlp(avgpool) + mlp(maxpool))."""
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Learns where to focus spatially using channel-wise max and mean statistics.

    Args:
        kernel_size: Convolution kernel size (default 7)

    Returns:
        Attention weights of shape (B, 1, H, W) in range [0, 1]

    References:
        - Woo et al. "CBAM: Convolutional Block Attention Module" (2018)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        # Padding to preserve spatial dimensions
        padding = kernel_size // 2

        # Conv takes concatenated [max, mean] -> 1 channel attention map
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention: sigmoid(conv(concat(maxpool, avgpool)))."""
        # Channel-wise statistics: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # Concatenate and apply convolution
        concat = torch.cat([max_out, avg_out], dim=1)  # (B, 2, H, W)
        return torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention sequentially (channel-first).

    Args:
        channels: Number of input channels
        reduction: Channel attention reduction ratio (default 16)
        kernel_size: Spatial attention kernel size (default 7)

    Returns:
        Refined features with same shape as input

    References:
        - Woo et al. "CBAM: Convolutional Block Attention Module" (2018)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention, then spatial attention."""
        # Channel attention: refine which channels to focus on
        x = x * self.channel_attention(x)

        # Spatial attention: refine where to focus spatially
        x = x * self.spatial_attention(x)

        return x


def test_blocks():
    """Test all blocks."""
    print("Testing building blocks...")

    x = torch.randn(2, 64, 128, 128)

    # Test ConvBlock
    conv_block = ConvBlock(64, 128, stride=2)
    y = conv_block(x)
    assert y.shape == (2, 128, 64, 64), f"ConvBlock wrong shape: {y.shape}"
    print("[OK] ConvBlock")

    # Test DeconvBlock
    deconv_block = DeconvBlock(128, 64, stride=2)
    z = deconv_block(y)
    assert z.shape == (2, 64, 128, 128), f"DeconvBlock wrong shape: {z.shape}"
    print("[OK] DeconvBlock")

    # Test ResidualBlock
    res_block = ResidualBlock(64)
    r = res_block(x)
    assert r.shape == x.shape, f"ResidualBlock wrong shape: {r.shape}"
    print("[OK] ResidualBlock")

    # Test PreActResidualBlock (stride=1)
    preact = PreActResidualBlock(64, 64, stride=1)
    p = preact(x)
    assert p.shape == x.shape, f"PreActResidualBlock stride=1 wrong shape: {p.shape}"
    print("[OK] PreActResidualBlock (stride=1)")

    # Test PreActResidualBlock (stride=2)
    preact_down = PreActResidualBlock(64, 128, stride=2)
    pd = preact_down(x)
    assert pd.shape == (2, 128, 64, 64), f"PreActResidualBlock stride=2 wrong shape: {pd.shape}"
    print("[OK] PreActResidualBlock (stride=2)")

    # Test PreActResidualBlockDown
    down_block = PreActResidualBlockDown(64, 128)
    d = down_block(x)
    assert d.shape == (2, 128, 64, 64), f"PreActResidualBlockDown wrong shape: {d.shape}"
    print("[OK] PreActResidualBlockDown")

    # Test PreActResidualBlockUp
    up_block = PreActResidualBlockUp(128, 64)
    u = up_block(pd)
    assert u.shape == x.shape, f"PreActResidualBlockUp wrong shape: {u.shape}"
    print("[OK] PreActResidualBlockUp")

    # Test ChannelAttention
    ca = ChannelAttention(64, reduction=16)
    ca_out = ca(x)
    assert ca_out.shape == (2, 64, 1, 1), f"ChannelAttention wrong shape: {ca_out.shape}"
    assert (ca_out >= 0).all() and (ca_out <= 1).all(), "ChannelAttention not in [0,1]"
    print("[OK] ChannelAttention")

    # Test SpatialAttention
    sa = SpatialAttention(kernel_size=7)
    sa_out = sa(x)
    assert sa_out.shape == (2, 1, 128, 128), f"SpatialAttention wrong shape: {sa_out.shape}"
    assert (sa_out >= 0).all() and (sa_out <= 1).all(), "SpatialAttention not in [0,1]"
    print("[OK] SpatialAttention")

    # Test CBAM
    cbam = CBAM(64, reduction=16, kernel_size=7)
    cbam_out = cbam(x)
    assert cbam_out.shape == x.shape, f"CBAM wrong shape: {cbam_out.shape}"
    print("[OK] CBAM")

    # Test CBAM with small channels (edge case)
    cbam_small = CBAM(8, reduction=16)
    small_x = torch.randn(2, 8, 32, 32)
    small_out = cbam_small(small_x)
    assert small_out.shape == small_x.shape, "CBAM small channels failed"
    print("[OK] CBAM (small channels)")

    print("\nAll block tests passed!")


if __name__ == "__main__":
    test_blocks()
