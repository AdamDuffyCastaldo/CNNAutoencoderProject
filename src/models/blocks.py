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
        
        # TODO: Implement ConvBlock
        #
        # self.conv = nn.Conv2d(...)
        # self.bn = nn.BatchNorm2d(...) if use_bn else nn.Identity()
        # self.activation = nn.LeakyReLU(negative_slope)
        #
        # Note: If use_bn=True, set bias=False in Conv2d (BN has its own bias)
        
        raise NotImplementedError("TODO: Implement ConvBlock")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv → bn → activation"""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


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
        
        # TODO: Implement DeconvBlock
        #
        # self.deconv = nn.ConvTranspose2d(...)
        # self.bn = nn.BatchNorm2d(...) if use_bn else nn.Identity()
        # self.activation = nn.ReLU()
        
        raise NotImplementedError("TODO: Implement DeconvBlock")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: deconv → bn → activation"""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


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
        
        # TODO: Implement ResidualBlock
        #
        # padding = kernel_size // 2
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        # self.bn1 = nn.BatchNorm2d(channels)
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        # self.bn2 = nn.BatchNorm2d(channels)
        
        raise NotImplementedError("TODO: Implement ResidualBlock")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        # TODO: Implement forward pass
        #
        # residual = x
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out = out + residual  # Skip connection
        # out = F.relu(out)
        # return out
        
        raise NotImplementedError("TODO: Implement forward pass")


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
        
        # TODO: Implement ResidualBlockWithDownsample
        #
        # Main path: two convolutions (first with stride)
        # Skip path: 1×1 conv with same stride to match dimensions
        
        raise NotImplementedError("TODO: Implement ResidualBlockWithDownsample")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with downsampling."""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


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
        
        # TODO: Implement ResidualBlockWithUpsample
        #
        # Main path: transposed conv for upsampling, then regular conv
        # Skip path: transposed 1×1 conv to match dimensions
        
        raise NotImplementedError("TODO: Implement ResidualBlockWithUpsample")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with upsampling."""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


# ============================================================================
# ATTENTION BLOCKS (Day 4, Advanced)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    
    Learns to weight channels based on their importance.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for bottleneck (default 16)
    
    References:
        - Day 4, Section 4.4 of learning guide
        - Hu et al. "Squeeze-and-Excitation Networks" (2018)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # TODO: Implement ChannelAttention
        #
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channels, channels // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channels // reduction, channels, bias=False),
        # )
        
        raise NotImplementedError("TODO: Implement ChannelAttention")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    
    Learns where to focus spatially using channel-wise statistics.
    
    Args:
        kernel_size: Convolution kernel size (default 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        # TODO: Implement SpatialAttention
        raise NotImplementedError("TODO: Implement SpatialAttention")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention."""
        raise NotImplementedError("TODO: Implement forward pass")


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel and spatial attention sequentially.
    
    Args:
        channels: Number of input channels
        reduction: Channel attention reduction ratio
        kernel_size: Spatial attention kernel size
    
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
        
        # TODO: Implement CBAM
        #
        # self.channel_attention = ChannelAttention(channels, reduction)
        # self.spatial_attention = SpatialAttention(kernel_size)
        
        raise NotImplementedError("TODO: Implement CBAM")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel then spatial attention."""
        # TODO: Implement forward pass
        raise NotImplementedError("TODO: Implement forward pass")


def test_blocks():
    """Test all blocks."""
    print("Testing building blocks...")
    
    x = torch.randn(2, 64, 128, 128)
    
    # Test ConvBlock
    conv_block = ConvBlock(64, 128, stride=2)
    y = conv_block(x)
    assert y.shape == (2, 128, 64, 64), f"ConvBlock wrong shape: {y.shape}"
    print("✓ ConvBlock")
    
    # Test DeconvBlock
    deconv_block = DeconvBlock(128, 64, stride=2)
    z = deconv_block(y)
    assert z.shape == (2, 64, 128, 128), f"DeconvBlock wrong shape: {z.shape}"
    print("✓ DeconvBlock")
    
    # Test ResidualBlock
    res_block = ResidualBlock(64)
    r = res_block(x)
    assert r.shape == x.shape, f"ResidualBlock wrong shape: {r.shape}"
    print("✓ ResidualBlock")
    
    print("All block tests passed!")


if __name__ == "__main__":
    test_blocks()
