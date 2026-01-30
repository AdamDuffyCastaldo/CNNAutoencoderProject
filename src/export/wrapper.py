"""
ONNX Export Wrapper Model with Embedded Preprocessing

Wraps the ResNetAutoencoder with preprocessing embedded in the model graph,
enabling self-contained ONNX models that accept raw SAR linear intensity values.

The wrapper embeds:
- dB conversion (10 * log10(x))
- Range clipping to [vmin, vmax]
- Normalization to [0, 1]
- Inverse operations for decoding

This enables ONNX models to be deployed without external preprocessing logic.

Note: This wrapper handles single 256x256 patches. Tiled inference for full
images uses the SARCompressor which orchestrates patch extraction, ONNX
inference, and reassembly with blending.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple


class SARAutoencoderWithPreprocess(nn.Module):
    """
    Wrapper model that embeds preprocessing into the ONNX graph.

    The model accepts raw SAR linear intensity values and:
    - Encoder: converts to dB, normalizes, compresses to latent space
    - Decoder: decompresses from latent, converts back to linear SAR

    All operations use pure torch functions for ONNX compatibility.

    Args:
        encoder: Encoder module from ResNetAutoencoder
        decoder: Decoder module from ResNetAutoencoder
        vmin: Minimum dB value for normalization
        vmax: Maximum dB value for normalization
        noise_floor: Minimum value to clamp inputs (avoids log(0))

    Example:
        >>> model = SARAutoencoderWithPreprocess.from_checkpoint("best.pth")
        >>> x = torch.rand(2, 1, 256, 256) * 300  # Simulated linear SAR
        >>> z = model.encode(x)  # Raw SAR -> latent
        >>> y = model.decode(z)  # Latent -> raw SAR
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vmin: float,
        vmax: float,
        noise_floor: float = 1e-10,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Register as buffers so they export to ONNX
        self.register_buffer("vmin", torch.tensor(vmin, dtype=torch.float32))
        self.register_buffer("vmax", torch.tensor(vmax, dtype=torch.float32))
        self.register_buffer("noise_floor", torch.tensor(noise_floor, dtype=torch.float32))

        # Store raw values for easy access
        self._vmin_val = vmin
        self._vmax_val = vmax
        self._noise_floor_val = noise_floor

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw SAR linear intensity to normalized [0, 1] range.

        Steps:
            1. Clamp to noise floor (avoid log(0))
            2. Convert to dB: 10 * log10(x)
            3. Clip to [vmin, vmax]
            4. Normalize to [0, 1]

        Args:
            x: Raw SAR linear intensity (B, 1, H, W)

        Returns:
            Normalized tensor in [0, 1] range
        """
        # Clamp to noise floor
        x = torch.clamp(x, min=self.noise_floor)

        # Convert to dB
        x_db = 10.0 * torch.log10(x)

        # Clip to range
        x_db = torch.clamp(x_db, self.vmin, self.vmax)

        # Normalize to [0, 1]
        normalized = (x_db - self.vmin) / (self.vmax - self.vmin)

        return normalized

    def inverse_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized [0, 1] back to raw SAR linear intensity.

        Steps:
            1. Denormalize from [0, 1] to [vmin, vmax] (dB scale)
            2. Convert from dB to linear: 10^(x_db/10)

        Args:
            x: Normalized tensor in [0, 1] range

        Returns:
            Raw SAR linear intensity values
        """
        # Denormalize to dB
        x_db = x * (self.vmax - self.vmin) + self.vmin

        # Convert to linear
        x_linear = torch.pow(10.0, x_db / 10.0)

        return x_linear

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw SAR linear intensity to latent representation.

        Applies preprocessing then encoder.

        Args:
            x: Raw SAR linear intensity (B, 1, 256, 256)

        Returns:
            Latent representation (B, C, 16, 16)
        """
        normalized = self.preprocess(x)
        return self.encoder(normalized)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to raw SAR linear intensity.

        Applies decoder then inverse preprocessing.

        Args:
            z: Latent representation (B, C, 16, 16)

        Returns:
            Raw SAR linear intensity (B, 1, 256, 256)
        """
        normalized = self.decoder(z)
        return self.inverse_preprocess(normalized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode-decode round-trip.

        Args:
            x: Raw SAR linear intensity (B, 1, 256, 256)

        Returns:
            Reconstructed raw SAR linear intensity (B, 1, 256, 256)
        """
        z = self.encode(x)
        return self.decode(z)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "SARAutoencoderWithPreprocess":
        """
        Load wrapped model from a saved checkpoint.

        Args:
            checkpoint_path: Path to PyTorch checkpoint file
            device: Device to load model on ('cpu' or 'cuda')

        Returns:
            SARAutoencoderWithPreprocess instance in eval mode

        Example:
            >>> model = SARAutoencoderWithPreprocess.from_checkpoint(
            ...     "notebooks/checkpoints/resnet_c32_b64_cr8x_*/best.pth"
            ... )
        """
        from src.models.resnet_autoencoder import ResNetAutoencoder

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract config
        config = checkpoint.get("config", {})

        # Extract preprocessing params
        if "preprocessing_params" in config:
            preprocess_params = config["preprocessing_params"]
        elif "preprocess_params" in checkpoint:
            preprocess_params = checkpoint["preprocess_params"]
        else:
            # Fallback to default values
            preprocess_params = {"vmin": 14.7688, "vmax": 24.5407}

        vmin = float(preprocess_params.get("vmin", 14.7688))
        vmax = float(preprocess_params.get("vmax", 24.5407))

        # Extract model parameters
        latent_channels = config.get("latent_channels", 16)
        base_channels = config.get("base_channels", 64)

        # Instantiate autoencoder
        autoencoder = ResNetAutoencoder(
            in_channels=1,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        # Load state dict
        autoencoder.load_state_dict(checkpoint["model_state_dict"])

        # Create wrapped model
        wrapped = cls(
            encoder=autoencoder.encoder,
            decoder=autoencoder.decoder,
            vmin=vmin,
            vmax=vmax,
        )

        # Move to device and set eval mode
        wrapped.to(device)
        wrapped.eval()

        return wrapped

    def get_model_info(self) -> dict:
        """
        Get model configuration information.

        Returns:
            Dictionary with model configuration
        """
        return {
            "vmin": self._vmin_val,
            "vmax": self._vmax_val,
            "noise_floor": self._noise_floor_val,
            "latent_channels": self.encoder.latent_channels,
            "base_channels": self.encoder.base_channels,
            "input_shape": [1, 1, 256, 256],
        }


def test_wrapper():
    """Test the wrapper model."""
    import os

    print("Testing SARAutoencoderWithPreprocess...")

    # Find a checkpoint
    checkpoint_paths = [
        "notebooks/checkpoints/resnet_c32_b64_cr8x_20260128_213848/best.pth",
        "notebooks/checkpoints/resnet_c16_b64_cr16x_20260128_003926/best.pth",
    ]

    checkpoint_path = None
    for p in checkpoint_paths:
        if os.path.exists(p):
            checkpoint_path = p
            break

    if checkpoint_path is None:
        print("  No checkpoint found, skipping test")
        return

    print(f"  Loading: {checkpoint_path}")

    # Load model
    model = SARAutoencoderWithPreprocess.from_checkpoint(checkpoint_path)
    print(f"  Model info: {model.get_model_info()}")

    # Test with simulated SAR values
    x = torch.rand(2, 1, 256, 256) * 300  # Simulated linear SAR values

    # Test encode
    z = model.encode(x)
    print(f"  Input: {x.shape}, Latent: {z.shape}")

    # Test decode
    y = model.decode(z)
    print(f"  Output: {y.shape}")

    # Verify shapes
    assert z.shape[0] == 2, f"Batch size mismatch: {z.shape[0]}"
    assert z.shape[2] == 16 and z.shape[3] == 16, f"Latent spatial size mismatch: {z.shape}"
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"

    # Test full round-trip
    y_rt = model(x)
    assert y_rt.shape == x.shape, f"Round-trip shape mismatch: {y_rt.shape}"

    print("  All tests PASSED!")


if __name__ == "__main__":
    test_wrapper()
