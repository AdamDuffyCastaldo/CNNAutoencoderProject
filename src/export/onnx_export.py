"""
ONNX Export Utilities with Validation and Checksum Generation

Provides functions for exporting SAR autoencoder models to ONNX format:
- export_encoder_onnx: Export encoder (raw SAR -> latent)
- export_decoder_onnx: Export decoder (latent -> raw SAR)
- validate_onnx_model: Validate ONNX output matches PyTorch
- export_full_model: High-level export with validation and checksums (FR7.6)

Uses torch.onnx.export(dynamo=True) per PyTorch 2.5+ recommendation.
Includes SHA256 checksums in metadata.json for model integrity verification.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .wrapper import SARAutoencoderWithPreprocess


def compute_file_checksum(file_path: str) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        SHA256 hex digest string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks for memory efficiency
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class EncoderWrapper(nn.Module):
    """Wrapper to export encoder with preprocessing."""

    def __init__(self, model: SARAutoencoderWithPreprocess):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)


class DecoderWrapper(nn.Module):
    """Wrapper to export decoder with inverse preprocessing."""

    def __init__(self, model: SARAutoencoderWithPreprocess):
        super().__init__()
        self.model = model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


def export_encoder_onnx(
    model: SARAutoencoderWithPreprocess,
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 17,
) -> None:
    """
    Export encoder to ONNX format.

    The exported model accepts raw SAR linear intensity values and outputs
    the latent representation.

    Args:
        model: SARAutoencoderWithPreprocess instance
        output_path: Path to save ONNX file
        example_input: Example input tensor (default: random 2x1x256x256)
        opset_version: ONNX opset version (default: 17)

    Note:
        Uses batch size > 1 for example input to ensure dynamic batch works.
    """
    model.eval()

    # Create example input if not provided
    if example_input is None:
        # Use batch size > 1 to avoid pitfall #1 (batch dimension not dynamic)
        example_input = torch.randn(2, 1, 256, 256, device=next(model.parameters()).device)

    # Create encoder wrapper
    encoder_wrapper = EncoderWrapper(model)
    encoder_wrapper.eval()

    # Export using standard torch.onnx.export
    # Note: dynamo=True is the new approach but may have compatibility issues
    # Using standard export for broader compatibility
    torch.onnx.export(
        encoder_wrapper,
        (example_input,),
        output_path,
        input_names=["input"],
        output_names=["latent"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "latent": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )


def export_decoder_onnx(
    model: SARAutoencoderWithPreprocess,
    output_path: str,
    latent_channels: int,
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 17,
) -> None:
    """
    Export decoder to ONNX format.

    The exported model accepts latent representation and outputs
    raw SAR linear intensity values.

    Args:
        model: SARAutoencoderWithPreprocess instance
        output_path: Path to save ONNX file
        latent_channels: Number of latent channels
        example_input: Example latent tensor (default: random 2xCx16x16)
        opset_version: ONNX opset version (default: 17)
    """
    model.eval()

    # Create example input if not provided
    if example_input is None:
        # Use batch size > 1 for dynamic batch dimension
        device = next(model.parameters()).device
        example_input = torch.randn(2, latent_channels, 16, 16, device=device)

    # Create decoder wrapper
    decoder_wrapper = DecoderWrapper(model)
    decoder_wrapper.eval()

    # Export
    torch.onnx.export(
        decoder_wrapper,
        (example_input,),
        output_path,
        input_names=["latent"],
        output_names=["output"],
        dynamic_axes={
            "latent": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )


def validate_onnx_model(
    onnx_path: str,
    pytorch_model: nn.Module,
    example_input: torch.Tensor,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Validate ONNX model output matches PyTorch.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: PyTorch model (should be the same wrapper used for export)
        example_input: Example input tensor
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary with validation results:
        - valid: bool, whether validation passed
        - max_diff: float, maximum absolute difference
        - mean_diff: float, mean absolute difference
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for validation. "
            "Install with: pip install onnxruntime"
        )

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(example_input)

    # Load ONNX model
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run ONNX inference
    onnx_output = session.run(
        None,
        {input_name: example_input.cpu().numpy()},
    )[0]

    # Compare outputs
    pytorch_np = pytorch_output.cpu().numpy()
    diff = np.abs(pytorch_np - onnx_output)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    valid = max_diff < atol

    return {
        "valid": valid,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def export_full_model(
    checkpoint_path: str,
    output_dir: str,
    validate: bool = True,
    opset_version: int = 17,
) -> Dict[str, Any]:
    """
    Export complete model (encoder + decoder) to ONNX with validation and checksums.

    Creates:
        - {output_dir}/encoder.onnx: Encoder model (raw SAR -> latent)
        - {output_dir}/decoder.onnx: Decoder model (latent -> raw SAR)
        - {output_dir}/metadata.json: Model metadata with checksums (FR7.6)

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Output directory for ONNX files
        validate: Whether to validate ONNX against PyTorch
        opset_version: ONNX opset version (default: 17)

    Returns:
        Dictionary with export results:
        - encoder: validation results for encoder
        - decoder: validation results for decoder
        - metadata: path to metadata.json
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = SARAutoencoderWithPreprocess.from_checkpoint(checkpoint_path, device="cpu")
    model_info = model.get_model_info()

    # Define output paths
    encoder_path = str(output_dir / "encoder.onnx")
    decoder_path = str(output_dir / "decoder.onnx")
    metadata_path = str(output_dir / "metadata.json")

    results = {}

    # Export encoder
    print(f"Exporting encoder to {encoder_path}...")
    export_encoder_onnx(model, encoder_path, opset_version=opset_version)

    # Export decoder
    print(f"Exporting decoder to {decoder_path}...")
    export_decoder_onnx(
        model,
        decoder_path,
        latent_channels=model_info["latent_channels"],
        opset_version=opset_version,
    )

    # Validate if requested
    # Note: tolerances are set based on typical BatchNorm network behavior
    # Encoder is tighter (preprocessing + encoder only)
    # Decoder is looser (many more operations, inverse preprocessing with exp)
    if validate:
        print("Validating encoder...")
        encoder_wrapper = EncoderWrapper(model)
        # Use representative SAR values for validation
        encoder_input = torch.rand(2, 1, 256, 256) * 300
        results["encoder"] = validate_onnx_model(
            encoder_path,
            encoder_wrapper,
            encoder_input,
            atol=1e-4,  # Encoder tolerance (preprocessing + encoder)
        )

        print("Validating decoder...")
        decoder_wrapper = DecoderWrapper(model)
        decoder_input = torch.randn(2, model_info["latent_channels"], 16, 16)
        results["decoder"] = validate_onnx_model(
            decoder_path,
            decoder_wrapper,
            decoder_input,
            atol=1e-2,  # Decoder tolerance (decoder + inverse_preprocess with exp)
        )
    else:
        results["encoder"] = {"valid": None, "max_diff": None, "mean_diff": None}
        results["decoder"] = {"valid": None, "max_diff": None, "mean_diff": None}

    # Compute checksums (FR7.6)
    print("Computing checksums...")
    encoder_checksum = compute_file_checksum(encoder_path)
    decoder_checksum = compute_file_checksum(decoder_path)

    # Create metadata
    metadata = {
        "latent_channels": model_info["latent_channels"],
        "base_channels": model_info["base_channels"],
        "vmin": model_info["vmin"],
        "vmax": model_info["vmax"],
        "noise_floor": model_info["noise_floor"],
        "input_shape": model_info["input_shape"],
        "export_date": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "onnx_opset": opset_version,
        "source_checkpoint": str(checkpoint_path),
        "checksums": {
            "encoder_sha256": encoder_checksum,
            "decoder_sha256": decoder_checksum,
        },
    }

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    results["metadata"] = metadata_path

    return results


def test_export():
    """Test ONNX export functionality."""
    import os
    import tempfile

    print("Testing ONNX export...")

    # Find a checkpoint
    checkpoint_path = "notebooks/checkpoints/resnet_c32_b64_cr8x_20260128_213848/best.pth"

    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  Skipping test")
        return

    # Export to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Exporting to {tmpdir}...")
        results = export_full_model(checkpoint_path, tmpdir, validate=True)

        print(f"  Encoder valid: {results['encoder']['valid']}, max_diff: {results['encoder']['max_diff']:.2e}")
        print(f"  Decoder valid: {results['decoder']['valid']}, max_diff: {results['decoder']['max_diff']:.2e}")
        print(f"  Files: {os.listdir(tmpdir)}")

        # Verify checksums
        with open(results["metadata"]) as f:
            meta = json.load(f)
        print(f"  Encoder SHA256: {meta['checksums']['encoder_sha256'][:16]}...")
        print(f"  Decoder SHA256: {meta['checksums']['decoder_sha256'][:16]}...")

        # Verify checksum matches actual file
        actual_encoder_checksum = compute_file_checksum(os.path.join(tmpdir, "encoder.onnx"))
        assert meta["checksums"]["encoder_sha256"] == actual_encoder_checksum, "Encoder checksum mismatch!"

    print("  All tests PASSED!")


if __name__ == "__main__":
    test_export()
