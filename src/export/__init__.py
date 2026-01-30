"""
ONNX Export Module for SAR Autoencoder

Provides utilities for exporting trained SAR autoencoder models to ONNX format
with embedded preprocessing, enabling deployment outside the PyTorch ecosystem.

Key exports:
    - SARAutoencoderWithPreprocess: Wrapper model with preprocessing embedded
    - export_encoder_onnx: Export encoder to ONNX
    - export_decoder_onnx: Export decoder to ONNX
    - validate_onnx_model: Validate ONNX model against PyTorch
    - export_full_model: High-level export with validation and checksums
"""

from .wrapper import SARAutoencoderWithPreprocess

# Import ONNX export utilities (may not be available yet)
try:
    from .onnx_export import (
        export_encoder_onnx,
        export_decoder_onnx,
        validate_onnx_model,
        export_full_model,
    )
    __all__ = [
        "SARAutoencoderWithPreprocess",
        "export_encoder_onnx",
        "export_decoder_onnx",
        "validate_onnx_model",
        "export_full_model",
    ]
except ImportError:
    __all__ = ["SARAutoencoderWithPreprocess"]
