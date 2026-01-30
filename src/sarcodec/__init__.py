"""
SAR Codec - SAR Image Compression using Learned Autoencoders

This package provides tools for compressing and decompressing SAR (Synthetic
Aperture Radar) satellite imagery using trained autoencoder models.

Basic usage:
    from sarcodec import SARCompressor

    # Load model and compress
    compressor = SARCompressor("path/to/checkpoint.pth")
    latent, metadata = compressor.compress(image)

    # Decompress
    reconstructed = compressor.decompress(latent, metadata)

CLI usage:
    sarcodec compress input.tif -o output.npz
    sarcodec decompress output.npz -o reconstructed.tif

For more information, see: https://github.com/username/sarcodec
"""

__version__ = "1.0.0"
__author__ = "SAR Codec Authors"

# Re-export main classes
from src.inference.compressor import SARCompressor
from src.inference.geotiff import (
    GeoMetadata,
    read_geotiff,
    write_geotiff,
)

__all__ = [
    "__version__",
    "SARCompressor",
    "GeoMetadata",
    "read_geotiff",
    "write_geotiff",
]


# Lazy imports for optional modules
def get_api_app():
    """Get FastAPI application (requires sarcodec[api])."""
    try:
        from src.api.app import app
        return app
    except ImportError as e:
        raise ImportError(
            "FastAPI not installed. Install with: pip install sarcodec[api]"
        ) from e


def get_onnx_exporter():
    """Get ONNX export functions (requires sarcodec[export])."""
    try:
        from src.export.onnx_export import export_full_model
        return export_full_model
    except ImportError as e:
        raise ImportError(
            "ONNX not installed. Install with: pip install sarcodec[export]"
        ) from e
