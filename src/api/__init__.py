"""
FastAPI REST API for SAR Codec.

This module provides HTTP endpoints for SAR image compression and decompression.

Usage:
    python -m src.api

Endpoints:
    GET  /health     - Health check with loaded models
    POST /encode     - Compress GeoTIFF to NPZ
    POST /decode     - Decompress NPZ to GeoTIFF
    POST /compress   - Alias for /encode
    POST /decompress - Alias for /decode
    GET  /docs       - OpenAPI documentation
"""

# Lazy imports - only import when accessed
def __getattr__(name):
    if name == "app":
        from .app import app
        return app
    elif name == "lifespan":
        from .app import lifespan
        return lifespan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "lifespan"]
