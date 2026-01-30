"""
Pydantic models for SAR Codec API request/response types.

These models define the structure of API payloads for health checks,
compression statistics, model information, and error responses.
"""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    """Information about a loaded compression model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "resnet_8x",
                "latent_channels": 32,
                "compression_ratio": 8.0,
                "checkpoint_path": "notebooks/checkpoints/resnet_c32_b64_cr8x_20260128_213848/best.pth",
            }
        }
    )

    name: str
    latent_channels: int
    compression_ratio: float
    checkpoint_path: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": ["4x", "8x", "16x"],
                "device": "cuda",
                "cuda_available": True,
            }
        }
    )

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    models_loaded: list[str]
    device: str
    cuda_available: bool


class CompressionStats(BaseModel):
    """Statistics about a compression operation."""

    model_config = ConfigDict(
        # Allow arbitrary types (for tuples) and serialize tuples as lists
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "compression_ratio": 8.0,
                "bpp": 1.0,
                "original_shape": [512, 512],
                "latent_shape": [9, 32, 16, 16],
                "n_tiles": 9,
                "time_seconds": 1.5,
            }
        },
    )

    compression_ratio: float
    bpp: float
    original_shape: tuple[int, int]
    latent_shape: tuple[int, ...]
    n_tiles: int
    time_seconds: float


class ErrorResponse(BaseModel):
    """Response model for error conditions."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Model not found",
                "detail": "Model '32x' is not loaded. Available models: 4x, 8x, 16x",
            }
        }
    )

    error: str
    detail: Optional[str] = None
