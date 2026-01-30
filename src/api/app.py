"""
FastAPI application for SAR image compression.

This module defines the FastAPI application instance and the lifespan
context manager that handles model loading at startup and cleanup at shutdown.

All endpoint logic is in endpoints.py - this file handles only:
- Application instantiation
- Lifespan context manager for model loading/cleanup
- Router inclusion
"""

import glob
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .endpoints import router

# Global model storage (populated at startup)
compressors: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup, cleanup on shutdown.

    This context manager:
    1. Finds available checkpoint files
    2. Loads SARCompressor instances for each model variant (4x, 8x, 16x)
    3. Stores them in the global compressors dict
    4. Cleans up on shutdown
    """
    from src.inference import SARCompressor

    # Find checkpoints directory
    checkpoint_dir = os.environ.get(
        "SARCODEC_CHECKPOINT_DIR",
        "notebooks/checkpoints"
    )

    # Model configurations: name -> glob pattern
    model_configs = {
        "4x": "resnet_c64_b64_cr4x_*/best.pth",
        "8x": "resnet_c32_b64_cr8x_*/best.pth",
        "16x": "resnet_c16_b64_cr16x_*/best.pth",
    }

    # Load available models
    for name, pattern in model_configs.items():
        matches = glob.glob(os.path.join(checkpoint_dir, pattern))
        if matches:
            # Use the first match (most recent by convention)
            checkpoint_path = matches[0]
            try:
                compressors[name] = SARCompressor(checkpoint_path)
                print(f"Loaded model: {name} from {checkpoint_path}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")

    if not compressors:
        print("WARNING: No models loaded! Check SARCODEC_CHECKPOINT_DIR or checkpoint paths.")

    yield

    # Cleanup
    compressors.clear()
    print("Models unloaded.")


app = FastAPI(
    title="SAR Codec API",
    description="SAR image compression using learned autoencoders. "
                "Upload GeoTIFF files for compression, or upload compressed NPZ "
                "files for decompression. Supports 4x, 8x, and 16x compression models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# Allow running directly: python -m src.api.app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
