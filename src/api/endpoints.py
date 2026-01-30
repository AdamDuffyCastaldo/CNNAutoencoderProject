"""
API endpoints for SAR Codec.

This module contains all endpoint implementations:
- /health - Health check with loaded models
- /encode - Compress GeoTIFF to NPZ
- /decode - Decompress NPZ to GeoTIFF
- /compress - Alias for /encode
- /decompress - Alias for /decode
"""

import io
import json
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from .models import HealthResponse

router = APIRouter()


def get_compressors():
    """Get the compressors dict from app module."""
    from .app import compressors
    return compressors


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the API status, loaded models, and device information.
    """
    import torch
    compressors = get_compressors()

    return HealthResponse(
        status="healthy" if compressors else "degraded",
        version="1.0.0",
        models_loaded=list(compressors.keys()),
        device="cuda" if torch.cuda.is_available() else "cpu",
        cuda_available=torch.cuda.is_available(),
    )


@router.post("/encode")
async def encode_image(
    file: UploadFile = File(..., description="GeoTIFF image to compress"),
    model: str = Query("8x", enum=["4x", "8x", "16x"], description="Model variant"),
):
    """
    Encode a GeoTIFF to latent representation.

    Upload a GeoTIFF file and receive a compressed NPZ file containing
    the latent representation and metadata needed for decompression.
    """
    compressors = get_compressors()

    # Validate model exists
    if model not in compressors:
        available = ", ".join(compressors.keys()) if compressors else "none"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not loaded. Available models: {available}"
        )

    # Validate file type
    if file.filename and not file.filename.lower().endswith(('.tif', '.tiff')):
        raise HTTPException(
            status_code=400,
            detail="File must be GeoTIFF (.tif or .tiff)"
        )

    compressor = compressors[model]
    start_time = time.time()

    try:
        # Read file content
        content = await file.read()

        # Read GeoTIFF from memory
        from rasterio.io import MemoryFile
        with MemoryFile(content) as memfile:
            with memfile.open() as src:
                data = src.read(1).astype(np.float32)
                geo_metadata = _extract_geo_metadata(src)

        # Compress
        latent, tile_metadata = compressor.compress(data)

        # Build response NPZ
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            latent=latent,
            metadata=json.dumps({
                "geo_metadata": _serialize_geo_metadata(geo_metadata),
                "tile_metadata": tile_metadata,
            })
        )
        buffer.seek(0)

        processing_time = time.time() - start_time

        # Derive output filename
        output_name = file.filename if file.filename else "compressed"
        if not output_name.endswith('.npz'):
            output_name = output_name + ".npz"

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={output_name}",
                "X-Processing-Time-Ms": str(int(processing_time * 1000)),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Compression failed: {str(e)}"
        )


@router.post("/decode")
async def decode_latent(
    file: UploadFile = File(..., description="Compressed NPZ file"),
    model: str = Query("8x", enum=["4x", "8x", "16x"], description="Model variant"),
):
    """
    Decode latent representation to GeoTIFF.

    Upload a compressed NPZ file (from /encode) and receive the
    reconstructed GeoTIFF image.
    """
    compressors = get_compressors()

    if model not in compressors:
        available = ", ".join(compressors.keys()) if compressors else "none"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not loaded. Available models: {available}"
        )

    if file.filename and not file.filename.lower().endswith('.npz'):
        raise HTTPException(
            status_code=400,
            detail="File must be NPZ format"
        )

    compressor = compressors[model]
    start_time = time.time()

    try:
        # Read NPZ from memory
        content = await file.read()
        buffer = io.BytesIO(content)
        npz_data = np.load(buffer, allow_pickle=True)

        latent = npz_data['latent']
        metadata = json.loads(str(npz_data['metadata']))
        geo_metadata = _deserialize_geo_metadata(metadata['geo_metadata'])
        tile_metadata = metadata['tile_metadata']

        # Decompress
        reconstructed = compressor.decompress(latent, tile_metadata)

        # Write GeoTIFF to memory
        from rasterio.io import MemoryFile

        profile = {
            'driver': 'GTiff',
            'height': reconstructed.shape[0],
            'width': reconstructed.shape[1],
            'count': 1,
            'dtype': 'float32',
        }

        # Add geo metadata if available
        if geo_metadata:
            if geo_metadata.get('crs'):
                from rasterio.crs import CRS
                profile['crs'] = CRS.from_wkt(geo_metadata['crs'])
            if geo_metadata.get('transform'):
                from rasterio.transform import Affine
                profile['transform'] = Affine(*geo_metadata['transform'])
            if geo_metadata.get('nodata') is not None:
                profile['nodata'] = geo_metadata['nodata']

        output_buffer = io.BytesIO()
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(reconstructed, 1)
                if geo_metadata and geo_metadata.get('tags'):
                    dst.update_tags(**geo_metadata['tags'])
            output_buffer.write(memfile.read())

        output_buffer.seek(0)
        processing_time = time.time() - start_time

        # Derive output filename
        output_name = file.filename if file.filename else "reconstructed.tif"
        output_name = output_name.replace('.npz', '_reconstructed.tif')

        return StreamingResponse(
            output_buffer,
            media_type="image/tiff",
            headers={
                "Content-Disposition": f"attachment; filename={output_name}",
                "X-Processing-Time-Ms": str(int(processing_time * 1000)),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decompression failed: {str(e)}"
        )


# Aliases for user convenience
@router.post("/compress")
async def compress(
    file: UploadFile = File(..., description="GeoTIFF image to compress"),
    model: str = Query("8x", enum=["4x", "8x", "16x"], description="Model variant"),
):
    """Alias for /encode - compress GeoTIFF to NPZ."""
    return await encode_image(file, model)


@router.post("/decompress")
async def decompress(
    file: UploadFile = File(..., description="Compressed NPZ file"),
    model: str = Query("8x", enum=["4x", "8x", "16x"], description="Model variant"),
):
    """Alias for /decode - decompress NPZ to GeoTIFF."""
    return await decode_latent(file, model)


# Helper functions for geo metadata handling
def _extract_geo_metadata(src):
    """Extract metadata from rasterio dataset."""
    return {
        'crs': src.crs.to_wkt() if src.crs else None,
        'transform': tuple(src.transform)[:6] if src.transform else None,
        'nodata': src.nodata,
        'tags': dict(src.tags()),
    }


def _serialize_geo_metadata(meta):
    """Convert geo metadata to JSON-serializable dict."""
    if meta is None:
        return None
    return {
        'crs': meta.get('crs'),
        'transform': list(meta['transform']) if meta.get('transform') else None,
        'nodata': float(meta['nodata']) if meta.get('nodata') is not None else None,
        'tags': meta.get('tags', {}),
    }


def _deserialize_geo_metadata(data):
    """Reconstruct geo metadata from JSON."""
    if data is None:
        return None
    return {
        'crs': data.get('crs'),
        'transform': tuple(data['transform']) if data.get('transform') else None,
        'nodata': data.get('nodata'),
        'tags': data.get('tags', {}),
    }
