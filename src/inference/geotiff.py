"""
GeoTIFF I/O for SAR Compression Pipeline

Handles reading and writing GeoTIFF files with full metadata preservation:
- CRS (Coordinate Reference System)
- Affine transform (georeferencing)
- Nodata values
- Band descriptions and tags

References:
    - rasterio documentation: https://rasterio.readthedocs.io/
    - COG specification: https://www.cogeo.org/
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass
class GeoMetadata:
    """
    Container for GeoTIFF geospatial metadata.

    Captures all information needed to preserve georeferencing
    when reading and writing GeoTIFF files.

    Attributes:
        crs: Coordinate Reference System (rasterio CRS or None)
        transform: Affine transformation matrix (georeferencing)
        nodata: Value representing missing/invalid pixels
        dtype: NumPy-compatible data type string
        count: Number of bands
        width: Image width in pixels
        height: Image height in pixels
        tags: Dictionary of metadata tags
        descriptions: Tuple of band descriptions (one per band)
    """
    crs: Optional[Any]  # rasterio CRS or None
    transform: Optional[Any]  # rasterio Affine or None
    nodata: Optional[float]
    dtype: str
    count: int
    width: int
    height: int
    tags: Dict[str, str]
    descriptions: Optional[Tuple[str, ...]]

    def has_georef(self) -> bool:
        """Check if metadata contains valid georeferencing."""
        return self.crs is not None and self.transform is not None


def read_geotiff(path: Union[str, Path]) -> Tuple[np.ndarray, GeoMetadata]:
    """
    Read a GeoTIFF file with full metadata extraction.

    Args:
        path: Path to the GeoTIFF file

    Returns:
        Tuple of:
            - data: NumPy array of shape (count, H, W) for multi-band
                   or (H, W) for single-band images
            - metadata: GeoMetadata object with all geospatial info

    Raises:
        FileNotFoundError: If the file doesn't exist
        rasterio.errors.RasterioIOError: If the file can't be read

    Example:
        >>> data, meta = read_geotiff('sentinel1.tif')
        >>> print(f"Shape: {data.shape}, CRS: {meta.crs}")
    """
    path = Path(path)

    with rasterio.open(path, 'r') as src:
        # Read all bands
        data = src.read()  # Shape: (count, H, W)

        # Extract CRS with graceful handling
        crs = src.crs
        if crs is None:
            warnings.warn(
                f"No CRS found in '{path.name}'. "
                "Output will not be georeferenced.",
                UserWarning
            )

        # Extract metadata
        metadata = GeoMetadata(
            crs=crs,
            transform=src.transform,
            nodata=src.nodata,
            dtype=str(src.dtypes[0]),  # Primary band dtype
            count=src.count,
            width=src.width,
            height=src.height,
            tags=dict(src.tags()),
            descriptions=src.descriptions if any(src.descriptions) else None
        )

    # For single-band images, squeeze to 2D
    if data.shape[0] == 1:
        data = data.squeeze(0)

    return data, metadata


def write_geotiff(
    data: np.ndarray,
    metadata: GeoMetadata,
    path: Union[str, Path],
    compress: str = 'lzw'
) -> None:
    """
    Write a GeoTIFF file with metadata preservation.

    Args:
        data: NumPy array of shape (H, W) or (count, H, W)
        metadata: GeoMetadata object with geospatial info
        path: Output file path
        compress: Compression method ('lzw', 'deflate', 'zstd', None)

    Example:
        >>> write_geotiff(processed_data, original_meta, 'output.tif')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle 2D (single band) and 3D (multi-band) data
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add band dimension
        count = 1
    else:
        count = data.shape[0]

    height, width = data.shape[1], data.shape[2]

    # Warn if no CRS
    if metadata.crs is None:
        warnings.warn(
            f"No CRS in metadata. Output '{path.name}' will not be georeferenced.",
            UserWarning
        )

    # Build profile
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype.name,
        'width': width,
        'height': height,
        'count': count,
        'crs': metadata.crs,
        'transform': metadata.transform,
        'nodata': metadata.nodata,
    }

    # Add compression if specified
    if compress:
        profile['compress'] = compress

    # Write file
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)

        # Write tags if present
        if metadata.tags:
            dst.update_tags(**metadata.tags)

        # Write band descriptions if present
        if metadata.descriptions:
            for i, desc in enumerate(metadata.descriptions[:count], start=1):
                if desc:
                    dst.set_band_description(i, desc)


def create_nodata_mask(
    data: np.ndarray,
    nodata: Optional[float]
) -> np.ndarray:
    """
    Create a boolean mask identifying nodata pixels.

    Args:
        data: Image data array
        nodata: Nodata value (None = no nodata pixels)

    Returns:
        Boolean mask where True = nodata pixel

    Example:
        >>> mask = create_nodata_mask(data, -9999.0)
        >>> valid_data = data[~mask]
    """
    if nodata is None:
        return np.zeros(data.shape, dtype=bool)

    # Handle NaN nodata specially
    if np.isnan(nodata):
        return np.isnan(data)

    # Standard comparison for finite values
    return data == nodata


def apply_nodata_mask(
    data: np.ndarray,
    mask: np.ndarray,
    nodata_value: float
) -> np.ndarray:
    """
    Apply nodata mask to data array.

    Args:
        data: Image data array
        mask: Boolean mask (True = set to nodata)
        nodata_value: Value to use for nodata pixels

    Returns:
        Copy of data with masked pixels set to nodata_value

    Example:
        >>> output = apply_nodata_mask(compressed, original_mask, -9999.0)
    """
    result = data.copy()
    result[mask] = nodata_value
    return result


# COG support flag - set at module level
_COG_AVAILABLE = False
try:
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles
    _COG_AVAILABLE = True
except ImportError:
    pass


def write_cog(
    data: np.ndarray,
    metadata: GeoMetadata,
    path: Union[str, Path]
) -> None:
    """
    Write a Cloud Optimized GeoTIFF (COG) with metadata preservation.

    COGs are optimized for cloud storage and HTTP range requests,
    enabling efficient partial reads of large images.

    If rio-cogeo is not installed, falls back to standard GeoTIFF
    with a warning.

    Args:
        data: NumPy array of shape (H, W) or (count, H, W)
        metadata: GeoMetadata object with geospatial info
        path: Output file path

    Example:
        >>> write_cog(compressed_data, original_meta, 'output_cog.tif')
    """
    if not _COG_AVAILABLE:
        warnings.warn(
            "rio-cogeo not installed. Falling back to standard GeoTIFF. "
            "Install with: pip install rio-cogeo",
            UserWarning
        )
        write_geotiff(data, metadata, path, compress='deflate')
        return

    from rasterio.io import MemoryFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle 2D (single band) and 3D (multi-band) data
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        count = 1
    else:
        count = data.shape[0]

    height, width = data.shape[1], data.shape[2]

    # Build profile for temp file
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype.name,
        'width': width,
        'height': height,
        'count': count,
        'crs': metadata.crs,
        'transform': metadata.transform,
        'nodata': metadata.nodata,
    }

    # Write to memory file first, then convert to COG
    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            mem.write(data)

            # Write tags if present
            if metadata.tags:
                mem.update_tags(**metadata.tags)

            # Write band descriptions if present
            if metadata.descriptions:
                for i, desc in enumerate(metadata.descriptions[:count], start=1):
                    if desc:
                        mem.set_band_description(i, desc)

        # Use deflate profile for COG
        cog_profile = cog_profiles.get('deflate')

        # Translate to COG
        cog_translate(
            memfile,
            str(path),
            cog_profile,
            quiet=True
        )


def is_cog_available() -> bool:
    """Check if COG support is available."""
    return _COG_AVAILABLE


def test_geotiff_io():
    """
    Test GeoTIFF I/O with round-trip verification.

    Creates synthetic data, writes to temp file, reads back,
    and verifies data and metadata preservation.
    """
    import tempfile
    import os

    print("Testing GeoTIFF I/O module...")

    # Create synthetic test data
    height, width = 256, 256
    data = np.random.rand(height, width).astype(np.float32)
    data[0:10, 0:10] = -9999.0  # Add some nodata pixels

    # Create mock metadata with WGS84 CRS
    metadata = GeoMetadata(
        crs=CRS.from_epsg(4326),
        transform=Affine(0.001, 0, 10.0, 0, -0.001, 50.0),  # Mock transform
        nodata=-9999.0,
        dtype='float32',
        count=1,
        width=width,
        height=height,
        tags={'source': 'test', 'processing': 'synthetic'},
        descriptions=('Test Band',)
    )

    # Test nodata mask creation
    mask = create_nodata_mask(data, metadata.nodata)
    assert mask.shape == data.shape, "Mask shape mismatch"
    assert mask[0, 0] == True, "Nodata pixel not detected"
    assert mask[100, 100] == False, "Valid pixel marked as nodata"
    print("  - create_nodata_mask: OK")

    # Test apply_nodata_mask
    test_data = np.ones((10, 10), dtype=np.float32)
    test_mask = np.zeros((10, 10), dtype=bool)
    test_mask[0, 0] = True
    result = apply_nodata_mask(test_data, test_mask, -9999.0)
    assert result[0, 0] == -9999.0, "Nodata not applied"
    assert result[1, 1] == 1.0, "Valid pixel modified"
    print("  - apply_nodata_mask: OK")

    # Test write/read round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, 'test.tif')

        # Write
        write_geotiff(data, metadata, test_path, compress='lzw')
        assert os.path.exists(test_path), "Output file not created"
        print("  - write_geotiff: OK")

        # Read back
        data_read, meta_read = read_geotiff(test_path)
        print("  - read_geotiff: OK")

        # Verify data
        np.testing.assert_array_almost_equal(
            data, data_read, decimal=5,
            err_msg="Data mismatch after round-trip"
        )
        print("  - Data preservation: OK")

        # Verify metadata
        assert meta_read.crs == metadata.crs, "CRS mismatch"
        assert meta_read.transform == metadata.transform, "Transform mismatch"
        assert meta_read.nodata == metadata.nodata, "Nodata mismatch"
        assert meta_read.width == metadata.width, "Width mismatch"
        assert meta_read.height == metadata.height, "Height mismatch"
        print("  - Metadata preservation: OK")

        # Verify tags
        assert 'source' in meta_read.tags, "Tags not preserved"
        assert meta_read.tags['source'] == 'test', "Tag value mismatch"
        print("  - Tags preservation: OK")

        # Test multi-band
        multi_band_data = np.random.rand(3, 64, 64).astype(np.float32)
        multi_meta = GeoMetadata(
            crs=CRS.from_epsg(32632),  # UTM zone 32N
            transform=Affine(10.0, 0, 500000.0, 0, -10.0, 5500000.0),
            nodata=None,
            dtype='float32',
            count=3,
            width=64,
            height=64,
            tags={},
            descriptions=('Band 1', 'Band 2', 'Band 3')
        )

        multi_path = os.path.join(tmpdir, 'multi.tif')
        write_geotiff(multi_band_data, multi_meta, multi_path)
        multi_read, multi_meta_read = read_geotiff(multi_path)

        assert multi_read.shape == (3, 64, 64), f"Multi-band shape wrong: {multi_read.shape}"
        np.testing.assert_array_almost_equal(multi_band_data, multi_read, decimal=5)
        print("  - Multi-band support: OK")

        # Test COG support
        cog_path = os.path.join(tmpdir, 'test_cog.tif')
        write_cog(data, metadata, cog_path)
        assert os.path.exists(cog_path), "COG file not created"

        if is_cog_available():
            # Verify COG is readable
            cog_read, cog_meta = read_geotiff(cog_path)
            # COG compression may cause minor differences, use lower precision
            np.testing.assert_array_almost_equal(data, cog_read, decimal=4)
            print("  - write_cog (with rio-cogeo): OK")
        else:
            print("  - write_cog (fallback to GeoTIFF): OK")

    print("\nAll GeoTIFF I/O tests passed!")


if __name__ == '__main__':
    test_geotiff_io()
