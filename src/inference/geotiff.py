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
from rasterio.transform import Affine, from_gcps, from_bounds
from scipy.interpolate import RBFInterpolator


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
        gcps: List of Ground Control Points (for Sentinel-1 SAFE format)
        gcps_crs: CRS for GCPs (may differ from image CRS)
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
    gcps: Optional[list] = None  # List of GroundControlPoint objects
    gcps_crs: Optional[Any] = None  # CRS for GCPs

    def has_georef(self) -> bool:
        """Check if metadata contains valid georeferencing."""
        # Valid if has CRS+transform OR has GCPs
        has_affine = self.crs is not None and self.transform is not None
        has_gcps = self.gcps is not None and len(self.gcps) > 0
        return has_affine or has_gcps


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

        # Extract GCPs (used by Sentinel-1 SAFE format)
        gcps = None
        gcps_crs = None
        if src.gcps[0]:  # gcps returns (gcps_list, crs)
            gcps, gcps_crs = src.gcps

        # Warn only if no CRS AND no GCPs
        if crs is None and gcps is None:
            warnings.warn(
                f"No CRS or GCPs found in '{path.name}'. "
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
            descriptions=src.descriptions if any(src.descriptions) else None,
            gcps=gcps,
            gcps_crs=gcps_crs,
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

    # Warn if no CRS and no GCPs
    has_gcps = metadata.gcps is not None and len(metadata.gcps) > 0
    if metadata.crs is None and not has_gcps:
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

        # Write GCPs if present (Sentinel-1 SAFE format)
        if metadata.gcps is not None and len(metadata.gcps) > 0:
            dst.gcps = (metadata.gcps, metadata.gcps_crs)


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

            # Write GCPs if present (Sentinel-1 SAFE format)
            if metadata.gcps is not None and len(metadata.gcps) > 0:
                mem.gcps = (metadata.gcps, metadata.gcps_crs)

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


def warp_with_gcps(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    resolution: Optional[float] = None,
    progress_callback: Optional[callable] = None
) -> None:
    """
    Warp a GeoTIFF using GCPs with thin-plate spline interpolation.

    Uses coarse grid + bilinear upsampling for speed while maintaining accuracy.

    Args:
        input_path: Path to input GeoTIFF with GCPs
        output_path: Path to output warped GeoTIFF
        resolution: Output resolution in CRS units (default: derive from GCPs)
        progress_callback: Optional callback(current, total) for progress

    Raises:
        ValueError: If input has no GCPs
    """
    from scipy.ndimage import map_coordinates
    from scipy.interpolate import RegularGridInterpolator

    input_path = Path(input_path)
    output_path = Path(output_path)

    if progress_callback:
        progress_callback(0, 100)

    with rasterio.open(input_path) as src:
        gcps, gcps_crs = src.gcps

        if not gcps:
            raise ValueError(f"No GCPs found in {input_path}")

        data = src.read(1)
        height, width = data.shape
        nodata = src.nodata

        if progress_callback:
            progress_callback(5, 100)

        # Build RBF interpolators from GCPs
        pixel_coords = np.array([[g.col, g.row] for g in gcps])
        geo_coords = np.array([[g.x, g.y] for g in gcps])

        inv_col = RBFInterpolator(geo_coords, pixel_coords[:, 0], kernel='thin_plate_spline')
        inv_row = RBFInterpolator(geo_coords, pixel_coords[:, 1], kernel='thin_plate_spline')

        if progress_callback:
            progress_callback(10, 100)

        # Calculate output bounds
        min_x, min_y = geo_coords.min(axis=0)
        max_x, max_y = geo_coords.max(axis=0)

        if resolution is None:
            affine_approx = from_gcps(gcps)
            resolution = abs(affine_approx.a)

        out_width = int(np.ceil((max_x - min_x) / resolution))
        out_height = int(np.ceil((max_y - min_y) / resolution))
        out_transform = from_bounds(min_x, min_y, max_x, max_y, out_width, out_height)

        if progress_callback:
            progress_callback(15, 100)

        # OPTIMIZATION: Compute RBF on coarse grid, then interpolate per-chunk
        coarse_step = 64
        coarse_h = (out_height + coarse_step - 1) // coarse_step + 1
        coarse_w = (out_width + coarse_step - 1) // coarse_step + 1

        coarse_rows = np.linspace(0, out_height - 1, coarse_h)
        coarse_cols = np.linspace(0, out_width - 1, coarse_w)
        coarse_cols_grid, coarse_rows_grid = np.meshgrid(coarse_cols, coarse_rows)

        coarse_xs = min_x + (coarse_cols_grid + 0.5) * resolution
        coarse_ys = max_y - (coarse_rows_grid + 0.5) * resolution
        coarse_coords = np.column_stack([coarse_xs.ravel(), coarse_ys.ravel()])

        if progress_callback:
            progress_callback(20, 100)

        coarse_src_cols = inv_col(coarse_coords).reshape(coarse_h, coarse_w)

        if progress_callback:
            progress_callback(35, 100)

        coarse_src_rows = inv_row(coarse_coords).reshape(coarse_h, coarse_w)

        if progress_callback:
            progress_callback(50, 100)

        # Create interpolators for the coarse grid
        col_interp = RegularGridInterpolator(
            (coarse_rows, coarse_cols), coarse_src_cols,
            method='linear', bounds_error=False, fill_value=-1
        )
        row_interp = RegularGridInterpolator(
            (coarse_rows, coarse_cols), coarse_src_rows,
            method='linear', bounds_error=False, fill_value=-1
        )

        # Process in chunks to avoid memory issues
        chunk_size = 2048  # rows per chunk
        output = np.zeros((out_height, out_width), dtype=data.dtype)
        n_chunks = (out_height + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * chunk_size
            row_end = min(row_start + chunk_size, out_height)
            chunk_h = row_end - row_start

            if progress_callback:
                progress_callback(50 + int(40 * chunk_idx / n_chunks), 100)

            # Generate coordinates for this chunk
            chunk_rows = np.arange(row_start, row_end)
            chunk_cols = np.arange(out_width)
            cols_grid, rows_grid = np.meshgrid(chunk_cols, chunk_rows)
            points = np.column_stack([rows_grid.ravel(), cols_grid.ravel()])

            # Interpolate source coordinates
            src_cols = col_interp(points).reshape(chunk_h, out_width)
            src_rows = row_interp(points).reshape(chunk_h, out_width)

            # Bilinear resampling
            src_cols_int = np.clip(src_cols.astype(np.int32), 0, width - 2)
            src_rows_int = np.clip(src_rows.astype(np.int32), 0, height - 2)
            col_frac = np.clip(src_cols - src_cols_int, 0, 1).astype(np.float32)
            row_frac = np.clip(src_rows - src_rows_int, 0, 1).astype(np.float32)

            valid = (src_cols >= 0) & (src_cols < width - 1) & \
                    (src_rows >= 0) & (src_rows < height - 1)

            v00 = data[src_rows_int, src_cols_int]
            v01 = data[src_rows_int, src_cols_int + 1]
            v10 = data[src_rows_int + 1, src_cols_int]
            v11 = data[src_rows_int + 1, src_cols_int + 1]

            chunk_out = (v00 * (1 - col_frac) * (1 - row_frac) +
                         v01 * col_frac * (1 - row_frac) +
                         v10 * (1 - col_frac) * row_frac +
                         v11 * col_frac * row_frac)

            if nodata is not None:
                chunk_out[~valid] = nodata
            else:
                chunk_out[~valid] = 0

            output[row_start:row_end, :] = chunk_out

        if progress_callback:
            progress_callback(90, 100)

        # Write output
        profile = {
            'driver': 'GTiff',
            'dtype': output.dtype,
            'width': out_width,
            'height': out_height,
            'count': 1,
            'crs': gcps_crs,
            'transform': out_transform,
            'nodata': nodata,
            'compress': 'lzw',
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output, 1)

        if progress_callback:
            progress_callback(100, 100)


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
