"""Properly warp a GeoTIFF using GCPs (accurate georeferencing).

This script creates a properly orthorectified image by warping using GCPs
instead of approximating with an affine transform.

Usage:
    python scripts/warp_with_gcps.py input.tif output.tif
"""
import sys
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.transform import from_gcps, from_bounds
from scipy.interpolate import RBFInterpolator


def create_gcp_transform_functions(gcps):
    """Create forward and inverse transform functions from GCPs using RBF interpolation."""
    # Extract GCP coordinates
    pixel_coords = np.array([[g.col, g.row] for g in gcps])
    geo_coords = np.array([[g.x, g.y] for g in gcps])

    # Create RBF interpolators for forward transform (pixel -> geo)
    forward_x = RBFInterpolator(pixel_coords, geo_coords[:, 0], kernel='thin_plate_spline')
    forward_y = RBFInterpolator(pixel_coords, geo_coords[:, 1], kernel='thin_plate_spline')

    # Create RBF interpolators for inverse transform (geo -> pixel)
    inverse_col = RBFInterpolator(geo_coords, pixel_coords[:, 0], kernel='thin_plate_spline')
    inverse_row = RBFInterpolator(geo_coords, pixel_coords[:, 1], kernel='thin_plate_spline')

    return (forward_x, forward_y), (inverse_col, inverse_row)


def warp_image_with_gcps(input_path, output_path, resolution=None):
    """Warp image using GCPs with thin plate spline interpolation."""

    with rasterio.open(input_path) as src:
        gcps, gcps_crs = src.gcps

        if not gcps:
            print(f"No GCPs found in {input_path}")
            sys.exit(1)

        print(f"Found {len(gcps)} GCPs with CRS: {gcps_crs}")

        # Get source data
        data = src.read(1)
        height, width = data.shape
        nodata = src.nodata

        # Create transform functions
        print("Creating GCP-based transform functions...")
        (fwd_x, fwd_y), (inv_col, inv_row) = create_gcp_transform_functions(gcps)

        # Calculate output bounds from GCPs
        geo_coords = np.array([[g.x, g.y] for g in gcps])
        min_x, min_y = geo_coords.min(axis=0)
        max_x, max_y = geo_coords.max(axis=0)

        # Determine resolution
        if resolution is None:
            # Use approximate resolution from affine fit
            affine_approx = from_gcps(gcps)
            resolution = abs(affine_approx.a)

        print(f"Output resolution: {resolution:.8f} degrees")
        print(f"Bounds: ({min_x:.6f}, {min_y:.6f}) - ({max_x:.6f}, {max_y:.6f})")

        # Calculate output dimensions
        out_width = int(np.ceil((max_x - min_x) / resolution))
        out_height = int(np.ceil((max_y - min_y) / resolution))

        print(f"Output size: {out_width} x {out_height}")

        # Create output transform
        out_transform = from_bounds(min_x, min_y, max_x, max_y, out_width, out_height)

        # Create output array
        output = np.full((out_height, out_width), nodata if nodata else 0, dtype=data.dtype)

        # Generate output pixel coordinates
        print("Warping image...")
        rows, cols = np.meshgrid(np.arange(out_height), np.arange(out_width), indexing='ij')

        # Convert output pixels to geographic coordinates
        xs = min_x + cols * resolution + resolution / 2
        ys = max_y - rows * resolution - resolution / 2

        # Convert to source pixel coordinates using inverse GCP transform
        coords = np.column_stack([xs.ravel(), ys.ravel()])
        src_cols = inv_col(coords).reshape(out_height, out_width)
        src_rows = inv_row(coords).reshape(out_height, out_width)

        # Bilinear interpolation
        src_cols_int = src_cols.astype(int)
        src_rows_int = src_rows.astype(int)

        # Get fractional parts for interpolation
        col_frac = src_cols - src_cols_int
        row_frac = src_rows - src_rows_int

        # Mask for valid source coordinates
        valid = (src_cols_int >= 0) & (src_cols_int < width - 1) & \
                (src_rows_int >= 0) & (src_rows_int < height - 1)

        # Bilinear interpolation for valid pixels
        for i in range(out_height):
            if i % 500 == 0:
                print(f"  Row {i}/{out_height}...")
            for j in range(out_width):
                if valid[i, j]:
                    c, r = src_cols_int[i, j], src_rows_int[i, j]
                    cf, rf = col_frac[i, j], row_frac[i, j]

                    # Bilinear interpolation
                    output[i, j] = (
                        data[r, c] * (1 - cf) * (1 - rf) +
                        data[r, c + 1] * cf * (1 - rf) +
                        data[r + 1, c] * (1 - cf) * rf +
                        data[r + 1, c + 1] * cf * rf
                    )

        # Write output
        profile = {
            'driver': 'GTiff',
            'dtype': data.dtype,
            'width': out_width,
            'height': out_height,
            'count': 1,
            'crs': gcps_crs,
            'transform': out_transform,
            'nodata': nodata,
            'compress': 'lzw',
        }

        print(f"Writing {output_path}...")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output, 1)

        print("Done!")
        return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python warp_with_gcps.py input.tif [output.tif] [resolution]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".tif", "_warped.tif")
    resolution = float(sys.argv[3]) if len(sys.argv) > 3 else None

    warp_image_with_gcps(input_file, output_file, resolution)
