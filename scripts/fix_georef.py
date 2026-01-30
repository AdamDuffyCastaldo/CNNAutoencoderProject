"""Convert GCPs to geotransform for QGIS compatibility (memory efficient)."""
import sys
import rasterio
from rasterio.transform import from_gcps

input_file = sys.argv[1] if len(sys.argv) > 1 else "reconstructed.tif"
output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".tif", "_georef.tif")

print(f"Converting {input_file} -> {output_file}")

with rasterio.open(input_file) as src:
    gcps, gcps_crs = src.gcps
    if not gcps:
        print(f"No GCPs in {input_file}")
        sys.exit(1)

    print(f"Found {len(gcps)} GCPs with CRS: {gcps_crs}")
    transform = from_gcps(gcps)

    profile = src.profile.copy()
    profile.update(crs=gcps_crs, transform=transform)

    # Process in windows to save memory
    with rasterio.open(output_file, 'w', **profile) as dst:
        for i, window in enumerate(src.block_windows(1)):
            idx, win = window
            data = src.read(window=win)
            dst.write(data, window=win)
            if i % 100 == 0:
                print(f"  Processing block {i}...")

print(f"Done! Created {output_file}")
