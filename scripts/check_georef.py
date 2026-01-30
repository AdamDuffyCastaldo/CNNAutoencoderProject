"""Check georeferencing of a GeoTIFF file."""
import sys
import rasterio

path = sys.argv[1] if len(sys.argv) > 1 else 'test_reconstructed.tiff'

with rasterio.open(path) as src:
    print(f"File: {path}")
    print(f"Size: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")

    gcps, gcps_crs = src.gcps
    if gcps:
        print(f"\nGCPs: {len(gcps)} points")
        print(f"GCPs CRS: {gcps_crs}")
        print(f"\nFirst 3 GCPs:")
        for i, g in enumerate(gcps[:3]):
            print(f"  [{i}] pixel=({g.col:.1f}, {g.row:.1f}) -> geo=({g.x:.6f}, {g.y:.6f})")
    else:
        print("\nNo GCPs found")
