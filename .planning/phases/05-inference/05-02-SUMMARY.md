---
phase: 05-inference
plan: 02
subsystem: inference
tags: [geotiff, rasterio, io, metadata, cog]
dependency-graph:
  requires: []
  provides: [geotiff-io, metadata-preservation, nodata-handling, cog-support]
  affects: [05-03, 05-04, 05-05]
tech-stack:
  added: [rich]
  patterns: [dataclass-metadata, graceful-fallback]
key-files:
  created:
    - src/inference/geotiff.py
  modified:
    - requirements.txt
decisions:
  - id: geometa-dataclass
    choice: "Dataclass for GeoMetadata"
    rationale: "Clean container for CRS, transform, nodata, tags"
  - id: cog-optional
    choice: "rio-cogeo as optional dependency"
    rationale: "Not all users need COG, keep core deps minimal"
  - id: rich-for-cli
    choice: "Add rich for CLI progress bars"
    rationale: "Better user experience in later plans"
metrics:
  duration: ~15 minutes
  completed: 2026-01-26
---

# Phase 05 Plan 02: GeoTIFF I/O Summary

**One-liner:** GeoTIFF read/write with full metadata preservation (CRS, transform, nodata, tags) plus optional COG support.

## What Was Built

### GeoMetadata Dataclass
Container for all geospatial metadata needed for round-trip preservation:
- `crs`: Coordinate Reference System (rasterio CRS)
- `transform`: Affine transformation matrix (georeferencing)
- `nodata`: Value for missing/invalid pixels
- `dtype`, `count`, `width`, `height`: Image dimensions
- `tags`: Arbitrary key-value metadata
- `descriptions`: Band names/descriptions

### Core I/O Functions

| Function | Purpose |
|----------|---------|
| `read_geotiff(path)` | Read GeoTIFF with full metadata extraction |
| `write_geotiff(data, metadata, path)` | Write GeoTIFF with metadata preservation |
| `create_nodata_mask(data, nodata)` | Create boolean mask for nodata pixels |
| `apply_nodata_mask(data, mask, value)` | Apply nodata values to output |
| `write_cog(data, metadata, path)` | Write Cloud Optimized GeoTIFF (optional) |
| `is_cog_available()` | Check if rio-cogeo is installed |

### Key Features
- **Metadata Preservation:** CRS, transform, nodata, tags all preserved on round-trip
- **Multi-band Support:** Handles single-band (H, W) and multi-band (C, H, W) images
- **Graceful Degradation:** Missing CRS generates warning, doesn't crash
- **COG Fallback:** If rio-cogeo not installed, falls back to standard GeoTIFF with warning
- **Compression:** LZW default for GeoTIFF, DEFLATE for COG

## Implementation Details

```python
# Read with metadata
data, meta = read_geotiff('input.tif')

# Create nodata mask before processing
mask = create_nodata_mask(data, meta.nodata)

# ... process data ...

# Restore nodata pixels
output = apply_nodata_mask(processed, mask, meta.nodata)

# Write with same metadata
write_geotiff(output, meta, 'output.tif')

# Or write as COG (if rio-cogeo available)
write_cog(output, meta, 'output_cog.tif')
```

## Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| rich | >=13.0.0 | CLI progress bars and formatting |
| rio-cogeo | >=5.4.0 | COG support (optional, commented) |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 82631ee | feat | Implement GeoTIFF I/O module with metadata preservation |
| fa71b0e | feat | Add optional COG output support and update requirements |

## Verification Results

```
Testing GeoTIFF I/O module...
  - create_nodata_mask: OK
  - apply_nodata_mask: OK
  - write_geotiff: OK
  - read_geotiff: OK
  - Data preservation: OK
  - Metadata preservation: OK
  - Tags preservation: OK
  - Multi-band support: OK
  - write_cog (fallback to GeoTIFF): OK

All GeoTIFF I/O tests passed!
```

## Deviations from Plan

None - plan executed exactly as written.

## Next Steps

This module will be used by:
- **05-03:** Tiler module for extracting patches from full images
- **05-04:** Compressor for full pipeline integration
- **05-05:** CLI for file I/O operations

The GeoTIFF I/O provides the foundation for the end-to-end compression pipeline, ensuring Sentinel-1 images retain their georeferencing after compression/decompression.
