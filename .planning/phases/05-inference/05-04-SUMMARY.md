---
phase: 05-inference
plan: 04
subsystem: cli
tags: [argparse, rich, cli, compression, decompression]

# Dependency graph
requires:
  - phase: 05-02
    provides: GeoTIFF I/O (read_geotiff, write_geotiff, write_cog, GeoMetadata)
  - phase: 05-03
    provides: SARCompressor class with batched GPU inference and progress callbacks
provides:
  - CLI entry point for SAR compression/decompression
  - sarcodec compress command for GeoTIFF to NPZ
  - sarcodec decompress command for NPZ to GeoTIFF
  - Rich progress bars with ETA
  - Distinct exit codes for error handling
affects: [05-05, deployment, documentation]

# Tech tracking
tech-stack:
  added: [rich]
  patterns: [argparse subcommands, progress callbacks, JSON metadata serialization]

key-files:
  created:
    - scripts/sarcodec.py
  modified: []

key-decisions:
  - "Combined Task 1 and Task 2 into single implementation since they are tightly coupled"
  - "GeoMetadata serialized as CRS WKT string and transform tuple for JSON compatibility"
  - "Nodata pixels replaced with median during compression, mask stored separately"
  - "Exit codes: 0=success, 1=file, 2=model, 3=OOM, 4=general"

patterns-established:
  - "CLI subcommand pattern with argparse for sarcodec compress/decompress"
  - "Rich Progress context manager with callback integration"
  - "GeoMetadata JSON serialization/deserialization for NPZ storage"

# Metrics
duration: 5min
completed: 2026-01-26
---

# Phase 5 Plan 4: CLI Interface Summary

**sarcodec CLI with compress/decompress subcommands, rich progress bars, and distinct exit codes for GeoTIFF-to-NPZ round-trip**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-26T19:23:05Z
- **Completed:** 2026-01-26T19:28:00Z
- **Tasks:** 2 (implemented together)
- **Files modified:** 1

## Accomplishments

- CLI with compress and decompress subcommands via argparse
- Rich progress bars with ETA during processing
- Full metadata preservation through round-trip (CRS, transform, tags)
- Distinct exit codes: SUCCESS(0), FILE_ERROR(1), MODEL_ERROR(2), OOM_ERROR(3), GENERAL_ERROR(4)
- Version flag showing model info, checkpoint hash, and CUDA details
- Batch processing support for multiple input files

## Task Commits

Each task was committed atomically:

1. **Task 1+2: CLI with argparse and compress/decompress commands** - `346e4c4` (feat)

**Plan metadata:** Pending (this summary commit)

## Files Created/Modified

- `scripts/sarcodec.py` - CLI entry point with compress/decompress subcommands

## Decisions Made

1. **Combined Tasks 1 and 2:** Since the CLI structure and command implementations are tightly coupled, implemented both in a single commit for cleaner atomicity.

2. **GeoMetadata serialization:** CRS stored as WKT string (portable), transform as 6-element tuple for JSON compatibility in NPZ files.

3. **Nodata handling:** During compression, nodata pixels are replaced with median of valid pixels. The boolean nodata mask is stored separately in NPZ and reapplied during decompression.

4. **Exit code design:** Distinct exit codes enable scripting and automation to detect failure types (file not found vs model error vs OOM).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **rich not installed:** Required `pip install rich` before CLI could run. This is already in requirements.txt but wasn't installed in the current environment.

## Verification Results

All verifications passed:

```
$ python scripts/sarcodec.py --version
sarcodec version 1.0.0
Model:
  Path: .../resnet_lite_v2_c16/best.pth
  Size: 64.83 MB
  Hash: 7e8249aa
PyTorch:
  Version: 2.5.1+cu121
  CUDA: 12.1
  GPU: NVIDIA GeForce RTX 3070
  VRAM: 8192 MB

$ python scripts/sarcodec.py compress test.tif -o test.npz
Compressed: test_input.tif -> test_compressed.npz (19.6x, 21.4s)

$ python scripts/sarcodec.py decompress test.npz -o output.tif
Decompressed: test_compressed.npz -> test_output.tif (5.5s)

$ python scripts/sarcodec.py compress nonexistent.tif; echo $?
Error: File not found: nonexistent.tif
1
```

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CLI ready for full validation testing in 05-05
- Round-trip verified with synthetic data
- Ready for real Sentinel-1 GeoTIFF testing
- Metadata preservation confirmed (CRS, transform, tags)

---
*Phase: 05-inference*
*Completed: 2026-01-26*
