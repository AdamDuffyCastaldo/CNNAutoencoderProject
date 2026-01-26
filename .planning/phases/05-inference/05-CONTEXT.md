# Phase 5: Full Image Inference - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement tiled inference with blending that processes complete Sentinel-1 scenes (10000x10000+ pixels) without visible seams or memory issues. Support raw GeoTIFF input for end-to-end compression pipeline, preserving geospatial metadata.

</domain>

<decisions>
## Implementation Decisions

### Tile Overlap & Blending
- 64 pixel overlap (25%) as default — good balance of quality and speed
- Overlap size configurable via CLI flag for power users
- Cosine ramp blending for smooth tile transitions
- Skip problematic tiles (NaN, extreme values) and fill with original input — graceful degradation
- Allow GPU-optimized tile processing order (non-deterministic) for performance
- Pad small images (< 256x256) to tile size with reflection — accept any input size
- Optional blending weight map visualization for debugging (CLI flag)

### CLI Interface Design
- Single script with subcommands: `sarcodec compress` / `sarcodec decompress`
- Progress bar with ETA during processing (rich progress)
- Support stdin/stdout for piping workflows
- NumPy .npz format for compressed output
- Optional output path with auto-naming: `input.tif` → `input.npz` if `-o` omitted
- Default model path (`best.pth` in standard location) with `--model` override
- Print compression ratio and estimated metrics summary after completion
- Accept multiple input files: `sarcodec compress *.tif`
- Detailed exit codes (0=success, 1=file error, 2=model error, 3=OOM, etc.)
- Rich help output with usage examples
- `--version` flag showing tool version, model version, checkpoint hash

### Memory Management
- Auto-detect GPU memory and adapt batch size accordingly
- Adaptive tile batching based on available VRAM — maximize throughput
- Full output image assembled in RAM (sufficient for expected image sizes)
- CPU fallback with warning when no GPU available
- Auto-enable AMP (FP16) on GPUs with Tensor Core support, FP32 otherwise
- Pre-validate input file (format, size) before processing

### GeoTIFF Handling
- Preserve full metadata: CRS, transform, nodata, description, tags
- Nodata pixels pass through unchanged — mask during inference, preserve in output
- Configurable TIFF compression via `--tiff-compress` flag (default: LZW)
- Prompt user for CRS if input has missing/invalid coordinate reference system
- Process all bands independently for multi-band GeoTIFFs, reassemble in output
- Configurable output data type via `--output-dtype` flag
- Optional Cloud Optimized GeoTIFF (COG) output via `--cog` flag

### Claude's Discretion
- Edge padding strategy (will use reflect padding)
- OOM recovery strategy (reduce batch and retry vs fail with message)
- NPZ metadata inclusion (whether to embed geo metadata for standalone decompression)
- Exact exit code assignments
- Progress bar library choice (tqdm vs rich)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for tiled inference and CLI design.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-inference*
*Context gathered: 2026-01-26*
