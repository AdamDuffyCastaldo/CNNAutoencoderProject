# Phase 5: Full Image Inference - Research

**Researched:** 2026-01-26
**Domain:** Tiled inference, GeoTIFF I/O, CLI design, GPU memory management
**Confidence:** HIGH

## Summary

This phase implements tiled inference with blending for processing complete Sentinel-1 scenes (10000x10000+ pixels) without memory issues or visible seams. The research covers four interconnected domains:

1. **Tiled Inference with Blending** - Processing large images by splitting into overlapping tiles, running inference on each tile, and blending them back together using cosine ramp weights. Well-established patterns exist in the deep learning community with PyTorch-native solutions.

2. **GeoTIFF I/O with Metadata Preservation** - Using rasterio for reading/writing GeoTIFFs while preserving CRS, transform, nodata values, and other geospatial metadata. rio-cogeo provides Cloud Optimized GeoTIFF output capability.

3. **CLI Interface Design** - Python's argparse with rich library for progress bars and formatted output. Subcommand pattern (`sarcodec compress`, `sarcodec decompress`) is well-supported.

4. **GPU Memory Management** - PyTorch provides APIs for detecting VRAM capacity and monitoring usage, enabling adaptive batch sizing during tiled inference.

**Primary recommendation:** Build custom tiling logic using NumPy/PyTorch rather than external tiling libraries (simpler dependency management, full control over blending). Use rasterio for GeoTIFF I/O and rich for CLI progress display.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rasterio | 1.3+ | GeoTIFF read/write with metadata | Industry standard for geospatial raster I/O, already in project |
| rio-cogeo | 5.4+ | Cloud Optimized GeoTIFF creation | Official COG creation tool, validates output |
| rich | 14.1+ | CLI progress bars, formatted output | Modern, beautiful terminal output, active development |
| numpy | existing | Array tiling, blending math | Already in project, no new dependency |
| torch | existing | GPU inference with AMP | Already in project for model inference |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| argparse | stdlib | CLI argument parsing | Subcommand structure, help generation |
| hashlib | stdlib | Checkpoint hash for --version | MD5 of model weights for version tracking |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rich | tqdm | tqdm simpler but less pretty; rich already covers progress + formatted output |
| Custom tiling | blended-tiling PyPI | External dependency; custom code gives full control over blend weights |
| Custom tiling | tiler PyPI | N-dimensional but overkill; our use case is 2D images only |
| argparse | click/typer | More features but adds dependency; argparse sufficient for our needs |

**Installation:**
```bash
pip install rich rio-cogeo
```

Note: rasterio already in requirements.txt; rio-cogeo adds COG creation capability.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── inference/
│   ├── __init__.py
│   ├── compressor.py      # Core SARCompressor class (extend existing stub)
│   ├── tiling.py          # Tile extraction, blending, reconstruction
│   └── geotiff.py         # GeoTIFF I/O with metadata preservation
scripts/
├── sarcodec.py            # CLI entry point with subcommands
```

### Pattern 1: Cosine Ramp Blending Weights

**What:** Weight function that smoothly transitions from 0 at tile edges to 1 at tile center in overlap regions.

**When to use:** For all tile boundary blending to eliminate visible seams.

**Formula:**
```python
def create_cosine_ramp_weights(tile_size: int, overlap: int) -> np.ndarray:
    """
    Create 2D blending weights with cosine ramp in overlap regions.

    Center of tile = 1.0, edges in overlap region = cosine ramp to 0.
    """
    weights = np.ones((tile_size, tile_size), dtype=np.float32)

    # Create 1D cosine ramp for overlap region
    ramp = np.linspace(0, np.pi / 2, overlap)
    ramp = np.sin(ramp) ** 2  # Cosine-squared ramp (0 -> 1)

    # Apply ramp to all four edges
    # Top edge
    weights[:overlap, :] *= ramp[:, np.newaxis]
    # Bottom edge
    weights[-overlap:, :] *= ramp[::-1, np.newaxis]
    # Left edge
    weights[:, :overlap] *= ramp[np.newaxis, :]
    # Right edge
    weights[:, -overlap:] *= ramp[np.newaxis, ::-1]

    return weights
```

**Why cosine-squared:** Smooth derivative at boundaries (unlike linear), sums to 1.0 when overlapping regions are added together.

### Pattern 2: Tile Grid Extraction with Reflection Padding

**What:** Extract tiles with configurable overlap, padding image edges with reflection.

**When to use:** Before inference on full image.

**Example:**
```python
def extract_tiles(
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64
) -> Tuple[np.ndarray, Dict]:
    """
    Extract overlapping tiles from image with edge padding.

    Args:
        image: 2D numpy array (H, W)
        tile_size: Tile dimension
        overlap: Overlap in pixels

    Returns:
        tiles: (N, tile_size, tile_size) array
        metadata: Dict with grid_shape, original_shape, padding
    """
    stride = tile_size - overlap
    h, w = image.shape

    # Compute padding needed
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride

    # Pad with reflection (handles edges naturally for SAR)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Extract tiles
    tiles = []
    n_rows = (padded.shape[0] - overlap) // stride
    n_cols = (padded.shape[1] - overlap) // stride

    for i in range(n_rows):
        for j in range(n_cols):
            y = i * stride
            x = j * stride
            tile = padded[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)

    metadata = {
        'grid_shape': (n_rows, n_cols),
        'original_shape': (h, w),
        'padded_shape': padded.shape,
        'padding': (pad_h, pad_w),
        'tile_size': tile_size,
        'overlap': overlap
    }

    return np.array(tiles), metadata
```

### Pattern 3: Weighted Tile Reconstruction

**What:** Reconstruct full image from processed tiles using weighted blending.

**When to use:** After running inference on all tiles.

**Example:**
```python
def reconstruct_from_tiles(
    tiles: np.ndarray,
    metadata: Dict,
    blend_weights: np.ndarray
) -> np.ndarray:
    """
    Reconstruct image from overlapping tiles with blending.

    Args:
        tiles: (N, tile_size, tile_size) processed tiles
        metadata: From extract_tiles()
        blend_weights: (tile_size, tile_size) blending weights

    Returns:
        Reconstructed image at original size
    """
    grid_shape = metadata['grid_shape']
    padded_shape = metadata['padded_shape']
    original_shape = metadata['original_shape']
    tile_size = metadata['tile_size']
    stride = tile_size - metadata['overlap']

    # Initialize accumulator and weight sum
    output = np.zeros(padded_shape, dtype=np.float32)
    weight_sum = np.zeros(padded_shape, dtype=np.float32)

    idx = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            y = i * stride
            x = j * stride

            output[y:y+tile_size, x:x+tile_size] += tiles[idx] * blend_weights
            weight_sum[y:y+tile_size, x:x+tile_size] += blend_weights
            idx += 1

    # Normalize by weight sum (avoid division by zero)
    output = np.divide(output, weight_sum, where=weight_sum > 0)

    # Crop to original size
    return output[:original_shape[0], :original_shape[1]]
```

### Pattern 4: GeoTIFF Metadata Preservation

**What:** Read GeoTIFF with full metadata, process, write back preserving georeferencing.

**When to use:** For all GeoTIFF operations.

**Example:**
```python
import rasterio
from rasterio.crs import CRS

def read_geotiff_with_metadata(path: str) -> Tuple[np.ndarray, Dict]:
    """Read GeoTIFF and extract all metadata for preservation."""
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)

        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'nodata': src.nodata,
            'dtype': src.dtypes[0],
            'count': src.count,
            'width': src.width,
            'height': src.height,
            'driver': 'GTiff',
            'description': src.descriptions,
            'tags': src.tags(),
        }

    return data, metadata

def write_geotiff_with_metadata(
    data: np.ndarray,
    metadata: Dict,
    path: str,
    compress: str = 'lzw'
) -> None:
    """Write GeoTIFF preserving metadata from read operation."""
    # Update dimensions for potentially different output
    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'width': data.shape[-1],
        'height': data.shape[-2],
        'count': data.shape[0] if data.ndim == 3 else 1,
        'crs': metadata['crs'],
        'transform': metadata['transform'],
        'nodata': metadata.get('nodata'),
        'compress': compress,
    }

    with rasterio.open(path, 'w', **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)

        # Restore tags
        if metadata.get('tags'):
            dst.update_tags(**metadata['tags'])
```

### Pattern 5: Adaptive GPU Batch Size

**What:** Auto-detect VRAM and set batch size for tile processing.

**When to use:** At initialization of compressor, before processing tiles.

**Example:**
```python
import torch

def get_optimal_batch_size(
    model: torch.nn.Module,
    tile_size: int = 256,
    latent_channels: int = 16,
    safety_factor: float = 0.7
) -> int:
    """
    Estimate optimal batch size based on available VRAM.

    Args:
        model: The autoencoder model
        tile_size: Tile dimension
        latent_channels: Latent space channels
        safety_factor: Fraction of free VRAM to use (default 70%)

    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 1  # CPU: conservative batch size

    # Get device properties
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory

    # Get currently allocated
    allocated = torch.cuda.memory_allocated(0)
    free_vram = total_vram - allocated

    # Estimate memory per tile (rough heuristic)
    # Input: tile_size^2 * 4 bytes (float32)
    # Intermediate activations: ~10x input (conservative)
    # Output: same as input
    bytes_per_tile = tile_size * tile_size * 4 * 12  # ~3MB for 256x256

    # Calculate batch size
    usable_vram = free_vram * safety_factor
    batch_size = int(usable_vram / bytes_per_tile)

    # Clamp to reasonable range
    return max(1, min(batch_size, 64))
```

### Pattern 6: AMP Inference

**What:** Use automatic mixed precision for faster inference on supported GPUs.

**When to use:** Always for CUDA inference on GPUs with Tensor Cores.

**Example:**
```python
@torch.inference_mode()
def process_tiles_batched(
    model: torch.nn.Module,
    tiles: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool = True
) -> np.ndarray:
    """
    Process tiles through model with batching and optional AMP.

    Args:
        model: Encoder or full autoencoder
        tiles: (N, H, W) numpy array of tiles
        batch_size: Tiles per batch
        device: torch device
        use_amp: Whether to use FP16 inference

    Returns:
        Processed tiles as numpy array
    """
    model.eval()
    outputs = []

    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i+batch_size]

        # Add channel dimension: (B, H, W) -> (B, 1, H, W)
        x = torch.from_numpy(batch[:, np.newaxis, :, :]).to(device)

        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(x)
        else:
            out = model(x)

        # Handle tuple output (x_hat, z) vs single output
        if isinstance(out, tuple):
            out = out[0]  # Get reconstructed image

        outputs.append(out.cpu().numpy())

    return np.concatenate(outputs, axis=0)
```

### Anti-Patterns to Avoid

- **Loading full 10000x10000 image to GPU:** Will cause OOM. Always tile first.
- **Non-overlapping tiles:** Will show visible seams at tile boundaries.
- **Linear blending weights:** Visible discontinuities at overlap edges. Use cosine ramp.
- **Ignoring nodata pixels:** Nodata regions should not influence blending. Mask them.
- **Hardcoded batch sizes:** Different GPUs have different VRAM. Auto-detect.
- **FP32 inference on Tensor Core GPUs:** Wastes performance. Use AMP.
- **Processing tiles in sequential order:** GPU can't parallelize. Use batched processing.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GeoTIFF I/O | Custom TIFF reader | rasterio | Handles all TIFF variants, CRS, transforms |
| COG creation | Manual TIFF structure | rio-cogeo | COG spec is complex, validation matters |
| Progress bars | print statements | rich.progress | ETA, multiple tasks, beautiful output |
| CLI parsing | manual sys.argv | argparse | Subcommands, help, type validation |
| CRS handling | String parsing | pyproj via rasterio | EPSG codes, WKT, projections |
| Mixed precision | Manual casting | torch.autocast | Handles op-level dtype selection |

**Key insight:** GeoTIFF and COG specifications are complex with many edge cases. Rasterio and rio-cogeo handle these correctly; custom implementations will have subtle bugs with certain files.

## Common Pitfalls

### Pitfall 1: Blend Weight Accumulation Overflow

**What goes wrong:** When many tiles overlap at corners, weight sums can become very large, causing numerical issues.

**Why it happens:** Each corner pixel may be covered by 4 tiles with non-zero weights.

**How to avoid:** Always normalize by weight sum: `output / weight_sum`. Use `np.divide(output, weight_sum, where=weight_sum > 0)` to avoid division by zero.

**Warning signs:** Very bright or dark artifacts at tile corners.

### Pitfall 2: Edge Padding Mode Selection

**What goes wrong:** Zero padding creates dark artifacts at image boundaries. Constant padding creates visible rectangles.

**Why it happens:** SAR images don't have natural black/white borders.

**How to avoid:** Use `mode='reflect'` for padding, which mirrors edge pixels naturally.

**Warning signs:** Dark/bright borders around the reconstructed image.

### Pitfall 3: Nodata Value Propagation

**What goes wrong:** Nodata regions (NaN, 0, or specific value) participate in blending, creating artifacts.

**Why it happens:** Model processes nodata as real data, blending mixes valid and invalid pixels.

**How to avoid:**
1. Track nodata mask before processing
2. Replace nodata with interpolated/padded values for inference
3. Restore nodata mask in final output

**Warning signs:** Strange values or gradients near image edges or gaps.

### Pitfall 4: CRS Loss During Processing

**What goes wrong:** Output GeoTIFF cannot be overlaid correctly in GIS software.

**Why it happens:** CRS/transform not properly copied from input to output.

**How to avoid:** Extract full metadata at read time, pass through processing, apply at write time. Use rasterio's profile copy pattern.

**Warning signs:** Output file opens but has wrong location or "Unknown CRS".

### Pitfall 5: Memory Leak in Tile Loop

**What goes wrong:** VRAM usage grows during processing, eventually OOM.

**Why it happens:** PyTorch tensors not properly released, CUDA cache not cleared.

**How to avoid:**
1. Use `@torch.inference_mode()` (more efficient than `@torch.no_grad()`)
2. Explicitly delete tensors and call `torch.cuda.empty_cache()` periodically
3. Process tiles in batches, not one huge list

**Warning signs:** Increasing VRAM usage reported by `nvidia-smi` during run.

### Pitfall 6: Incorrect Tile Count Calculation

**What goes wrong:** Last row/column of tiles missing or out of bounds access.

**Why it happens:** Off-by-one errors in grid calculation with overlap.

**How to avoid:**
- Number of tiles = `ceil((image_size - overlap) / stride)` where `stride = tile_size - overlap`
- Pad image first to ensure all tiles are full-size

**Warning signs:** Index out of bounds error, or missing strip at bottom/right of output.

### Pitfall 7: NPZ Metadata Not Preserved for Decompression

**What goes wrong:** Compressed .npz file cannot be decompressed standalone (missing geo metadata).

**Why it happens:** Only latent representation saved, not the metadata needed to write GeoTIFF output.

**How to avoid:** Save to NPZ:
- `latent`: Compressed latent representation
- `metadata`: JSON string with CRS, transform, nodata, original_shape, tile_metadata

**Warning signs:** Decompression works but output has no georeferencing.

## Code Examples

Verified patterns from official sources and established projects:

### Rich Progress Bar for Tile Processing

```python
# Source: https://rich.readthedocs.io/en/latest/progress.html
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

def process_with_progress(tiles, model, batch_size):
    """Process tiles with rich progress bar."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing tiles...", total=len(tiles))

        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i+batch_size]
            # Process batch...
            progress.advance(task, len(batch))
```

### argparse Subcommands Pattern

```python
# Source: https://docs.python.org/3/library/argparse.html
import argparse

def create_parser():
    """Create CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='sarcodec',
        description='SAR image compression using neural autoencoder'
    )
    parser.add_argument('--version', action='store_true', help='Show version info')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Compress subcommand
    compress_parser = subparsers.add_parser('compress', help='Compress SAR images')
    compress_parser.add_argument('input', nargs='+', help='Input GeoTIFF file(s)')
    compress_parser.add_argument('-o', '--output', help='Output path (default: auto)')
    compress_parser.add_argument('--model', help='Model checkpoint path')
    compress_parser.add_argument('--overlap', type=int, default=64, help='Tile overlap')

    # Decompress subcommand
    decompress_parser = subparsers.add_parser('decompress', help='Decompress SAR images')
    decompress_parser.add_argument('input', nargs='+', help='Input .npz file(s)')
    decompress_parser.add_argument('-o', '--output', help='Output path')
    decompress_parser.add_argument('--tiff-compress', default='lzw', help='TIFF compression')
    decompress_parser.add_argument('--cog', action='store_true', help='Output as COG')

    return parser
```

### COG Creation with rio-cogeo

```python
# Source: https://guide.cloudnativegeo.org/cloud-optimized-geotiffs/writing-cogs-in-python.html
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.io import MemoryFile

def write_as_cog(data: np.ndarray, metadata: Dict, output_path: str):
    """Write array as Cloud Optimized GeoTIFF."""
    profile = cog_profiles.get("deflate")

    # Create temporary GeoTIFF in memory
    mem_profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'width': data.shape[-1],
        'height': data.shape[-2],
        'count': 1 if data.ndim == 2 else data.shape[0],
        'crs': metadata['crs'],
        'transform': metadata['transform'],
    }

    with MemoryFile() as memfile:
        with memfile.open(**mem_profile) as mem:
            if data.ndim == 2:
                mem.write(data, 1)
            else:
                mem.write(data)

        # Translate to COG
        cog_translate(
            memfile,
            output_path,
            profile,
            in_memory=True,
            quiet=True,
        )
```

### VRAM Detection and Tensor Core Check

```python
# Source: https://docs.pytorch.org/docs/stable/cuda.html
import torch

def get_gpu_info() -> Dict:
    """Get GPU information for adaptive processing."""
    if not torch.cuda.is_available():
        return {'available': False, 'device': 'cpu'}

    props = torch.cuda.get_device_properties(0)

    # Check for Tensor Cores (compute capability >= 7.0)
    has_tensor_cores = props.major >= 7

    return {
        'available': True,
        'device': 'cuda',
        'name': props.name,
        'total_memory_gb': props.total_memory / (1024**3),
        'compute_capability': f"{props.major}.{props.minor}",
        'has_tensor_cores': has_tensor_cores,
        'use_amp': has_tensor_cores,  # Recommend AMP if Tensor Cores available
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.cuda.amp.autocast` | `torch.amp.autocast("cuda")` | PyTorch 2.0 | Device-agnostic AMP API |
| `@torch.no_grad()` | `@torch.inference_mode()` | PyTorch 1.9 | More efficient, disables version tracking |
| Manual COG structure | rio-cogeo 5.x | GDAL 3.1+ COG driver | Reliable COG validation |

**Deprecated/outdated:**
- `torch.cuda.amp.autocast()`: Deprecated, use `torch.amp.autocast(device_type="cuda")`
- `torch.cuda.amp.GradScaler()`: Not needed for inference (only training)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal overlap percentage for SAR**
   - What we know: 25% (64px for 256px tiles) is commonly used
   - What's unclear: Whether SAR speckle needs more/less overlap than natural images
   - Recommendation: Start with 64px (25%), make configurable, test empirically

2. **NPZ vs custom format for compressed output**
   - What we know: NPZ is simple, built-in, supports compression
   - What's unclear: Whether to embed geo metadata or keep sidecar JSON
   - Recommendation: Embed as JSON string in NPZ for self-contained files

3. **OOM recovery strategy**
   - What we know: Can catch OOM, reduce batch size, retry
   - What's unclear: Whether retry is worth the complexity vs just failing with good message
   - Recommendation: Implement retry with halved batch size, max 2 retries

## Sources

### Primary (HIGH confidence)
- [Rasterio Documentation](https://rasterio.readthedocs.io/en/stable/) - GeoTIFF I/O patterns, CRS handling
- [Rich Progress Documentation](https://rich.readthedocs.io/en/latest/progress.html) - Progress bar API
- [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html) - Autocast, inference mode
- [Python argparse Documentation](https://docs.python.org/3/library/argparse.html) - Subcommand pattern
- [rio-cogeo Documentation](https://cogeotiff.github.io/rio-cogeo/) - COG creation

### Secondary (MEDIUM confidence)
- [Cloud-Optimized Geospatial Formats Guide](https://guide.cloudnativegeo.org/cloud-optimized-geotiffs/writing-cogs-in-python.html) - COG Python patterns
- [blended-tiling GitHub](https://github.com/ProGamerGov/blended-tiling) - Tile blending concepts
- [PyTorch CUDA Memory Documentation](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) - VRAM monitoring

### Tertiary (LOW confidence)
- WebSearch results for cosine blending weights - Formula derived from general knowledge

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are well-documented, widely used
- Architecture patterns: HIGH - Tiling/blending is well-established technique
- Pitfalls: MEDIUM - Based on general deep learning experience, some SAR-specific items need validation
- Code examples: HIGH - Verified against official documentation

**Research date:** 2026-01-26
**Valid until:** 2026-02-26 (stable domain, 30 days)
