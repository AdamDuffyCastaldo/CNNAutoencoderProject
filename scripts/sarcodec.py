#!/usr/bin/env python
"""
SAR Image Codec CLI

Command-line interface for compressing and decompressing SAR images
using a trained autoencoder model.

Usage:
    sarcodec compress INPUT [OPTIONS]
    sarcodec decompress INPUT [OPTIONS]
    sarcodec --version

Examples:
    # Compress a GeoTIFF
    sarcodec compress sentinel1.tif -o compressed.npz

    # Decompress back to GeoTIFF
    sarcodec decompress compressed.npz -o reconstructed.tif

    # Batch compress multiple files
    sarcodec compress *.tif --model custom_model.pth
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console

from src.inference import SARCompressor
from src.inference.geotiff import (
    GeoMetadata,
    read_geotiff,
    write_geotiff,
    write_cog,
    create_nodata_mask,
    apply_nodata_mask,
)

# Version info
__version__ = "1.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FILE_ERROR = 1
EXIT_MODEL_ERROR = 2
EXIT_OOM_ERROR = 3
EXIT_GENERAL_ERROR = 4

# Default paths
DEFAULT_MODEL_PATH = str(
    PROJECT_ROOT / "notebooks" / "checkpoints" / "resnet_lite_v2_c16" / "best.pth"
)

# Console for rich output
console = Console()


def get_checkpoint_hash(model_path: str, bytes_to_read: int = 1024) -> str:
    """Get MD5 hash of first N bytes of checkpoint file."""
    try:
        with open(model_path, 'rb') as f:
            data = f.read(bytes_to_read)
        return hashlib.md5(data).hexdigest()[:8]
    except Exception:
        return "unknown"


def show_version(model_path: str) -> None:
    """Print version information."""
    import torch

    console.print(f"[bold]sarcodec[/bold] version {__version__}")
    console.print()

    # Model info
    console.print("[bold]Model:[/bold]")
    if os.path.exists(model_path):
        checkpoint_hash = get_checkpoint_hash(model_path)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        console.print(f"  Path: {model_path}")
        console.print(f"  Size: {file_size_mb:.2f} MB")
        console.print(f"  Hash: {checkpoint_hash}")
    else:
        console.print(f"  Path: {model_path} [red](not found)[/red]")

    console.print()

    # PyTorch info
    console.print("[bold]PyTorch:[/bold]")
    console.print(f"  Version: {torch.__version__}")
    if torch.cuda.is_available():
        console.print(f"  CUDA: {torch.version.cuda}")
        console.print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        console.print(f"  VRAM: {vram_mb:.0f} MB")
    else:
        console.print("  CUDA: not available")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    # Main parser
    parser = argparse.ArgumentParser(
        prog="sarcodec",
        description="SAR Image Compression/Decompression Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sarcodec compress image.tif                    Compress to image.npz
  sarcodec compress image.tif -o out.npz         Compress with custom output name
  sarcodec compress *.tif                        Batch compress multiple files
  sarcodec decompress data.npz                   Decompress to data.tif
  sarcodec decompress data.npz --cog             Output as Cloud Optimized GeoTIFF
  sarcodec --version                             Show version and model info
        """
    )

    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version, model info, and PyTorch/CUDA details"
    )

    # Subparsers
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Compress subcommand
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress GeoTIFF to NPZ format",
        description="Compress SAR GeoTIFF images using a trained autoencoder model."
    )
    compress_parser.add_argument(
        "input",
        nargs="+",
        help="Input GeoTIFF file(s)"
    )
    compress_parser.add_argument(
        "-o", "--output",
        help="Output NPZ file path (default: input name with .npz extension)"
    )
    compress_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Model checkpoint path (default: {DEFAULT_MODEL_PATH})"
    )
    compress_parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Tile overlap in pixels (default: 64)"
    )
    compress_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tiles per batch (default: auto-detect based on GPU memory)"
    )

    # Decompress subcommand
    decompress_parser = subparsers.add_parser(
        "decompress",
        help="Decompress NPZ to GeoTIFF format",
        description="Decompress NPZ files back to GeoTIFF format with georeferencing."
    )
    decompress_parser.add_argument(
        "input",
        nargs="+",
        help="Input NPZ file(s)"
    )
    decompress_parser.add_argument(
        "-o", "--output",
        help="Output GeoTIFF file path (default: input name with .tif extension)"
    )
    decompress_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Model checkpoint path (default: {DEFAULT_MODEL_PATH})"
    )
    decompress_parser.add_argument(
        "--tiff-compress",
        choices=["lzw", "deflate", "none"],
        default="lzw",
        help="TIFF compression method (default: lzw)"
    )
    decompress_parser.add_argument(
        "--cog",
        action="store_true",
        help="Output as Cloud Optimized GeoTIFF"
    )

    return parser


def create_progress_callback(
    progress: Progress,
    task_id: Any
) -> Callable[[int, int], None]:
    """Create a progress callback for the compressor."""
    def callback(current: int, total: int) -> None:
        progress.update(task_id, completed=current, total=total)
    return callback


def serialize_geo_metadata(metadata: GeoMetadata) -> Dict:
    """Serialize GeoMetadata to JSON-compatible dict."""
    result = {
        "nodata": metadata.nodata,
        "dtype": metadata.dtype,
        "count": metadata.count,
        "width": metadata.width,
        "height": metadata.height,
        "tags": metadata.tags,
        "descriptions": list(metadata.descriptions) if metadata.descriptions else None,
    }

    # Serialize CRS as WKT string if present
    if metadata.crs is not None:
        result["crs_wkt"] = metadata.crs.to_wkt()
    else:
        result["crs_wkt"] = None

    # Serialize transform as tuple if present
    if metadata.transform is not None:
        # Affine has 6 parameters: a, b, c, d, e, f
        result["transform"] = tuple(metadata.transform)[:6]
    else:
        result["transform"] = None

    return result


def deserialize_geo_metadata(data: Dict) -> GeoMetadata:
    """Deserialize JSON dict to GeoMetadata."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    # Deserialize CRS
    crs = None
    if data.get("crs_wkt"):
        crs = CRS.from_wkt(data["crs_wkt"])

    # Deserialize transform
    transform = None
    if data.get("transform"):
        transform = Affine(*data["transform"])

    return GeoMetadata(
        crs=crs,
        transform=transform,
        nodata=data.get("nodata"),
        dtype=data.get("dtype", "float32"),
        count=data.get("count", 1),
        width=data.get("width", 0),
        height=data.get("height", 0),
        tags=data.get("tags", {}),
        descriptions=tuple(data["descriptions"]) if data.get("descriptions") else None,
    )


def compress_file(
    input_path: str,
    output_path: str,
    compressor: SARCompressor,
    progress: Progress
) -> Dict:
    """
    Compress a single GeoTIFF file.

    Returns:
        Dictionary with compression statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read GeoTIFF
    task_read = progress.add_task(
        f"[cyan]Reading {input_path.name}...",
        total=None
    )
    data, geo_metadata = read_geotiff(input_path)
    progress.update(task_read, completed=1, total=1)
    progress.remove_task(task_read)

    # Get nodata mask if applicable
    nodata_mask = None
    if geo_metadata.nodata is not None:
        nodata_mask = create_nodata_mask(data, geo_metadata.nodata)
        # Replace nodata with median for compression
        if nodata_mask.any():
            valid_median = np.median(data[~nodata_mask])
            data = data.copy()
            data[nodata_mask] = valid_median

    # Ensure 2D (single band)
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data.squeeze(0)
        else:
            console.print(
                f"[yellow]Warning: Multi-band image. Using first band only.[/yellow]"
            )
            data = data[0]

    # Compress with progress bar
    task_compress = progress.add_task(
        f"[green]Compressing...",
        total=100  # Will be updated by callback
    )

    start_time = time.time()
    latent, tile_metadata = compressor.compress(
        data,
        progress_callback=create_progress_callback(progress, task_compress)
    )
    compress_time = time.time() - start_time

    progress.remove_task(task_compress)

    # Build full metadata
    full_metadata = {
        "geo_metadata": serialize_geo_metadata(geo_metadata),
        "tile_metadata": tile_metadata,
        "original_dtype": str(data.dtype),
        "sarcodec_version": __version__,
    }

    # Include nodata mask if present
    if nodata_mask is not None and nodata_mask.any():
        full_metadata["has_nodata_mask"] = True

    # Save to NPZ
    task_save = progress.add_task(
        f"[cyan]Saving {output_path.name}...",
        total=None
    )

    # Save arrays and metadata
    save_dict = {
        "latent": latent,
        "metadata": json.dumps(full_metadata),
    }
    if nodata_mask is not None and nodata_mask.any():
        save_dict["nodata_mask"] = nodata_mask.astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    progress.update(task_save, completed=1, total=1)
    progress.remove_task(task_save)

    # Calculate stats
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    compression_ratio = input_size / output_size if output_size > 0 else 0

    stats = compressor.get_compression_stats(data, latent)

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_size_mb": input_size / (1024 * 1024),
        "output_size_mb": output_size / (1024 * 1024),
        "file_compression_ratio": compression_ratio,
        "data_compression_ratio": stats["compression_ratio"],
        "bpp": stats["bpp"],
        "time_seconds": compress_time,
        "n_tiles": stats["n_tiles"],
        "original_shape": stats["original_shape"],
        "latent_shape": stats["latent_shape"],
    }


def decompress_file(
    input_path: str,
    output_path: str,
    compressor: SARCompressor,
    progress: Progress,
    tiff_compress: str = "lzw",
    use_cog: bool = False
) -> Dict:
    """
    Decompress a single NPZ file.

    Returns:
        Dictionary with decompression statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load NPZ
    task_load = progress.add_task(
        f"[cyan]Loading {input_path.name}...",
        total=None
    )

    npz_data = np.load(input_path, allow_pickle=True)
    latent = npz_data["latent"]
    metadata = json.loads(str(npz_data["metadata"]))

    # Load nodata mask if present
    nodata_mask = None
    if "nodata_mask" in npz_data:
        nodata_mask = npz_data["nodata_mask"].astype(bool)

    progress.update(task_load, completed=1, total=1)
    progress.remove_task(task_load)

    # Extract metadata components
    geo_metadata = deserialize_geo_metadata(metadata["geo_metadata"])
    tile_metadata = metadata["tile_metadata"]

    # Convert tile_metadata lists back to tuples
    if "grid_shape" in tile_metadata:
        tile_metadata["grid_shape"] = tuple(tile_metadata["grid_shape"])
    if "original_shape" in tile_metadata:
        tile_metadata["original_shape"] = tuple(tile_metadata["original_shape"])
    if "padded_shape" in tile_metadata:
        tile_metadata["padded_shape"] = tuple(tile_metadata["padded_shape"])
    if "padding" in tile_metadata:
        tile_metadata["padding"] = tuple(tuple(p) for p in tile_metadata["padding"])

    # Decompress with progress bar
    task_decompress = progress.add_task(
        f"[green]Decompressing...",
        total=100
    )

    start_time = time.time()
    reconstructed = compressor.decompress(
        latent,
        tile_metadata,
        progress_callback=create_progress_callback(progress, task_decompress)
    )
    decompress_time = time.time() - start_time

    progress.remove_task(task_decompress)

    # Apply nodata mask if present
    if nodata_mask is not None and geo_metadata.nodata is not None:
        reconstructed = apply_nodata_mask(
            reconstructed, nodata_mask, geo_metadata.nodata
        )

    # Write output
    task_write = progress.add_task(
        f"[cyan]Writing {output_path.name}...",
        total=None
    )

    # Convert compression option
    compress_opt = None if tiff_compress == "none" else tiff_compress

    if use_cog:
        write_cog(reconstructed, geo_metadata, output_path)
    else:
        write_geotiff(reconstructed, geo_metadata, output_path, compress=compress_opt)

    progress.update(task_write, completed=1, total=1)
    progress.remove_task(task_write)

    output_size = output_path.stat().st_size

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "output_size_mb": output_size / (1024 * 1024),
        "time_seconds": decompress_time,
        "output_shape": reconstructed.shape,
        "is_cog": use_cog,
    }


def compress_command(args: argparse.Namespace) -> int:
    """Execute the compress command."""
    # Validate input files exist
    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if path.exists():
            input_files.append(path)
        else:
            console.print(f"[red]Error: File not found: {pattern}[/red]")
            return EXIT_FILE_ERROR

    if not input_files:
        console.print("[red]Error: No input files specified[/red]")
        return EXIT_FILE_ERROR

    # Validate model exists
    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model not found: {args.model}[/red]")
        return EXIT_MODEL_ERROR

    # Load model
    try:
        console.print(f"[dim]Loading model: {args.model}[/dim]")
        compressor = SARCompressor(
            model_path=args.model,
            overlap=args.overlap,
            batch_size=args.batch_size
        )
        console.print(
            f"[dim]Device: {compressor.device}, "
            f"Batch size: {compressor.batch_size}[/dim]"
        )
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return EXIT_MODEL_ERROR

    # Process files
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for i, input_file in enumerate(input_files):
            # Determine output path
            if args.output and len(input_files) == 1:
                output_path = args.output
            else:
                output_path = input_file.with_suffix(".npz")

            try:
                stats = compress_file(
                    str(input_file),
                    str(output_path),
                    compressor,
                    progress
                )
                results.append(stats)

                # Print summary for this file
                console.print(
                    f"[green]Compressed:[/green] {input_file.name} -> "
                    f"{Path(output_path).name} "
                    f"({stats['file_compression_ratio']:.1f}x, "
                    f"{stats['time_seconds']:.1f}s)"
                )

            except FileNotFoundError as e:
                console.print(f"[red]File error: {e}[/red]")
                return EXIT_FILE_ERROR
            except torch.cuda.OutOfMemoryError:
                console.print("[red]GPU out of memory. Try --batch-size 1[/red]")
                return EXIT_OOM_ERROR
            except Exception as e:
                console.print(f"[red]Error processing {input_file}: {e}[/red]")
                traceback.print_exc()
                return EXIT_GENERAL_ERROR

    # Print overall summary
    if len(results) > 1:
        total_input = sum(r["input_size_mb"] for r in results)
        total_output = sum(r["output_size_mb"] for r in results)
        total_time = sum(r["time_seconds"] for r in results)
        console.print()
        console.print(
            f"[bold]Summary:[/bold] {len(results)} files, "
            f"{total_input:.2f} MB -> {total_output:.2f} MB "
            f"({total_input/total_output:.1f}x), "
            f"{total_time:.1f}s total"
        )

    return EXIT_SUCCESS


def decompress_command(args: argparse.Namespace) -> int:
    """Execute the decompress command."""
    # Validate input files exist
    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if path.exists():
            input_files.append(path)
        else:
            console.print(f"[red]Error: File not found: {pattern}[/red]")
            return EXIT_FILE_ERROR

    if not input_files:
        console.print("[red]Error: No input files specified[/red]")
        return EXIT_FILE_ERROR

    # Validate model exists
    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model not found: {args.model}[/red]")
        return EXIT_MODEL_ERROR

    # Load model
    try:
        console.print(f"[dim]Loading model: {args.model}[/dim]")
        compressor = SARCompressor(model_path=args.model)
        console.print(
            f"[dim]Device: {compressor.device}, "
            f"Batch size: {compressor.batch_size}[/dim]"
        )
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return EXIT_MODEL_ERROR

    # Process files
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for i, input_file in enumerate(input_files):
            # Determine output path
            if args.output and len(input_files) == 1:
                output_path = args.output
            else:
                output_path = input_file.with_suffix(".tif")

            try:
                stats = decompress_file(
                    str(input_file),
                    str(output_path),
                    compressor,
                    progress,
                    tiff_compress=args.tiff_compress,
                    use_cog=args.cog
                )
                results.append(stats)

                # Print summary for this file
                cog_note = " (COG)" if stats["is_cog"] else ""
                console.print(
                    f"[green]Decompressed:[/green] {input_file.name} -> "
                    f"{Path(output_path).name}{cog_note} "
                    f"({stats['time_seconds']:.1f}s)"
                )

            except FileNotFoundError as e:
                console.print(f"[red]File error: {e}[/red]")
                return EXIT_FILE_ERROR
            except torch.cuda.OutOfMemoryError:
                console.print("[red]GPU out of memory.[/red]")
                return EXIT_OOM_ERROR
            except Exception as e:
                console.print(f"[red]Error processing {input_file}: {e}[/red]")
                traceback.print_exc()
                return EXIT_GENERAL_ERROR

    # Print overall summary
    if len(results) > 1:
        total_output = sum(r["output_size_mb"] for r in results)
        total_time = sum(r["time_seconds"] for r in results)
        console.print()
        console.print(
            f"[bold]Summary:[/bold] {len(results)} files, "
            f"{total_output:.2f} MB total output, "
            f"{total_time:.1f}s total"
        )

    return EXIT_SUCCESS


def main() -> int:
    """Main entry point."""
    # Import torch here to allow --help to work without GPU
    global torch
    import torch

    parser = create_parser()
    args = parser.parse_args()

    # Handle --version flag
    if args.version:
        model_path = DEFAULT_MODEL_PATH
        show_version(model_path)
        return EXIT_SUCCESS

    # If no command, show help
    if args.command is None:
        parser.print_help()
        return EXIT_SUCCESS

    # Route to appropriate command
    if args.command == "compress":
        return compress_command(args)
    elif args.command == "decompress":
        return decompress_command(args)
    else:
        parser.print_help()
        return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
