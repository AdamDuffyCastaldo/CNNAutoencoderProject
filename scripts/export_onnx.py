#!/usr/bin/env python
"""
CLI Script for ONNX Export of SAR Autoencoder Models

Export trained SAR autoencoder checkpoints to ONNX format for deployment
outside the PyTorch ecosystem.

Usage:
    # Export single model
    python scripts/export_onnx.py notebooks/checkpoints/resnet_c32_b64_cr8x_*/best.pth models/onnx/8x/

    # Export all ResNet models
    python scripts/export_onnx.py --all models/onnx/

    # Skip validation (faster)
    python scripts/export_onnx.py --no-validate checkpoint.pth output/

Example verification using checksums from metadata.json:
    python -c "import hashlib; print(hashlib.sha256(open('encoder.onnx','rb').read()).hexdigest())"
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.export.onnx_export import export_full_model, compute_file_checksum

console = Console()


def get_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """
    Extract model name from checkpoint path.

    Looks for patterns like:
    - resnet_c64_b64_cr4x_* -> resnet_4x
    - resnet_c32_b64_cr8x_* -> resnet_8x
    - resnet_c16_b64_cr16x_* -> resnet_16x
    """
    path = Path(checkpoint_path)
    parent_name = path.parent.name

    # Extract compression ratio from directory name
    if "cr4x" in parent_name:
        return "resnet_4x"
    elif "cr8x" in parent_name:
        return "resnet_8x"
    elif "cr16x" in parent_name:
        return "resnet_16x"
    else:
        # Use directory name as fallback
        return parent_name.split("_")[0]


def find_best_checkpoints() -> dict:
    """
    Find the best checkpoint for each ResNet compression ratio.

    Returns:
        Dictionary mapping model name to checkpoint path
    """
    checkpoints_dir = project_root / "notebooks" / "checkpoints"

    # Define the best checkpoints (from STATE.md)
    best_checkpoints = {
        "resnet_4x": "resnet_c64_b64_cr4x_20260129_141535",
        "resnet_8x": "resnet_c32_b64_cr8x_20260128_213848",
        "resnet_16x": "resnet_c16_b64_cr16x_20260128_003926",
    }

    result = {}
    for name, dir_name in best_checkpoints.items():
        checkpoint_path = checkpoints_dir / dir_name / "best.pth"
        if checkpoint_path.exists():
            result[name] = str(checkpoint_path)
        else:
            # Try to find any matching checkpoint
            pattern = f"resnet_*_cr{name.split('_')[1]}*"
            matches = sorted(glob(str(checkpoints_dir / pattern / "best.pth")))
            if matches:
                result[name] = matches[-1]  # Use most recent

    return result


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def export_single_model(
    checkpoint_path: str,
    output_dir: str,
    validate: bool = True,
    opset_version: int = 17,
) -> dict:
    """
    Export a single model to ONNX.

    Args:
        checkpoint_path: Path to checkpoint file
        output_dir: Output directory
        validate: Whether to validate ONNX against PyTorch
        opset_version: ONNX opset version

    Returns:
        Export results dictionary
    """
    console.print(f"\n[bold blue]Exporting:[/bold blue] {checkpoint_path}")
    console.print(f"[bold blue]Output:[/bold blue] {output_dir}")

    try:
        results = export_full_model(
            checkpoint_path,
            output_dir,
            validate=validate,
            opset_version=opset_version,
        )

        # Print results
        output_path = Path(output_dir)
        encoder_path = output_path / "encoder.onnx"
        decoder_path = output_path / "decoder.onnx"
        metadata_path = output_path / "metadata.json"

        # File sizes
        encoder_size = encoder_path.stat().st_size
        decoder_size = decoder_path.stat().st_size
        total_size = encoder_size + decoder_size

        # Print validation results
        if validate:
            encoder_status = "[green]PASSED[/green]" if results["encoder"]["valid"] else "[red]FAILED[/red]"
            decoder_status = "[green]PASSED[/green]" if results["decoder"]["valid"] else "[red]FAILED[/red]"

            console.print(f"\n[bold]Validation Results:[/bold]")
            console.print(f"  Encoder: {encoder_status} (max_diff: {results['encoder']['max_diff']:.2e})")
            console.print(f"  Decoder: {decoder_status} (max_diff: {results['decoder']['max_diff']:.2e})")

        # Print file info
        console.print(f"\n[bold]Files Created:[/bold]")
        console.print(f"  encoder.onnx: {format_file_size(encoder_size)}")
        console.print(f"  decoder.onnx: {format_file_size(decoder_size)}")
        console.print(f"  metadata.json: contains checksums for integrity verification")
        console.print(f"  [bold]Total size: {format_file_size(total_size)}[/bold]")

        # Print checksum info (FR7.6)
        import json
        with open(metadata_path) as f:
            meta = json.load(f)
        console.print(f"\n[bold]Checksums (FR7.6):[/bold]")
        console.print(f"  Encoder SHA256: {meta['checksums']['encoder_sha256'][:32]}...")
        console.print(f"  Decoder SHA256: {meta['checksums']['decoder_sha256'][:32]}...")

        return results

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        return None
    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        return None


def export_all_models(
    output_dir: str,
    validate: bool = True,
    opset_version: int = 17,
) -> dict:
    """
    Export all ResNet models (4x, 8x, 16x) to ONNX.

    Args:
        output_dir: Base output directory
        validate: Whether to validate ONNX against PyTorch
        opset_version: ONNX opset version

    Returns:
        Dictionary of results per model
    """
    checkpoints = find_best_checkpoints()

    if not checkpoints:
        console.print("[red]Error:[/red] No ResNet checkpoints found")
        return {}

    console.print(f"\n[bold]Found {len(checkpoints)} checkpoints to export:[/bold]")
    for name, path in checkpoints.items():
        console.print(f"  {name}: {path}")

    results = {}
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for name, checkpoint_path in checkpoints.items():
        model_output_dir = output_base / name
        console.print(f"\n{'='*60}")
        result = export_single_model(
            checkpoint_path,
            str(model_output_dir),
            validate=validate,
            opset_version=opset_version,
        )
        results[name] = result

    # Print summary table
    console.print(f"\n{'='*60}")
    console.print("\n[bold]Export Summary:[/bold]\n")

    table = Table(box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("Encoder", justify="center")
    table.add_column("Decoder", justify="center")
    table.add_column("Encoder Size", justify="right")
    table.add_column("Decoder Size", justify="right")
    table.add_column("Encoder SHA256", style="dim")

    for name, result in results.items():
        if result is None:
            table.add_row(name, "[red]FAILED[/red]", "[red]FAILED[/red]", "-", "-", "-")
            continue

        encoder_valid = "[green]OK[/green]" if result["encoder"]["valid"] or not validate else "[red]FAIL[/red]"
        decoder_valid = "[green]OK[/green]" if result["decoder"]["valid"] or not validate else "[red]FAIL[/red]"

        model_output_dir = output_base / name
        encoder_size = format_file_size((model_output_dir / "encoder.onnx").stat().st_size)
        decoder_size = format_file_size((model_output_dir / "decoder.onnx").stat().st_size)

        import json
        with open(model_output_dir / "metadata.json") as f:
            meta = json.load(f)
        encoder_sha = meta["checksums"]["encoder_sha256"][:16] + "..."

        table.add_row(name, encoder_valid, decoder_valid, encoder_size, decoder_size, encoder_sha)

    console.print(table)

    # Print output structure
    console.print(f"\n[bold]Output Structure:[/bold]")
    console.print(f"  {output_dir}/")
    for name in results.keys():
        console.print(f"    {name}/")
        console.print(f"      encoder.onnx")
        console.print(f"      decoder.onnx")
        console.print(f"      metadata.json")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export SAR autoencoder models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single model
  python scripts/export_onnx.py notebooks/checkpoints/resnet_c32_b64_cr8x_*/best.pth models/onnx/8x/

  # Export all ResNet models
  python scripts/export_onnx.py --all models/onnx/

  # Skip validation (faster export)
  python scripts/export_onnx.py --no-validate checkpoint.pth output/

  # Verify model integrity using checksum from metadata.json
  python -c "import hashlib,json; m=json.load(open('metadata.json')); \\
      actual=hashlib.sha256(open('encoder.onnx','rb').read()).hexdigest(); \\
      print('Match' if actual==m['checksums']['encoder_sha256'] else 'MISMATCH')"
""",
    )

    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Path to checkpoint file (required unless --all is used)",
    )
    parser.add_argument(
        "output",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all ResNet checkpoints (4x, 8x, 16x)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ONNX validation step",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.all:
        if args.checkpoint:
            console.print("[yellow]Warning:[/yellow] checkpoint argument ignored when using --all")
    else:
        if not args.checkpoint:
            console.print("[red]Error:[/red] checkpoint path required (or use --all)")
            parser.print_help()
            sys.exit(1)

    # Print banner
    console.print(
        Panel.fit(
            "[bold blue]SAR Autoencoder ONNX Export[/bold blue]\n"
            "Export trained models for deployment",
            border_style="blue",
        )
    )

    validate = not args.no_validate

    if args.all:
        results = export_all_models(
            args.output,
            validate=validate,
            opset_version=args.opset,
        )
        # Check if any failed
        if not results or any(r is None for r in results.values()):
            sys.exit(1)
    else:
        result = export_single_model(
            args.checkpoint,
            args.output,
            validate=validate,
            opset_version=args.opset,
        )
        if result is None:
            sys.exit(1)

    console.print("\n[bold green]Export complete![/bold green]")


if __name__ == "__main__":
    main()
