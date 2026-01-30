#!/usr/bin/env python3
"""
Bitrate-matched evaluation: Compare autoencoders vs JPEG-2000 at equivalent BPP.

This script:
1. Loads all 6 autoencoder checkpoints
2. Extracts latents and computes actual BPP via entropy estimation
3. Generates fine-grained JPEG-2000 R-D curve (25 quality points)
4. Interpolates JPEG-2000 metrics at autoencoder BPP values
5. Produces fair comparison figures and data

Usage:
    python scripts/run_bitrate_matched_evaluation.py --output-dir reports/bitrate_matched
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from src.data.datamodule import SARDataModule
from src.evaluation.metrics import SARMetrics
from src.evaluation.codec_baselines import JPEG2000Codec
from src.evaluation.bitrate import (
    quantize_latent,
    dequantize_latent,
    compute_latent_bpp,
    estimate_latent_entropy
)


# =============================================================================
# Helper Functions (reused from run_evaluation_sweep.py)
# =============================================================================

def discover_checkpoints():
    """
    Dynamically discover checkpoint paths using glob patterns.
    Returns dict mapping model name to checkpoint path.

    Discovers ALL 6 autoencoder checkpoints:
    - baseline_4x, baseline_8x, baseline_16x
    - resnet_4x, resnet_8x, resnet_16x
    """
    checkpoint_dir = Path('notebooks/checkpoints')

    # Pattern -> model name mapping (all 6 models)
    patterns = {
        'baseline_4x': 'baseline_c64_b64_cr4x_*',
        'baseline_8x': 'baseline_c32_b64_cr8x_*',
        'baseline_16x': 'baseline_c16_b64_cr16x_*',
        'resnet_4x': 'resnet_c64_b64_cr4x_*',
        'resnet_8x': 'resnet_c32_b64_cr8x_*',
        'resnet_16x': 'resnet_c16_b64_cr16x_*',
    }

    checkpoints = {}
    for name, pattern in patterns.items():
        matches = sorted(glob.glob(str(checkpoint_dir / pattern)))
        if matches:
            # Find the most recent directory that actually has best.pth
            found = False
            for match in reversed(matches):
                ckpt_path = Path(match) / 'best.pth'
                if ckpt_path.exists():
                    checkpoints[name] = str(ckpt_path)
                    print(f"  Found {name}: {ckpt_path}")
                    found = True
                    break
            if not found:
                print(f"  Warning: {name} directories exist but none have best.pth")
        else:
            print(f"  Warning: No checkpoint found for {name} (pattern: {pattern})")

    return checkpoints


def get_compression_ratio_from_name(model_name: str) -> float:
    """Extract compression ratio from model name."""
    if '4x' in model_name:
        return 4.0
    elif '8x' in model_name:
        return 8.0
    elif '16x' in model_name:
        return 16.0
    return 16.0  # default


def get_latent_channels_from_name(model_name: str) -> int:
    """Infer latent channels from model name (based on compression ratio)."""
    # For 256x256 -> 16x16 bottleneck:
    # 4x: 64 channels, 8x: 32 channels, 16x: 16 channels
    if '4x' in model_name:
        return 64
    elif '8x' in model_name:
        return 32
    elif '16x' in model_name:
        return 16
    return 16


def load_model_from_checkpoint(checkpoint_path: str, model_name: str, device: torch.device):
    """Load model from checkpoint, auto-detecting model type."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or infer from name
    model_config = checkpoint.get('model_config', {})
    latent_channels = model_config.get('latent_channels', get_latent_channels_from_name(model_name))
    base_channels = model_config.get('base_channels', 64)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_resnet_keys = any('stage1' in k or 'stage2' in k for k in state_dict.keys())

    if has_resnet_keys:
        from src.models.resnet_autoencoder import ResNetAutoencoder
        model = ResNetAutoencoder(
            in_channels=1,
            base_channels=base_channels,
            latent_channels=latent_channels
        )
    else:
        from src.models.autoencoder import SARAutoencoder
        model = SARAutoencoder(latent_channels=latent_channels)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    preprocessing_params = checkpoint.get('preprocessing_params', {})
    return model, preprocessing_params


def get_test_images(dataloader, n_samples: int):
    """
    Extract test images for evaluation.
    Returns list of (index, image_array) tuples.
    """
    images = []
    idx = 0
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        for i in range(batch.shape[0]):
            images.append((idx, batch[i, 0].numpy()))
            idx += 1
            if len(images) >= n_samples:
                return images
    return images


# =============================================================================
# New Functions for Bitrate-Matched Evaluation
# =============================================================================

def evaluate_autoencoder_bpp(model, dataloader, device, n_samples: int) -> dict:
    """
    Evaluate autoencoder with entropy-based BPP calculation.

    For each sample:
    - Extract latent via model.encoder
    - Quantize latent
    - Compute BPP via compute_latent_bpp
    - Decode from quantized latent for quality metrics

    Returns:
        Dict with:
        - 'bpp_mean', 'bpp_std': aggregate BPP statistics
        - 'psnr_mean', 'psnr_std': quality from quantized reconstruction
        - 'ssim_mean', 'ssim_std': quality from quantized reconstruction
        - 'samples': list of per-sample (bpp, psnr, ssim)
    """
    samples = []
    sample_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating autoencoder BPP"):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)

            for i in range(batch.shape[0]):
                if sample_idx >= n_samples:
                    break

                # Get single sample
                x = batch[i:i+1]  # Keep batch dimension

                # Encode to latent
                latent = model.encode(x)

                # Convert to numpy for quantization
                latent_np = latent.cpu().numpy()

                # Compute entropy-based BPP
                bpp = compute_latent_bpp(latent_np, n_bins=256, input_shape=(256, 256))

                # Quantize and dequantize latent
                quantized, ranges = quantize_latent(latent_np, n_bins=256)
                dequantized = dequantize_latent(quantized, ranges, n_bins=256)

                # Decode from quantized latent
                dequantized_tensor = torch.from_numpy(dequantized).to(device)
                reconstruction = model.decode(dequantized_tensor)

                # Compute quality metrics on quantized reconstruction
                orig_np = x[0, 0].cpu().numpy()
                recon_np = reconstruction[0, 0].cpu().numpy()

                psnr = SARMetrics.psnr(orig_np, recon_np)
                ssim = SARMetrics.ssim(orig_np, recon_np)

                samples.append({
                    'bpp': float(bpp),
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                })

                sample_idx += 1

            if sample_idx >= n_samples:
                break

    # Aggregate statistics
    bpp_values = [s['bpp'] for s in samples]
    psnr_values = [s['psnr'] for s in samples]
    ssim_values = [s['ssim'] for s in samples]

    return {
        'bpp_mean': float(np.mean(bpp_values)),
        'bpp_std': float(np.std(bpp_values)),
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'n_samples': len(samples),
        'samples': samples,
    }


def generate_jpeg2000_rd_curve(test_images: list, n_points: int = 25) -> list:
    """
    Generate fine-grained JPEG-2000 rate-distortion curve.

    Args:
        test_images: List of (index, image_array) tuples
        n_points: Number of quality levels to evaluate (default 25)

    Returns:
        List of dicts with: quality, bpp, bpp_std, psnr, psnr_std, ssim, ssim_std
    """
    codec = JPEG2000Codec()

    # Quality range: 1 to 1000 (full JPEG-2000 range)
    # Log-space for even BPP coverage
    qualities = np.logspace(np.log10(1), np.log10(1000), n_points).astype(int)
    qualities = np.unique(qualities)  # Remove duplicates

    rd_data = []

    for quality in tqdm(qualities, desc="JPEG-2000 R-D curve"):
        bpp_values = []
        psnr_values = []
        ssim_values = []

        for _, img in test_images:
            try:
                # Encode and decode
                encoded = codec.encode(img, int(quality))
                decoded = codec.decode(encoded)

                # BPP = encoded_bits / input_pixels
                encoded_bits = len(encoded) * 8
                input_pixels = img.shape[0] * img.shape[1]
                bpp = encoded_bits / input_pixels

                psnr = SARMetrics.psnr(img, decoded)
                ssim = SARMetrics.ssim(img, decoded)

                bpp_values.append(bpp)
                psnr_values.append(psnr)
                ssim_values.append(ssim)

            except Exception as e:
                print(f"  Warning: JPEG-2000 encoding failed at quality={quality}: {e}")
                continue

        if bpp_values:
            rd_data.append({
                'quality': int(quality),
                'bpp': float(np.mean(bpp_values)),
                'bpp_std': float(np.std(bpp_values)),
                'psnr': float(np.mean(psnr_values)),
                'psnr_std': float(np.std(psnr_values)),
                'ssim': float(np.mean(ssim_values)),
                'ssim_std': float(np.std(ssim_values)),
                'n_samples': len(bpp_values),
            })

    return rd_data


def interpolate_at_bpp(rd_curve: list, target_bpp: float) -> dict:
    """
    Interpolate JPEG-2000 R-D curve at target BPP.

    Args:
        rd_curve: List of dicts with bpp, psnr, ssim
        target_bpp: Target bits-per-pixel value

    Returns:
        Dict with bpp, psnr, ssim, extrapolated flag
    """
    # Sort by BPP
    sorted_rd = sorted(rd_curve, key=lambda x: x['bpp'])
    bpp_values = [r['bpp'] for r in sorted_rd]
    psnr_values = [r['psnr'] for r in sorted_rd]
    ssim_values = [r['ssim'] for r in sorted_rd]

    # Check if target is within range
    if target_bpp < min(bpp_values) or target_bpp > max(bpp_values):
        # Extrapolation needed
        return {
            'bpp': target_bpp,
            'psnr': np.nan,
            'ssim': np.nan,
            'extrapolated': True,
            'note': f'BPP {target_bpp:.4f} outside range [{min(bpp_values):.4f}, {max(bpp_values):.4f}]'
        }

    # Linear interpolation
    psnr_interp = interp1d(bpp_values, psnr_values, kind='linear')
    ssim_interp = interp1d(bpp_values, ssim_values, kind='linear')

    return {
        'bpp': target_bpp,
        'psnr': float(psnr_interp(target_bpp)),
        'ssim': float(ssim_interp(target_bpp)),
        'extrapolated': False,
    }


def plot_rd_curves(ae_results: dict, jp2_rd_curve: list, output_dir: Path, metric: str = 'psnr'):
    """
    Plot rate-distortion curves comparing autoencoders and JPEG-2000.

    Args:
        ae_results: Dict mapping model_name to result dict (with bpp_mean, psnr_mean, etc.)
        jp2_rd_curve: List of JPEG-2000 R-D points
        output_dir: Output directory for figures
        metric: 'psnr' or 'ssim'
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort JPEG-2000 curve by BPP
    sorted_rd = sorted(jp2_rd_curve, key=lambda x: x['bpp'])
    jp2_bpp = [r['bpp'] for r in sorted_rd]
    jp2_metric = [r[metric] for r in sorted_rd]
    jp2_std = [r[f'{metric}_std'] for r in sorted_rd]

    # Plot JPEG-2000 curve (continuous with error band)
    ax.plot(jp2_bpp, jp2_metric, 'o-', color='gray', label='JPEG-2000',
            linewidth=2, markersize=4, alpha=0.8)
    ax.fill_between(
        jp2_bpp,
        [m - s for m, s in zip(jp2_metric, jp2_std)],
        [m + s for m, s in zip(jp2_metric, jp2_std)],
        alpha=0.2, color='gray'
    )

    # Colors and markers for autoencoders
    colors = {'resnet': 'tab:orange', 'baseline': 'tab:blue'}
    markers = {'resnet': 's', 'baseline': 'o'}

    # Plot autoencoder points
    for model_name, result in ae_results.items():
        model_type = 'resnet' if 'resnet' in model_name else 'baseline'
        bpp = result['bpp_mean']
        metric_val = result[f'{metric}_mean']
        metric_std = result[f'{metric}_std']

        ax.errorbar(
            bpp, metric_val, yerr=metric_std,
            fmt=markers.get(model_type, '^'),
            color=colors.get(model_type, 'tab:green'),
            markersize=10, capsize=4, capthick=1,
            label=model_name.replace('_', ' ').title(),
            zorder=5
        )

        # Annotate with model name
        offset = (5, 5) if metric_val > 25 else (5, -10)
        ax.annotate(
            model_name.replace('_', ' '),
            (bpp, metric_val),
            textcoords='offset points', xytext=offset,
            fontsize=8, alpha=0.8
        )

    ax.set_xlabel('Bits Per Pixel (BPP)', fontsize=12)
    ax.set_ylabel('PSNR (dB)' if metric == 'psnr' else 'SSIM', fontsize=12)
    ax.set_title('Bitrate-Matched Rate-Distortion Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set axis limits to focus on the data
    all_bpp = [result['bpp_mean'] for result in ae_results.values()] + jp2_bpp
    bpp_min, bpp_max = min(all_bpp), max(all_bpp)
    bpp_margin = (bpp_max - bpp_min) * 0.1
    ax.set_xlim(max(0, bpp_min - bpp_margin), bpp_max + bpp_margin)

    plt.tight_layout()

    output_path = output_dir / 'figures' / f'rd_curve_bitrate_matched_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def create_summary_csv(ae_results: dict, jp2_rd_curve: list, output_path: Path):
    """
    Create summary CSV table with bitrate-matched comparison.

    Columns:
    - model: autoencoder name
    - compression_ratio: 4, 8, or 16
    - ae_bpp: actual autoencoder BPP from entropy
    - ae_psnr: autoencoder PSNR (from quantized reconstruction)
    - ae_ssim: autoencoder SSIM
    - jp2_psnr_at_bpp: JPEG-2000 PSNR interpolated at ae_bpp
    - jp2_ssim_at_bpp: JPEG-2000 SSIM interpolated at ae_bpp
    - psnr_diff: ae_psnr - jp2_psnr_at_bpp
    - ssim_diff: ae_ssim - jp2_ssim_at_bpp
    """
    rows = []

    for model_name, result in ae_results.items():
        ae_bpp = result['bpp_mean']
        ae_psnr = result['psnr_mean']
        ae_ssim = result['ssim_mean']

        # Interpolate JPEG-2000 at same BPP
        jp2_interp = interpolate_at_bpp(jp2_rd_curve, ae_bpp)

        jp2_psnr = jp2_interp['psnr']
        jp2_ssim = jp2_interp['ssim']

        psnr_diff = ae_psnr - jp2_psnr if not np.isnan(jp2_psnr) else np.nan
        ssim_diff = ae_ssim - jp2_ssim if not np.isnan(jp2_ssim) else np.nan

        rows.append({
            'model': model_name,
            'compression_ratio': get_compression_ratio_from_name(model_name),
            'ae_bpp': ae_bpp,
            'ae_psnr': ae_psnr,
            'ae_ssim': ae_ssim,
            'jp2_psnr_at_bpp': jp2_psnr,
            'jp2_ssim_at_bpp': jp2_ssim,
            'psnr_diff': psnr_diff,
            'ssim_diff': ssim_diff,
            'extrapolated': jp2_interp.get('extrapolated', False),
        })

    # Sort by compression ratio, then model type
    rows.sort(key=lambda x: (x['compression_ratio'], x['model']))

    # Write CSV
    columns = ['model', 'compression_ratio', 'ae_bpp', 'ae_psnr', 'ae_ssim',
               'jp2_psnr_at_bpp', 'jp2_ssim_at_bpp', 'psnr_diff', 'ssim_diff', 'extrapolated']

    with open(output_path, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, '')
                if isinstance(val, float):
                    if np.isnan(val):
                        values.append('NaN')
                    else:
                        values.append(f'{val:.6f}')
                elif isinstance(val, bool):
                    values.append(str(val))
                else:
                    values.append(str(val))
            f.write(','.join(values) + '\n')

    print(f"  Saved: {output_path}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bitrate-matched evaluation: Compare autoencoders vs JPEG-2000 at equivalent BPP'
    )
    parser.add_argument('--output-dir', type=str, default='reports/bitrate_matched',
                        help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Number of test samples to evaluate')
    parser.add_argument('--n-jp2-points', type=int, default=25,
                        help='Number of JPEG-2000 quality points for R-D curve')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for autoencoder evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available)')
    parser.add_argument('--data-path', type=str, default='data/patches/metadata.npy',
                        help='Path to dataset metadata')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("BITRATE-MATCHED EVALUATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Setup device
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'data').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)

    # Discover checkpoints
    print("\nDiscovering checkpoints...")
    checkpoints = discover_checkpoints()

    if len(checkpoints) < 6:
        print(f"\nWarning: Found only {len(checkpoints)} checkpoints, expected 6")
        print("Continuing with available checkpoints...")

    # Load test data
    print("\nLoading test data...")
    datamodule = SARDataModule(
        patches_path=args.data_path,
        batch_size=args.batch_size,
        val_fraction=0.1,
        num_workers=0,
        lazy=True,
        max_samples=args.n_samples
    )
    test_loader = datamodule.val_dataloader()
    print(f"Test samples: {datamodule.val_size}")

    # Extract test images for JPEG-2000 evaluation
    print("\nExtracting test images for JPEG-2000 evaluation...")
    test_images = get_test_images(test_loader, args.n_samples)
    actual_samples = len(test_images)
    print(f"Extracted {actual_samples} images")

    # ==========================================================================
    # Step 1: Evaluate Autoencoders with Entropy-Based BPP
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING AUTOENCODERS (Entropy-Based BPP)")
    print("=" * 70)

    ae_results = {}

    for model_name, ckpt_path in checkpoints.items():
        print(f"\n--- {model_name} ---")
        print(f"Checkpoint: {ckpt_path}")

        try:
            model, _ = load_model_from_checkpoint(ckpt_path, model_name, device)

            # Evaluate with entropy-based BPP
            result = evaluate_autoencoder_bpp(model, test_loader, device, actual_samples)

            ae_results[model_name] = result

            print(f"  BPP: {result['bpp_mean']:.4f} +/- {result['bpp_std']:.4f}")
            print(f"  PSNR: {result['psnr_mean']:.2f} +/- {result['psnr_std']:.2f} dB")
            print(f"  SSIM: {result['ssim_mean']:.4f} +/- {result['ssim_std']:.4f}")

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save autoencoder BPP results
    ae_bpp_path = output_dir / 'data' / 'autoencoder_bpp.json'
    # Convert samples list to JSON-serializable format
    ae_results_json = {}
    for name, result in ae_results.items():
        ae_results_json[name] = {
            'bpp_mean': result['bpp_mean'],
            'bpp_std': result['bpp_std'],
            'psnr_mean': result['psnr_mean'],
            'psnr_std': result['psnr_std'],
            'ssim_mean': result['ssim_mean'],
            'ssim_std': result['ssim_std'],
            'n_samples': result['n_samples'],
        }
    with open(ae_bpp_path, 'w') as f:
        json.dump(ae_results_json, f, indent=2)
    print(f"\nSaved autoencoder BPP results: {ae_bpp_path}")

    # ==========================================================================
    # Step 2: Generate JPEG-2000 R-D Curve
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"GENERATING JPEG-2000 R-D CURVE ({args.n_jp2_points} points)")
    print("=" * 70)

    jp2_rd_curve = generate_jpeg2000_rd_curve(test_images, n_points=args.n_jp2_points)

    print(f"\nGenerated {len(jp2_rd_curve)} R-D points")
    print(f"  BPP range: [{jp2_rd_curve[0]['bpp']:.4f}, {jp2_rd_curve[-1]['bpp']:.4f}]")
    print(f"  PSNR range: [{jp2_rd_curve[0]['psnr']:.2f}, {jp2_rd_curve[-1]['psnr']:.2f}]")

    # Save JPEG-2000 R-D curve
    jp2_rd_path = output_dir / 'data' / 'jpeg2000_rd_curve.json'
    with open(jp2_rd_path, 'w') as f:
        json.dump(jp2_rd_curve, f, indent=2)
    print(f"Saved JPEG-2000 R-D curve: {jp2_rd_path}")

    # ==========================================================================
    # Step 3: Create Bitrate-Matched Comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CREATING BITRATE-MATCHED COMPARISON")
    print("=" * 70)

    bitrate_matched_results = {}

    for model_name, result in ae_results.items():
        ae_bpp = result['bpp_mean']

        # Interpolate JPEG-2000 at same BPP
        jp2_interp = interpolate_at_bpp(jp2_rd_curve, ae_bpp)

        bitrate_matched_results[model_name] = {
            'ae_bpp': ae_bpp,
            'ae_psnr': result['psnr_mean'],
            'ae_ssim': result['ssim_mean'],
            'jp2_psnr_at_bpp': jp2_interp['psnr'],
            'jp2_ssim_at_bpp': jp2_interp['ssim'],
            'extrapolated': jp2_interp.get('extrapolated', False),
        }

        if not jp2_interp.get('extrapolated', False):
            psnr_diff = result['psnr_mean'] - jp2_interp['psnr']
            ssim_diff = result['ssim_mean'] - jp2_interp['ssim']
            print(f"  {model_name}: BPP={ae_bpp:.4f}, AE PSNR={result['psnr_mean']:.2f}, "
                  f"JP2 PSNR={jp2_interp['psnr']:.2f}, Diff={psnr_diff:+.2f} dB")
        else:
            print(f"  {model_name}: BPP={ae_bpp:.4f} (outside JPEG-2000 range)")

    # Save bitrate-matched results
    bm_results_path = output_dir / 'data' / 'bitrate_matched_results.json'
    with open(bm_results_path, 'w') as f:
        json.dump(bitrate_matched_results, f, indent=2)
    print(f"\nSaved bitrate-matched results: {bm_results_path}")

    # ==========================================================================
    # Step 4: Generate Figures
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    print("\nGenerating PSNR R-D curve...")
    plot_rd_curves(ae_results, jp2_rd_curve, output_dir, metric='psnr')

    print("Generating SSIM R-D curve...")
    plot_rd_curves(ae_results, jp2_rd_curve, output_dir, metric='ssim')

    # ==========================================================================
    # Step 5: Create Summary Table
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CREATING SUMMARY TABLE")
    print("=" * 70)

    summary_path = output_dir / 'tables' / 'bitrate_matched_summary.csv'
    rows = create_summary_csv(ae_results, jp2_rd_curve, summary_path)

    # Print summary table
    print("\n" + "-" * 90)
    print(f"{'Model':<16} {'CR':>4} {'BPP':>8} {'AE PSNR':>10} {'JP2 PSNR':>10} {'Diff':>10}")
    print("-" * 90)
    for row in rows:
        diff_str = f"{row['psnr_diff']:+.2f} dB" if not np.isnan(row['psnr_diff']) else "N/A"
        jp2_str = f"{row['jp2_psnr_at_bpp']:.2f}" if not np.isnan(row['jp2_psnr_at_bpp']) else "N/A"
        print(f"{row['model']:<16} {row['compression_ratio']:>4.0f} {row['ae_bpp']:>8.4f} "
              f"{row['ae_psnr']:>10.2f} {jp2_str:>10} {diff_str:>10}")
    print("-" * 90)

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"\nOutputs:")
    print(f"  Data: {output_dir / 'data'}")
    print(f"  Figures: {output_dir / 'figures'}")
    print(f"  Tables: {output_dir / 'tables'}")
    print(f"\nModels evaluated: {len(ae_results)}")
    print(f"JPEG-2000 R-D points: {len(jp2_rd_curve)}")
    print(f"Test samples: {actual_samples}")


if __name__ == '__main__':
    main()
