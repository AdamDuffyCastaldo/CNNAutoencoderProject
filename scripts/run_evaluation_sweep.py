#!/usr/bin/env python3
"""
Systematic evaluation sweep for all autoencoder models and JPEG-2000 codec.

Evaluates all models on the SAME test samples to enable paired statistical tests.
Outputs structured JSON and CSV for analysis.

Usage:
    python scripts/run_evaluation_sweep.py --output-dir reports --n-samples 2000
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

from src.data.datamodule import SARDataModule
from src.evaluation.metrics import (
    SARMetrics, compute_ms_ssim, enl_ratio, edge_preservation_index
)
from src.evaluation.codec_baselines import JPEG2000Codec, CodecEvaluator


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


def get_test_images_with_indices(dataloader, n_samples: int):
    """
    Extract test images with indices for per-sample pairing.
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


def compute_sample_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Compute all metrics for a single sample pair.

    Args:
        original: Original image (H, W) numpy array
        reconstructed: Reconstructed image (H, W) numpy array

    Returns:
        Dictionary with all metrics
    """
    # Basic metrics via SARMetrics
    psnr = SARMetrics.psnr(original, reconstructed)
    ssim = SARMetrics.ssim(original, reconstructed)

    # MS-SSIM
    ms_ssim_val = compute_ms_ssim(original, reconstructed)

    # ENL ratio
    enl_result = enl_ratio(original, reconstructed)
    enl_ratio_val = enl_result.get('enl_ratio', np.nan)

    # Edge Preservation Index
    epi = edge_preservation_index(original, reconstructed)

    return {
        'psnr': float(psnr) if not np.isnan(psnr) else None,
        'ssim': float(ssim) if not np.isnan(ssim) else None,
        'ms_ssim': float(ms_ssim_val) if not np.isnan(ms_ssim_val) else None,
        'enl_ratio': float(enl_ratio_val) if not np.isnan(enl_ratio_val) else None,
        'epi': float(epi) if not np.isnan(epi) else None,
    }


def evaluate_autoencoder_per_sample(model, dataloader, device, n_samples: int):
    """
    Evaluate autoencoder and return per-sample metrics.
    Returns dict mapping sample_index -> metrics dict.
    """
    per_sample = {}
    sample_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating autoencoder"):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)

            recon, _ = model(batch)

            for i in range(batch.shape[0]):
                if sample_idx >= n_samples:
                    return per_sample

                orig_np = batch[i, 0].cpu().numpy()
                recon_np = recon[i, 0].cpu().numpy()

                # Compute all metrics for this sample
                per_sample[sample_idx] = compute_sample_metrics(orig_np, recon_np)
                sample_idx += 1

    return per_sample


def evaluate_codec_per_sample(codec_evaluator, test_images, target_ratio: float):
    """
    Evaluate codec on test images and return per-sample metrics.
    test_images is list of (index, image_array) tuples.
    Returns dict mapping sample_index -> metrics dict.
    """
    per_sample = {}

    for idx, img in tqdm(test_images, desc=f"Evaluating JPEG-2000 @ {target_ratio}x"):
        # Compress and decompress
        compressed = codec_evaluator.codec.encode(img, codec_evaluator.calibrated_params.get(target_ratio, 100))
        reconstructed = codec_evaluator.codec.decode(compressed)

        per_sample[idx] = compute_sample_metrics(img, reconstructed)

    return per_sample


def aggregate_metrics(per_sample: dict):
    """Aggregate per-sample metrics into mean/std/min/max."""
    if not per_sample:
        return {}

    metrics_list = list(per_sample.values())
    metric_names = metrics_list[0].keys()

    aggregated = {}
    for metric in metric_names:
        values = [m[metric] for m in metrics_list if m[metric] is not None]
        if values:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
            }
        else:
            aggregated[metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'count': 0,
            }

    return aggregated


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation sweep on all models')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of test samples to evaluate')
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
    print("EVALUATION SWEEP")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Setup device
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'data').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

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

    # Extract test images with indices for codec evaluation
    print("\nExtracting test images for codec evaluation...")
    test_images = get_test_images_with_indices(test_loader, args.n_samples)
    actual_samples = len(test_images)
    print(f"Extracted {actual_samples} images")

    # Results storage
    all_results = {}  # model_name -> aggregated metrics
    per_sample_metrics = {}  # model_name -> {sample_idx -> metrics}

    # Evaluate each autoencoder
    print("\n" + "=" * 70)
    print("EVALUATING AUTOENCODERS")
    print("=" * 70)

    for model_name, ckpt_path in checkpoints.items():
        print(f"\n--- {model_name} ---")
        print(f"Checkpoint: {ckpt_path}")

        try:
            model, preproc_params = load_model_from_checkpoint(ckpt_path, model_name, device)
            compression_ratio = get_compression_ratio_from_name(model_name)

            # Per-sample evaluation
            per_sample = evaluate_autoencoder_per_sample(
                model, test_loader, device, actual_samples
            )

            # Aggregate
            aggregated = aggregate_metrics(per_sample)

            all_results[model_name] = {
                'type': 'autoencoder',
                'architecture': 'resnet' if 'resnet' in model_name else 'baseline',
                'checkpoint': ckpt_path,
                'compression_ratio': compression_ratio,
                'n_samples': len(per_sample),
                'metrics': aggregated,
            }
            per_sample_metrics[model_name] = per_sample

            psnr_mean = aggregated.get('psnr', {}).get('mean')
            psnr_std = aggregated.get('psnr', {}).get('std')
            ssim_mean = aggregated.get('ssim', {}).get('mean')
            ssim_std = aggregated.get('ssim', {}).get('std')

            print(f"  PSNR: {psnr_mean:.2f} +/- {psnr_std:.2f} dB" if psnr_mean else "  PSNR: N/A")
            print(f"  SSIM: {ssim_mean:.4f} +/- {ssim_std:.4f}" if ssim_mean else "  SSIM: N/A")

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Evaluate JPEG-2000 at each compression ratio
    print("\n" + "=" * 70)
    print("EVALUATING JPEG-2000")
    print("=" * 70)

    try:
        codec = JPEG2000Codec()
        codec_evaluator = CodecEvaluator(codec)

        # Calibrate for target ratios
        target_ratios = [4.0, 8.0, 16.0]
        sample_images = [img for _, img in test_images[:5]]
        codec_evaluator.calibrate(target_ratios, sample_images)

        for ratio in target_ratios:
            model_name = f'jpeg2000_{int(ratio)}x'
            print(f"\n--- {model_name} ---")

            # Evaluate on SAME samples as autoencoders (using indices)
            per_sample = evaluate_codec_per_sample(codec_evaluator, test_images, ratio)
            aggregated = aggregate_metrics(per_sample)

            all_results[model_name] = {
                'type': 'codec',
                'codec': 'JPEG-2000',
                'compression_ratio': ratio,
                'n_samples': len(per_sample),
                'metrics': aggregated,
            }
            per_sample_metrics[model_name] = per_sample

            psnr_mean = aggregated.get('psnr', {}).get('mean')
            psnr_std = aggregated.get('psnr', {}).get('std')
            ssim_mean = aggregated.get('ssim', {}).get('mean')
            ssim_std = aggregated.get('ssim', {}).get('std')

            print(f"  PSNR: {psnr_mean:.2f} +/- {psnr_std:.2f} dB" if psnr_mean else "  PSNR: N/A")
            print(f"  SSIM: {ssim_mean:.4f} +/- {ssim_std:.4f}" if ssim_mean else "  SSIM: N/A")

    except Exception as e:
        print(f"JPEG-2000 evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # All results JSON
    all_results_path = output_dir / 'data' / 'all_results.json'
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results: {all_results_path}")

    # Per-sample metrics JSON (convert int keys to strings for JSON)
    per_sample_json = {
        model: {str(k): v for k, v in samples.items()}
        for model, samples in per_sample_metrics.items()
    }
    per_sample_path = output_dir / 'data' / 'per_sample_metrics.json'
    with open(per_sample_path, 'w') as f:
        json.dump(per_sample_json, f, indent=2)
    print(f"Per-sample metrics: {per_sample_path}")

    # Summary CSV
    rows = []
    for name, result in all_results.items():
        metrics = result['metrics']
        rows.append({
            'name': name,
            'type': result['type'],
            'compression_ratio': result['compression_ratio'],
            'n_samples': result['n_samples'],
            'psnr_mean': metrics.get('psnr', {}).get('mean'),
            'psnr_std': metrics.get('psnr', {}).get('std'),
            'ssim_mean': metrics.get('ssim', {}).get('mean'),
            'ssim_std': metrics.get('ssim', {}).get('std'),
            'ms_ssim_mean': metrics.get('ms_ssim', {}).get('mean'),
            'enl_ratio_mean': metrics.get('enl_ratio', {}).get('mean'),
            'epi_mean': metrics.get('epi', {}).get('mean'),
        })

    # Sort by compression ratio, then type, then name
    rows.sort(key=lambda x: (x['compression_ratio'], x['type'], x['name']))

    # Write CSV manually (avoid pandas dependency)
    summary_path = output_dir / 'tables' / 'results_summary.csv'
    csv_columns = ['name', 'type', 'compression_ratio', 'n_samples', 'psnr_mean', 'psnr_std',
                   'ssim_mean', 'ssim_std', 'ms_ssim_mean', 'enl_ratio_mean', 'epi_mean']
    with open(summary_path, 'w') as f:
        f.write(','.join(csv_columns) + '\n')
        for row in rows:
            values = [str(row.get(col, '')) if row.get(col) is not None else '' for col in csv_columns]
            f.write(','.join(values) + '\n')
    print(f"Summary CSV: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Name':<20} {'Ratio':>6} {'PSNR':>10} {'SSIM':>10} {'EPI':>10}")
    print("-" * 60)
    for row in rows:
        psnr_str = f"{row['psnr_mean']:.2f}" if row['psnr_mean'] else "N/A"
        ssim_str = f"{row['ssim_mean']:.4f}" if row['ssim_mean'] else "N/A"
        epi_str = f"{row['epi_mean']:.4f}" if row['epi_mean'] else "N/A"
        print(f"{row['name']:<20} {row['compression_ratio']:>6.0f} {psnr_str:>10} {ssim_str:>10} {epi_str:>10}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"\nTotal models evaluated: {len(all_results)}")
    print(f"  Autoencoders: {len(checkpoints)}")
    print(f"  Codecs: {len([r for r in all_results.values() if r['type'] == 'codec'])}")


if __name__ == '__main__':
    main()
