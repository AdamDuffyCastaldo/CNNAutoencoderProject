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
    """
    checkpoint_dir = Path('notebooks/checkpoints')

    # Pattern -> model name mapping
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
            # Take most recent (last in sorted order by timestamp)
            ckpt_path = Path(matches[-1]) / 'best.pth'
            if ckpt_path.exists():
                checkpoints[name] = str(ckpt_path)
                print(f"  Found {name}: {ckpt_path}")
            else:
                print(f"  Warning: {name} directory exists but no best.pth: {matches[-1]}")
        else:
            print(f"  Warning: No checkpoint found for {name} (pattern: {pattern})")

    return checkpoints


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, auto-detecting model type."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get('model_config', {})
    latent_channels = model_config.get('latent_channels', 16)
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
        model = SARAutoencoder(in_channels=1, latent_channels=latent_channels)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    preprocessing_params = checkpoint.get('preprocessing_params', {})
    return model, preprocessing_params


def get_compression_ratio_from_name(model_name: str) -> float:
    """Extract compression ratio from model name like 'baseline_4x' -> 4.0"""
    if '4x' in model_name:
        return 4.0
    elif '8x' in model_name:
        return 8.0
    elif '16x' in model_name:
        return 16.0
    return 16.0  # default


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

            recon = model(batch)

            for i in range(batch.shape[0]):
                if sample_idx >= n_samples:
                    return per_sample

                # Convert to numpy for metrics
                orig_np = batch[i, 0].cpu().numpy()
                rec_np = recon[i, 0].cpu().numpy()

                # Compute all metrics for this sample
                try:
                    ms_ssim_val = compute_ms_ssim(orig_np, rec_np, data_range=1.0)
                except Exception:
                    ms_ssim_val = float('nan')

                try:
                    enl_result = enl_ratio(orig_np, rec_np)
                    enl_val = enl_result.get('ratio', float('nan')) if isinstance(enl_result, dict) else float(enl_result)
                except Exception:
                    enl_val = float('nan')

                per_sample[sample_idx] = {
                    'psnr': SARMetrics.psnr(orig_np, rec_np, data_range=1.0),
                    'ssim': SARMetrics.ssim(orig_np, rec_np, data_range=1.0),
                    'ms_ssim': ms_ssim_val,
                    'enl_ratio': enl_val,
                    'epi': edge_preservation_index(orig_np, rec_np),
                }
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
        compressed = codec_evaluator.codec.compress(img, target_ratio)
        reconstructed = codec_evaluator.codec.decompress(compressed)

        # Compute metrics using numpy arrays
        try:
            ms_ssim_val = compute_ms_ssim(img, reconstructed, data_range=1.0)
        except Exception:
            ms_ssim_val = float('nan')

        try:
            enl_result = enl_ratio(img, reconstructed)
            enl_val = enl_result.get('ratio', float('nan')) if isinstance(enl_result, dict) else float(enl_result)
        except Exception:
            enl_val = float('nan')

        per_sample[idx] = {
            'psnr': SARMetrics.psnr(img, reconstructed, data_range=1.0),
            'ssim': SARMetrics.ssim(img, reconstructed, data_range=1.0),
            'ms_ssim': ms_ssim_val,
            'enl_ratio': enl_val,
            'epi': edge_preservation_index(img, reconstructed),
        }

    return per_sample


def aggregate_metrics(per_sample: dict):
    """Aggregate per-sample metrics into mean/std/min/max."""
    if not per_sample:
        return {}

    metrics_list = list(per_sample.values())
    metric_names = metrics_list[0].keys()

    aggregated = {}
    for metric in metric_names:
        values = [m[metric] for m in metrics_list]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
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

    if len(checkpoints) == 0:
        print("\nERROR: No checkpoints found!")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoints")

    # Load test data
    print("\nLoading test data...")
    datamodule = SARDataModule(
        patches_path='data/patches/metadata.npy',
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
    print(f"Extracted {len(test_images)} images")

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

        model, preproc_params = load_model_from_checkpoint(ckpt_path, device)
        compression_ratio = get_compression_ratio_from_name(model_name)

        # Per-sample evaluation
        per_sample = evaluate_autoencoder_per_sample(
            model, test_loader, device, args.n_samples
        )

        # Aggregate
        aggregated = aggregate_metrics(per_sample)

        all_results[model_name] = {
            'type': 'autoencoder',
            'checkpoint': ckpt_path,
            'compression_ratio': compression_ratio,
            'n_samples': len(per_sample),
            'metrics': aggregated,
        }
        per_sample_metrics[model_name] = per_sample

        print(f"  PSNR: {aggregated['psnr']['mean']:.2f} +/- {aggregated['psnr']['std']:.2f} dB")
        print(f"  SSIM: {aggregated['ssim']['mean']:.4f} +/- {aggregated['ssim']['std']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Evaluate JPEG-2000 at each compression ratio
    print("\n" + "=" * 70)
    print("EVALUATING JPEG-2000")
    print("=" * 70)

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

        print(f"  PSNR: {aggregated['psnr']['mean']:.2f} +/- {aggregated['psnr']['std']:.2f} dB")
        print(f"  SSIM: {aggregated['ssim']['mean']:.4f} +/- {aggregated['ssim']['std']:.4f}")

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
    import pandas as pd
    rows = []
    for name, result in all_results.items():
        metrics = result['metrics']
        rows.append({
            'name': name,
            'type': result['type'],
            'compression_ratio': result['compression_ratio'],
            'n_samples': result['n_samples'],
            'psnr_mean': metrics['psnr']['mean'],
            'psnr_std': metrics['psnr']['std'],
            'ssim_mean': metrics['ssim']['mean'],
            'ssim_std': metrics['ssim']['std'],
            'ms_ssim_mean': metrics['ms_ssim']['mean'],
            'enl_ratio_mean': metrics['enl_ratio']['mean'],
            'epi_mean': metrics['epi']['mean'],
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(['compression_ratio', 'type', 'name'])
    summary_path = output_dir / 'tables' / 'results_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV: {summary_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal models evaluated: {len(all_results)}")
    print(f"  Autoencoders: {len(checkpoints)}")
    print(f"  Codecs: {len(target_ratios)}")


if __name__ == '__main__':
    main()
