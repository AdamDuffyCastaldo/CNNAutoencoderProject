#!/usr/bin/env python3
"""
Evaluation sweep with 8-bit quantization for fair JPEG-2000 comparison.

Both autoencoders and JPEG-2000 operate on 8-bit quantized data:
- Input: float32 [0,1] -> uint8 [0,255] -> float32 [0,1]
- Output: float32 [0,1] -> uint8 [0,255] -> float32 [0,1]
- Metrics computed on quantized data

This ensures apples-to-apples comparison with JPEG-2000.

Usage:
    python scripts/run_evaluation_sweep_8bit.py --output-dir reports/8bit --n-samples 500
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


def quantize_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Quantize float32 [0,1] image to uint8 [0,255] and back to float32 [0,1].

    This simulates the precision loss of 8-bit storage.
    """
    # Clip to valid range, scale to 0-255, round, clip, convert to uint8
    img_clipped = np.clip(img, 0.0, 1.0)
    img_uint8 = np.round(img_clipped * 255.0).astype(np.uint8)
    # Convert back to float32 [0,1]
    img_quantized = img_uint8.astype(np.float32) / 255.0
    return img_quantized


def discover_checkpoints():
    """Dynamically discover checkpoint paths using glob patterns."""
    checkpoint_dir = Path('notebooks/checkpoints')

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
            for match in reversed(matches):
                ckpt_path = Path(match) / 'best.pth'
                if ckpt_path.exists():
                    checkpoints[name] = str(ckpt_path)
                    print(f"  Found {name}: {ckpt_path}")
                    break

    return checkpoints


def get_compression_ratio_from_name(model_name: str) -> float:
    """Extract compression ratio from model name."""
    if '4x' in model_name:
        return 4.0
    elif '8x' in model_name:
        return 8.0
    elif '16x' in model_name:
        return 16.0
    return 16.0


def get_latent_channels_from_name(model_name: str) -> int:
    """Infer latent channels from model name."""
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

    return model


def get_test_images_with_indices(dataloader, n_samples: int):
    """Extract test images with indices, applying 8-bit quantization."""
    images = []
    idx = 0
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        for i in range(batch.shape[0]):
            # Get original float32 image
            img_float = batch[i, 0].numpy()
            # Quantize to 8-bit and back
            img_quantized = quantize_to_8bit(img_float)
            images.append((idx, img_quantized))
            idx += 1
            if len(images) >= n_samples:
                return images
    return images


def compute_sample_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute all metrics for a single sample pair."""
    psnr = SARMetrics.psnr(original, reconstructed)
    ssim = SARMetrics.ssim(original, reconstructed)
    ms_ssim_val = compute_ms_ssim(original, reconstructed)
    enl_result = enl_ratio(original, reconstructed)
    enl_ratio_val = enl_result.get('enl_ratio', np.nan)
    epi = edge_preservation_index(original, reconstructed)

    return {
        'psnr': float(psnr) if not np.isnan(psnr) else None,
        'ssim': float(ssim) if not np.isnan(ssim) else None,
        'ms_ssim': float(ms_ssim_val) if not np.isnan(ms_ssim_val) else None,
        'enl_ratio': float(enl_ratio_val) if not np.isnan(enl_ratio_val) else None,
        'epi': float(epi) if not np.isnan(epi) else None,
    }


def evaluate_autoencoder_per_sample_8bit(model, test_images, device):
    """
    Evaluate autoencoder with 8-bit quantization on input AND output.

    Pipeline:
    1. Input already quantized to 8-bit (from test_images)
    2. Process through autoencoder
    3. Quantize output to 8-bit
    4. Compute metrics
    """
    per_sample = {}

    model.eval()
    with torch.no_grad():
        for idx, img_quantized in tqdm(test_images, desc="Evaluating autoencoder (8-bit)"):
            # Convert to tensor
            img_tensor = torch.from_numpy(img_quantized).unsqueeze(0).unsqueeze(0).float().to(device)

            # Process through autoencoder
            recon_tensor, _ = model(img_tensor)
            recon_float = recon_tensor[0, 0].cpu().numpy()

            # Quantize output to 8-bit
            recon_quantized = quantize_to_8bit(recon_float)

            # Compute metrics on 8-bit quantized data
            per_sample[idx] = compute_sample_metrics(img_quantized, recon_quantized)

    return per_sample


def evaluate_codec_per_sample_8bit(codec_evaluator, test_images, target_ratio: float):
    """
    Evaluate JPEG-2000 codec (already operates on 8-bit internally).

    Pipeline:
    1. Input already quantized to 8-bit
    2. Compress/decompress with JPEG-2000
    3. Output is 8-bit
    4. Compute metrics
    """
    per_sample = {}

    for idx, img_quantized in tqdm(test_images, desc=f"Evaluating JPEG-2000 @ {target_ratio}x (8-bit)"):
        # JPEG-2000 codec expects [0,1] float, converts to uint8 internally
        compressed = codec_evaluator.codec.encode(img_quantized, codec_evaluator.calibrated_params.get(target_ratio, 100))
        reconstructed = codec_evaluator.codec.decode(compressed)

        # Reconstructed is already effectively 8-bit quantized
        recon_quantized = quantize_to_8bit(reconstructed)

        per_sample[idx] = compute_sample_metrics(img_quantized, recon_quantized)

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
            aggregated[metric] = {'mean': None, 'std': None, 'min': None, 'max': None, 'count': 0}

    return aggregated


def parse_args():
    parser = argparse.ArgumentParser(description='Run 8-bit quantized evaluation sweep')
    parser.add_argument('--output-dir', type=str, default='reports/8bit',
                        help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of test samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for data loading')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available)')
    parser.add_argument('--data-path', type=str, default='data/patches/metadata.npy',
                        help='Path to dataset metadata')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("8-BIT QUANTIZED EVALUATION SWEEP")
    print("=" * 70)
    print("Fair comparison: both autoencoders and JPEG-2000 on 8-bit data")
    print(f"Started: {datetime.now().isoformat()}")

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'data').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)

    # Discover checkpoints
    print("\nDiscovering checkpoints...")
    checkpoints = discover_checkpoints()

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

    # Extract and quantize test images
    print("\nExtracting and quantizing test images to 8-bit...")
    test_images = get_test_images_with_indices(test_loader, args.n_samples)
    actual_samples = len(test_images)
    print(f"Extracted {actual_samples} 8-bit quantized images")

    # Measure quantization loss
    sample_img = test_images[0][1]
    quant_loss = np.mean(np.abs(sample_img - quantize_to_8bit(sample_img)))
    print(f"Quantization precision: ~{quant_loss:.6f} (should be ~0 for already-quantized)")

    # Results storage
    all_results = {}
    per_sample_metrics = {}

    # Evaluate autoencoders
    print("\n" + "=" * 70)
    print("EVALUATING AUTOENCODERS (8-bit I/O)")
    print("=" * 70)

    for model_name, ckpt_path in checkpoints.items():
        print(f"\n--- {model_name} ---")

        try:
            model = load_model_from_checkpoint(ckpt_path, model_name, device)
            compression_ratio = get_compression_ratio_from_name(model_name)

            per_sample = evaluate_autoencoder_per_sample_8bit(model, test_images, device)
            aggregated = aggregate_metrics(per_sample)

            all_results[model_name] = {
                'type': 'autoencoder',
                'architecture': 'resnet' if 'resnet' in model_name else 'baseline',
                'checkpoint': ckpt_path,
                'compression_ratio': compression_ratio,
                'n_samples': len(per_sample),
                'quantization': '8-bit',
                'metrics': aggregated,
            }
            per_sample_metrics[model_name] = per_sample

            psnr_mean = aggregated.get('psnr', {}).get('mean')
            ssim_mean = aggregated.get('ssim', {}).get('mean')
            print(f"  PSNR: {psnr_mean:.2f} dB" if psnr_mean else "  PSNR: N/A")
            print(f"  SSIM: {ssim_mean:.4f}" if ssim_mean else "  SSIM: N/A")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Evaluate JPEG-2000
    print("\n" + "=" * 70)
    print("EVALUATING JPEG-2000 (8-bit)")
    print("=" * 70)

    try:
        codec = JPEG2000Codec()
        codec_evaluator = CodecEvaluator(codec)

        target_ratios = [4.0, 8.0, 16.0]
        sample_images = [img for _, img in test_images[:5]]
        codec_evaluator.calibrate(target_ratios, sample_images)

        for ratio in target_ratios:
            model_name = f'jpeg2000_{int(ratio)}x'
            print(f"\n--- {model_name} ---")

            per_sample = evaluate_codec_per_sample_8bit(codec_evaluator, test_images, ratio)
            aggregated = aggregate_metrics(per_sample)

            all_results[model_name] = {
                'type': 'codec',
                'codec': 'JPEG-2000',
                'compression_ratio': ratio,
                'n_samples': len(per_sample),
                'quantization': '8-bit',
                'metrics': aggregated,
            }
            per_sample_metrics[model_name] = per_sample

            psnr_mean = aggregated.get('psnr', {}).get('mean')
            ssim_mean = aggregated.get('ssim', {}).get('mean')
            print(f"  PSNR: {psnr_mean:.2f} dB" if psnr_mean else "  PSNR: N/A")
            print(f"  SSIM: {ssim_mean:.4f}" if ssim_mean else "  SSIM: N/A")

    except Exception as e:
        print(f"JPEG-2000 evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    all_results_path = output_dir / 'data' / 'all_results_8bit.json'
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"All results: {all_results_path}")

    per_sample_json = {
        model: {str(k): v for k, v in samples.items()}
        for model, samples in per_sample_metrics.items()
    }
    per_sample_path = output_dir / 'data' / 'per_sample_metrics_8bit.json'
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

    rows.sort(key=lambda x: (x['compression_ratio'], x['type'], x['name']))

    summary_path = output_dir / 'tables' / 'results_summary_8bit.csv'
    csv_columns = ['name', 'type', 'compression_ratio', 'n_samples', 'psnr_mean', 'psnr_std',
                   'ssim_mean', 'ssim_std', 'ms_ssim_mean', 'enl_ratio_mean', 'epi_mean']
    with open(summary_path, 'w') as f:
        f.write(','.join(csv_columns) + '\n')
        for row in rows:
            values = [str(row.get(col, '')) if row.get(col) is not None else '' for col in csv_columns]
            f.write(','.join(values) + '\n')
    print(f"Summary CSV: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("8-BIT QUANTIZED COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Name':<20} {'Ratio':>6} {'PSNR (dB)':>12} {'SSIM':>10} {'EPI':>10}")
    print("-" * 62)
    for row in rows:
        psnr_str = f"{row['psnr_mean']:.2f}" if row['psnr_mean'] else "N/A"
        ssim_str = f"{row['ssim_mean']:.4f}" if row['ssim_mean'] else "N/A"
        epi_str = f"{row['epi_mean']:.4f}" if row['epi_mean'] else "N/A"
        print(f"{row['name']:<20} {row['compression_ratio']:>6.0f} {psnr_str:>12} {ssim_str:>10} {epi_str:>10}")

    print("\n" + "=" * 70)
    print("8-BIT EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
