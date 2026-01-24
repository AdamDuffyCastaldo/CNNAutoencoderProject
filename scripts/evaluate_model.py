#!/usr/bin/env python3
"""
Evaluate trained SAR autoencoder model.

This script provides comprehensive evaluation of trained models including:
- Full metrics computation (PSNR, SSIM, MS-SSIM, ENL ratio, EPI)
- Visual comparisons with zoomed crops
- Rate-distortion curve comparison with traditional codecs
- JSON output for reproducible experiments

Usage:
    # Basic evaluation
    python scripts/evaluate_model.py --checkpoint notebooks/checkpoints/resnet_lite_v2_c16/best.pth

    # With codec comparison
    python scripts/evaluate_model.py --checkpoint path/to/best.pth --compare-codecs

    # Limited samples for quick test
    python scripts/evaluate_model.py --checkpoint path/to/best.pth --n-samples 100 --n-visualizations 3

Output Structure:
    evaluations/{model_name}/
        {model_name}_eval.json       # Summary metrics (mean/std/min/max)
        {model_name}_detailed.json   # Per-sample metrics
        rate_distortion.csv          # R-D data (if --compare-codecs)
        rate_distortion.png          # R-D plot (if --compare-codecs)
        comparisons/
            sample_01.png
            sample_02.png
            ...
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.data.datamodule import SARDataModule
from src.evaluation import Evaluator, Visualizer
from src.evaluation.evaluator import print_evaluation_report
from src.evaluation.codec_baselines import JPEG2000Codec, JPEGCodec, CodecEvaluator
from src.evaluation.metrics import compute_bpp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained SAR autoencoder model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')

    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: evaluations/{checkpoint_name})')

    # Data configuration
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to test data (default: uses datamodule default)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')

    # Codec comparison
    parser.add_argument('--compare-codecs', action='store_true',
                        help='Also evaluate JPEG-2000 and JPEG at same compression ratio')
    parser.add_argument('--compression-ratios', type=str, default='8,16,32',
                        help='Comma-separated compression ratios for codec comparison')

    # Visualization
    parser.add_argument('--n-visualizations', type=int, default=5,
                        help='Number of comparison images to generate (default: 5)')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Skip visualization generation')

    # Device configuration
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available)')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint, supporting both SARAutoencoder and ResNetAutoencoder.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, model_name, preprocessing_params)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    latent_channels = model_config.get('latent_channels', 16)
    base_channels = model_config.get('base_channels', 64)
    model_type = model_config.get('model_type', None)

    # Try to determine model type from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Check for ResNet-style keys
    has_resnet_keys = any('stage1' in k or 'stage2' in k for k in state_dict.keys())

    if model_type == 'resnet' or has_resnet_keys:
        from src.models.resnet_autoencoder import ResNetAutoencoder
        model = ResNetAutoencoder(
            in_channels=1,
            base_channels=base_channels,
            latent_channels=latent_channels
        )
        model_name = f"resnet_c{latent_channels}"
    else:
        from src.models.autoencoder import SARAutoencoder
        model = SARAutoencoder(
            in_channels=1,
            latent_channels=latent_channels
        )
        model_name = f"baseline_c{latent_channels}"

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Extract preprocessing params
    preprocessing_params = checkpoint.get('preprocessing_params', {})

    # Try to get model name from checkpoint metadata
    saved_name = checkpoint.get('model_name', None)
    if saved_name:
        model_name = saved_name

    return model, model_name, preprocessing_params


def get_test_images(dataloader, n_samples: int = 100):
    """Extract images from dataloader for codec evaluation."""
    images = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        for i in range(batch.shape[0]):
            images.append(batch[i, 0].numpy())
            if len(images) >= n_samples:
                return images
    return images


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 70)
    print("SAR AUTOENCODER EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading model from checkpoint...")
    try:
        model, model_name, preprocessing_params = load_model_from_checkpoint(
            args.checkpoint, device
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    print(f"Model: {model_name}")

    # Get compression ratio
    if hasattr(model, 'get_compression_ratio'):
        compression_ratio = model.get_compression_ratio()
        print(f"Compression ratio: {compression_ratio:.1f}x")
    else:
        compression_ratio = 16.0  # Default assumption
        print(f"Compression ratio: {compression_ratio:.1f}x (assumed)")

    # Count parameters
    if hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        print(f"Parameters: {params['total']:,}")
    else:
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Use checkpoint name as output subdirectory
        ckpt_name = Path(args.checkpoint).parent.name
        if ckpt_name == 'checkpoints':
            ckpt_name = Path(args.checkpoint).stem
        output_dir = Path('evaluations') / ckpt_name

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load test data
    print("\nLoading test data...")
    if args.data_dir:
        data_path = args.data_dir
    else:
        # Try common locations
        for path in ['data/patches/metadata.npy', 'data/patches/patches.npy']:
            if Path(path).exists():
                data_path = path
                break
        else:
            print("Error: No data found. Specify --data-dir")
            sys.exit(1)

    try:
        datamodule = SARDataModule(
            patches_path=data_path,
            batch_size=args.batch_size,
            val_fraction=0.1,
            num_workers=0,  # Windows compatibility
            lazy=Path(data_path).name == 'metadata.npy',
            max_samples=args.n_samples
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Use validation set for evaluation (separate from training)
    test_loader = datamodule.val_dataloader()
    print(f"Evaluation samples: {datamodule.val_size}")

    # Create evaluator
    evaluator = Evaluator(
        model,
        device=str(device),
        model_name=model_name,
        checkpoint_path=str(args.checkpoint),
        preprocessing_params=preprocessing_params
    )

    # Evaluate model
    print("\nEvaluating model...")
    results = evaluator.evaluate_dataset(test_loader, show_progress=True)

    # Print report
    print_evaluation_report(results)

    # Save results - CRITICAL: produces JSON files
    print("\nSaving results...")
    saved_paths = evaluator.save_results(results, str(output_dir), model_name)
    print(f"  Summary: {saved_paths['summary']}")
    print(f"  Detailed: {saved_paths['detailed']}")

    # Generate visualizations
    if not args.no_visualizations and args.n_visualizations > 0:
        print(f"\nGenerating {args.n_visualizations} visualizations...")

        comparisons_dir = output_dir / 'comparisons'
        comparisons_dir.mkdir(exist_ok=True)
        visualizer = Visualizer(save_dir=str(comparisons_dir))

        # Get sample reconstructions
        samples = evaluator.get_reconstructions(test_loader, n_samples=args.n_visualizations)

        for i, (orig, recon, metrics) in enumerate(samples):
            visualizer.plot_comparison(
                orig, recon, metrics,
                save_path=f'sample_{i+1:02d}.png',
                auto_zoom=True,
                show=False
            )
        print(f"  Saved to: {comparisons_dir}")

    # Codec comparison
    if args.compare_codecs:
        print("\nComparing with traditional codecs...")

        # Parse compression ratios
        target_ratios = [float(r) for r in args.compression_ratios.split(',')]
        print(f"Target ratios: {target_ratios}")

        # Collect rate-distortion data
        rd_data = []

        # Add autoencoder point
        ae_rd = evaluator.collect_rd_point(results)
        rd_data.append(ae_rd)
        print(f"\n{model_name}:")
        print(f"  BPP: {ae_rd['bpp']:.2f}, PSNR: {ae_rd['psnr']:.2f} dB, SSIM: {ae_rd['ssim']:.4f}")

        # Get test images for codec evaluation
        print("\nExtracting test images for codec evaluation...")
        test_images = get_test_images(test_loader, n_samples=min(200, datamodule.val_size))
        print(f"Using {len(test_images)} images for codec evaluation")

        # Evaluate each codec
        for CodecClass, codec_name in [(JPEG2000Codec, 'JPEG-2000'), (JPEGCodec, 'JPEG')]:
            print(f"\nEvaluating {codec_name}...")
            try:
                codec = CodecClass()
                codec_evaluator = CodecEvaluator(codec)

                # Calibrate for all target ratios
                codec_evaluator.calibrate(target_ratios, test_images[:5])

                for target_ratio in target_ratios:
                    codec_results = codec_evaluator.evaluate_batch(
                        test_images[:100],  # Limit for speed
                        target_ratio,
                        show_progress=True
                    )

                    # Merge codec results into rd_data
                    metrics = codec_results['metrics']
                    rd_data.append({
                        'name': codec_name,
                        'compression_ratio': target_ratio,
                        'bpp': 32.0 / target_ratio,
                        'psnr': metrics['psnr']['mean'],
                        'ssim': metrics['ssim']['mean'],
                        'achieved_ratio': metrics['achieved_ratio']['mean']
                    })

                    print(f"  {target_ratio}x: PSNR={metrics['psnr']['mean']:.2f} dB, "
                          f"SSIM={metrics['ssim']['mean']:.4f}")

            except Exception as e:
                print(f"  Warning: {codec_name} evaluation failed: {e}")

        # Save rate-distortion data to CSV
        try:
            import pandas as pd
            rd_df = pd.DataFrame(rd_data)
            rd_csv_path = output_dir / 'rate_distortion.csv'
            rd_df.to_csv(rd_csv_path, index=False)
            print(f"\nRate-distortion data saved to: {rd_csv_path}")
        except ImportError:
            # Fallback: save as JSON
            rd_json_path = output_dir / 'rate_distortion.json'
            with open(rd_json_path, 'w') as f:
                json.dump(rd_data, f, indent=2)
            print(f"\nRate-distortion data saved to: {rd_json_path}")

        # Generate rate-distortion plot
        visualizer = Visualizer(save_dir=str(output_dir))
        visualizer.plot_rate_distortion(
            rd_data,
            output_path='rate_distortion.png',
            title=f'Rate-Distortion: {model_name} vs Traditional Codecs',
            show=False
        )

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey metrics:")
    metrics = results.get('metrics', {})
    print(f"  PSNR: {metrics.get('psnr', {}).get('mean', 'N/A'):.2f} dB")
    print(f"  SSIM: {metrics.get('ssim', {}).get('mean', 'N/A'):.4f}")
    print(f"  MS-SSIM: {metrics.get('ms_ssim', {}).get('mean', 'N/A'):.4f}")
    print(f"  EPI: {metrics.get('epi', {}).get('mean', 'N/A'):.4f}")
    print(f"  ENL ratio: {metrics.get('enl_ratio', {}).get('mean', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
