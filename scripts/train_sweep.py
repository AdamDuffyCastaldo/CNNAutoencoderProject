#!/usr/bin/env python
"""
Training sweep script for SAR autoencoder architecture comparison.

Supports single runs and YAML-based sweep configurations for automated
multi-config training (overnight batches).

Usage:
    # Single run
    python scripts/train_sweep.py --model baseline --latent-channels 16 --base-channels 64

    # Sweep from YAML config
    python scripts/train_sweep.py --sweep configs/sweep_baseline_ratios.yaml

    # Smoke test (1 epoch, small data subset)
    python scripts/train_sweep.py --model baseline --latent-channels 16 --epochs 1 --max-samples 1000
"""

import argparse
import gc
import sys
import time
import yaml
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datamodule import SARDataModule
from src.losses.combined import CombinedLoss
from src.training.trainer import Trainer
from src.models.autoencoder import SARAutoencoder
from src.models.resnet_autoencoder import ResNetAutoencoder
from src.models.residual_autoencoder import ResidualAutoencoder
from src.models.attention_autoencoder import AttentionAutoencoder


MODEL_REGISTRY = {
    'baseline': SARAutoencoder,
    'resnet': ResNetAutoencoder,
    'residual': ResidualAutoencoder,
    'attention': AttentionAutoencoder,
}


def create_model(model_name: str, latent_channels: int, base_channels: int):
    """Create model by name with appropriate constructor args."""
    cls = MODEL_REGISTRY[model_name]
    if model_name == 'baseline':
        return cls(latent_channels=latent_channels, base_channels=base_channels)
    elif model_name == 'attention':
        return cls(in_channels=1, base_channels=base_channels,
                   latent_channels=latent_channels, reduction=16)
    else:
        # resnet, residual
        return cls(in_channels=1, base_channels=base_channels,
                   latent_channels=latent_channels)


def compute_compression_ratio(latent_channels: int) -> float:
    """Compute compression ratio from latent channels. 256x256x1 -> 16x16xLC."""
    return (256 * 256) / (16 * 16 * latent_channels)


def make_run_name(model_name: str, latent_channels: int, base_channels: int) -> str:
    """Generate standardized run name: {model}_c{LC}_b{BC}_cr{ratio}x."""
    ratio = compute_compression_ratio(latent_channels)
    return f"{model_name}_c{latent_channels}_b{base_channels}_cr{ratio:.0f}x"


def run_training(run_cfg: dict, data_module: SARDataModule) -> dict:
    """Execute a single training run. Returns summary dict."""
    model_name = run_cfg['model']
    latent_channels = run_cfg['latent_channels']
    base_channels = run_cfg['base_channels']
    lr = run_cfg['learning_rate']
    epochs = run_cfg['epochs']
    early_stopping = run_cfg['early_stopping_patience']
    mse_weight = run_cfg.get('mse_weight', 0.5)
    ssim_weight = run_cfg.get('ssim_weight', 0.5)

    ratio = compute_compression_ratio(latent_channels)
    run_name = make_run_name(model_name, latent_channels, base_channels)

    print("\n" + "=" * 70)
    print(f"  TRAINING: {run_name}")
    print(f"  Model: {model_name} | LC={latent_channels} | BC={base_channels} | {ratio:.0f}x compression")
    print(f"  LR={lr} | Epochs={epochs} | Patience={early_stopping}")
    print("=" * 70)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create model
    model = create_model(model_name, latent_channels, base_channels)
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,}")

    # Loss
    loss_fn = CombinedLoss(mse_weight=mse_weight, ssim_weight=ssim_weight)

    # Trainer config
    config = {
        'learning_rate': lr,
        'optimizer': run_cfg.get('optimizer', 'adamw'),
        'scheduler': run_cfg.get('scheduler', 'plateau'),
        'lr_patience': run_cfg.get('lr_patience', 10),
        'lr_factor': run_cfg.get('lr_factor', 0.5),
        'max_grad_norm': run_cfg.get('max_grad_norm', 1.0),
        'use_amp': run_cfg.get('use_amp', True),
        'warmup_epochs': run_cfg.get('warmup_epochs', 0),
        'run_name': run_name,
        'preprocessing_params': data_module.preprocessing_params,
        # Store hyperparams for reproducibility
        'model_type': model_name,
        'latent_channels': latent_channels,
        'base_channels': base_channels,
        'mse_weight': mse_weight,
        'ssim_weight': ssim_weight,
        'batch_size': data_module.batch_size,
        'compression_ratio': ratio,
    }

    # OneCycleLR needs epochs count in config
    if config['scheduler'] == 'onecycle':
        config['epochs'] = epochs

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )

    # Train
    t0 = time.time()
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping)
    elapsed = time.time() - t0

    # Build summary
    summary = {
        'run_name': trainer.run_name,
        'model': model_name,
        'latent_channels': latent_channels,
        'base_channels': base_channels,
        'compression_ratio': ratio,
        'parameters': params['total'],
        'learning_rate': lr,
        'epochs_trained': len(history),
        'elapsed_seconds': elapsed,
        'checkpoint': str(trainer.checkpoint_dir / 'best.pth'),
    }

    if history:
        best_epoch = min(history, key=lambda h: h.get('val_loss', float('inf')))
        summary['best_val_loss'] = best_epoch.get('val_loss', None)
        summary['best_psnr'] = best_epoch.get('val_psnr', None)
        summary['best_ssim'] = best_epoch.get('val_ssim', None)

        final = history[-1]
        summary['final_psnr'] = final.get('val_psnr', None)
        summary['final_ssim'] = final.get('val_ssim', None)

    # Print summary
    print(f"\n  --- Run Complete: {trainer.run_name} ---")
    print(f"  Epochs: {summary['epochs_trained']} | Time: {elapsed/60:.1f} min")
    if summary.get('best_psnr') is not None:
        print(f"  Best PSNR: {summary['best_psnr']:.2f} dB | Best SSIM: {summary['best_ssim']:.4f}")
    print(f"  Checkpoint: {summary['checkpoint']}")

    # Cleanup GPU memory
    del model, trainer, loss_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def load_sweep_config(sweep_path: str) -> list:
    """Load sweep YAML and expand into list of run configs."""
    with open(sweep_path, 'r') as f:
        sweep = yaml.safe_load(f)

    defaults = sweep.get('defaults', {})
    runs = []

    for run in sweep['runs']:
        # Merge defaults with per-run overrides
        cfg = {**defaults, **run}
        runs.append(cfg)

    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description='SAR Autoencoder Training Sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single baseline run at 16x compression
  python scripts/train_sweep.py --model baseline --latent-channels 16

  # Sweep all baseline compression ratios
  python scripts/train_sweep.py --sweep configs/sweep_baseline_ratios.yaml

  # Quick smoke test
  python scripts/train_sweep.py --model baseline --latent-channels 16 --epochs 1 --max-samples 1000
        """)

    # Mode selection
    parser.add_argument('--sweep', type=str, default=None,
                        help='Path to sweep YAML config (overrides single-run args)')

    # Single run params
    parser.add_argument('--model', type=str, default='baseline',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture (default: baseline)')
    parser.add_argument('--latent-channels', type=int, default=16,
                        help='Latent channels (4=64x, 8=32x, 16=16x, 32=8x, 64=4x)')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base feature channels (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs (default: 50)')
    parser.add_argument('--early-stopping', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'onecycle'],
                        help='LR scheduler (default: plateau)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer (default: adamw)')

    # Data
    parser.add_argument('--data-path', type=str,
                        default='D:/Projects/CNNAutoencoderProject/data/patches/metadata.npy',
                        help='Path to metadata.npy')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit dataset size (for testing)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for Windows)')

    # Skip existing
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip runs that already have checkpoints')

    return parser.parse_args()


def check_existing_run(run_name: str) -> bool:
    """Check if a run with this base name already has a checkpoint."""
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        return False
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith(run_name):
            best = d / 'best.pth'
            if best.exists():
                return True
    return False


def main():
    args = parse_args()

    print("=" * 70)
    print("  SAR Autoencoder Training Sweep")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # Build run list
    if args.sweep:
        print(f"\n  Loading sweep config: {args.sweep}")
        runs = load_sweep_config(args.sweep)
        print(f"  Total runs: {len(runs)}")
    else:
        runs = [{
            'model': args.model,
            'latent_channels': args.latent_channels,
            'base_channels': args.base_channels,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'early_stopping_patience': args.early_stopping,
            'batch_size': args.batch_size,
            'scheduler': args.scheduler,
            'optimizer': args.optimizer,
        }]

    # Apply CLI overrides to sweep runs (epochs, max-samples can override)
    if args.sweep and args.epochs != 50:
        print(f"  CLI override: epochs={args.epochs}")
        for r in runs:
            r['epochs'] = args.epochs

    # Print run plan
    print("\n  Run Plan:")
    for i, r in enumerate(runs, 1):
        name = make_run_name(r['model'], r['latent_channels'], r.get('base_channels', 64))
        ratio = compute_compression_ratio(r['latent_channels'])
        print(f"    {i}. {name} (LR={r.get('learning_rate', 1e-4)}, {ratio:.0f}x)")

    # Load data once (shared across all runs)
    batch_size = runs[0].get('batch_size', args.batch_size)
    print(f"\n  Loading data (batch_size={batch_size})...")
    dm = SARDataModule(
        patches_path=args.data_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        val_fraction=0.1,
        max_samples=args.max_samples,
    )
    print(f"  Preprocessing params: {dm.preprocessing_params}")

    # Execute runs
    summaries = []
    skipped = 0

    for i, run_cfg in enumerate(runs, 1):
        # Fill in defaults
        run_cfg.setdefault('base_channels', 64)
        run_cfg.setdefault('learning_rate', 1e-4)
        run_cfg.setdefault('epochs', 50)
        run_cfg.setdefault('early_stopping_patience', 15)
        run_cfg.setdefault('batch_size', batch_size)
        run_cfg.setdefault('scheduler', 'plateau')
        run_cfg.setdefault('optimizer', 'adamw')

        run_name = make_run_name(run_cfg['model'], run_cfg['latent_channels'],
                                 run_cfg['base_channels'])

        # Check if run needs different batch size — reload data if needed
        if run_cfg['batch_size'] != dm.batch_size:
            print(f"\n  Reloading data with batch_size={run_cfg['batch_size']}...")
            dm = SARDataModule(
                patches_path=args.data_path,
                batch_size=run_cfg['batch_size'],
                num_workers=args.num_workers,
                val_fraction=0.1,
                max_samples=args.max_samples,
            )

        # Skip check
        if args.skip_existing and check_existing_run(run_name):
            print(f"\n  [{i}/{len(runs)}] SKIP: {run_name} (checkpoint exists)")
            skipped += 1
            continue

        print(f"\n  [{i}/{len(runs)}] Starting: {run_name}")

        try:
            summary = run_training(run_cfg, dm)
            summaries.append(summary)
        except Exception as e:
            print(f"\n  [ERROR] {run_name} failed: {e}")
            summaries.append({
                'run_name': run_name,
                'model': run_cfg['model'],
                'latent_channels': run_cfg['latent_channels'],
                'base_channels': run_cfg['base_channels'],
                'compression_ratio': compute_compression_ratio(run_cfg['latent_channels']),
                'error': str(e),
            })
            # Clean up after failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 70)
    print("  SWEEP COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if skipped:
        print(f"  Skipped: {skipped} (existing checkpoints)")

    print(f"\n  {'Model':<35} {'Ratio':>6} {'PSNR':>8} {'SSIM':>8} {'Time':>8}")
    print("  " + "-" * 67)

    for s in summaries:
        name = s.get('run_name', 'unknown')
        ratio = f"{s.get('compression_ratio', 0):.0f}x"
        if 'error' in s:
            print(f"  {name:<35} {ratio:>6} {'FAILED':>8} {'':>8} {'':>8}")
        else:
            psnr = f"{s.get('best_psnr', 0):.2f}" if s.get('best_psnr') else "N/A"
            ssim = f"{s.get('best_ssim', 0):.4f}" if s.get('best_ssim') else "N/A"
            mins = f"{s.get('elapsed_seconds', 0)/60:.0f}m"
            print(f"  {name:<35} {ratio:>6} {psnr:>8} {ssim:>8} {mins:>8}")

    print()


if __name__ == '__main__':
    main()
