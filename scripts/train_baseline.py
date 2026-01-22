#!/usr/bin/env python
"""
Train baseline SAR autoencoder (Phase 2).

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --epochs 100 --latent_channels 16
    python scripts/train_baseline.py --resume checkpoints/baseline_c16/latest.pth
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datamodule import SARDataModule
from src.models.autoencoder import SARAutoencoder
from src.losses.combined import CombinedLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline SAR autoencoder')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='D:/Projects/CNNAutoencoderProject/data/processed/metadata.npy',
                        help='Path to metadata.npy or patches.npy')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8 for 8GB VRAM)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for Windows)')

    # Model
    parser.add_argument('--latent_channels', type=int, default=16,
                        help='Latent channels (16 = 16x compression)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count')

    # Loss
    parser.add_argument('--mse_weight', type=float, default=0.5,
                        help='MSE loss weight')
    parser.add_argument('--ssim_weight', type=float, default=0.5,
                        help='SSIM loss weight')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Output
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name (default: auto-generated)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Print config
    print("=" * 60)
    print("Baseline SAR Autoencoder Training")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent channels: {args.latent_channels} ({256*256 // (16*16*args.latent_channels):.0f}x compression)")
    print(f"Loss weights: MSE={args.mse_weight}, SSIM={args.ssim_weight}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    # Data
    print("\nLoading data...")
    dm = SARDataModule(
        patches_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=0.1,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("\nCreating model...")
    model = SARAutoencoder(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
    )
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,}")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"Compression ratio: {model.get_compression_ratio():.1f}x")

    # Loss
    loss_fn = CombinedLoss(
        mse_weight=args.mse_weight,
        ssim_weight=args.ssim_weight,
    )

    # Config for trainer
    config = {
        'learning_rate': args.learning_rate,
        'lr_patience': 10,
        'lr_factor': 0.5,
        'max_grad_norm': 1.0,
        'run_name': args.run_name or f'baseline_c{args.latent_channels}',
        'preprocessing_params': dm.preprocessing_params,
        # Store hyperparams for reproducibility
        'latent_channels': args.latent_channels,
        'base_channels': args.base_channels,
        'mse_weight': args.mse_weight,
        'ssim_weight': args.ssim_weight,
        'batch_size': args.batch_size,
    }

    # Trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
    )

    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir={trainer.log_dir.parent}")
    print(f"Checkpoints: {trainer.checkpoint_dir}")
    print()

    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    if history:
        final = history[-1]
        print(f"Final epoch: {final['epoch'] + 1}")
        print(f"Best val loss: {trainer.best_val_loss:.4f}")
        print(f"Final val PSNR: {final['val_psnr']:.2f} dB")
        print(f"Final val SSIM: {final['val_ssim']:.4f}")

        # Check success criterion
        if final['val_psnr'] >= 25:
            print("\n[SUCCESS] PSNR > 25 dB achieved!")
        else:
            print(f"\n[WARNING] PSNR {final['val_psnr']:.2f} < 25 dB target")

    print(f"\nBest checkpoint: {trainer.checkpoint_dir / 'best.pth'}")


if __name__ == '__main__':
    main()
