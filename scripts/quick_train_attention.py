#!/usr/bin/env python
"""
Quick training script for AttentionAutoencoder (Variant C)
Runs minimal training to verify setup, then exits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Config
BATCH_SIZE = 8  # Small batch for quick test
MAX_BATCHES = 50  # Just 50 batches for verification
LATENT_CHANNELS = 16
BASE_CHANNELS = 64
MSE_WEIGHT = 0.7
SSIM_WEIGHT = 0.3
LR = 1e-4
SEED = 42
RUN_NAME = 'attention_v1_c16'

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("=" * 60)
    print("Quick Training Test: AttentionAutoencoder (Variant C)")
    print("=" * 60)

    # Import
    from src.data.datamodule import SARDataModule
    from src.models import AttentionAutoencoder
    from src.losses.combined import CombinedLoss

    # Data
    DATA_PATH = "D:/Projects/CNNAutoencoderProject/data/patches/metadata.npy"
    print(f"\nLoading data from: {DATA_PATH}")
    dm = SARDataModule(patches_path=DATA_PATH, batch_size=BATCH_SIZE, num_workers=0, val_fraction=0.1)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")

    # Model
    print("\nCreating AttentionAutoencoder...")
    model = AttentionAutoencoder(latent_channels=LATENT_CHANNELS, base_channels=BASE_CHANNELS)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    cbam_count = sum(1 for m in model.modules() if m.__class__.__name__ == 'CBAM')
    print(f"CBAM modules: {cbam_count}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Device: {device}")

    # Loss and optimizer
    loss_fn = CombinedLoss(mse_weight=MSE_WEIGHT, ssim_weight=SSIM_WEIGHT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\nLoss: {MSE_WEIGHT} MSE + {SSIM_WEIGHT} SSIM")
    print(f"Learning rate: {LR}")

    # Quick training
    print(f"\n{'=' * 60}")
    print(f"Training {MAX_BATCHES} batches...")
    print("=" * 60)

    model.train()
    train_losses = []
    train_psnrs = []

    for i, batch in enumerate(tqdm(train_loader, total=MAX_BATCHES, desc="Training")):
        if i >= MAX_BATCHES:
            break

        x = batch.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            x_hat, z = model(x)
            loss, metrics = loss_fn(x_hat, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(metrics['loss'])
        train_psnrs.append(metrics['psnr'])

    # Quick validation
    print("\nValidating...")
    model.eval()
    val_losses = []
    val_psnrs = []
    val_ssims = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=MAX_BATCHES, desc="Validation")):
            if i >= MAX_BATCHES:
                break

            x = batch.to(device)
            with torch.amp.autocast('cuda'):
                x_hat, z = model(x)
            x_hat = x_hat.float()
            x = x.float()
            loss, metrics = loss_fn(x_hat, x)

            val_losses.append(metrics['loss'])
            val_psnrs.append(metrics['psnr'])
            val_ssims.append(metrics['ssim'])

    # Summary
    print("\n" + "=" * 60)
    print("QUICK TRAINING RESULTS")
    print("=" * 60)
    print(f"Train Loss: {np.mean(train_losses):.4f}")
    print(f"Train PSNR: {np.mean(train_psnrs):.2f} dB")
    print(f"Val Loss: {np.mean(val_losses):.4f}")
    print(f"Val PSNR: {np.mean(val_psnrs):.2f} dB")
    print(f"Val SSIM: {np.mean(val_ssims):.4f}")

    # Save checkpoint
    checkpoint_dir = Path("notebooks/checkpoints") / RUN_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0,  # Not a full epoch
        'best_val_loss': np.mean(val_losses),
        'preprocessing_params': dm.preprocessing_params,
        'config': {
            'model_type': 'AttentionAutoencoder-Variant-C',
            'latent_channels': LATENT_CHANNELS,
            'base_channels': BASE_CHANNELS,
            'mse_weight': MSE_WEIGHT,
            'ssim_weight': SSIM_WEIGHT,
            'batch_size': BATCH_SIZE,
        },
        'quick_test': True,
        'batches_trained': MAX_BATCHES,
    }

    checkpoint_path = checkpoint_dir / "quick_test.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("Quick training test PASSED!")
    print("Model and training pipeline verified.")
    print("=" * 60)
    print("\nFor full training, run: notebooks/train_attention.ipynb")
    print("Expected full training time: ~60+ hours for 30 epochs")

    return np.mean(val_psnrs), np.mean(val_ssims)


if __name__ == "__main__":
    main()
