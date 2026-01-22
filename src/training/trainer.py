"""
Training Manager for SAR Autoencoder

This module implements the main training loop with:
- Automatic GPU detection
- Learning rate scheduling
- Checkpointing (best and latest)
- TensorBoard logging
- Early stopping
- Gradient clipping

References:
    - Day 2, Section 2.7 of the learning guide
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import json
from tqdm import tqdm
import logging


class Trainer:
    """
    Training manager for SAR autoencoder.

    Features:
    - Automatic GPU detection
    - Learning rate scheduling (ReduceLROnPlateau)
    - Checkpointing (saves best and latest models)
    - TensorBoard logging
    - Early stopping
    - Gradient clipping

    Args:
        model: Autoencoder model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: Loss function (should return (loss, metrics))
        config: Training configuration dict
        device: Device ('cuda', 'cpu', or None for auto)

    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, loss_fn, config)
        >>> trainer.train(epochs=50, early_stopping_patience=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        config: Dict,
        device: Optional[str] = None
    ):
        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Move model and loss to device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        self.config = config

        # Store preprocessing params (critical for SAR data - enables correct inference)
        self.preprocessing_params = config.get('preprocessing_params', None)

        # Optimizer (FR3.3: Adam with configurable lr, default 1e-4)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0),
        )

        # Scheduler (FR3.4: ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 10),
            verbose=False,  # Handle logging manually via TensorBoard
        )

        # Logging setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = config.get('run_name', f'baseline_{timestamp}')
        self.log_dir = Path(config.get('log_dir', 'runs')) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # File logging (per CONTEXT.md: save training logs to file)
        self.log_file = self.log_dir / 'training.log'

        # Configure file handler for this trainer's logger
        self.logger = logging.getLogger(f'trainer.{self.run_name}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(stream_handler)

        # Save config to log directory (convert numpy types for JSON serialization)
        def convert_to_json_serializable(obj):
            """Convert numpy types to Python native types for JSON."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(x) for x in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(convert_to_json_serializable(config), f, indent=2)

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints')) / self.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history: List[Dict] = []

        # Store fixed sample batch for consistent visualization
        self._sample_batch = None

        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict of averaged metrics for the epoch
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]",
                    leave=True, dynamic_ncols=True)

        for batch in pbar:
            x = batch.to(self.device, non_blocking=True)

            # Forward
            self.optimizer.zero_grad()
            x_hat, z = self.model(x)
            loss, metrics = self.loss_fn(x_hat, x)

            # Backward
            loss.backward()

            # Gradient clipping (FR3.5: max norm 1.0)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )

            # Update
            self.optimizer.step()

            # Accumulate metrics
            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1
            self.global_step += 1

            # Update progress bar (per CONTEXT.md: show loss + PSNR + SSIM)
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'ssim': f"{metrics['ssim']:.4f}"
            })

        return {k: v / num_batches for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict of averaged metrics
        """
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]",
                    leave=True, dynamic_ncols=True)

        for batch in pbar:
            x = batch.to(self.device, non_blocking=True)
            x_hat, z = self.model(x)
            loss, metrics = self.loss_fn(x_hat, x)

            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'ssim': f"{metrics['ssim']:.4f}"
            })

        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': self.config,
            'preprocessing_params': self.preprocessing_params,  # Critical for SAR!
            'history': self.history,
        }

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        self.history = checkpoint.get('history', [])

        self.logger.info(f"Resumed from epoch {checkpoint['epoch']} (best_val_loss={self.best_val_loss:.4f})")

    @torch.no_grad()
    def log_images(self, num_images: int = 4):
        """Log sample reconstructions to TensorBoard."""
        self.model.eval()

        # Use fixed sample batch for consistent visualization
        if self._sample_batch is None:
            self._sample_batch = next(iter(self.val_loader))[:num_images]

        x = self._sample_batch.to(self.device)
        x_hat, z = self.model(x)

        # Create triple view: original | reconstructed | difference (per CONTEXT.md)
        diff = torch.abs(x - x_hat)

        # Stack vertically: [originals, reconstructions, differences]
        # Each row is num_images wide
        combined = torch.cat([x, x_hat, diff], dim=0)

        # Make grid: nrow=num_images means each row has num_images samples
        grid = vutils.make_grid(combined, nrow=num_images, normalize=True, padding=2)
        self.writer.add_image('Reconstructions/triple_view', grid, self.epoch)

    def _log_weight_histograms(self):
        """Log weight histograms to TensorBoard (per CONTEXT.md: every 10 epochs)."""
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, self.epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, self.epoch)

    def _log_gpu_memory(self) -> Dict[str, float]:
        """Log GPU memory usage (per CONTEXT.md: log GPU memory in epoch summary)."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return {'gpu_allocated_gb': allocated, 'gpu_reserved_gb': reserved}
        return {}

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def train(self, epochs: int, early_stopping_patience: int = 20) -> List[Dict]:
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs

        Returns:
            List of epoch history dictionaries
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Config: {self.config}")

        start_epoch = self.epoch

        for epoch in range(start_epoch, epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log scalars to TensorBoard (per CONTEXT.md: every epoch)
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

            # Log learning rate
            current_lr = self._get_lr()
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)

            # Log GPU memory (per CONTEXT.md)
            gpu_mem = self._log_gpu_memory()
            for key, value in gpu_mem.items():
                self.writer.add_scalar(f'system/{key}', value, epoch)

            # Log reconstruction images (per CONTEXT.md: every epoch)
            self.log_images(num_images=4)

            # Log weight histograms (per CONTEXT.md: every 10 epochs)
            if epoch % 10 == 0:
                self._log_weight_histograms()

            # Update scheduler (FR3.4)
            self.scheduler.step(val_metrics['loss'])

            # Check for improvement (FR3.8: best model tracking)
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint (FR3.7)
            self.save_checkpoint(is_best=is_best)

            # Store history
            self.history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_psnr': train_metrics.get('psnr', 0),
                'val_psnr': val_metrics.get('psnr', 0),
                'train_ssim': train_metrics.get('ssim', 0),
                'val_ssim': val_metrics.get('ssim', 0),
                'learning_rate': current_lr,
            })

            # Print epoch summary (per CONTEXT.md: with GPU memory)
            gpu_str = f" | GPU: {gpu_mem.get('gpu_allocated_gb', 0):.2f}GB" if gpu_mem else ""
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: loss={train_metrics['loss']:.4f}, psnr={train_metrics.get('psnr', 0):.2f}, ssim={train_metrics.get('ssim', 0):.4f} | "
                f"Val: loss={val_metrics['loss']:.4f}, psnr={val_metrics.get('psnr', 0):.2f}, ssim={val_metrics.get('ssim', 0):.4f} | "
                f"LR: {current_lr:.2e}{gpu_str}"
            )

            # Early stopping (FR3.6)
            if self.epochs_without_improvement >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                break

        # Cleanup
        self.writer.close()
        self.logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

        return self.history


def test_trainer():
    """Test trainer with dummy data."""
    print("Testing Trainer...")
    print("(Requires model and data to be implemented)")


if __name__ == "__main__":
    test_trainer()
