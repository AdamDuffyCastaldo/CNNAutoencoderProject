"""
Training Manager for SAR Autoencoder

This module implements the main training loop with:
- Automatic GPU detection
- Learning rate scheduling with optional warmup
- Checkpointing (best and latest)
- TensorBoard logging
- Early stopping
- Gradient clipping
- Mixed Precision Training (AMP) for ~2x speedup

References:
    - Day 2, Section 2.7 of the learning guide
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
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
    - Timestamped checkpoint archives (prevents accidental overwrites)
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

        # Check GPU memory at startup (warn about contention)
        if torch.cuda.is_available() and device == 'cuda':
            self._check_gpu_memory()

        # Move model and loss to device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        self.config = config

        # Store preprocessing params (critical for SAR data - enables correct inference)
        self.preprocessing_params = config.get('preprocessing_params', None)

        # Learning rate and warmup config
        self.base_lr = config.get('learning_rate', 1e-4)
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.warmup_start_lr = self.base_lr / 10  # Start at 10% of target LR

        # Optimizer (FR3.3: Adam/AdamW with configurable lr)
        optimizer_type = config.get('optimizer', 'adam').lower()
        weight_decay = config.get('weight_decay', 0)

        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.base_lr,
                betas=config.get('betas', (0.9, 0.999)),
                eps=1e-8,
                weight_decay=weight_decay if weight_decay > 0 else 1e-5,
            )
            print(f"Using AdamW optimizer with weight_decay={self.optimizer.param_groups[0]['weight_decay']}")
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.base_lr,
                betas=config.get('betas', (0.9, 0.999)),
                weight_decay=weight_decay,
            )

        if self.warmup_epochs > 0:
            print(f"Learning rate warmup: {self.warmup_epochs} epochs ({self.warmup_start_lr:.2e} -> {self.base_lr:.2e})")

        # Scheduler (FR3.4: ReduceLROnPlateau, applied after warmup)
        self.scheduler_type = config.get('scheduler', 'plateau').lower()

        if self.scheduler_type == 'onecycle':
            # OneCycleLR: steps per batch, includes built-in warmup
            steps_per_epoch = len(train_loader)
            total_steps = config.get('epochs', 50) * steps_per_epoch
            
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.base_lr,
                total_steps=total_steps,
                pct_start=config.get('pct_start', 0.05),  # 5% warmup
                div_factor=config.get('div_factor', 25),  # initial_lr = max_lr/25
                final_div_factor=config.get('final_div_factor', 1e4),
            )
            # OneCycleLR has built-in warmup, disable manual warmup
            self.warmup_epochs = 0
            print(f"Using OneCycleLR: max_lr={self.base_lr:.2e}, {total_steps} total steps")
        else:
            # ReduceLROnPlateau: steps per epoch based on val_loss
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.get('lr_factor', 0.5),
                patience=config.get('lr_patience', 10),
                verbose=False,
            )
            print(f"Using ReduceLROnPlateau: patience={config.get('lr_patience', 10)}")

        # Logging setup - always append timestamp for unique TensorBoard runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_name = config.get('run_name', 'baseline')

        # Always generate unique run name unless explicitly disabled
        if config.get('unique_run_name', True):
            self.run_name = f"{base_run_name}_{timestamp}"
        else:
            self.run_name = base_run_name

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

        # Mixed Precision Training (AMP) - ~2x speedup on modern GPUs
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            print("Mixed Precision (AMP) enabled - ~2x training speedup")

        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Mixed Precision (AMP): {'enabled' if self.use_amp else 'disabled'}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict of averaged metrics for the epoch
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        nan_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]",
                    leave=True, dynamic_ncols=True)

        for batch in pbar:
            x = batch.to(self.device, non_blocking=True)

            # Forward with mixed precision
            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                x_hat, z = self.model(x)
                loss, metrics = self.loss_fn(x_hat, x)

            # Skip batch if loss is NaN (numerical instability protection)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue

            # Backward with gradient scaling for AMP
            self.scaler.scale(loss).backward()

            # Gradient clipping (FR3.5: max norm 1.0)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )

            # Update with scaler
            # Update with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step OneCycleLR per batch (not per epoch)
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()

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

        if nan_batches > 0:
            self.logger.warning(f"Skipped {nan_batches} batches with NaN loss")

        # Handle case where all batches produced NaN
        if num_batches == 0:
            self.logger.error("All training batches produced NaN!")
            return {'loss': float('inf'), 'psnr': 0, 'ssim': 0, 'mse': float('inf')}

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
        nan_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]",
                    leave=True, dynamic_ncols=True)

        for batch in pbar:
            x = batch.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                x_hat, z = self.model(x)

            # Cast to float32 for stable loss computation (fixes AMP NaN issues)
            x_hat = x_hat.float()
            x = x.float()
            loss, metrics = self.loss_fn(x_hat, x)

            # Skip NaN batches (numerical instability protection)
            if any(v != v for v in metrics.values()):  # NaN check
                nan_batches += 1
                continue

            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'ssim': f"{metrics['ssim']:.4f}"
            })

        if nan_batches > 0:
            self.logger.warning(f"Skipped {nan_batches} batches with NaN values")

        if num_batches == 0:
            self.logger.error("All validation batches produced NaN!")
            return {'loss': float('inf'), 'psnr': 0, 'ssim': 0, 'mse': float('inf')}

        return {k: v / num_batches for k, v in epoch_metrics.items()}

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.

        Saves:
        - latest.pth: Always updated (for resumption)
        - best.pth: Updated when is_best=True
        - archive/best_YYYYMMDD_HHMMSS.pth: Timestamped copy when is_best=True
          (prevents accidental overwrites from re-running training)
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # AMP scaler state
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': self.config,
            'preprocessing_params': self.preprocessing_params,  # Critical for SAR!
            'history': self.history,
            'saved_at': datetime.now().isoformat(),  # Timestamp for reference
        }

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best (with timestamped archive)
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)

            # Also save timestamped archive copy (prevents accidental overwrites)
            if self.config.get('archive_best_checkpoints', True):
                archive_dir = self.checkpoint_dir / 'archive'
                archive_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = archive_dir / f'best_{timestamp}_epoch{self.epoch:03d}_loss{self.best_val_loss:.4f}.pth'
                torch.save(checkpoint, archive_path)
                self.logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f}) + archived to {archive_path.name}")
            else:
                self.logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load AMP scaler state if available
        if 'scaler_state_dict' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

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

        with autocast('cuda', enabled=self.use_amp):
            x_hat, z = self.model(x)

        # Convert to float32 for visualization
        x_hat = x_hat.float()

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

    def _update_warmup_lr(self, epoch: int):
        """Update learning rate during warmup phase (linear warmup)."""
        if epoch < self.warmup_epochs:
            # Linear interpolation from warmup_start_lr to base_lr
            progress = (epoch + 1) / self.warmup_epochs
            new_lr = self.warmup_start_lr + progress * (self.base_lr - self.warmup_start_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            return True  # Still in warmup
        elif epoch == self.warmup_epochs:
            # Warmup just finished, set to base_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr
            self.logger.info(f"Warmup complete. Learning rate set to {self.base_lr:.2e}")
            return False  # Warmup finished
        return False  # Past warmup

    def _check_gpu_memory(self):
        """Check GPU memory at startup and warn if significant memory already in use."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                usage_pct = used / total * 100
                free_gb = (total - used) / 1024

                if usage_pct > 50:
                    print(f"\n[WARNING] GPU memory already {usage_pct:.0f}% used ({used}MB / {total}MB)")
                    print(f"          Only {free_gb:.1f} GB available for training.")
                    print(f"          If training is slow, restart Jupyter kernel to free GPU memory.")
                    print(f"          To kill all kernels: 'Kernel > Shutdown All Kernels' in Jupyter\n")
                else:
                    print(f"GPU memory: {used}MB / {total}MB ({usage_pct:.0f}% used, {free_gb:.1f} GB free)")
        except Exception:
            pass  # Don't fail if nvidia-smi unavailable

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

            # Update warmup LR at start of epoch (before training)
            in_warmup = self._update_warmup_lr(epoch) if self.warmup_epochs > 0 else False

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

            if self.scheduler_type != 'onecycle':
                if not in_warmup and epoch >= self.warmup_epochs:
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

            # Store history (use .get() for safety when all batches are NaN)
            self.history.append({
                'epoch': epoch,
                'train_loss': train_metrics.get('loss', float('inf')),
                'val_loss': val_metrics.get('loss', float('inf')),
                'train_psnr': train_metrics.get('psnr', 0),
                'val_psnr': val_metrics.get('psnr', 0),
                'train_ssim': train_metrics.get('ssim', 0),
                'val_ssim': val_metrics.get('ssim', 0),
                'learning_rate': current_lr,
            })

            # Early abort if training completely diverged (all NaN)
            if train_metrics.get('loss', 0) == float('inf') and val_metrics.get('loss', 0) == float('inf'):
                self.logger.error("Training completely diverged (all NaN). Aborting.")
                self.logger.error("Try: lower learning rate, fresh start, or check for data issues.")
                break

            # Print epoch summary (per CONTEXT.md: with GPU memory)
            gpu_str = f" | GPU: {gpu_mem.get('gpu_allocated_gb', 0):.2f}GB" if gpu_mem else ""
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: loss={train_metrics.get('loss', float('inf')):.4f}, psnr={train_metrics.get('psnr', 0):.2f}, ssim={train_metrics.get('ssim', 0):.4f} | "
                f"Val: loss={val_metrics.get('loss', float('inf')):.4f}, psnr={val_metrics.get('psnr', 0):.2f}, ssim={val_metrics.get('ssim', 0):.4f} | "
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
