# Phase 2: Baseline Model - Research

**Researched:** 2026-01-21
**Domain:** PyTorch CNN Autoencoder Architecture and Training
**Confidence:** HIGH

## Summary

Phase 2 implements a plain 4-layer encoder-decoder autoencoder with combined MSE+SSIM loss and complete training infrastructure. The existing codebase contains skeleton files with comprehensive TODO comments that define the architecture precisely. Research confirms that the standard PyTorch stack (torch 2.5.1, pytorch-msssim, TensorBoard, tqdm) is already installed and well-suited for this phase.

The architecture follows established patterns: ConvBlock (Conv2d + BatchNorm + LeakyReLU) for encoding, DeconvBlock (ConvTranspose2d + BatchNorm + ReLU) for decoding, with 5x5 kernels, stride 2, and configurable latent channels. Critical implementation details include: output_padding=1 for exact 2x upsampling, no activation on final encoder layer, sigmoid output activation, and Kaiming initialization with mode='fan_out' for LeakyReLU.

The training infrastructure requires: Adam optimizer (lr=1e-4), ReduceLROnPlateau scheduler (patience=10, factor=0.5), gradient clipping (max_norm=1.0), checkpointing (model + optimizer + scheduler + config + preprocessing params), and comprehensive TensorBoard logging (scalars per epoch, images per epoch, weight histograms every 10 epochs).

**Primary recommendation:** Use pytorch-msssim (already installed) for SSIM loss rather than hand-rolling, and follow the existing config.json pattern for configuration management.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.5.1+cu121 | Model development, training | Verified installed, excellent GPU support |
| pytorch-msssim | installed | SSIM loss computation | Fast, differentiable, GPU-accelerated |
| TensorBoard | installed | Training visualization | Already configured in codebase |
| tqdm | installed | Progress bars | Standard for training loops |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchvision | installed | make_grid for image logging | TensorBoard image visualization |
| numpy | installed | Metrics computation | Evaluation, data handling |
| json | stdlib | Config management | Save/load training configuration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled SSIM | pytorch-msssim | pytorch-msssim is faster, tested, GPU-accelerated |
| Hand-rolled scheduler | torch.optim.lr_scheduler | Built-in is well-tested, no reason to customize |
| PyTorch Lightning | Vanilla PyTorch | Lightning adds abstraction; vanilla is simpler for this scope |

**Installation:**
```bash
# All required packages already installed
# No additional installation needed
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── models/
│   ├── blocks.py         # ConvBlock, DeconvBlock (implement TODO)
│   ├── encoder.py        # SAREncoder (implement TODO)
│   ├── decoder.py        # SARDecoder (implement TODO)
│   └── autoencoder.py    # SARAutoencoder wrapper (implement TODO)
├── losses/
│   ├── mse.py            # MSELoss (already implemented)
│   ├── ssim.py           # SSIMLoss (implement or use pytorch-msssim)
│   └── combined.py       # CombinedLoss (implement TODO)
├── training/
│   └── trainer.py        # Trainer class (implement TODO)
└── evaluation/
    └── metrics.py        # PSNR, SSIM computation (partial)
```

### Pattern 1: ConvBlock Architecture
**What:** Encapsulates Conv2d + BatchNorm + LeakyReLU as reusable unit
**When to use:** All encoder layers except final
**Example:**
```python
# Source: PyTorch documentation + autoencoder best practices
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=2, padding=2, use_bn=True, negative_slope=0.2):
        super().__init__()
        # When using BatchNorm, set bias=False (BN has its own bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
```

### Pattern 2: DeconvBlock with output_padding
**What:** ConvTranspose2d + BatchNorm + ReLU for upsampling
**When to use:** All decoder layers except final
**Critical detail:** output_padding=1 required for exact 2x upsampling with kernel=5, stride=2
**Example:**
```python
# Source: PyTorch ConvTranspose2d documentation
# Formula: H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
# For 2x upsampling: H_out = (H_in - 1) * 2 - 2*2 + 5 + 1 = 2*H_in
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=2, padding=2, output_padding=1, use_bn=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.bn(self.deconv(x)))
```

### Pattern 3: SSIM Loss using pytorch-msssim
**What:** Use established library instead of hand-rolling
**Why:** Faster (separable kernels), tested, compatible with autograd
**Example:**
```python
# Source: https://github.com/VainF/pytorch-msssim
from pytorch_msssim import SSIM

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=1.0, channel=1):
        super().__init__()
        self.ssim_module = SSIM(data_range=data_range, size_average=True,
                                 channel=channel, win_size=window_size,
                                 nonnegative_ssim=True)

    def forward(self, x_hat, x):
        # Returns 1 - SSIM (loss form)
        return 1 - self.ssim_module(x_hat, x)
```

### Pattern 4: Kaiming Initialization for LeakyReLU
**What:** Proper weight initialization for LeakyReLU networks
**Why:** Preserves gradient magnitudes, enables training of deeper networks
**Example:**
```python
# Source: torch.nn.init documentation
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out',
                                     nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
```

### Pattern 5: Checkpoint with Full State
**What:** Save all training state for complete resumption
**Why:** Preserves optimizer momentum, scheduler state, preprocessing params
**Example:**
```python
# Source: PyTorch Saving and Loading Models tutorial
checkpoint = {
    'epoch': self.epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'best_val_loss': self.best_val_loss,
    'config': self.config,
    'preprocessing_params': self.preprocessing_params,  # Critical for SAR!
}
torch.save(checkpoint, path)
```

### Pattern 6: TensorBoard Image Logging
**What:** Log reconstruction samples as grids
**When:** Every epoch per CONTEXT.md decisions
**Example:**
```python
# Source: PyTorch TensorBoard tutorial
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

def log_images(self, writer, epoch, num_samples=4):
    self.model.eval()
    with torch.no_grad():
        x = self.sample_batch[:num_samples].to(self.device)
        x_hat, _ = self.model(x)

        # Create triple view: original | reconstructed | difference
        diff = torch.abs(x - x_hat)
        combined = torch.cat([x, x_hat, diff], dim=0)
        grid = vutils.make_grid(combined, nrow=num_samples, normalize=True)
        writer.add_image('Reconstructions/triple_view', grid, epoch)
```

### Anti-Patterns to Avoid
- **Checkerboard artifacts from ConvTranspose2d:** Can occur with wrong kernel/stride combinations. Use kernel=5, stride=2, padding=2, output_padding=1 for clean 2x upsampling. Alternative: nn.Upsample + Conv2d if artifacts persist.
- **Saving entire model with torch.save(model):** Use state_dict instead for architecture flexibility.
- **Forgetting model.eval() during validation:** BatchNorm behavior differs between train/eval modes.
- **Using verbose=True on ReduceLROnPlateau:** Creates noisy logs; handle LR logging manually via TensorBoard.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSIM computation | Custom Gaussian window convolution | pytorch_msssim.SSIM | Separable kernels for speed, tested edge cases, GPU-optimized |
| Learning rate scheduling | Custom decay logic | torch.optim.lr_scheduler.ReduceLROnPlateau | Handles all edge cases, integrates with optimizer |
| Image grid visualization | Manual tensor stacking | torchvision.utils.make_grid | Handles padding, normalization, channel ordering |
| Progress bars | Print statements | tqdm | ETA, dynamic updates, configurable format |
| Gradient clipping | Manual norm computation | torch.nn.utils.clip_grad_norm_ | In-place, handles parameter groups |

**Key insight:** PyTorch ecosystem has mature solutions for all training infrastructure. Focus implementation effort on model architecture and SAR-specific concerns.

## Common Pitfalls

### Pitfall 1: ConvTranspose2d Output Size Mismatch
**What goes wrong:** Decoder output shape doesn't match encoder input shape (256x256)
**Why it happens:** Incorrect output_padding calculation for kernel=5, stride=2
**How to avoid:** Use output_padding=1 with kernel=5, stride=2, padding=2. Verify: `(16-1)*2 - 2*2 + 5 + 1 = 32` per layer.
**Warning signs:** RuntimeError about tensor size mismatch in forward pass

### Pitfall 2: BatchNorm with Small Batches
**What goes wrong:** Unstable statistics with batch_size < 8
**Why it happens:** BatchNorm needs sufficient samples for mean/var estimation
**How to avoid:** Use drop_last=True in DataLoader (already set in SARDataModule). batch_size=8 is adequate.
**Warning signs:** NaN losses in first few epochs, especially with batch_size=1-2

### Pitfall 3: SSIM Loss Returning Negative Values
**What goes wrong:** Loss becomes negative, training fails
**Why it happens:** SSIM can be negative for very dissimilar images
**How to avoid:** Use nonnegative_ssim=True in pytorch_msssim.SSIM
**Warning signs:** Negative loss values in early training

### Pitfall 4: Optimizer State Lost on Resume
**What goes wrong:** Training "jumps" after checkpoint load
**Why it happens:** Optimizer momentum buffers not restored
**How to avoid:** Always save and load optimizer.state_dict(), scheduler.state_dict()
**Warning signs:** Validation loss spikes after resuming

### Pitfall 5: Preprocessing Params Not Saved
**What goes wrong:** Reconstructions have wrong scale on inference
**Why it happens:** vmin/vmax used during training not available at inference
**How to avoid:** Save preprocessing_params (from SARDataModule.preprocessing_params) in checkpoint
**Warning signs:** Inference outputs look washed out or saturated

### Pitfall 6: Early Stopping Triggers with LR Scheduler
**What goes wrong:** Training stops immediately after LR reduction
**Why it happens:** Same patience value for both scheduler and early stopping
**How to avoid:** Set early_stopping_patience > lr_patience (e.g., patience=20 > lr_patience=10)
**Warning signs:** Training stops after first LR reduction

### Pitfall 7: Progress Bar Metrics Not Updating
**What goes wrong:** tqdm shows stale values
**Why it happens:** Using tqdm.write() instead of postfix update
**How to avoid:** Use pbar.set_postfix({'loss': value, 'psnr': value})
**Warning signs:** Progress bar shows same metrics throughout epoch

## Code Examples

Verified patterns from official sources:

### Combined Loss Function
```python
# Source: Project CONTEXT.md decisions + pytorch-msssim documentation
class CombinedLoss(nn.Module):
    """Combined MSE + SSIM loss per Phase 2 requirements."""

    def __init__(self, mse_weight=0.5, ssim_weight=0.5, window_size=11):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = nn.MSELoss()
        # Use pytorch_msssim for robust SSIM computation
        from pytorch_msssim import SSIM
        self.ssim_module = SSIM(data_range=1.0, size_average=True,
                                 channel=1, win_size=window_size,
                                 nonnegative_ssim=True)

    def forward(self, x_hat, x):
        mse = self.mse_loss(x_hat, x)
        ssim_val = self.ssim_module(x_hat, x)
        ssim_loss = 1 - ssim_val

        loss = self.mse_weight * mse + self.ssim_weight * ssim_loss

        # Compute PSNR for metrics
        with torch.no_grad():
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

        metrics = {
            'loss': loss.item(),
            'mse': mse.item(),
            'ssim': ssim_val.item(),
            'psnr': psnr.item(),
        }
        return loss, metrics
```

### Training Loop with Progress Bar
```python
# Source: tqdm documentation + PyTorch best practices
def train_epoch(self):
    self.model.train()
    epoch_metrics = defaultdict(float)
    num_batches = 0

    pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]",
                leave=True, dynamic_ncols=True)

    for batch in pbar:
        x = batch.to(self.device)

        self.optimizer.zero_grad()
        x_hat, z = self.model(x)
        loss, metrics = self.loss_fn(x_hat, x)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                        self.config.get('max_grad_norm', 1.0))
        self.optimizer.step()

        # Accumulate metrics
        for key, value in metrics.items():
            epoch_metrics[key] += value
        num_batches += 1

        # Update progress bar with current batch metrics
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'psnr': f"{metrics['psnr']:.2f}",
            'ssim': f"{metrics['ssim']:.4f}"
        })

    return {k: v / num_batches for k, v in epoch_metrics.items()}
```

### GPU Memory Logging
```python
# Source: torch.cuda documentation
def log_gpu_memory(self):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        return {'gpu_allocated_gb': allocated, 'gpu_reserved_gb': reserved}
    return {}
```

### Weight Histogram Logging
```python
# Source: TensorBoard documentation
def log_weight_histograms(self, writer, epoch):
    if epoch % 10 == 0:  # Every 10 epochs per CONTEXT.md
        for name, param in self.model.named_parameters():
            writer.add_histogram(f'weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom SSIM | pytorch-msssim library | 2020+ | 5-6x speedup, numerical consistency |
| Manual LR decay | ReduceLROnPlateau | Standard practice | Automatic response to training dynamics |
| Print-based logging | TensorBoard | Standard practice | Rich visualization, comparison across runs |
| Single checkpoint | model + optimizer + scheduler | Standard practice | Seamless training resume |

**Deprecated/outdated:**
- Using Conv2d bias when BatchNorm follows (wastes parameters, BN has its own bias)
- verbose=True on schedulers (noisy, use TensorBoard instead)
- torch.save(model) for entire model (breaks on architecture changes)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal loss weight balance**
   - What we know: Default 0.5/0.5 MSE/SSIM from CONTEXT.md decisions
   - What's unclear: Whether SAR speckle needs different balance
   - Recommendation: Start with 0.5/0.5, tune based on ENL ratio in Phase 3

2. **Number of sample images per epoch**
   - What we know: CONTEXT.md marks this as Claude's discretion
   - What's unclear: Balance between logging overhead and insight
   - Recommendation: 4-8 samples (one batch worth), fixed samples for consistency

3. **Log file format and location**
   - What we know: CONTEXT.md requires log file in addition to TensorBoard
   - What's unclear: Exact format specification
   - Recommendation: JSON lines format in runs/{timestamp}/training.log

## Sources

### Primary (HIGH confidence)
- PyTorch 2.9 documentation - torch.nn, torch.optim, torch.utils.tensorboard
- pytorch-msssim GitHub (VainF/pytorch-msssim) - SSIM/MS-SSIM implementation
- Existing codebase - config.json pattern, SARDataModule interface

### Secondary (MEDIUM confidence)
- PyTorch Lightning tutorials - Autoencoder patterns (for reference, not using Lightning)
- WebSearch results - Best practices for gradient clipping, checkpointing, TensorBoard

### Tertiary (LOW confidence)
- Community patterns for progress bar formatting
- Exact optimal learning rate values (need experimental validation)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified installed and working
- Architecture: HIGH - Patterns match existing skeletons and established practices
- Training infrastructure: HIGH - PyTorch standard patterns well-documented
- Pitfalls: MEDIUM-HIGH - Based on documentation + common failure modes

**Research date:** 2026-01-21
**Valid until:** 60 days (stable patterns, no fast-moving dependencies)
