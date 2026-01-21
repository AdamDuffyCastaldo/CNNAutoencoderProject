# Architecture

**Analysis Date:** 2026-01-21

## Pattern Overview

**Overall:** Modular encoder-decoder autoencoder architecture with clear separation of concerns between model components, data handling, training, evaluation, and inference.

**Key Characteristics:**
- **Encoder-Decoder Structure**: CNN autoencoder compresses 256×256×1 SAR images to 16×16×C latent representation (4× spatial compression per layer × 4 layers = 256× total)
- **Layered Abstraction**: Clear separation between models, data handling, training logic, loss functions, evaluation, and inference
- **Configuration-Driven**: YAML-based configuration (`configs/default.yaml`) controls all training and model parameters
- **Learning-First Design**: Code includes extensive TODO placeholders and docstrings guiding implementation decisions

## Layers

**Model Layer:**
- Purpose: Define neural network architectures for encoding/decoding
- Location: `src/models/`
- Contains: `autoencoder.py`, `encoder.py`, `decoder.py`, `blocks.py`
- Depends on: PyTorch core modules
- Used by: Training and inference layers

**Data Layer:**
- Purpose: Handle data loading, preprocessing, and augmentation
- Location: `src/data/`
- Contains: `dataset.py` (SARPatchDataset, SARImageDataset), `datamodule.py` (SARDataModule), `preprocessing.py`
- Depends on: NumPy, PyTorch Dataset/DataLoader, scikit-image
- Used by: Training layer

**Loss Layer:**
- Purpose: Define training objectives combining multiple metrics
- Location: `src/losses/`
- Contains: `combined.py` (CombinedLoss, EdgePreservingLoss), `mse.py`, `ssim.py`
- Depends on: PyTorch, scikit-image for SSIM
- Used by: Training layer

**Training Layer:**
- Purpose: Manage training loops, checkpointing, and early stopping
- Location: `src/training/trainer.py` (Trainer class)
- Contains: Epoch training, validation, checkpoint saving, TensorBoard logging
- Depends on: Model, data, and loss layers
- Used by: `scripts/train.py`

**Evaluation Layer:**
- Purpose: Compute comprehensive quality metrics and analyze model performance
- Location: `src/evaluation/`
- Contains: `evaluator.py` (Evaluator, SARMetrics), `metrics.py`, `visualizer.py`
- Depends on: Model layer, NumPy, SciPy, scikit-image
- Used by: Training loops and evaluation scripts

**Inference Layer:**
- Purpose: Production-ready compression/decompression pipeline
- Location: `src/inference/compressor.py` (SARCompressor)
- Contains: Full image compression, preprocessing, patch tiling, decompression
- Depends on: Model layer
- Used by: `scripts/evaluate.py`, deployment

**Compression Utilities Layer:**
- Purpose: Information theory and entropy coding utilities
- Location: `src/compression/`
- Contains: `entropy.py` (entropy calculation), `histogram.py`
- Depends on: NumPy, SciPy
- Used by: Compression analysis and codec development

**CLI/Script Layer:**
- Purpose: User-facing entry points for training, evaluation, and inference
- Location: `scripts/`
- Contains: `train.py`, `evaluate.py`, `download_sentinel_data.py`
- Depends on: All above layers
- Used by: End users and deployment pipelines

## Data Flow

**Training Pipeline:**

1. Configuration loaded from `configs/default.yaml` or command-line args
2. `SARDataModule` loads patches from `.npy` file, splits into train/val (90/10 by default)
3. Creates `SARPatchDataset` for augmentation (flips, 90° rotations only - SAR-safe)
4. `DataLoader` batches with shuffle=True for training, shuffle=False for validation
5. `Trainer` processes batches:
   - Forward: `x → model.encode() → model.decode() → x̂`
   - Loss: `CombinedLoss(x̂, x) → loss + metrics`
   - Backward: gradient computation + clipping
   - Update: Adam optimizer step
6. Validation run every epoch with no augmentation
7. Checkpoints saved (best + latest) based on validation loss
8. Early stopping if no improvement for N epochs

**Inference Pipeline:**

1. Raw SAR image loaded
2. `SARCompressor.preprocess()`: convert to dB, clip to range (-25 to 5 dB), normalize to [0, 1]
3. `SARCompressor.compress()`:
   - Pad image to stride-divisible size
   - Extract overlapping patches (256×256 with 32-px overlap)
   - Batch process through encoder
   - Return latent patches + metadata (original shape, preprocessing params)
4. `SARCompressor.decompress()`:
   - Batch process latent through decoder
   - Blend overlapping patches using cosine ramp weights
   - Remove padding
   - Inverse preprocess (denormalize, convert from dB back to linear)

**State Management:**

- **Training State**: Managed by `Trainer` class (epoch, global_step, best_val_loss, history)
- **Model State**: PyTorch model parameters + optimizer state + scheduler state (saved/loaded via checkpoints)
- **Data State**: Patches in memory during training, loaded once via NumPy `.load()`
- **Inference State**: Stateless - model in eval mode, preprocessing params stored in checkpoint metadata

## Key Abstractions

**SARAutoencoder:**
- Purpose: Complete encoder-decoder model with utility methods
- Location: `src/models/autoencoder.py`
- Pattern: PyTorch nn.Module composition
- Methods: `forward()`, `encode()`, `decode()`, `get_compression_ratio()`, `get_latent_size()`, `count_parameters()`, `analyze_latent()`

**Encoder/Decoder:**
- Purpose: Spatial compression (encoder) and reconstruction (decoder)
- Location: `src/models/encoder.py`, `src/models/decoder.py`
- Architecture:
  - Encoder: 4 strided (stride=2) ConvBlocks with LeakyReLU
    - 256×256 → 128×128 → 64×64 → 32×32 → 16×16 (spatial dims)
    - 1 → 64 → 128 → 256 → C channels (latent_channels determines final width)
  - Decoder: Mirror of encoder using DeconvBlocks with ReLU + Sigmoid output
- Pattern: Sequential layer stacking with He weight initialization

**Building Blocks:**
- Purpose: Reusable conv/deconv/residual building blocks
- Location: `src/models/blocks.py`
- Blocks: ConvBlock, DeconvBlock, ResidualBlock, ResidualBlockWithDownsample, ResidualBlockWithUpsample
- Attention blocks (optional): ChannelAttention, SpatialAttention, CBAM
- Pattern: Composition of Conv/ConvTranspose + BatchNorm + Activation

**SARDataModule:**
- Purpose: Train/val data pipeline abstraction
- Location: `src/data/datamodule.py`
- Pattern: PyTorch Lightning-inspired (provides train_dataloader(), val_dataloader())
- Handles: Loading `.npy`, splitting with seeded randomness, DataLoader creation

**SARPatchDataset:**
- Purpose: PyTorch Dataset for patch iteration with optional augmentation
- Location: `src/data/dataset.py`
- Pattern: PyTorch Dataset subclass (__len__, __getitem__)
- Augmentations: Horizontal flip, vertical flip, 90° rotations (all SAR-safe)

**CombinedLoss:**
- Purpose: Balanced MSE + SSIM loss for reconstruction quality
- Location: `src/losses/combined.py`
- Formula: `loss = λ_MSE × MSE(x̂, x) + λ_SSIM × (1 - SSIM(x̂, x))`
- Returns: (scalar loss, metrics dict with loss/mse/ssim/psnr)

**Trainer:**
- Purpose: Complete training loop with checkpointing and early stopping
- Location: `src/training/trainer.py`
- Pattern: Stateful manager class
- Features:
  - Auto GPU detection
  - ReduceLROnPlateau scheduling
  - Gradient clipping
  - TensorBoard logging (runs directory)
  - Best + latest checkpointing
  - Early stopping

**Evaluator:**
- Purpose: Comprehensive image quality assessment
- Location: `src/evaluation/evaluator.py`
- Metrics: MSE, PSNR, SSIM, ENL (Equivalent Number of Looks), EPI (Edge Preservation Index), MAE, histogram similarity, correlation, local variance
- Methods: evaluate_batch(), evaluate_dataset(), analyze_latent_space(), find_failure_cases(), find_best_cases()

**SARCompressor:**
- Purpose: Production inference pipeline
- Location: `src/inference/compressor.py`
- Pipeline: Preprocess → Pad → Extract patches → Encode → Store metadata
- Features: Overlapping patches with cosine-ramp blending, preprocessing param storage, file I/O

## Entry Points

**Training Script:**
- Location: `scripts/train.py`
- Triggers: User execution with args or config file
- Responsibilities:
  1. Parse CLI arguments and/or load YAML config
  2. Create SARDataModule from patches file
  3. Instantiate SARAutoencoder with config params
  4. Create CombinedLoss with MSE/SSIM weights
  5. Create Trainer and call train() method
  6. TODO: Complete implementation (currently raises NotImplementedError)

**Evaluation Script:**
- Location: `scripts/evaluate.py` (referenced but not provided)
- Triggers: User execution on trained model
- Responsibilities: Load checkpoint, evaluate on test data, generate metrics report

**Download Script:**
- Location: `scripts/download_sentinel_data.py`
- Triggers: User execution to fetch SAR patches
- Responsibilities: Query Sentinel-1 data, extract patches, save to .npy

**Configuration File:**
- Location: `configs/default.yaml`
- Purpose: Central control of all hyperparameters and paths
- Sections: data, preprocessing, model, loss, training, logging, inference

## Error Handling

**Strategy:** Try/except blocks with informative logging (planned via trainer.py)

**Patterns:**

- **Weight initialization**: HeNormal for ReLU-family activations, implemented in encoder/decoder `_initialize_weights()` methods
- **Invalid tensor dimensions**: Assertions in test functions validate shape contracts (e.g., encoder output shape in test_encoder())
- **NaN/Inf detection**: SSIM and other metrics check for edge cases (e.g., division by zero protection in local_variance_ratio)
- **Preprocessing edge cases**: Clipping bounds for SAR values, handling of invalid pixels in entropy.py

## Cross-Cutting Concerns

**Logging:**
- Framework: TensorBoard via SummaryWriter (`runs/{timestamp}/` directories)
- Implementation: Trainer writes scalars (train/val loss, metrics) and images every N epochs
- Console output: TQDM progress bars in training loops

**Validation:**
- Data validation: SARPatchDataset checks input shape (N, H, W) and value range [0, 1]
- Model validation: Test functions (test_autoencoder, test_encoder, etc.) verify output shapes and gradient flow
- Configuration validation: YAML loaded with pyyaml, CLI args with argparse

**Authentication:** Not applicable - local training project

**Preprocessing:**
- Input: Raw SAR intensity images (linear or dB scale)
- Steps: Handle invalid pixels → Convert to dB if needed → Clip to range (-25 to 5 dB) → Normalize [0, 1]
- Location: `src/inference/compressor.py` preprocess() method (TODO)
- Inverse: Denormalize → Clip range back → Convert from dB to linear
- Location: `src/inference/compressor.py` inverse_preprocess() method (TODO)

---

*Architecture analysis: 2026-01-21*
