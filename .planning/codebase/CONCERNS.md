# Codebase Concerns

**Analysis Date:** 2026-01-21

## Tech Debt

**Pervasive NotImplementedError Pattern:**
- Issue: Core functionality across the entire codebase uses placeholder implementations with TODO comments instead of actual code. This blocks all end-to-end workflows.
- Files: `src/models/blocks.py`, `src/models/encoder.py`, `src/models/decoder.py`, `src/models/autoencoder.py`, `src/data/dataset.py`, `src/data/datamodule.py`, `src/data/preprocessing.py`, `src/losses/ssim.py`, `src/losses/combined.py`, `src/training/trainer.py`, `src/inference/compressor.py`, `scripts/train.py`, `scripts/evaluate.py`
- Impact: No training, evaluation, or inference possible. Every import of these modules raises NotImplementedError.
- Fix approach: Systematically implement each module, starting with:
  1. Build blocks (`src/models/blocks.py` - ConvBlock, DeconvBlock, ResidualBlock variants, Attention modules)
  2. Encoder/Decoder (`src/models/encoder.py`, `src/models/decoder.py`)
  3. Complete Autoencoder (`src/models/autoencoder.py`)
  4. Loss functions (`src/losses/ssim.py`, `src/losses/combined.py`)
  5. Data pipeline (`src/data/dataset.py`, `src/data/datamodule.py`, `src/data/preprocessing.py`)
  6. Training loop (`src/training/trainer.py`)
  7. Inference pipeline (`src/inference/compressor.py`)

**Incomplete Model Components:**
- Issue: `ConvBlock`, `DeconvBlock`, `ResidualBlock`, `ResidualBlockWithDownsample`, `ResidualBlockWithUpsample`, `ChannelAttention`, `SpatialAttention`, and `CBAM` modules are all stubs in `src/models/blocks.py`.
- Files: `src/models/blocks.py` (lines 21-327)
- Impact: Cannot build encoder/decoder architecture without these blocks.
- Fix approach: Implement each block class with proper __init__ and forward methods. Reference implementations provided in docstrings and learning guide.

**Incomplete Encoder/Decoder:**
- Issue: `SAREncoder` and `SARDecoder` in `src/models/encoder.py` and `src/models/decoder.py` don't initialize layers or implement forward passes.
- Files: `src/models/encoder.py` (lines 62-132), `src/models/decoder.py` (lines 63-107)
- Impact: Autoencoder cannot process input without working encoder/decoder.
- Fix approach: Implement layer stacking and forward passes using the block components once they're available.

**Missing Preprocessing Implementation:**
- Issue: `handle_invalid_values()`, `from_db()`, and `compute_clip_bounds()` are unimplemented in `src/data/preprocessing.py`.
- Files: `src/data/preprocessing.py` (lines 43-113)
- Impact: Cannot properly prepare raw SAR data for model training. Invalid values, dB conversion, and clipping are critical for SAR image handling.
- Fix approach: Implement each function following the docstring specifications and domain knowledge for SAR data.

**Loss Function Chain Broken:**
- Issue: Both `SSIMLoss` (src/losses/ssim.py) and `CombinedLoss` (src/losses/combined.py) are unimplemented. SSIM window creation and forward computation missing.
- Files: `src/losses/ssim.py` (lines 66-128), `src/losses/combined.py` (lines 60-104)
- Impact: No loss to train the model with. Combined loss balances MSE and SSIM.
- Fix approach: Implement Gaussian window creation in SSIMLoss, then implement forward pass. Build CombinedLoss on top.

**Dataset and DataModule Stubs:**
- Issue: `SARPatchDataset` and `SARDataModule` don't initialize, don't load data, don't return items.
- Files: `src/data/dataset.py` (lines 56-132), `src/data/datamodule.py` (lines 54-95)
- Impact: No way to load training data or create DataLoaders.
- Fix approach: Implement numpy array loading, train/val splitting, dataset indexing, and DataLoader creation.

**Training Pipeline Not Functional:**
- Issue: `Trainer` class in `src/training/trainer.py` has initialization incomplete (optimizer, scheduler, logging, checkpointing all commented).
- Files: `src/training/trainer.py` (lines 73-269)
- Impact: Cannot train model. Training loop, validation loop, checkpointing, and logging all missing.
- Fix approach: Complete optimizer/scheduler initialization, implement train_epoch/validate/checkpoint methods.

**Inference Pipeline Incomplete:**
- Issue: `SARCompressor` in `src/inference/compressor.py` doesn't load models or implement compression/decompression.
- Files: `src/inference/compressor.py` (lines 62-193)
- Impact: No way to compress/decompress full-size SAR images post-training.
- Fix approach: Implement model loading, preprocessing, patching, blending, and statistics computation.

**Script Entry Points Disabled:**
- Issue: Training and evaluation scripts (`scripts/train.py`, `scripts/evaluate.py`) are completely commented out with TODO markers.
- Files: `scripts/train.py` (lines 79-112), `scripts/evaluate.py` (lines 38-46)
- Impact: No CLI interface to run training or evaluation.
- Fix approach: Uncomment and verify once core modules are implemented.

## Known Bugs

**Redundant Imports in Visualization:**
- Symptoms: Multiple import of typing in `src/evaluation/visualizer.py`
- Files: `src/evaluation/visualizer.py` (lines 14 duplicates 12-14)
- Trigger: Import the module
- Workaround: Removed manually before use; no functional impact

**Data Range Assumption in Metrics:**
- Symptoms: `SARMetrics` assumes data in [0, 1] range, but preprocessing may produce different ranges
- Files: `src/evaluation/metrics.py` (various metrics assume normalized input)
- Trigger: Run evaluation on raw or non-normalized data
- Workaround: Always normalize to [0, 1] before evaluation

## Security Considerations

**No Input Validation:**
- Risk: Preprocessing and dataset modules don't validate input arrays (shape, dtype, range)
- Files: `src/data/preprocessing.py`, `src/data/dataset.py`, `src/models/blocks.py`
- Current mitigation: None
- Recommendations: Add shape/dtype/range checks in dataset init and preprocessing functions

**Unsafe File Operations:**
- Risk: Model checkpoint loading doesn't validate file existence or integrity
- Files: `src/inference/compressor.py` (line 62-86), `src/training/trainer.py` (line 200)
- Current mitigation: None
- Recommendations: Add try/except blocks and file validation before loading checkpoints

**YAML Config Injection:**
- Risk: `scripts/train.py` uses `yaml.safe_load()` which is safe, but doesn't validate required config keys
- Files: `scripts/train.py` (lines 68-72)
- Current mitigation: Using safe_load is good practice
- Recommendations: Add schema validation for config dictionaries

## Performance Bottlenecks

**No Batch Processing Optimization:**
- Problem: Dataset doesn't use pinned memory or optimized data loading
- Files: `src/data/datamodule.py` (lines 77-95)
- Cause: DataLoader created without optimization flags
- Improvement path: Add `pin_memory=True`, `persistent_workers=True` for GPU training

**Lack of Gradient Accumulation:**
- Problem: Training loop (once implemented) may need gradient accumulation for large batch effective sizes
- Files: `src/training/trainer.py` (training loop not implemented)
- Cause: Standard training loop doesn't support gradient accumulation
- Improvement path: Add configurable gradient accumulation steps

**No Model Quantization Support:**
- Problem: Large models may be slow in inference
- Files: `src/inference/compressor.py`
- Cause: No int8 or fp16 inference paths
- Improvement path: Add torch.quantization or mixed precision support

**Patch Blending Inefficiency:**
- Problem: Patch reconstruction with cosine ramp blending may have memory overhead
- Files: `src/inference/compressor.py` (lines 127-133)
- Cause: Blending weights computed per-image, potentially repeatedly
- Improvement path: Cache blending weights as buffers

## Fragile Areas

**Data Preprocessing Chain:**
- Files: `src/data/preprocessing.py`, `src/data/dataset.py`, `src/data/datamodule.py`
- Why fragile: Preprocessing parameters (vmin, vmax, clip method) must be consistent across train/val/test. If inconsistent, model will fail silently or produce poor results. No validation that parameters match.
- Safe modification: Always store preprocessing params in checkpoint metadata and validate before inference.
- Test coverage: No tests for preprocessing edge cases (NaN values, negative values, all-zero images).

**Loss Function Weighting:**
- Files: `src/losses/combined.py`, `src/losses/ssim.py`
- Why fragile: MSE/SSIM balance is critical but loss weights are hardcoded in config. Different batch sizes or image distributions may need different weights.
- Safe modification: Validate loss components individually in isolation before combining. Use test batches to verify loss behavior.
- Test coverage: No tests verifying SSIM computation correctness or gradient flow.

**Latent Space Dimensionality:**
- Files: `src/models/autoencoder.py`, `src/models/encoder.py`, `src/models/decoder.py`
- Why fragile: Latent channels control compression ratio. Changing this breaks checkpoints and inference code that assumes specific spatial dimensions (16x16 latent).
- Safe modification: Document the encoder spatial reduction steps. Test dimension consistency end-to-end.
- Test coverage: No dimension consistency tests.

**Batch Normalization Dependency:**
- Files: `src/models/blocks.py`, `src/training/trainer.py`
- Why fragile: Batch normalization requires batch_size > 1 and different train/eval behavior. No code path for batch_size=1 inference.
- Safe modification: Allow disabling BN with `use_bn=False` config. Test with both BN enabled/disabled.
- Test coverage: No BN behavior tests.

**Config-Model Mismatch:**
- Files: `configs/default.yaml`, `src/models/autoencoder.py`, `src/training/trainer.py`
- Why fragile: Model architecture (base_channels, latent_channels) must match what was used to train. No version checking.
- Safe modification: Store architecture config in checkpoints. Validate config matches checkpoint before loading.
- Test coverage: None.

## Scaling Limits

**In-Memory Dataset Loading:**
- Current capacity: Entire dataset loaded as single numpy array in `SARDataModule.__init__`
- Limit: 32GB+ datasets won't fit in RAM (common for satellite data)
- Scaling path: Implement streaming dataset that loads patches on-demand from disk or cloud storage

**Full Image Processing Memory:**
- Current capacity: Patch-based processing assumes patches fit in GPU memory
- Limit: Very large SAR scenes may require smaller patches than 256x256
- Scaling path: Make patch size and overlap configurable, add memory-aware patching

**Single-GPU Training:**
- Current capacity: No distributed training support in Trainer
- Limit: Training limited to single GPU memory
- Scaling path: Add DistributedDataParallel support, gradient accumulation

**No Mixed Precision:**
- Current capacity: FP32 only
- Limit: Large models may be slow on older GPUs
- Scaling path: Add torch.cuda.amp for automatic mixed precision

## Dependencies at Risk

**Redundant Dependency (torch imported twice):**
- Risk: `requirements.txt` lists `torch` twice (lines 5 and 9) and `torchvision` twice (lines 6 and 10)
- Impact: Minor - pip handles duplicates, but creates confusion
- Migration plan: Clean up requirements.txt, remove duplicates, use version constraints

**SAR Library (rasterio) Not Used:**
- Risk: `rasterio>=1.3.0` listed but no actual usage in codebase for loading SAR data
- Impact: Scripts expect `.npy` patches, not raw GeoTIFF data
- Migration plan: Either implement rasterio loading in preprocessing or remove dependency

**PyWavelets Not Used:**
- Risk: `PyWavelets` listed but no wavelet transforms in any module
- Impact: Dead dependency, adds bloat
- Migration plan: Remove or document if future use planned

**pytorch-msssim Optional:**
- Risk: Listed in requirements but actual SSIM loss implemented from scratch in `src/losses/ssim.py`
- Impact: Could use pytorch-msssim instead of custom implementation to reduce bugs
- Migration plan: Consider switching to pytorch-msssim or removing dependency

## Missing Critical Features

**No Checkpoint Version Management:**
- Problem: No way to migrate old checkpoints if model architecture changes
- Blocks: Cannot iterate on model design without losing trained weights
- Files affected: `src/training/trainer.py`, `src/inference/compressor.py`

**No Hyperparameter Logging:**
- Problem: Training config not saved with checkpoints
- Blocks: Cannot reproduce results - unknown what hyperparams were used
- Files affected: `src/training/trainer.py` (checkpoint saving not implemented)

**No Early Stopping Implementation:**
- Problem: Trainer has early_stopping_patience in config but no implementation
- Blocks: Training may not stop even if validation loss plateaus
- Files affected: `src/training/trainer.py` (line 104 initialized but never used)

**No Learning Rate Scheduling Beyond ReduceLROnPlateau:**
- Problem: Only ReduceLROnPlateau supported, no step or cosine annealing despite being in config
- Blocks: Cannot use more sophisticated LR schedules
- Files affected: `configs/default.yaml` (mentions options not implemented), `src/training/trainer.py`

**No Validation During Training:**
- Problem: Trainer.train() not implemented, no validation loop
- Blocks: Cannot monitor val loss during training
- Files affected: `src/training/trainer.py` (lines 160-177)

**No Image Logging to TensorBoard:**
- Problem: `log_images()` stub in Trainer but not implemented
- Blocks: Cannot visualize reconstructions during training
- Files affected: `src/training/trainer.py` (lines 206-207)

**No Inference Batching:**
- Problem: SARCompressor processes one patch at a time
- Blocks: Slow inference on large images
- Files affected: `src/inference/compressor.py` (no batch processing in compress/decompress)

## Test Coverage Gaps

**No Unit Tests for Data Preprocessing:**
- What's not tested: Invalid value handling, dB conversion, clipping, normalization
- Files: `src/data/preprocessing.py` (entire module)
- Risk: Silent corruption of SAR data (wrong dB conversion, NaN propagation)
- Priority: High - data preprocessing is critical for SAR

**No Unit Tests for Model Building Blocks:**
- What's not tested: ConvBlock, ResidualBlock, Attention modules don't have dimension consistency tests
- Files: `src/models/blocks.py`
- Risk: Architectural bugs (wrong padding, stride) not caught until training
- Priority: High - block bugs propagate to entire model

**No Integration Tests for End-to-End Pipeline:**
- What's not tested: Data loading -> Model forward -> Loss computation -> Gradient flow
- Files: All modules
- Risk: Unknown which integration point is broken when training fails
- Priority: Critical - needed before any training

**No Tests for Loss Functions:**
- What's not tested: SSIM computation correctness, gradient flow, numerical stability
- Files: `src/losses/ssim.py`, `src/losses/combined.py`
- Risk: Loss values could be NaN or incorrect, model won't train
- Priority: High - wrong loss means wrong training

**No Tests for Checkpoint Save/Load:**
- What's not tested: Model state saved/loaded correctly, preprocessing params preserved
- Files: `src/training/trainer.py`, `src/inference/compressor.py`
- Risk: Inference on loaded checkpoint produces different results
- Priority: High - affects production use

**No Tests for Dataset Augmentation:**
- What's not tested: Augmentations actually applied, shape/dtype consistent
- Files: `src/data/dataset.py` (lines 100-132)
- Risk: Augmentation silently disabled or produces wrong shapes
- Priority: Medium - affects generalization

**No Tests for Evaluation Metrics:**
- What's not tested: Metric computation correctness, consistency with standard libraries
- Files: `src/evaluation/metrics.py`
- Risk: Reported metrics don't match actual quality
- Priority: Medium - affects result interpretation

**No Regression Tests:**
- What's not tested: Model quality doesn't degrade in future changes
- Files: All
- Risk: Code changes break model without detection
- Priority: Medium - needed for CI/CD

---

*Concerns audit: 2026-01-21*
