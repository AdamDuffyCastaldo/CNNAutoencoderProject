# Codebase Structure

**Analysis Date:** 2026-01-21

## Directory Layout

```
CNNAutoencoderProject/
├── src/                          # Production source code
│   ├── models/                   # Neural network architectures
│   │   ├── autoencoder.py        # Complete encoder-decoder model
│   │   ├── encoder.py            # Spatial compression (256→16)
│   │   ├── decoder.py            # Spatial reconstruction (16→256)
│   │   ├── blocks.py             # Building blocks (Conv, Deconv, Residual, Attention)
│   │   └── __init__.py
│   │
│   ├── data/                     # Data loading and preprocessing
│   │   ├── dataset.py            # SARPatchDataset, SARImageDataset
│   │   ├── datamodule.py         # SARDataModule (train/val split)
│   │   ├── preprocessing.py      # SAR-specific preprocessing (TODO)
│   │   └── __init__.py
│   │
│   ├── training/                 # Training loops and management
│   │   ├── trainer.py            # Trainer class (main training loop)
│   │   └── __init__.py
│   │
│   ├── losses/                   # Loss functions
│   │   ├── combined.py           # CombinedLoss, EdgePreservingLoss
│   │   ├── mse.py                # MSE loss (TODO)
│   │   ├── ssim.py               # SSIM loss (TODO)
│   │   └── __init__.py
│   │
│   ├── evaluation/               # Evaluation metrics and analysis
│   │   ├── evaluator.py          # Evaluator, SARMetrics classes
│   │   ├── metrics.py            # Individual metric functions (TODO)
│   │   ├── visualizer.py         # Visualization tools (TODO)
│   │   └── __init__.py
│   │
│   ├── inference/                # Production inference pipeline
│   │   ├── compressor.py         # SARCompressor (full image compression)
│   │   └── __init__.py
│   │
│   ├── compression/              # Entropy and codec utilities
│   │   ├── entropy.py            # Entropy calculation
│   │   ├── histogram.py          # Histogram utilities (TODO)
│   │   └── __init__.py
│   │
│   ├── utils/                    # Utility functions
│   │   ├── io.py                 # File I/O utilities (TODO)
│   │   └── __init__.py
│   │
│   └── __init__.py               # Package entry point
│
├── scripts/                      # CLI entry points
│   ├── train.py                  # Training script (main entry)
│   ├── evaluate.py               # Evaluation script
│   └── download_sentinel_data.py # Download SAR data
│
├── configs/                      # Configuration files
│   ├── default.yaml              # Default hyperparameters
│   └── .gitkeep
│
├── checkpoints/                  # Model weights (gitignored)
│   ├── best.pth                  # Best model
│   └── latest.pth                # Latest checkpoint
│
├── runs/                         # TensorBoard logs (gitignored)
│   └── YYYYMMDD_HHMMSS/
│       └── events.pb
│
├── experiments/                  # Experiment results (gitignored)
│
├── learningnotebooks/            # Jupyter learning notebooks
│   ├── phase1_compression_fundamentals/
│   ├── phase2_learned_compression/
│   ├── phase3_sar_fundamentals/
│   └── phase4_sar_codec/
│
├── data/                         # Local data (gitignored)
│   └── patches/
│       └── patches.npy           # Training patches
│
├── results/                      # Evaluation results (gitignored)
│
├── .vscode/                      # VS Code workspace settings
│
├── .planning/                    # GSD planning documents
│   └── codebase/
│       ├── ARCHITECTURE.md
│       └── STRUCTURE.md
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .git/                         # Git repository
```

## Directory Purposes

**`src/models/`:**
- Purpose: Define all neural network architectures
- Contains: PyTorch nn.Module subclasses
- Key files: `autoencoder.py` (main model), `encoder.py`/`decoder.py` (components), `blocks.py` (reusable units)
- Dependency: Only PyTorch, no other src/ dependencies

**`src/data/`:**
- Purpose: Data loading, splitting, augmentation
- Contains: PyTorch Dataset classes, DataModule orchestrator
- Key files: `dataset.py` (dataset classes), `datamodule.py` (train/val split + DataLoader creation)
- Dependency: Models layer (indirect - via datamodule usage)

**`src/training/`:**
- Purpose: Main training orchestration
- Contains: Trainer class with epoch loop, validation, checkpointing, early stopping
- Key files: `trainer.py` (single file with Trainer class)
- Dependency: Models, data, and losses layers

**`src/losses/`:**
- Purpose: Training objectives combining multiple quality metrics
- Contains: Loss function classes (CombinedLoss, EdgePreservingLoss)
- Key files: `combined.py` (main losses), `mse.py`/`ssim.py` (component losses)
- Dependency: PyTorch, scikit-image for SSIM

**`src/evaluation/`:**
- Purpose: Comprehensive image quality assessment
- Contains: Evaluator class and metric calculation
- Key files: `evaluator.py` (main module with SARMetrics and Evaluator), `metrics.py`/`visualizer.py` (TODO)
- Dependency: NumPy, SciPy, scikit-image, Matplotlib

**`src/inference/`:**
- Purpose: Production-ready compression/decompression
- Contains: SARCompressor class (full image handling with tiling)
- Key files: `compressor.py` (single file)
- Dependency: Models layer

**`src/compression/`:**
- Purpose: Information theory and entropy utilities
- Contains: Entropy calculation, histogram analysis
- Key files: `entropy.py` (implemented), `histogram.py` (TODO)
- Dependency: NumPy, SciPy (independent from training pipeline)

**`src/utils/`:**
- Purpose: Shared utilities
- Contains: File I/O helpers, misc utilities
- Key files: `io.py` (TODO)
- Dependency: None (utilities are dependencies for other modules)

**`scripts/`:**
- Purpose: User-facing CLI entry points
- Contains: Python scripts (not modules)
- Key files: `train.py`, `evaluate.py`, `download_sentinel_data.py`
- Dependency: All src/ layers

**`configs/`:**
- Purpose: Centralized hyperparameter control
- Contains: YAML configuration files
- Key files: `default.yaml` (primary config)
- Format: YAML with sections for data, preprocessing, model, loss, training, logging, inference

**`checkpoints/`:**
- Purpose: Saved model weights and training state
- Contains: PyTorch .pth checkpoint files
- Generated: During training via Trainer.save_checkpoint()
- Files: `best.pth` (lowest validation loss), `latest.pth` (most recent)

**`runs/`:**
- Purpose: TensorBoard event logs
- Contains: TensorBoard event files (events.pb)
- Generated: During training via Trainer (SummaryWriter)
- Organization: Subdirectory per run with timestamp

**`learningnotebooks/`:**
- Purpose: Educational Jupyter notebooks
- Contains: Phase-based learning materials
- Organization: 4 phases, each with multiple days/sessions

**`data/`:**
- Purpose: Local training data storage
- Contains: Preprocessed SAR patch .npy files
- Key: `patches.npy` (N×256×256 array of normalized patches)
- Note: Gitignored due to size

## Key File Locations

**Entry Points:**
- `scripts/train.py`: Main training entry point (user runs this to train)
- `scripts/evaluate.py`: Evaluation on test data
- `scripts/download_sentinel_data.py`: Fetch SAR patches from Sentinel-1

**Configuration:**
- `configs/default.yaml`: Central hyperparameter config (data paths, model size, training params, logging dirs)

**Core Logic:**
- `src/models/autoencoder.py`: Complete model combining encoder + decoder
- `src/models/encoder.py`: 256×256 → 16×16 compression
- `src/models/decoder.py`: 16×16 → 256×256 reconstruction
- `src/training/trainer.py`: Training loop with all features (checkpointing, early stopping, logging)
- `src/evaluation/evaluator.py`: Comprehensive metrics (MSE, PSNR, SSIM, ENL, EPI, etc.)
- `src/inference/compressor.py`: Production inference (full image tiling + blending)

**Testing:**
- Each module has a `test_*()` function at the bottom
- Examples:
  - `src/models/autoencoder.py::test_autoencoder()`
  - `src/data/dataset.py::test_dataset()`
  - `src/training/trainer.py::test_trainer()`
  - `src/losses/combined.py::test_losses()`

## Naming Conventions

**Files:**
- `snake_case.py` for all Python files
- Single class per file (except `__init__.py`): `encoder.py` contains SAREncoder, `decoder.py` contains SARDecoder
- Grouping: Related classes in single file when appropriate (evaluator.py has Evaluator + SARMetrics)

**Directories:**
- `snake_case/` for all directories
- Functional grouping: `models/`, `data/`, `training/`, `losses/`, `evaluation/`, `inference/`, `compression/`
- Semantic naming: Clear purpose from name alone

**Classes:**
- `PascalCase` for all classes
- Prefix `SAR` for domain-specific classes: SARAutoencoder, SAREncoder, SARDecoder, SARDataModule, SARPatchDataset, etc.
- Suffix `Loss` for loss functions: CombinedLoss, EdgePreservingLoss, MSELoss, SSIMLoss
- Suffix `Block` for building blocks: ConvBlock, DeconvBlock, ResidualBlock, ResidualBlockWithDownsample, etc.
- Utility classes: Trainer, Evaluator, SARCompressor, SARMetrics

**Functions:**
- `snake_case()` for all functions
- `test_*()` pattern for test functions at module level
- Private methods: `_method()` prefix (e.g., `_initialize_weights()`, `_augment()`)
- Dunder methods: `__init__()`, `__len__()`, `__getitem__()`, `forward()`

**Variables:**
- `snake_case` for all variables
- Single-letter allowed in math contexts: `x`, `z`, `C`, `H`, `W`
- SAR domain abbreviations: `ENL` (Equivalent Number of Looks), `EPI` (Edge Preservation Index), `dB` (decibels)
- Abbreviations: `val_` prefix for validation, `_loss` suffix, `_weight` suffix
- Configuration: All lowercase in YAML, converted to snake_case in Python

## Where to Add New Code

**New Feature (e.g., new loss function):**
- Primary code: `src/losses/{feature_name}.py` or add to existing loss file
- Tests: Include `test_{feature}()` function at bottom of module
- Integration: Update `src/losses/__init__.py` to export new class
- Usage: Referenced in `Trainer` via config or explicit instantiation

**New Component/Module (e.g., attention encoder variant):**
- Implementation: `src/models/{component_name}.py`
- If reusable building block: Add to `src/models/blocks.py`
- Test function: Include in same file
- Entry point: Update `src/models/__init__.py`

**New Data Augmentation:**
- Implementation: Add method to `SARPatchDataset._augment()` in `src/data/dataset.py`
- Requirement: Must be SAR-safe (flips/rotations only, no arbitrary rotations or intensity changes)
- Control: Add boolean flag to `__init__` and YAML config

**New Metric:**
- Implementation: Static method in `SARMetrics` class in `src/evaluation/evaluator.py`
- Usage: Call from `Evaluator.evaluate_batch()` or `evaluate_dataset()`
- Integration: Add to metrics dict in Evaluator methods

**New Preprocessing Step:**
- Implementation: Method in `SARCompressor` class in `src/inference/compressor.py`
  - Or separate file: `src/data/preprocessing.py` if standalone utility
- Control: Store preprocessing params in checkpoint metadata (SARCompressor._load_model())

**Utility Functions:**
- Shared helpers: `src/utils/` directory
- File I/O: `src/utils/io.py`
- Math utilities: `src/utils/{utility_name}.py`

**Configuration Changes:**
- Edit: `configs/default.yaml` directly
- New sections: Add top-level key (e.g., `new_feature:`)
- New params: Use YAML defaults, override via CLI args or other config files

**Training Configuration:**
- Edit: `configs/default.yaml` sections: data, model, loss, training, logging
- Don't edit: Trainer class directly - rely on config dict

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD-generated codebase documentation
- Generated: False (human-written or GSD agent)
- Committed: Yes
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md (if quality focus), TESTING.md (if quality focus), CONCERNS.md (if concerns focus), STACK.md (if tech focus), INTEGRATIONS.md (if tech focus)

**`checkpoints/`:**
- Purpose: Model weight persistence
- Generated: True (by Trainer during training)
- Committed: No (.gitignore)
- Format: PyTorch .pth files (dict with model_state_dict, optimizer_state_dict, config, etc.)

**`runs/`:**
- Purpose: TensorBoard event logging
- Generated: True (by Trainer via SummaryWriter)
- Committed: No (.gitignore)
- Structure: `runs/{YYYYMMDD_HHMMSS}/` subdirectory per run

**`data/`:**
- Purpose: Local training data
- Generated: True (by download_sentinel_data.py or user)
- Committed: No (.gitignore)
- Key file: `patches/patches.npy` - NumPy array of shape (N, 256, 256) with float32 values in [0, 1]

**`experiments/`:**
- Purpose: Experiment metadata and results
- Generated: True (by experiment scripts)
- Committed: No (.gitignore)
- Planned use: Store experiment configs, metrics, comparison results

**`results/`:**
- Purpose: Final evaluation results
- Generated: True (by evaluation scripts)
- Committed: No (.gitignore)
- Planned content: Metrics JSON, reconstructed images, compression stats

**`.vscode/`:**
- Purpose: VS Code workspace configuration
- Generated: False (manually created)
- Committed: Yes
- Contents: Workspace settings, Python interpreter path, extensions

**`learningnotebooks/`:**
- Purpose: Educational progression
- Generated: False (human-written learning exercises)
- Committed: Yes
- Organization: Phase1 → Phase2 → Phase3 → Phase4, each with progressive complexity

**`venv/`:**
- Purpose: Python virtual environment
- Generated: True (by `python -m venv venv`)
- Committed: No (.gitignore)
- Usage: Isolated Python environment with dependencies from requirements.txt

---

*Structure analysis: 2026-01-21*
