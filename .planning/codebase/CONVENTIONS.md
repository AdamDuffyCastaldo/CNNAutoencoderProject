# Coding Conventions

**Analysis Date:** 2026-01-21

## Naming Patterns

**Files:**
- Module files use snake_case: `dataset.py`, `evaluator.py`, `preprocessing.py`
- Package directories use snake_case: `src/data/`, `src/models/`, `src/evaluation/`
- Configuration files use lowercase with underscores: `default.yaml`

**Classes:**
- PascalCase for all classes: `SARPatchDataset`, `SAREncoder`, `SARMetrics`, `Visualizer`, `Evaluator`
- Domain-specific prefixes used: `SAR*` classes indicate SAR (Synthetic Aperture Radar) domain knowledge
- PyTorch nn.Module subclasses: `SARAutoencoder`, `SAREncoder`, `ConvBlock`, `ResidualBlock`

**Functions:**
- snake_case for all module-level functions: `handle_invalid_values()`, `preprocess_sar_complete()`, `extract_patches()`, `analyze_sar_statistics()`
- Descriptive names with clear purpose: `test_dataset()`, `test_encoder()`, `test_evaluator()`, `test_blocks()`
- Private methods prefixed with underscore: `_initialize_weights()`, `_augment()`, `_gradient_magnitude()`

**Variables:**
- snake_case for local variables and instance attributes: `patches`, `latent_channels`, `base_channels`, `vmin`, `vmax`
- Single letters for loop counters and temporary values: `i`, `j`, `x`, `z` (follows PyTorch convention for tensors)
- numpy arrays typically lowercase: `image`, `patches`, `test_data`
- torch tensors typically lowercase: `x`, `z`, `loss`, `gradients`
- Mathematical variable names match domain literature: `x` (input), `x_hat` (reconstruction), `z` (latent), `mse`, `psnr`, `ssim`

**Types:**
- Type hints used throughout: `np.ndarray`, `torch.Tensor`, `Dict`, `List`, `Optional`, `Tuple`, `Callable`
- Type imports from `typing` module: `from typing import Dict, List, Tuple, Optional`

## Code Style

**Formatting:**
- No explicit formatter detected (no .prettierrc, black config, or isort config)
- Line length appears implicit, varies 50-150 characters
- Indentation: 4 spaces (Python standard)
- Trailing commas in multiline structures used inconsistently

**Comments:**
- Docstring style: Google-style docstrings with sections (Args, Returns, Example, Note, References)
- Triple-quoted docstrings for all classes and public functions: `"""Docstring here"""`
- Single-line comments use `#` at end of line or above
- Comment formatting: ` # Comment` with space after hash
- Section headers in comments: `# =========================================`
- Notes and warnings: `# TODO:`, `# Note:`, `# Warning:`

**Example docstring pattern from `src/data/dataset.py`:**
```python
"""
PyTorch Dataset Classes for SAR Autoencoder

This module provides Dataset classes for loading SAR patches.

References:
    - Day 2, Section 2.6 of the learning guide
"""

class SARPatchDataset(Dataset):
    """
    Dataset for SAR patches.

    Handles:
    - Numpy array to tensor conversion
    - Optional data augmentation (flips, 90° rotations)
    - Channel dimension addition

    Args:
        patches: Numpy array of shape (N, H, W), values in [0, 1]
        augment: Whether to apply random augmentation
        transform: Optional additional transform function

    Note:
        For SAR, safe augmentations are:
        - Horizontal flip
        - Vertical flip
        - 90° rotations

    Example:
        >>> patches = np.load('patches.npy')
        >>> dataset = SARPatchDataset(patches, augment=True)
        >>> sample = dataset[0]
    """
```

## Import Organization

**Order:**
1. Standard library imports: `import torch`, `import numpy as np`, `from pathlib import Path`
2. Third-party imports: `from torch.utils.data import Dataset`, `from scipy import ndimage`
3. Local/relative imports: `from .blocks import ConvBlock`, `from src.models import SARAutoencoder`

**Pattern from `src/evaluation/evaluator.py`:**
```python
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as skimage_ssim
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
```

**Path Aliases:**
- sys.path manipulation used in scripts: `sys.path.insert(0, str(Path(__file__).parent.parent))`
- No formal path aliases configured (no pyproject.toml or setup.cfg)

## Error Handling

**Patterns:**
- NotImplementedError with descriptive messages for stub implementations: `raise NotImplementedError("TODO: Implement forward pass")`
- Assertions for shape validation: `assert z.shape == (2, 64, 16, 16), f"Wrong shape: {z.shape}"`
- Optional type hints for nullable returns: `Optional[Callable]`, `Optional[str]`
- numpy's conditional assignment for edge cases: `np.where(image <= 0, noise_floor, image)`
- Division by zero protection with small epsilon: `local_var = np.maximum(local_var, 1e-10)`

**Example from `src/evaluation/evaluator.py`:**
```python
def psnr(x: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((x - x_hat) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(data_range ** 2 / mse))
```

## Logging

**Framework:** `print()` statements only - no logging library configured

**Patterns:**
- Progress reporting with print: `print(f"Evaluating dataset...")`
- Formatted output with f-strings: `print(f"✓ Shape correct: {z.shape}")`
- Section dividers: `print("=" * 70)`, `print("-" * 50)`
- Status indicators: `✓` (checkmark) for success, `⚠` (warning) for issues
- TensorBoard integration available via `tensorboard` dependency but not seen in source yet

**Example from `src/evaluation/evaluator.py`:**
```python
def print_evaluation_report(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
```

## Function Design

**Size:**
- Small focused functions preferred: most functions 5-50 lines
- Complex operations broken into steps with comments
- Maximum depth 2-3 levels of nesting

**Parameters:**
- Type hints required: all function parameters annotated
- Default values used: `kernel_size: int = 5`, `stride: int = 2`
- Kwargs for flexible parameters: `**kwargs` in `compute_clip_bounds()`
- Single responsibility: functions do one thing well

**Return Values:**
- Type hints on returns: `-> float`, `-> torch.Tensor`, `-> Dict[str, float]`
- Multiple returns via tuples: `Tuple[torch.Tensor, Dict]`, `Tuple[float, float]`
- Numpy float conversion: `return float(np.mean(...))` to avoid numpy scalar types
- Always return meaningful values, raise for errors

**Example from `src/data/preprocessing.py`:**
```python
def compute_clip_bounds(
    images: np.ndarray,
    method: str = 'percentile',
    **kwargs
) -> Tuple[float, float]:
    """
    Compute clip bounds from training data.

    Methods:
    - 'percentile': Use data-driven percentiles (low_pct, high_pct)
    - 'fixed': Use domain knowledge (vmin, vmax)
    - 'sigma': Use mean ± k×std

    Args:
        images: Array of images (in dB)
        method: Clipping method
        **kwargs: Method-specific parameters

    Returns:
        (vmin, vmax) clip bounds in dB
    """
```

## Module Design

**Exports:**
- Classes and functions at module level are public
- Underscore prefix for private implementation: `_initialize_weights()`, `_augment()`
- No `__all__` lists seen; import star discouraged by convention

**Barrel Files:**
- Package `__init__.py` files present but often empty or minimal
- Examples: `src/data/__init__.py`, `src/models/__init__.py`
- Explicit imports preferred: `from src.models import SARAutoencoder`

**Submodule Organization:**
- Domain-separated modules: `src/data/`, `src/models/`, `src/evaluation/`, `src/losses/`, `src/compression/`, `src/inference/`
- Each module focuses on specific responsibility
- Cross-module imports allowed: evaluator imports models and data utilities
- Circular imports avoided by this separation

## Device Handling (PyTorch Specific)

**Pattern:**
- Device passed as parameter: `device='cuda'` or `device='cpu'`
- Tensor movement explicit: `.to(self.device)`, `.cpu()`, `.numpy()`
- Context managers for inference: `@torch.no_grad()`
- Device agnostic default: `device = torch.device(device)` with string parameter

**Example from `src/evaluation/evaluator.py`:**
```python
def __init__(self, model, device='cuda'):
    """
    Args:
        model: Trained autoencoder
        device: Device for inference
    """
    self.model = model
    self.device = torch.device(device)
    self.model.to(self.device)
    self.model.eval()
```

---

*Convention analysis: 2026-01-21*
