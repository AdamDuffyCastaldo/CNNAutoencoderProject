# Phase 3: SAR Evaluation Framework - Research

**Researched:** 2026-01-24
**Domain:** SAR image quality metrics, codec comparison, evaluation infrastructure
**Confidence:** HIGH

## Summary

This phase implements SAR-specific quality metrics (ENL ratio, EPI) and evaluation tools for comparing autoencoder reconstruction quality against traditional codecs. The research covers three domains: (1) SAR-specific metrics implementation, (2) traditional codec baselines using OpenCV/Pillow, and (3) evaluation infrastructure patterns.

The existing codebase has partial implementations in `src/evaluation/` with PSNR, SSIM, correlation, and basic histogram similarity. ENL and EPI are stubbed but not implemented in `metrics.py`. The `evaluator.py` has a working batch evaluation framework. Key gaps are: ENL ratio computation with homogeneous region detection, proper EPI implementation, MS-SSIM integration, JPEG-2000/JPEG codec baselines, rate-distortion curve generation, and JSON output format.

**Primary recommendation:** Extend existing `SARMetrics` class with ENL ratio (using coefficient of variation for homogeneous region detection), implement codec baselines using OpenCV for JPEG-2000 (with `IMWRITE_JPEG2000_COMPRESSION_X1000` parameter), and add structured JSON output for machine-readable results.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-image | >=0.21 | SSIM, image metrics | Already used, skimage_ssim in metrics.py |
| scipy.ndimage | (part of scipy) | uniform_filter, sobel for ENL/EPI | Already imported in evaluator.py |
| pytorch-msssim | 1.0.0 | GPU-accelerated MS-SSIM | Already used in SSIMLoss, provides ms_ssim function |
| numpy | >=1.24 | Array operations | Foundation of existing code |
| matplotlib | >=3.7 | Visualization, rate-distortion plots | Already used in visualizer.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| OpenCV (cv2) | >=4.8 | JPEG-2000 encoding/decoding | Codec baselines - has JPEG2000_COMPRESSION_X1000 param |
| Pillow | >=10.0 | Alternative JPEG codec | JPEG baseline if OpenCV unavailable |
| json | stdlib | Metric output files | Machine-readable results |
| tqdm | >=4.65 | Progress bars for batch eval | Already used in trainer.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| OpenCV JPEG-2000 | Pillow J2K | Pillow has quality_mode='dB' option, but OpenCV more direct compression ratio control |
| skimage SSIM | pytorch-msssim SSIM | pytorch-msssim already integrated, GPU-accelerated, consistent with training |
| Custom rate-distortion | CompressAI plotting | CompressAI heavyweight dependency for simple plotting needs |

**Installation:**
```bash
# Already installed in project:
pip install scikit-image scipy pytorch-msssim numpy matplotlib opencv-python pillow tqdm
```

## Architecture Patterns

### Recommended Project Structure
```
src/evaluation/
    __init__.py              # Exports public API
    metrics.py               # ENL, EPI, histogram similarity implementations
    evaluator.py             # Batch evaluation with statistics
    visualizer.py            # Comparison image generation
    codec_baselines.py       # NEW: Traditional codec compression/evaluation

evaluations/                 # NEW: Output directory
    {model_name}/
        {model_name}_eval.json       # Summary metrics
        {model_name}_detailed.json   # Per-patch metrics
        rate_distortion.csv          # R-D data points
        rate_distortion.png          # R-D plot
        comparisons/                 # Visual comparisons
            sample_01.png
            ...
```

### Pattern 1: ENL Ratio with Homogeneous Region Detection
**What:** Compute ENL in homogeneous regions by detecting areas with low coefficient of variation (CV)
**When to use:** Always for ENL computation - avoids texture regions biasing results
**Example:**
```python
# Source: Knowledge document 06_SAR_QUALITY_METRICS.md + S1-NRB documentation
from scipy.ndimage import uniform_filter
import numpy as np

def find_homogeneous_regions(image: np.ndarray,
                             window_size: int = 15,
                             cv_threshold: float = 0.3) -> np.ndarray:
    """
    Find homogeneous regions using coefficient of variation.

    CV = sigma / mu
    Low CV indicates homogeneous (flat) areas.

    Returns:
        Boolean mask where True = homogeneous
    """
    local_mean = uniform_filter(image, size=window_size)
    local_sq_mean = uniform_filter(image**2, size=window_size)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))

    cv = local_std / (local_mean + 1e-10)
    return cv < cv_threshold

def compute_enl(image: np.ndarray, mask: np.ndarray) -> float:
    """
    ENL = mu^2 / sigma^2 in homogeneous regions.
    """
    if mask.sum() < 100:  # Need enough samples
        return np.nan

    values = image[mask]
    mean = np.mean(values)
    var = np.var(values)

    if var < 1e-10:
        return float('inf')

    return (mean ** 2) / var

def enl_ratio(original: np.ndarray,
              reconstructed: np.ndarray,
              window_size: int = 15,
              cv_threshold: float = 0.3) -> dict:
    """
    Compute ENL ratio: ENL_recon / ENL_orig.

    Returns dict with both linear and dB domain results.
    """
    # Find homogeneous regions in original
    mask = find_homogeneous_regions(original, window_size, cv_threshold)

    enl_orig = compute_enl(original, mask)
    enl_recon = compute_enl(reconstructed, mask)

    ratio = enl_recon / enl_orig if enl_orig > 0 else np.nan

    return {
        'enl_original': enl_orig,
        'enl_reconstructed': enl_recon,
        'enl_ratio': ratio,
        'homogeneous_pixels': int(mask.sum()),
        'homogeneous_fraction': float(mask.sum() / mask.size)
    }
```

### Pattern 2: Edge Preservation Index (EPI)
**What:** Correlation of gradient magnitudes between original and reconstruction
**When to use:** Always - measures edge/structure preservation
**Example:**
```python
# Source: Knowledge document 06_SAR_QUALITY_METRICS.md
from scipy.ndimage import sobel
import numpy as np

def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude."""
    gx = sobel(image, axis=1)  # Horizontal
    gy = sobel(image, axis=0)  # Vertical
    return np.sqrt(gx**2 + gy**2)

def edge_preservation_index(original: np.ndarray,
                           reconstructed: np.ndarray) -> float:
    """
    EPI = correlation of gradient magnitudes.

    EPI ~1: edges preserved
    EPI <1: edges smoothed
    EPI >1: edges enhanced/artifacts

    Returns:
        EPI value (typically 0.8-1.0 for good reconstructions)
    """
    grad_orig = compute_gradient_magnitude(original)
    grad_recon = compute_gradient_magnitude(reconstructed)

    go = grad_orig.flatten()
    gr = grad_recon.flatten()

    numerator = np.sum(go * gr)
    denominator = np.sqrt(np.sum(go**2) * np.sum(gr**2))

    return numerator / denominator if denominator > 0 else 0.0
```

### Pattern 3: JPEG-2000 Codec Baseline with Compression Matching
**What:** Encode/decode with traditional codecs at matching compression ratios
**When to use:** Codec baseline comparison - sweep quality parameter to match target compression
**Example:**
```python
# Source: OpenCV imgcodecs documentation
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

# Enable JPEG-2000 support (may be needed)
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'

def encode_jpeg2000(image: np.ndarray,
                    compression_x1000: int = 100) -> bytes:
    """
    Encode image to JPEG-2000 with specified compression.

    Args:
        image: Grayscale image, normalized [0,1]
        compression_x1000: Target compression * 1000 (lower = more compression)
                          Range 1-1000, where 1000 = lowest compression

    Returns:
        Encoded bytes
    """
    # Convert to uint8 or uint16 for encoding
    img_uint8 = (image * 255).astype(np.uint8)

    # Encode with JPEG-2000 compression parameter
    encode_params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_x1000]
    success, encoded = cv2.imencode('.jp2', img_uint8, encode_params)

    if not success:
        raise RuntimeError("JPEG-2000 encoding failed")

    return encoded.tobytes()

def decode_jpeg2000(encoded: bytes) -> np.ndarray:
    """Decode JPEG-2000 bytes to image."""
    arr = np.frombuffer(encoded, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return decoded.astype(np.float32) / 255.0

def find_compression_param(target_ratio: float,
                           sample_image: np.ndarray,
                           tolerance: float = 0.1) -> int:
    """
    Binary search for JPEG-2000 compression parameter matching target ratio.

    Args:
        target_ratio: Desired compression ratio (e.g., 16.0)
        sample_image: Representative image for calibration
        tolerance: Acceptable ratio difference fraction

    Returns:
        compression_x1000 parameter achieving closest match
    """
    original_bytes = sample_image.size * 4  # float32
    target_bytes = original_bytes / target_ratio

    low, high = 1, 1000
    best_param = 500
    best_diff = float('inf')

    while low <= high:
        mid = (low + high) // 2
        encoded = encode_jpeg2000(sample_image, mid)
        achieved_ratio = original_bytes / len(encoded)

        diff = abs(achieved_ratio - target_ratio) / target_ratio
        if diff < best_diff:
            best_diff = diff
            best_param = mid

        if achieved_ratio > target_ratio:
            high = mid - 1  # Need more compression
        else:
            low = mid + 1   # Need less compression

        if diff < tolerance:
            break

    return best_param
```

### Pattern 4: Structured JSON Output
**What:** Machine-readable metric output with summary and detailed per-patch files
**When to use:** Always for evaluation results
**Example:**
```python
# Source: Project CONTEXT.md decisions
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class EvaluationSummary:
    """Summary metrics for a model evaluation."""
    model_name: str
    checkpoint_path: str
    evaluation_date: str
    num_samples: int
    compression_ratio: float

    # Mean metrics
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    ms_ssim_mean: float
    ms_ssim_std: float
    enl_ratio_mean: float
    enl_ratio_std: float
    epi_mean: float
    epi_std: float

    # Metadata for reproducibility
    preprocessing_params: Dict

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'EvaluationSummary':
        with open(path, 'r') as f:
            return cls(**json.load(f))
```

### Anti-Patterns to Avoid
- **Computing ENL on entire image:** ENL is only meaningful in homogeneous regions. Always use coefficient of variation mask.
- **Using uint8 for intermediate calculations:** Maintain float32 precision for metric computation; convert to uint8 only for codec encoding.
- **Hardcoding codec compression parameters:** Compression ratios differ by image content. Always calibrate with sample images.
- **Single-scale SSIM only:** MS-SSIM provides more robust quality assessment; include both.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSIM computation | numpy loops | skimage.metrics.structural_similarity or pytorch-msssim | Proper windowing, constants, edge handling |
| MS-SSIM | Multi-scale SSIM from scratch | pytorch_msssim.ms_ssim | Numerically stable, consistent with published implementations |
| Local mean/variance | Manual convolution | scipy.ndimage.uniform_filter | Optimized, handles boundaries correctly |
| Gradient magnitude | Manual finite differences | scipy.ndimage.sobel | Standard Sobel kernels, proper edge handling |
| JSON serialization with numpy | Custom encoders | dataclasses + explicit float() conversion | Cleaner, less error-prone |
| Progress bars | print statements | tqdm | Already used in project, consistent UX |

**Key insight:** Image quality metrics have subtle implementation details (window sizes, boundary conditions, normalization constants) that affect reproducibility. Using established libraries ensures results match published benchmarks.

## Common Pitfalls

### Pitfall 1: ENL Computed on Textured Regions
**What goes wrong:** ENL values meaningless or highly variable when computed on areas with texture (edges, structures, varying backscatter)
**Why it happens:** ENL formula assumes homogeneous region where intensity variation comes only from speckle
**How to avoid:** Always detect homogeneous regions first using CV threshold (CV < 0.3 typical). Report homogeneous fraction so evaluator knows how much data contributed.
**Warning signs:** ENL values wildly different between patches; ENL ratio >> 2 or << 0.5

### Pitfall 2: JPEG-2000 Compression Ratio Mismatch
**What goes wrong:** Comparing autoencoder at 16x with JPEG-2000 at very different ratio invalidates comparison
**Why it happens:** JPEG-2000 compression parameter doesn't map linearly to compression ratio; varies by image content
**How to avoid:** Calibrate compression parameter per target ratio using binary search on sample images. Cache calibrated parameters.
**Warning signs:** Achieved compression ratios off by >20% from target

### Pitfall 3: Domain Mismatch (dB vs Linear)
**What goes wrong:** Metrics computed in wrong domain give misleading results
**Why it happens:** SAR data processed in dB domain for training but ENL traditionally computed in linear intensity domain
**How to avoid:** Per CONTEXT.md decision: compute ENL in BOTH domains and report both. Document which domain each metric uses.
**Warning signs:** ENL values don't match expected Sentinel-1 GRD range (4-5 looks)

### Pitfall 4: MS-SSIM NaN Values
**What goes wrong:** MS-SSIM returns NaN, breaking evaluation pipeline
**Why it happens:** MS-SSIM involves downsampling; fails on small images or extreme values
**How to avoid:** Use `normalize=True` in pytorch-msssim or catch NaN and fallback to single-scale SSIM. Input images must be >= 160x160 for default 5-scale MS-SSIM.
**Warning signs:** NaN in evaluation results; evaluation crashing

### Pitfall 5: Diverging Colormap Not Centered
**What goes wrong:** Difference visualization misleading because colormap center doesn't align with zero
**Why it happens:** Using imshow without setting vmin/vmax symmetrically around zero
**How to avoid:** For difference maps, always set `vmin=-max_abs, vmax=max_abs` where `max_abs = max(abs(diff))`. Use RdBu_r colormap.
**Warning signs:** White/neutral color doesn't represent zero difference

## Code Examples

Verified patterns from official sources:

### MS-SSIM Computation
```python
# Source: pytorch-msssim documentation (https://github.com/VainF/pytorch-msssim)
from pytorch_msssim import ms_ssim
import torch

def compute_ms_ssim(original: np.ndarray,
                    reconstructed: np.ndarray,
                    data_range: float = 1.0) -> float:
    """
    Compute MS-SSIM between two images.

    Note: Images must be at least 160x160 for default 5-scale computation.
    """
    # Convert to torch tensors: (1, 1, H, W)
    orig_t = torch.from_numpy(original).unsqueeze(0).unsqueeze(0).float()
    recon_t = torch.from_numpy(reconstructed).unsqueeze(0).unsqueeze(0).float()

    # MS-SSIM with normalization for stability
    return ms_ssim(orig_t, recon_t, data_range=data_range, size_average=True).item()
```

### Histogram Similarity (Intersection)
```python
# Source: Existing evaluator.py pattern + knowledge doc
import numpy as np

def histogram_similarity(original: np.ndarray,
                        reconstructed: np.ndarray,
                        bins: int = 256) -> dict:
    """
    Compare intensity distributions using multiple metrics.

    Returns:
        Dict with intersection, chi_square, kl_divergence
    """
    hist_orig, bin_edges = np.histogram(original.flatten(), bins=bins,
                                        range=(0, 1), density=True)
    hist_recon, _ = np.histogram(reconstructed.flatten(), bins=bin_edges,
                                 density=True)

    # Normalize
    hist_orig = hist_orig / (hist_orig.sum() + 1e-10)
    hist_recon = hist_recon / (hist_recon.sum() + 1e-10)

    # Intersection (1 = identical)
    intersection = float(np.minimum(hist_orig, hist_recon).sum())

    # Chi-square (0 = identical)
    chi_sq = float(np.sum((hist_orig - hist_recon)**2 /
                         (hist_orig + hist_recon + 1e-10)))

    # Bhattacharyya coefficient (1 = identical)
    bhatt = float(np.sum(np.sqrt(hist_orig * hist_recon)))

    return {
        'intersection': intersection,
        'chi_square': chi_sq,
        'bhattacharyya': bhatt
    }
```

### Rate-Distortion Curve Generation
```python
# Source: CompressAI pattern + knowledge doc 07_COMPRESSION_TRADEOFFS.md
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

def plot_rate_distortion(results: List[Dict],
                         output_path: str,
                         title: str = "Rate-Distortion Comparison"):
    """
    Plot PSNR vs BPP for multiple codecs/models.

    Args:
        results: List of dicts with keys: name, bpp, psnr, ssim
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by name
    by_name = {}
    for r in results:
        name = r['name']
        if name not in by_name:
            by_name[name] = {'bpp': [], 'psnr': [], 'ssim': []}
        by_name[name]['bpp'].append(r['bpp'])
        by_name[name]['psnr'].append(r['psnr'])
        by_name[name]['ssim'].append(r['ssim'])

    # Plot PSNR vs BPP
    for name, data in by_name.items():
        # Sort by BPP for line plot
        sorted_idx = np.argsort(data['bpp'])
        bpp = np.array(data['bpp'])[sorted_idx]
        psnr = np.array(data['psnr'])[sorted_idx]
        ssim = np.array(data['ssim'])[sorted_idx]

        axes[0].plot(bpp, psnr, 'o-', label=name, markersize=8)
        axes[1].plot(bpp, ssim, 'o-', label=name, markersize=8)

    axes[0].set_xlabel('Bits Per Pixel (BPP)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Bit Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Bits Per Pixel (BPP)')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Bit Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Difference Map Visualization with Diverging Colormap
```python
# Source: Matplotlib colormap documentation + CONTEXT.md decisions
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_with_diff(original: np.ndarray,
                              reconstructed: np.ndarray,
                              psnr: float,
                              ssim: float,
                              output_path: str,
                              zoom_region: tuple = None):
    """
    Plot side-by-side comparison with diverging difference map.

    Args:
        zoom_region: Optional (y_start, y_end, x_start, x_end) for detail crop
    """
    diff = reconstructed - original  # Preserve sign for over/under prediction
    max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)

    n_cols = 4 if zoom_region else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))

    # Original
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Reconstructed
    axes[1].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')

    # Difference (diverging colormap centered at 0)
    im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    axes[2].set_title('Difference (blue=under, red=over)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Zoomed region if provided
    if zoom_region:
        y0, y1, x0, x1 = zoom_region
        axes[3].imshow(diff[y0:y1, x0:x1], cmap='RdBu_r',
                      vmin=-max_abs, vmax=max_abs)
        axes[3].set_title(f'Zoomed Detail [{y0}:{y1}, {x0}:{x1}]')
        axes[3].axis('off')

    fig.suptitle(f'PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SSIM only | MS-SSIM + SSIM | Ongoing | MS-SSIM more robust to scale |
| ENL on full image | ENL on CV-detected regions | Standard practice | Meaningful ENL values |
| Manual codec param tuning | Binary search calibration | Best practice | Reproducible comparisons |
| Print metrics to console | Structured JSON output | Project decision | Machine-readable, reproducible |

**Deprecated/outdated:**
- OpenCV's IMWRITE_JPEG2000_COMPRESSION_X1000=1000 doesn't mean "no compression" - it's the lowest compression setting but still lossy

## Open Questions

Things that couldn't be fully resolved:

1. **ENL dB Domain Computation**
   - What we know: ENL formula ENL = mu^2/sigma^2 is for linear intensity. Project wants both domains.
   - What's unclear: Whether to convert dB back to linear first, or compute directly in dB domain (which changes the interpretation)
   - Recommendation: Compute in both - (1) convert normalized dB to linear using stored preprocessing params, compute ENL; (2) compute ENL directly on normalized dB values. Report both with clear labels.

2. **JPEG-2000 Multiprocessing Issues**
   - What we know: OpenCV JPEG-2000 has known issues with multiprocessing (GitHub issue #22974)
   - What's unclear: Whether this affects our batch evaluation
   - Recommendation: Run codec encoding in single-threaded mode initially. Cache encoded files to avoid re-encoding.

3. **CV Threshold for Homogeneous Detection**
   - What we know: CV < 0.3 is common threshold, but varies by application
   - What's unclear: Optimal threshold for normalized dB SAR data
   - Recommendation: Make configurable with 0.3 default. Report homogeneous fraction to allow assessment.

## Sources

### Primary (HIGH confidence)
- Knowledge document 06_SAR_QUALITY_METRICS.md - Complete ENL, EPI, metric formulas
- Knowledge document 07_COMPRESSION_TRADEOFFS.md - Rate-distortion patterns, BPP calculation
- pytorch-msssim GitHub (https://github.com/VainF/pytorch-msssim) - MS-SSIM API
- Existing src/evaluation/*.py - Current implementation patterns

### Secondary (MEDIUM confidence)
- OpenCV imgcodecs documentation - JPEG2000_COMPRESSION_X1000 parameter verified
- Matplotlib colormap documentation - RdBu diverging colormap usage
- S1-NRB documentation (https://s1-nrb.readthedocs.io/en/v1.6.0/general/enl.html) - ENL definition

### Tertiary (LOW confidence)
- WebSearch results for JPEG-2000 multiprocessing issues - flagged for caution during implementation
- Pillow JPEG2000 options - less direct compression control than OpenCV

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use or clearly documented
- Architecture: HIGH - Patterns derived from existing codebase and knowledge docs
- Pitfalls: MEDIUM - ENL threshold tuning may need adjustment for this specific dataset
- Codec baselines: MEDIUM - JPEG-2000 parameter mapping untested on SAR data specifically

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (30 days - stable domain, no expected library changes)
