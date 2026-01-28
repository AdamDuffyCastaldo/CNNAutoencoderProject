# Phase 6: Final Experiments - Research

**Researched:** 2026-01-28
**Domain:** Experiment execution, statistical analysis, rate-distortion comparison, reporting
**Confidence:** HIGH

## Summary

Phase 6 focuses on executing the final experiment matrix comparing autoencoder architectures against JPEG-2000 codec baselines, then producing a comprehensive comparison study with statistical rigor. The project already has a mature codebase with complete evaluation infrastructure (`Evaluator`, `CodecEvaluator`, `Visualizer`), trained models (ResNet, Baseline at various ratios), and established metrics (PSNR, SSIM, MS-SSIM, ENL ratio, EPI).

The primary work involves:
1. **Training missing models** at required compression ratios (ResNet 4x, 8x if needed)
2. **Systematic evaluation** using existing infrastructure with proper codec baseline comparison
3. **Statistical analysis** using scipy.stats for paired tests (t-test and Wilcoxon)
4. **Report generation** in Jupyter notebook + Markdown format with publication-quality figures

The key insight from Phase 4 is that **ResNet @16x (21.13 dB) achieves similar quality to Baseline @8x (21.34 dB)**, meaning ResNet achieves 2x better compression at equivalent quality. This gives a strong basis for the comparison study.

**Primary recommendation:** Leverage existing evaluation infrastructure rather than building new tools. Focus on systematic execution and statistical rigor.

## Standard Stack

The established libraries/tools for this domain:

### Core (Already in Project)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| pytorch | 2.x | Model training/inference | In use |
| pytorch-msssim | 1.0+ | MS-SSIM metric | In use |
| scipy | 1.10+ | Statistical tests (paired t-test, Wilcoxon) | Needs statistical functions |
| matplotlib | 3.7+ | Visualization and figures | In use |
| numpy | 1.24+ | Numerical operations | In use |
| pandas | 2.0+ | Data aggregation and CSV/tables | In use |
| opencv-python | 4.8+ | JPEG-2000/JPEG codec support | In use |

### Supporting (Add for Phase 6)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | (incl.) | `ttest_rel`, `wilcoxon`, `shapiro` | Statistical hypothesis tests |
| statsmodels | 0.14+ | Multiple comparison corrections | If comparing >2 methods per metric |
| scikit-image | 0.21+ | Additional metrics if needed | Already in requirements |

### Already Implemented
| Component | Location | Purpose |
|-----------|----------|---------|
| `Evaluator` | `src/evaluation/evaluator.py` | Model evaluation with all metrics |
| `CodecEvaluator` | `src/evaluation/codec_baselines.py` | JPEG-2000/JPEG comparison |
| `Visualizer` | `src/evaluation/visualizer.py` | Rate-distortion curves, comparisons |
| `compute_all_metrics` | `src/evaluation/metrics.py` | Unified metric computation |
| `train_sweep.py` | `scripts/train_sweep.py` | Batch training with YAML configs |
| `evaluate_model.py` | `scripts/evaluate_model.py` | CLI for model evaluation |

**Installation:**
```bash
# Already have most dependencies, just ensure scipy.stats is available
pip install scipy statsmodels  # If not present
```

## Architecture Patterns

### Recommended Project Structure
```
reports/
    final_comparison.ipynb           # Main analysis notebook
    final_comparison.md              # Markdown report (generated)
    figures/
        rate_distortion_psnr.png
        rate_distortion_ssim.png
        visual_comparison_16x.png
        visual_comparison_8x.png
        visual_comparison_4x.png
    tables/
        results_summary.csv
        statistical_tests.csv
        per_ratio_metrics.csv
    data/
        all_results.json             # Raw evaluation data
```

### Pattern 1: Systematic Evaluation Pipeline
**What:** Evaluate all models and codecs in a consistent manner
**When to use:** For each model/ratio combination
**Example:**
```python
# Existing infrastructure - USE THIS, don't rebuild
from src.evaluation import Evaluator
from src.evaluation.codec_baselines import JPEG2000Codec, CodecEvaluator

def evaluate_all_models(checkpoints: dict, test_loader):
    """Evaluate all models consistently."""
    results = []

    for name, ckpt_path in checkpoints.items():
        model, _, preproc = load_model_from_checkpoint(ckpt_path, device)
        evaluator = Evaluator(model, model_name=name, preprocessing_params=preproc)
        result = evaluator.evaluate_dataset(test_loader)
        results.append(evaluator.collect_rd_point(result))

    return results
```

### Pattern 2: Statistical Test Execution
**What:** Paired tests comparing autoencoder vs codec on same images
**When to use:** For formal hypothesis testing
**Example:**
```python
from scipy.stats import ttest_rel, wilcoxon, shapiro

def compare_methods_statistically(ae_metrics: list, codec_metrics: list):
    """Perform paired statistical tests."""
    # Check normality first
    _, p_normal = shapiro(np.array(ae_metrics) - np.array(codec_metrics))

    if p_normal > 0.05:
        # Use parametric t-test if normal
        stat, p_value = ttest_rel(ae_metrics, codec_metrics)
        test_name = "paired t-test"
    else:
        # Use non-parametric Wilcoxon if not normal
        stat, p_value = wilcoxon(ae_metrics, codec_metrics)
        test_name = "Wilcoxon signed-rank"

    return {
        'test': test_name,
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Pattern 3: Rate-Distortion Data Collection
**What:** Collect BPP and quality metrics for plotting
**When to use:** For generating R-D curves
**Example:**
```python
def collect_rd_data(models: dict, codec_evaluators: list, test_images: list):
    """Collect rate-distortion data for all methods."""
    rd_data = []

    # Autoencoders (fixed ratio per model)
    for name, result in models.items():
        rd_data.append({
            'name': name,
            'type': 'autoencoder',
            'bpp': 32.0 / result['compression_ratio'],
            'psnr': result['psnr']['mean'],
            'ssim': result['ssim']['mean'],
            'ms_ssim': result['ms_ssim']['mean'],
        })

    # Codecs (multiple quality levels)
    for codec_eval in codec_evaluators:
        for ratio in [4, 8, 16]:
            result = codec_eval.evaluate_batch(test_images, ratio)
            rd_data.append({
                'name': codec_eval.codec.name,
                'type': 'codec',
                'bpp': 32.0 / ratio,
                'psnr': result['metrics']['psnr']['mean'],
                'ssim': result['metrics']['ssim']['mean'],
            })

    return pd.DataFrame(rd_data)
```

### Anti-Patterns to Avoid
- **Rebuilding evaluation logic:** The `Evaluator` and `CodecEvaluator` classes already handle metric computation. Use them.
- **Hardcoding file paths:** Use Path objects and relative paths from project root.
- **Running evaluation without normalization match:** Codec evaluation must use same [0,1] normalization as autoencoders.
- **Ignoring per-sample data:** Statistical tests require per-sample metrics, not just means.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Metric computation | Custom PSNR/SSIM | `compute_all_metrics()` | Handles edge cases, consistent output |
| Codec comparison | Manual JPEG-2000 calls | `CodecEvaluator` | Calibrates quality params, caches results |
| R-D curve plotting | Manual matplotlib | `Visualizer.plot_rate_distortion()` | Handles grouping, styling |
| Visual comparisons | Custom layout | `Visualizer.plot_comparison()` | Auto-zoom, difference maps |
| Checkpoint loading | Manual torch.load | `load_model_from_checkpoint()` | Handles all model types |
| Statistical tests | Manual formulas | scipy.stats | Exact p-values, proper handling of ties |

**Key insight:** The Phase 3 evaluation infrastructure was specifically designed for this comparison study. Use it directly.

## Common Pitfalls

### Pitfall 1: Codec BPP Mismatch
**What goes wrong:** Comparing autoencoder at exact ratio vs codec at calibrated (approximate) ratio
**Why it happens:** Codec quality parameters don't map exactly to target ratios
**How to avoid:** Use the `achieved_ratio` from `CodecEvaluator` results, not target_ratio
**Warning signs:** Large discrepancy between target and achieved compression ratio

### Pitfall 2: Statistical Test Assumptions
**What goes wrong:** Using paired t-test when differences aren't normally distributed
**Why it happens:** Image quality metrics often have skewed distributions
**How to avoid:** Always run Shapiro-Wilk test first; use Wilcoxon if p < 0.05
**Warning signs:** PSNR differences showing heavy tails or outliers

### Pitfall 3: Multiple Comparison Problem
**What goes wrong:** Claiming significance when testing many metrics
**Why it happens:** Each test at alpha=0.05 has 5% false positive rate
**How to avoid:** Apply Bonferroni correction when comparing >3 metrics
**Warning signs:** Marginal p-values (0.01-0.05) across many comparisons

### Pitfall 4: Evaluation on Training Data
**What goes wrong:** Overly optimistic quality metrics
**Why it happens:** Using validation split that overlaps with training
**How to avoid:** Use dedicated test split OR new unseen Sentinel-1 scenes
**Warning signs:** Autoencoder metrics much better than training curves suggested

### Pitfall 5: JPEG-2000 Quality Parameter Range
**What goes wrong:** Unable to achieve high compression ratios
**Why it happens:** JPEG-2000 quality parameter (1-1000) has nonlinear effect
**How to avoid:** `CodecEvaluator.calibrate()` already handles this via binary search
**Warning signs:** `achieved_ratio` very different from `target_ratio`

### Pitfall 6: Small Test Set for Statistics
**What goes wrong:** Wide confidence intervals, unstable p-values
**Why it happens:** Insufficient samples for reliable statistics
**How to avoid:** Use at least 100 test samples; 500+ preferred for tight CIs
**Warning signs:** Standard error of mean > 0.5 dB for PSNR

## Code Examples

Verified patterns from existing codebase:

### Loading and Evaluating a Checkpoint
```python
# Source: scripts/evaluate_model.py
from src.evaluation import Evaluator
from src.evaluation.evaluator import print_evaluation_report

def evaluate_checkpoint(checkpoint_path: str, test_loader, device='cuda'):
    """Complete evaluation of a single checkpoint."""
    model, model_name, preproc = load_model_from_checkpoint(checkpoint_path, device)

    evaluator = Evaluator(
        model,
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        preprocessing_params=preproc
    )

    results = evaluator.evaluate_dataset(test_loader, show_progress=True)
    print_evaluation_report(results)

    return results, evaluator
```

### Codec Baseline Comparison
```python
# Source: src/evaluation/codec_baselines.py
from src.evaluation.codec_baselines import JPEG2000Codec, CodecEvaluator

def evaluate_codec_baseline(test_images: list, target_ratios: list):
    """Evaluate JPEG-2000 at multiple compression ratios."""
    codec = JPEG2000Codec()
    evaluator = CodecEvaluator(codec)

    # Calibrate quality parameters
    evaluator.calibrate(target_ratios, test_images[:5])

    # Evaluate at each ratio
    results = evaluator.evaluate_at_ratios(test_images, target_ratios)

    return results
```

### Rate-Distortion Curve
```python
# Source: src/evaluation/visualizer.py
from src.evaluation import Visualizer

def generate_rd_curve(rd_data: list, output_dir: str):
    """Generate publication-quality R-D curves."""
    viz = Visualizer(save_dir=output_dir)

    viz.plot_rate_distortion(
        rd_data,
        output_path='rate_distortion.png',
        title='Rate-Distortion: Autoencoder vs JPEG-2000',
        show=False
    )
```

### Statistical Comparison
```python
# Source: scipy.stats documentation
from scipy.stats import ttest_rel, wilcoxon, shapiro
import numpy as np

def statistical_comparison(ae_psnr: np.ndarray, codec_psnr: np.ndarray):
    """
    Perform paired statistical test between autoencoder and codec.

    Args:
        ae_psnr: Per-sample PSNR values for autoencoder
        codec_psnr: Per-sample PSNR values for codec (same samples)

    Returns:
        Dict with test results
    """
    diff = ae_psnr - codec_psnr

    # Normality test
    _, p_normal = shapiro(diff)

    # Select appropriate test
    if p_normal > 0.05 and len(diff) >= 20:
        stat, p_value = ttest_rel(ae_psnr, codec_psnr)
        test_name = "paired t-test"
    else:
        stat, p_value = wilcoxon(ae_psnr, codec_psnr, alternative='two-sided')
        test_name = "Wilcoxon signed-rank"

    return {
        'test': test_name,
        'statistic': float(stat),
        'p_value': float(p_value),
        'mean_difference': float(np.mean(diff)),
        'std_difference': float(np.std(diff)),
        'n_samples': len(diff),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
    }
```

### Publication-Quality Figure Setup
```python
# Source: matplotlib best practices
import matplotlib.pyplot as plt

def setup_publication_style():
    """Configure matplotlib for paper-ready figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (6.5, 4.0),  # Single column width
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PSNR-only comparison | Multi-metric (PSNR, SSIM, MS-SSIM, perceptual) | 2018+ | More complete quality assessment |
| Single-ratio models | Variable-rate or multi-ratio evaluation | 2020+ | Better R-D characterization |
| Informal comparison | Statistical hypothesis testing | Standard | Rigorous claims |
| JPEG baseline only | JPEG-2000, BPG, VVC comparison | 2020+ | Fair comparison to modern codecs |

**Current best practice for compression papers:**
- Report R-D curves (not just single-point comparisons)
- Include BD-rate (Bjontegaard Delta rate) for single-number comparison
- Use multiple quality metrics (at minimum PSNR + perceptual like SSIM/MS-SSIM)
- Statistical significance testing for claims of superiority
- Visual examples at multiple compression levels

**Deprecated/outdated:**
- Single compression ratio evaluation (use R-D curves)
- PSNR-only quality assessment (add SSIM minimum)
- Comparison to JPEG only (JPEG-2000 is the standard baseline)

## Open Questions

Things that couldn't be fully resolved:

1. **BD-Rate Calculation**
   - What we know: BD-rate is standard for single-number R-D comparison
   - What's unclear: Whether to implement custom or use existing package
   - Recommendation: Implement simple BD-rate calculation or use `bjontegaard` package if available. Not critical if R-D curves are clear.

2. **New Test Scenes Geographic Diversity**
   - What we know: CONTEXT.md requests new unseen Sentinel-1 scenes
   - What's unclear: Specific regions to sample from, how many scenes needed
   - Recommendation: Download 2-3 scenes from regions different from training (e.g., if training was Europe, test on Asia/Americas). 100+ patches per scene.

3. **Terrain Type Stratification**
   - What we know: CONTEXT.md requests metrics by terrain type
   - What's unclear: How to classify patches by terrain (manual vs automatic)
   - Recommendation: Start with visual inspection, group into obvious categories (water, urban, forest, agriculture). Can use simple heuristics (low variance = water, high edges = urban).

## Sources

### Primary (HIGH confidence)
- Existing codebase: `src/evaluation/` (evaluator.py, metrics.py, codec_baselines.py, visualizer.py)
- Existing codebase: `scripts/evaluate_model.py`, `scripts/train_sweep.py`
- [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
- [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)

### Secondary (MEDIUM confidence)
- [Matplotlib for publication figures](https://github.com/jbmouret/matplotlib_for_papers) - Best practices
- [Publication-Quality Plots in Python](https://www.fschuch.com/en/blog/2025/07/05/publication-quality-plots-in-python-with-matplotlib/) - Figure sizing
- [PSNR vs SSIM IEEE paper](https://ieeexplore.ieee.org/document/5596999/) - Metric interpretation

### Tertiary (LOW confidence)
- [Rate-Distortion Autoencoders](https://ar5iv.labs.arxiv.org/html/1908.05717) - R-D optimization approach
- [Variable Rate Deep Image Compression](https://ar5iv.labs.arxiv.org/html/1909.04802) - Multi-rate comparison methodology

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use or standard Python scientific stack
- Architecture: HIGH - Using existing evaluation infrastructure directly
- Pitfalls: HIGH - Based on actual codebase analysis and image quality research
- Statistical methods: HIGH - scipy.stats is well-documented and standard

**Research date:** 2026-01-28
**Valid until:** Indefinite - methodology is stable, infrastructure is complete

---

## Executive Summary for Planner

### What's Already Done
1. Complete evaluation infrastructure (`Evaluator`, `CodecEvaluator`, `Visualizer`)
2. Trained models: ResNet 16x (21.13 dB), Baseline 4x/8x/12x/16x
3. Metrics: PSNR, SSIM, MS-SSIM, ENL ratio, EPI all implemented
4. CLI tools: `evaluate_model.py`, `train_sweep.py`

### What Needs to Be Done
1. **Train missing models:** ResNet at 4x and 8x ratios (if needed)
2. **Run systematic evaluation:** All models + JPEG-2000 on test data
3. **Collect per-sample metrics:** For statistical testing
4. **Statistical tests:** Paired t-test/Wilcoxon for autoencoder vs codec
5. **Generate R-D curves:** Using existing `Visualizer.plot_rate_distortion()`
6. **Visual comparisons:** 5 examples per ratio (15 total)
7. **Write report:** Jupyter notebook + Markdown

### Key Constraints from CONTEXT.md
- Architectures: ResNet + Baseline only (2 architectures)
- Ratios: 4x, 8x, 16x (skip 12x)
- Codec baseline: JPEG-2000 only (skip JPEG)
- Statistics: Paired tests (t-test or Wilcoxon)
- Output: `reports/` directory, static matplotlib figures

### Critical Success Factors
1. Use existing `evaluate_model.py --compare-codecs` as starting point
2. Ensure per-sample metrics are saved for statistical tests
3. Match BPP levels for fair comparison (use achieved_ratio, not target)
4. Apply proper statistical tests based on normality check
