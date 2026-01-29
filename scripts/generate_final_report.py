#!/usr/bin/env python3
"""
Generate final comparison report with rate-distortion curves, statistical tests,
visual comparisons, and markdown report.

Usage:
    python scripts/generate_final_report.py --data-dir reports

Requires: reports/data/all_results.json and reports/data/per_sample_metrics.json
(outputs from run_evaluation_sweep.py)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(data_dir: Path):
    """Load evaluation data from JSON files."""
    with open(data_dir / 'data' / 'all_results.json') as f:
        all_results = json.load(f)
    with open(data_dir / 'data' / 'per_sample_metrics.json') as f:
        per_sample = json.load(f)
    summary_df = pd.read_csv(data_dir / 'tables' / 'results_summary.csv')
    return all_results, per_sample, summary_df


def prepare_rd_data(all_results: dict) -> pd.DataFrame:
    """Prepare rate-distortion data for plotting."""
    rd_data = []
    for name, result in all_results.items():
        ratio = result['compression_ratio']
        bpp = 32.0 / ratio  # 32-bit float -> bits per pixel
        metrics = result['metrics']

        # Determine model type and style
        if 'baseline' in name:
            model_type = 'Baseline'
            marker = 'o'
            color = 'tab:blue'
        elif 'resnet' in name:
            model_type = 'ResNet'
            marker = 's'
            color = 'tab:orange'
        else:  # codec
            model_type = 'JPEG-2000'
            marker = '^'
            color = 'tab:gray'

        rd_data.append({
            'name': name,
            'model_type': model_type,
            'ratio': ratio,
            'bpp': bpp,
            'psnr': metrics['psnr']['mean'],
            'psnr_std': metrics['psnr']['std'],
            'ssim': metrics['ssim']['mean'],
            'ssim_std': metrics['ssim']['std'],
            'marker': marker,
            'color': color,
        })

    return pd.DataFrame(rd_data)


def plot_rate_distortion_psnr(rd_df: pd.DataFrame, output_path: Path):
    """Generate PSNR rate-distortion curve."""
    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'figure.figsize': (8, 5), 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type in ['Baseline', 'ResNet', 'JPEG-2000']:
        subset = rd_df[rd_df['model_type'] == model_type].sort_values('bpp')
        if len(subset) == 0:
            continue

        marker = subset.iloc[0]['marker']
        color = subset.iloc[0]['color']

        ax.errorbar(
            subset['bpp'], subset['psnr'], yerr=subset['psnr_std'],
            marker=marker, color=color, label=model_type,
            capsize=3, linewidth=2, markersize=8
        )

    ax.set_xlabel('Bits Per Pixel (BPP)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Rate-Distortion: PSNR vs Compression')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_rate_distortion_ssim(rd_df: pd.DataFrame, output_path: Path):
    """Generate SSIM rate-distortion curve."""
    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'figure.figsize': (8, 5), 'figure.dpi': 150,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type in ['Baseline', 'ResNet', 'JPEG-2000']:
        subset = rd_df[rd_df['model_type'] == model_type].sort_values('bpp')
        if len(subset) == 0:
            continue

        marker = subset.iloc[0]['marker']
        color = subset.iloc[0]['color']

        ax.errorbar(
            subset['bpp'], subset['ssim'], yerr=subset['ssim_std'],
            marker=marker, color=color, label=model_type,
            capsize=3, linewidth=2, markersize=8
        )

    ax.set_xlabel('Bits Per Pixel (BPP)')
    ax.set_ylabel('SSIM')
    ax.set_title('Rate-Distortion: SSIM vs Compression')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def compare_methods(per_sample: dict, ae_name: str, codec_name: str, metric: str = 'psnr'):
    """Paired statistical test between autoencoder and codec on same samples."""
    if ae_name not in per_sample or codec_name not in per_sample:
        return None

    ae_samples = per_sample[ae_name]
    codec_samples = per_sample[codec_name]

    # Get values for matched sample indices
    common_indices = set(ae_samples.keys()) & set(codec_samples.keys())
    if len(common_indices) < 10:
        return None

    ae_vals = np.array([ae_samples[idx][metric] for idx in sorted(common_indices)])
    codec_vals = np.array([codec_samples[idx][metric] for idx in sorted(common_indices)])
    diff = ae_vals - codec_vals

    # Check normality (on subset for Shapiro limit)
    _, p_normal = shapiro(diff[:min(50, len(diff))])

    # Select appropriate test
    if p_normal > 0.05:
        stat, p_value = ttest_rel(ae_vals, codec_vals)
        test_name = "paired t-test"
    else:
        stat, p_value = wilcoxon(ae_vals, codec_vals)
        test_name = "Wilcoxon"

    return {
        'ae_model': ae_name,
        'codec': codec_name,
        'metric': metric,
        'test': test_name,
        'statistic': stat,
        'p_value': p_value,
        'ae_mean': np.mean(ae_vals),
        'codec_mean': np.mean(codec_vals),
        'difference': np.mean(diff),
        'n_samples': len(common_indices),
    }


def run_statistical_tests(per_sample: dict, output_path: Path) -> pd.DataFrame:
    """Run all statistical comparisons and save to CSV."""
    comparisons = []

    for ratio in [4, 8, 16]:
        codec_name = f'jpeg2000_{ratio}x'

        # ResNet vs JPEG-2000
        resnet_name = f'resnet_{ratio}x'
        for metric in ['psnr', 'ssim']:
            result = compare_methods(per_sample, resnet_name, codec_name, metric)
            if result:
                comparisons.append(result)

        # Baseline vs JPEG-2000
        baseline_name = f'baseline_{ratio}x'
        for metric in ['psnr', 'ssim']:
            result = compare_methods(per_sample, baseline_name, codec_name, metric)
            if result:
                comparisons.append(result)

    if not comparisons:
        print("Warning: No comparisons could be performed")
        return pd.DataFrame()

    # Apply Bonferroni correction
    n_tests = len(comparisons)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests

    for comp in comparisons:
        comp['bonferroni_alpha'] = bonferroni_alpha
        comp['significant_bonferroni'] = comp['p_value'] < bonferroni_alpha

    stat_df = pd.DataFrame(comparisons)
    stat_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")

    return stat_df


def generate_visual_comparisons(data_dir: Path):
    """Generate visual comparison grids for each compression ratio."""
    import torch
    import glob

    from src.data.datamodule import SARDataModule
    from src.models.autoencoder import SARAutoencoder
    from src.models.resnet_autoencoder import ResNetAutoencoder
    from src.evaluation.codec_baselines import JPEG2000Codec

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codec = JPEG2000Codec()

    # Load test data
    datamodule = SARDataModule(
        patches_path='data/patches/metadata.npy',
        batch_size=16, val_fraction=0.1, num_workers=0, lazy=True
    )
    test_loader = datamodule.val_dataloader()

    # Get 5 sample images
    sample_images = []
    for batch in test_loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        for i in range(min(5 - len(sample_images), batch.shape[0])):
            sample_images.append(batch[i, 0].numpy())
        if len(sample_images) >= 5:
            break

    print(f"Loaded {len(sample_images)} sample images for visual comparison")

    def load_best_ae_for_ratio(ratio: int):
        """Load best autoencoder for given ratio."""
        lc_map = {4: 64, 8: 32, 16: 16}
        lc = lc_map[ratio]

        # Try ResNet first
        resnet_pattern = f'notebooks/checkpoints/resnet_c{lc}_b64_cr{ratio}x_*'
        matches = sorted(glob.glob(resnet_pattern))
        if matches:
            ckpt_path = Path(matches[-1]) / 'best.pth'
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = ResNetAutoencoder(1, 64, lc)
                model.load_state_dict(ckpt['model_state_dict'])
                return model.to(device).eval(), 'ResNet'

        # Fallback to Baseline
        baseline_pattern = f'notebooks/checkpoints/baseline_c{lc}_b64_cr{ratio}x_*'
        matches = sorted(glob.glob(baseline_pattern))
        if matches:
            ckpt_path = Path(matches[-1]) / 'best.pth'
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = SARAutoencoder(1, lc)
                model.load_state_dict(ckpt['model_state_dict'])
                return model.to(device).eval(), 'Baseline'

        return None, None

    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'figure.dpi': 150, 'savefig.dpi': 300,
    })

    for ratio in [4, 8, 16]:
        print(f"\nGenerating visual comparison for {ratio}x compression...")

        model, model_type = load_best_ae_for_ratio(ratio)
        if model is None:
            print(f"  No model found for {ratio}x, skipping")
            continue

        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        fig.suptitle(f'{ratio}x Compression: Original | {model_type} | JPEG-2000 | Error Map', fontsize=14)

        for row, img in enumerate(sample_images):
            # Original
            axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, 0].set_title('Original' if row == 0 else '')
            axes[row, 0].axis('off')

            # Autoencoder reconstruction
            with torch.no_grad():
                img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
                ae_recon = model(img_t).cpu().numpy()[0, 0]
            axes[row, 1].imshow(ae_recon, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title(f'{model_type}' if row == 0 else '')
            axes[row, 1].axis('off')

            # JPEG-2000 reconstruction
            compressed = codec.compress(img, ratio)
            jp2_recon = codec.decompress(compressed)
            axes[row, 2].imshow(jp2_recon, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title('JPEG-2000' if row == 0 else '')
            axes[row, 2].axis('off')

            # Error map (AE error - JPEG2000 error): blue=AE better, red=JPEG2000 better
            ae_error = np.abs(ae_recon - img)
            jp2_error = np.abs(jp2_recon - img)
            error_diff = jp2_error - ae_error  # Positive = AE better
            axes[row, 3].imshow(error_diff, cmap='RdBu', vmin=-0.2, vmax=0.2)
            axes[row, 3].set_title('Error Diff (blue=AE better)' if row == 0 else '')
            axes[row, 3].axis('off')

        plt.tight_layout()
        save_path = data_dir / 'figures' / f'visual_comparison_{ratio}x.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        del model
        torch.cuda.empty_cache()


def generate_markdown_report(data_dir: Path, rd_df: pd.DataFrame, stat_df: pd.DataFrame):
    """Generate final markdown report with executive summary."""
    output_path = data_dir / 'final_comparison.md'

    # Build executive summary based on results
    summary_lines = []

    # Find best results at each ratio
    for ratio in [4, 8, 16]:
        resnet = rd_df[(rd_df['name'] == f'resnet_{ratio}x')]
        baseline = rd_df[(rd_df['name'] == f'baseline_{ratio}x')]
        codec = rd_df[(rd_df['name'] == f'jpeg2000_{ratio}x')]

        if not codec.empty:
            codec_psnr = codec['psnr'].values[0]
            codec_ssim = codec['ssim'].values[0]

            best_ae = None
            best_ae_psnr = -999

            if not resnet.empty:
                if resnet['psnr'].values[0] > best_ae_psnr:
                    best_ae = ('ResNet', resnet['psnr'].values[0], resnet['ssim'].values[0])
                    best_ae_psnr = resnet['psnr'].values[0]

            if not baseline.empty:
                if baseline['psnr'].values[0] > best_ae_psnr:
                    best_ae = ('Baseline', baseline['psnr'].values[0], baseline['ssim'].values[0])
                    best_ae_psnr = baseline['psnr'].values[0]

            if best_ae:
                psnr_diff = best_ae[1] - codec_psnr
                direction = "better" if psnr_diff > 0 else "worse"
                summary_lines.append(
                    f"- **{ratio}x compression:** {best_ae[0]} achieves {best_ae[1]:.2f} dB PSNR "
                    f"vs JPEG-2000's {codec_psnr:.2f} dB ({abs(psnr_diff):.2f} dB {direction})"
                )

    # Statistical significance summary
    sig_results = []
    if not stat_df.empty:
        sig = stat_df[stat_df['significant_bonferroni'] == True]
        for _, row in sig.iterrows():
            sig_results.append(
                f"- {row['ae_model']} vs {row['codec']} ({row['metric'].upper()}): "
                f"p = {row['p_value']:.2e}, diff = {row['difference']:.3f}"
            )

    # Generate report content
    report = f"""# SAR Image Compression: Autoencoder vs JPEG-2000 Comparison Study

## Executive Summary

This study compares CNN-based autoencoder compression against JPEG-2000 for Sentinel-1 SAR imagery at 4x, 8x, and 16x compression ratios.

### Key Findings

{chr(10).join(summary_lines) if summary_lines else "No comparison data available."}

### Statistically Significant Results (Bonferroni-corrected)

{chr(10).join(sig_results) if sig_results else "No statistically significant differences found, or insufficient data for testing."}

---

## Rate-Distortion Analysis

### PSNR vs Bits Per Pixel

![PSNR Rate-Distortion Curve](figures/rate_distortion_psnr.png)

### SSIM vs Bits Per Pixel

![SSIM Rate-Distortion Curve](figures/rate_distortion_ssim.png)

---

## Summary Table

| Model | Compression | BPP | PSNR (dB) | SSIM |
|-------|-------------|-----|-----------|------|
"""

    for _, row in rd_df.sort_values(['ratio', 'model_type']).iterrows():
        report += f"| {row['name']} | {row['ratio']:.0f}x | {row['bpp']:.2f} | {row['psnr']:.2f} +/- {row['psnr_std']:.2f} | {row['ssim']:.3f} +/- {row['ssim_std']:.3f} |\n"

    report += """
---

## Visual Comparisons

### 4x Compression

![Visual Comparison at 4x](figures/visual_comparison_4x.png)

### 8x Compression

![Visual Comparison at 8x](figures/visual_comparison_8x.png)

### 16x Compression

![Visual Comparison at 16x](figures/visual_comparison_16x.png)

---

## Statistical Test Details

See `tables/statistical_tests.csv` for full statistical analysis including:
- Test type (paired t-test or Wilcoxon signed-rank)
- Raw p-values and Bonferroni-corrected significance
- Per-metric comparisons at each compression ratio

---

## Conclusions

"""

    # Add conclusions based on data
    if not rd_df.empty:
        # Find overall best model
        ae_only = rd_df[rd_df['model_type'] != 'JPEG-2000']
        if not ae_only.empty:
            best_psnr_row = ae_only.loc[ae_only['psnr'].idxmax()]
            report += f"1. **Best overall autoencoder performance:** {best_psnr_row['name']} at {best_psnr_row['ratio']:.0f}x compression achieves {best_psnr_row['psnr']:.2f} dB PSNR\n\n"

        # Compare to JPEG-2000 at 16x
        ae_16x = rd_df[(rd_df['ratio'] == 16) & (rd_df['model_type'] != 'JPEG-2000')]
        jp2_16x = rd_df[(rd_df['name'] == 'jpeg2000_16x')]
        if not ae_16x.empty and not jp2_16x.empty:
            best_ae_16x = ae_16x.loc[ae_16x['psnr'].idxmax()]
            jp2_psnr = jp2_16x['psnr'].values[0]
            if best_ae_16x['psnr'] > jp2_psnr:
                report += f"2. **At 16x compression:** Autoencoders outperform JPEG-2000 by {best_ae_16x['psnr'] - jp2_psnr:.2f} dB PSNR\n\n"
            else:
                report += f"2. **At 16x compression:** JPEG-2000 outperforms autoencoders by {jp2_psnr - best_ae_16x['psnr']:.2f} dB PSNR\n\n"

    report += """3. **Recommendations:** See statistical test results to determine which method is recommended for specific use cases based on statistical significance.

---

*Report generated automatically from evaluation data.*
"""

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate final comparison report')
    parser.add_argument('--data-dir', type=str, default='reports',
                        help='Directory containing evaluation data')
    parser.add_argument('--skip-visuals', action='store_true',
                        help='Skip generating visual comparisons (faster)')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("GENERATING FINAL COMPARISON REPORT")
    print("=" * 70)

    # Check required files exist
    required_files = [
        data_dir / 'data' / 'all_results.json',
        data_dir / 'data' / 'per_sample_metrics.json',
        data_dir / 'tables' / 'results_summary.csv',
    ]
    for f in required_files:
        if not f.exists():
            print(f"ERROR: Required file not found: {f}")
            print("Run run_evaluation_sweep.py first")
            sys.exit(1)

    # Load data
    print("\nLoading evaluation data...")
    all_results, per_sample, summary_df = load_data(data_dir)
    print(f"  Loaded {len(all_results)} model results")

    # Prepare R-D data
    rd_df = prepare_rd_data(all_results)
    print(f"  Prepared R-D data for {len(rd_df)} configurations")

    # Generate rate-distortion curves
    print("\nGenerating rate-distortion curves...")
    plot_rate_distortion_psnr(rd_df, data_dir / 'figures' / 'rate_distortion_psnr.png')
    plot_rate_distortion_ssim(rd_df, data_dir / 'figures' / 'rate_distortion_ssim.png')

    # Run statistical tests
    print("\nRunning statistical tests...")
    stat_df = run_statistical_tests(per_sample, data_dir / 'tables' / 'statistical_tests.csv')

    # Generate visual comparisons (optional, slow)
    if not args.skip_visuals:
        print("\nGenerating visual comparisons...")
        generate_visual_comparisons(data_dir)
    else:
        print("\nSkipping visual comparisons (--skip-visuals)")

    # Generate markdown report
    print("\nGenerating markdown report...")
    generate_markdown_report(data_dir, rd_df, stat_df)

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - {data_dir / 'figures' / 'rate_distortion_psnr.png'}")
    print(f"  - {data_dir / 'figures' / 'rate_distortion_ssim.png'}")
    print(f"  - {data_dir / 'tables' / 'statistical_tests.csv'}")
    print(f"  - {data_dir / 'figures' / 'visual_comparison_*.png'}")
    print(f"  - {data_dir / 'final_comparison.md'}")


if __name__ == '__main__':
    main()
