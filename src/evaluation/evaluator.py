"""
Comprehensive Evaluation Framework for SAR Autoencoder

Implements multiple metrics and analysis tools for evaluating
autoencoder reconstruction quality on SAR images.

Key Features:
- Unified metric computation via compute_all_metrics
- ENL ratio tracking for speckle preservation
- Structured JSON output for reproducible experiments
- Rate-distortion data collection for codec comparison

References:
    - Day 3, Section 3.3 of the learning guide
"""

from datetime import datetime
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from .metrics import (
    SARMetrics, compute_all_metrics, compute_ms_ssim,
    enl_ratio, compute_bpp, compute_compression_ratio
)


class Evaluator:
    """
    Comprehensive evaluation pipeline for SAR autoencoders.

    Uses compute_all_metrics from metrics.py for consistent metric
    computation across autoencoder and codec evaluation.

    Attributes:
        model: Trained autoencoder model
        device: Device for inference
        model_name: Model identifier for output files
        checkpoint_path: Path to model checkpoint
        preprocessing_params: Dict with vmin/vmax for reproducibility

    Example:
        >>> model = ResNetAutoencoder(latent_channels=16)
        >>> model.load_state_dict(torch.load('best.pth')['model_state_dict'])
        >>> evaluator = Evaluator(model, device='cuda')
        >>> evaluator.model_name = 'resnet_lite_v2_c16'
        >>> results = evaluator.evaluate_dataset(test_loader)
        >>> evaluator.save_results(results, 'evaluations', 'resnet_lite_v2_c16')
    """

    def __init__(
        self,
        model,
        device: str = 'cuda',
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        preprocessing_params: Optional[Dict] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained autoencoder with encode() and forward() methods
            device: Device for inference ('cuda' or 'cpu')
            model_name: Model identifier for output files
            checkpoint_path: Path to checkpoint (for metadata)
            preprocessing_params: Dict with 'vmin' and 'vmax' for data preprocessing
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name or 'unnamed_model'
        self.checkpoint_path = checkpoint_path
        self.preprocessing_params = preprocessing_params or {}

    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: torch.Tensor,
        return_per_sample: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate a batch of images using compute_all_metrics.

        Uses the unified compute_all_metrics function for consistency
        with codec evaluation.

        Args:
            batch: Input tensor of shape (B, 1, H, W)
            return_per_sample: If True, also return per-sample metrics

        Returns:
            Tuple of (reconstructions, metrics_dict)
            metrics_dict contains averaged metrics with optional per_sample list
        """
        x = batch.to(self.device)
        x_hat, z = self.model(x)

        # Move to numpy for metric computation
        x_np = x.cpu().numpy()
        x_hat_np = x_hat.cpu().numpy()

        # Collect metrics per sample using compute_all_metrics
        batch_size = len(batch)
        per_sample_metrics = []

        # Aggregation containers for scalar metrics
        aggregated = {
            'mse': [], 'psnr': [], 'ssim': [], 'ms_ssim': [],
            'mae': [], 'epi': [],
            'enl_ratio': [], 'enl_original': [], 'enl_reconstructed': [],
            'hist_intersection': [], 'hist_bhattacharyya': [],
            'variance_ratio': [], 'variance_correlation': [],
            'pearson': [], 'spearman': []
        }

        for i in range(batch_size):
            orig = x_np[i, 0]
            recon = x_hat_np[i, 0]

            # Single call to compute_all_metrics for consistency
            try:
                sample_metrics = compute_all_metrics(orig, recon)
            except Exception as e:
                # Handle edge cases (NaN, small images, etc.)
                print(f"Warning: Metrics computation failed for sample {i}: {e}")
                continue

            # Extract scalar values from nested dicts
            sample_flat = {
                'mse': sample_metrics.get('mse', np.nan),
                'psnr': sample_metrics.get('psnr', np.nan),
                'ssim': sample_metrics.get('ssim', np.nan),
                'ms_ssim': sample_metrics.get('ms_ssim', np.nan),
                'mae': sample_metrics.get('mae', np.nan),
                'epi': sample_metrics.get('epi', np.nan),
            }

            # ENL ratio (nested dict)
            enl_data = sample_metrics.get('enl_ratio', {})
            sample_flat['enl_ratio'] = enl_data.get('enl_ratio', np.nan)
            sample_flat['enl_original'] = enl_data.get('enl_original', np.nan)
            sample_flat['enl_reconstructed'] = enl_data.get('enl_reconstructed', np.nan)

            # Histogram (nested dict)
            hist_data = sample_metrics.get('histogram', {})
            sample_flat['hist_intersection'] = hist_data.get('intersection', np.nan)
            sample_flat['hist_bhattacharyya'] = hist_data.get('bhattacharyya', np.nan)

            # Local variance (nested dict)
            var_data = sample_metrics.get('local_variance', {})
            sample_flat['variance_ratio'] = var_data.get('variance_ratio', np.nan)
            sample_flat['variance_correlation'] = var_data.get('variance_correlation', np.nan)

            # Correlation (nested dict)
            corr_data = sample_metrics.get('correlation', {})
            sample_flat['pearson'] = corr_data.get('pearson', np.nan)
            sample_flat['spearman'] = corr_data.get('spearman', np.nan)

            per_sample_metrics.append(sample_flat)

            # Aggregate (skip NaN values)
            for key in aggregated:
                val = sample_flat.get(key, np.nan)
                if not np.isnan(val):
                    aggregated[key].append(val)

        # Compute mean for each metric
        avg_metrics = {}
        for key, values in aggregated.items():
            if values:
                avg_metrics[key] = float(np.mean(values))
            else:
                avg_metrics[key] = np.nan

        if return_per_sample:
            avg_metrics['per_sample'] = per_sample_metrics

        return x_hat, avg_metrics

    def evaluate_dataset(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Evaluate entire dataset with comprehensive statistics.

        Returns both summary statistics (mean/std/min/max) and
        per-sample detailed results for deeper analysis.

        Args:
            dataloader: PyTorch DataLoader yielding batches
            max_batches: Optional limit on number of batches to process
            show_progress: Whether to show progress bar

        Returns:
            Dict with:
            - model_name: Model identifier
            - checkpoint_path: Path to checkpoint
            - evaluation_date: ISO timestamp
            - num_samples: Total samples evaluated
            - preprocessing_params: vmin/vmax for reproducibility
            - metrics: Dict with mean/std/min/max for each metric
            - per_sample: List of per-sample metric dicts
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc="Evaluating") if show_progress else dataloader
        except ImportError:
            iterator = dataloader

        all_sample_metrics = []
        batch_count = 0

        for batch in iterator:
            if max_batches and batch_count >= max_batches:
                break

            try:
                _, batch_metrics = self.evaluate_batch(batch, return_per_sample=True)
                per_sample = batch_metrics.pop('per_sample', [])
                all_sample_metrics.extend(per_sample)
            except Exception as e:
                print(f"Warning: Batch evaluation failed: {e}")
                continue

            batch_count += 1

        # Aggregate all sample metrics into statistics
        metric_keys = [
            'mse', 'psnr', 'ssim', 'ms_ssim', 'mae', 'epi',
            'enl_ratio', 'enl_original', 'enl_reconstructed',
            'hist_intersection', 'hist_bhattacharyya',
            'variance_ratio', 'variance_correlation',
            'pearson', 'spearman'
        ]

        metrics_stats = {}
        for key in metric_keys:
            values = [s.get(key, np.nan) for s in all_sample_metrics]
            values = [v for v in values if not np.isnan(v)]

            if values:
                metrics_stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                metrics_stats[key] = {
                    'mean': np.nan, 'std': np.nan,
                    'min': np.nan, 'max': np.nan, 'count': 0
                }

        # Get compression ratio from model if available
        compression_ratio = None
        if hasattr(self.model, 'get_compression_ratio'):
            compression_ratio = self.model.get_compression_ratio()

        results = {
            'model_name': self.model_name,
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'evaluation_date': datetime.now().isoformat(),
            'num_samples': len(all_sample_metrics),
            'preprocessing_params': self.preprocessing_params,
            'compression_ratio': compression_ratio,
            'metrics': metrics_stats,
            'per_sample': all_sample_metrics
        }

        return results

    def save_summary(self, results: Dict, output_path: str) -> None:
        """
        Save summary metrics to JSON.

        Creates a compact summary file with model metadata and
        aggregated statistics (excludes per-sample data).

        Args:
            results: Results dict from evaluate_dataset
            output_path: Path to output JSON file
        """
        # Create summary without per-sample data
        summary = {
            'model_name': results.get('model_name'),
            'checkpoint_path': results.get('checkpoint_path'),
            'evaluation_date': results.get('evaluation_date'),
            'num_samples': results.get('num_samples'),
            'preprocessing_params': results.get('preprocessing_params'),
            'compression_ratio': results.get('compression_ratio'),
            'metrics': results.get('metrics', {})
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_serializer)

        print(f"Summary saved to: {output_path}")

    def save_detailed(self, results: Dict, output_path: str) -> None:
        """
        Save per-sample detailed metrics to JSON.

        Creates a detailed file with metrics for each sample,
        useful for distribution analysis and outlier detection.

        Args:
            results: Results dict from evaluate_dataset
            output_path: Path to output JSON file
        """
        detailed = {
            'model_name': results.get('model_name'),
            'evaluation_date': results.get('evaluation_date'),
            'num_samples': results.get('num_samples'),
            'samples': []
        }

        for i, sample_metrics in enumerate(results.get('per_sample', [])):
            detailed['samples'].append({
                'index': i,
                **sample_metrics
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(detailed, f, indent=2, default=_json_serializer)

        print(f"Detailed results saved to: {output_path}")

    def save_results(
        self,
        results: Dict,
        output_dir: str,
        model_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save both summary and detailed results to evaluations directory.

        Creates the standard output structure:
        - {output_dir}/{model_name}_eval.json (summary)
        - {output_dir}/{model_name}_detailed.json (per-sample)

        Args:
            results: Results dict from evaluate_dataset
            output_dir: Output directory path
            model_name: Model name for filenames (uses self.model_name if None)

        Returns:
            Dict with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = model_name or self.model_name

        summary_path = output_dir / f"{name}_eval.json"
        detailed_path = output_dir / f"{name}_detailed.json"

        self.save_summary(results, str(summary_path))
        self.save_detailed(results, str(detailed_path))

        return {
            'summary': str(summary_path),
            'detailed': str(detailed_path)
        }

    def collect_rd_point(
        self,
        results: Dict,
        compression_ratio: Optional[float] = None
    ) -> Dict:
        """
        Format evaluation results as rate-distortion data point.

        Creates a dict suitable for rate-distortion curve plotting,
        with standardized keys matching codec evaluation output.

        Args:
            results: Results dict from evaluate_dataset
            compression_ratio: Override compression ratio (uses model's if None)

        Returns:
            Dict with name, compression_ratio, bpp, psnr, ssim, ms_ssim
        """
        metrics = results.get('metrics', {})

        # Get compression ratio
        if compression_ratio is None:
            compression_ratio = results.get('compression_ratio')
            if compression_ratio is None and hasattr(self.model, 'get_compression_ratio'):
                compression_ratio = self.model.get_compression_ratio()

        # Compute BPP from compression ratio
        # For 32-bit float input: bpp = 32 / compression_ratio
        bpp = 32.0 / compression_ratio if compression_ratio else None

        return {
            'name': self.model_name,
            'compression_ratio': compression_ratio,
            'bpp': bpp,
            'psnr': metrics.get('psnr', {}).get('mean'),
            'ssim': metrics.get('ssim', {}).get('mean'),
            'ms_ssim': metrics.get('ms_ssim', {}).get('mean'),
            'epi': metrics.get('epi', {}).get('mean'),
            'enl_ratio': metrics.get('enl_ratio', {}).get('mean'),
        }

    @torch.no_grad()
    def analyze_latent_space(self, dataloader, n_batches: int = 10) -> Dict:
        """
        Analyze latent space statistics.

        Args:
            dataloader: DataLoader for input data
            n_batches: Number of batches to analyze

        Returns:
            Dict with latent space statistics
        """
        all_latents = []

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            x = batch.to(self.device)
            z = self.model.encode(x)
            all_latents.append(z.cpu().numpy())

        latents = np.concatenate(all_latents, axis=0)

        # Per-channel statistics
        channel_means = np.mean(latents, axis=(0, 2, 3))
        channel_stds = np.std(latents, axis=(0, 2, 3))

        # Sparsity (fraction near zero)
        sparsity = np.mean(np.abs(latents) < 0.1)

        # Utilization (channels with significant variance)
        active_channels = np.sum(channel_stds > 0.1)

        return {
            'global_mean': float(np.mean(latents)),
            'global_std': float(np.std(latents)),
            'min': float(np.min(latents)),
            'max': float(np.max(latents)),
            'sparsity': float(sparsity),
            'active_channels': int(active_channels),
            'total_channels': len(channel_stds),
            'channel_std_mean': float(np.mean(channel_stds)),
            'channel_std_std': float(np.std(channel_stds)),
        }

    @torch.no_grad()
    def find_failure_cases(self, dataloader, n_worst: int = 10) -> List[Dict]:
        """
        Find samples with worst reconstruction quality.

        Args:
            dataloader: DataLoader for input data
            n_worst: Number of worst samples to return

        Returns:
            List of dicts with original, reconstructed, latent, mse
        """
        all_samples = []

        for batch in dataloader:
            x = batch.to(self.device)
            x_hat, z = self.model(x)

            # Compute per-sample MSE
            mse_per_sample = ((x - x_hat) ** 2).mean(dim=(1, 2, 3))

            for i in range(len(batch)):
                all_samples.append({
                    'original': x[i].cpu(),
                    'reconstructed': x_hat[i].cpu(),
                    'latent': z[i].cpu(),
                    'mse': mse_per_sample[i].item(),
                })

        # Sort by MSE (worst first)
        all_samples.sort(key=lambda x: x['mse'], reverse=True)

        return all_samples[:n_worst]

    @torch.no_grad()
    def find_best_cases(self, dataloader, n_best: int = 10) -> List[Dict]:
        """
        Find samples with best reconstruction quality.

        Args:
            dataloader: DataLoader for input data
            n_best: Number of best samples to return

        Returns:
            List of dicts with original, reconstructed, latent, mse
        """
        all_samples = []

        for batch in dataloader:
            x = batch.to(self.device)
            x_hat, z = self.model(x)

            mse_per_sample = ((x - x_hat) ** 2).mean(dim=(1, 2, 3))

            for i in range(len(batch)):
                all_samples.append({
                    'original': x[i].cpu(),
                    'reconstructed': x_hat[i].cpu(),
                    'latent': z[i].cpu(),
                    'mse': mse_per_sample[i].item(),
                })

        # Sort by MSE (best first)
        all_samples.sort(key=lambda x: x['mse'])

        return all_samples[:n_best]

    @torch.no_grad()
    def get_reconstructions(
        self,
        dataloader,
        n_samples: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Get sample reconstructions with metrics for visualization.

        Args:
            dataloader: DataLoader for input data
            n_samples: Number of samples to return

        Returns:
            List of (original, reconstructed, metrics) tuples
        """
        samples = []

        for batch in dataloader:
            x = batch.to(self.device)
            x_hat, z = self.model(x)

            x_np = x.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()

            for i in range(len(batch)):
                if len(samples) >= n_samples:
                    break

                orig = x_np[i, 0]
                recon = x_hat_np[i, 0]
                metrics = compute_all_metrics(orig, recon)

                samples.append((orig, recon, metrics))

            if len(samples) >= n_samples:
                break

        return samples


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def print_evaluation_report(results: Dict) -> None:
    """
    Print formatted evaluation results.

    Args:
        results: Results dict from Evaluator.evaluate_dataset()
    """
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"\nModel: {results.get('model_name', 'Unknown')}")
    print(f"Samples evaluated: {results.get('num_samples', 0)}")
    print(f"Compression ratio: {results.get('compression_ratio', 'N/A')}")
    if results.get('evaluation_date'):
        print(f"Date: {results['evaluation_date']}")

    metrics = results.get('metrics', {})

    print("\nReconstruction Quality Metrics:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)

    # Primary metrics
    for metric in ['psnr', 'ssim', 'ms_ssim', 'mse', 'mae', 'epi']:
        if metric in metrics:
            m = metrics[metric]
            mean_val = m.get('mean', np.nan)
            std_val = m.get('std', np.nan)
            min_val = m.get('min', np.nan)
            max_val = m.get('max', np.nan)

            # Format values, handling NaN
            mean_str = f"{mean_val:.4f}" if not np.isnan(mean_val) else "N/A"
            std_str = f"{std_val:.4f}" if not np.isnan(std_val) else "N/A"
            min_str = f"{min_val:.4f}" if not np.isnan(min_val) else "N/A"
            max_str = f"{max_val:.4f}" if not np.isnan(max_val) else "N/A"

            print(f"{metric.upper():<20} {mean_str:>10} {std_str:>10} "
                  f"{min_str:>10} {max_str:>10}")

    # SAR-specific metrics
    print("\nSAR-Specific Metrics:")
    print("-" * 60)

    for metric in ['enl_ratio', 'enl_original', 'enl_reconstructed',
                   'hist_intersection', 'hist_bhattacharyya',
                   'variance_ratio', 'variance_correlation']:
        if metric in metrics:
            m = metrics[metric]
            mean_val = m.get('mean', np.nan)

            if not np.isnan(mean_val):
                print(f"  {metric}: {mean_val:.4f}")

    print("=" * 70)


def test_evaluator():
    """Test evaluation framework."""
    print("=" * 60)
    print("EVALUATOR TEST")
    print("=" * 60)

    # Create dummy model and data
    from torch.utils.data import TensorDataset, DataLoader

    # Simple passthrough "model" for testing
    class DummyModel(torch.nn.Module):
        def __init__(self, noise_level=0.05):
            super().__init__()
            self.noise_level = noise_level
            self.encoder = torch.nn.Conv2d(1, 16, 1)
            self.latent_channels = 16

        def forward(self, x):
            z = self.encoder(x)
            x_hat = x + self.noise_level * torch.randn_like(x)
            x_hat = x_hat.clamp(0, 1)
            return x_hat, z

        def encode(self, x):
            return self.encoder(x)

        def get_compression_ratio(self):
            return 16.0

    model = DummyModel(noise_level=0.05)

    # Create test data (256x256 for MS-SSIM)
    test_data = torch.rand(20, 1, 256, 256)
    dataset = TensorDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=4)

    # Wrap dataloader to return just the tensor (not tuple)
    class SimpleLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                yield batch[0]
        def __len__(self):
            return len(self.loader)

    simple_loader = SimpleLoader(dataloader)

    # Create evaluator
    evaluator = Evaluator(
        model,
        device='cpu',
        model_name='test_model',
        preprocessing_params={'vmin': 14.7688, 'vmax': 24.5407}
    )

    # Test evaluate_dataset
    results = evaluator.evaluate_dataset(simple_loader, show_progress=False)
    print_evaluation_report(results)

    # Test JSON output
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = evaluator.save_results(results, tmpdir, 'test_model')
        print(f"\nSaved files: {paths}")

        # Verify JSON is valid
        with open(paths['summary']) as f:
            summary = json.load(f)
            print(f"Summary has {len(summary['metrics'])} metrics")

        with open(paths['detailed']) as f:
            detailed = json.load(f)
            print(f"Detailed has {len(detailed['samples'])} samples")

    # Test R-D point collection
    rd_point = evaluator.collect_rd_point(results)
    print(f"\nR-D point: {rd_point}")

    # Test latent analysis
    latent_stats = evaluator.analyze_latent_space(simple_loader, n_batches=3)
    print("\nLatent Space Analysis:")
    for key, value in latent_stats.items():
        print(f"  {key}: {value}")

    print("\nEvaluator test passed!")


if __name__ == "__main__":
    test_evaluator()
