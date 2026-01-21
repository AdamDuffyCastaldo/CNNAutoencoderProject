"""
Comprehensive Evaluation Framework for SAR Autoencoder

Implements multiple metrics and analysis tools.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class SARMetrics:
    """
    Collection of metrics for SAR image quality assessment.
    """
    
    @staticmethod
    def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((x - x_hat) ** 2))
    
    @staticmethod
    def psnr(x: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio in dB."""
        mse = np.mean((x - x_hat) ** 2)
        if mse == 0:
            return float('inf')
        return float(10 * np.log10(data_range ** 2 / mse))
    
    @staticmethod
    def ssim(x: np.ndarray, x_hat: np.ndarray, data_range: float = 1.0) -> float:
        """Structural Similarity Index."""
        return float(skimage_ssim(x, x_hat, data_range=data_range))
    
    @staticmethod
    def enl(image: np.ndarray, window_size: int = 32) -> np.ndarray:
        """
        Equivalent Number of Looks.
        
        ENL = μ² / σ² computed in local windows.
        Higher values indicate more smoothing.
        """
        from scipy.ndimage import uniform_filter
        
        # Local mean
        local_mean = uniform_filter(image, size=window_size)
        
        # Local variance
        local_sq_mean = uniform_filter(image ** 2, size=window_size)
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 1e-10)  # Avoid division by zero
        
        # ENL
        enl = local_mean ** 2 / local_var
        
        return enl
    
    @staticmethod
    def edge_preservation_index(x: np.ndarray, x_hat: np.ndarray) -> float:
        """
        Edge Preservation Index.
        
        Ratio of gradient magnitudes: EPI = Σ|∇x̂| / Σ|∇x|
        """
        # Compute gradients using Sobel
        def gradient_magnitude(img):
            gx = ndimage.sobel(img, axis=1)
            gy = ndimage.sobel(img, axis=0)
            return np.sqrt(gx**2 + gy**2)
        
        grad_x = gradient_magnitude(x)
        grad_x_hat = gradient_magnitude(x_hat)
        
        return float(np.sum(grad_x_hat) / (np.sum(grad_x) + 1e-10))
    
    @staticmethod
    def mean_absolute_error(x: np.ndarray, x_hat: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(x - x_hat)))
    
    @staticmethod
    def correlation(x: np.ndarray, x_hat: np.ndarray) -> Dict[str, float]:
        """Pearson and Spearman correlation coefficients."""
        x_flat = x.flatten()
        x_hat_flat = x_hat.flatten()
        
        pearson_r, pearson_p = pearsonr(x_flat, x_hat_flat)
        spearman_r, spearman_p = spearmanr(x_flat, x_hat_flat)
        
        return {
            'pearson': float(pearson_r),
            'spearman': float(spearman_r),
        }
    
    @staticmethod
    def histogram_similarity(x: np.ndarray, x_hat: np.ndarray, 
                            bins: int = 256) -> float:
        """
        Histogram intersection similarity.
        
        Measures how well the intensity distribution is preserved.
        """
        hist_x, _ = np.histogram(x.flatten(), bins=bins, range=(0, 1), density=True)
        hist_x_hat, _ = np.histogram(x_hat.flatten(), bins=bins, range=(0, 1), density=True)
        
        # Intersection
        intersection = np.minimum(hist_x, hist_x_hat).sum()
        
        # Normalize
        return float(intersection / hist_x.sum())
    
    @staticmethod
    def local_variance_ratio(x: np.ndarray, x_hat: np.ndarray, 
                            window_size: int = 16) -> Dict[str, float]:
        """
        Compare local variance statistics.
        
        Helps detect over-smoothing.
        """
        from scipy.ndimage import uniform_filter
        
        def local_variance(img):
            local_mean = uniform_filter(img, size=window_size)
            local_sq_mean = uniform_filter(img ** 2, size=window_size)
            return local_sq_mean - local_mean ** 2
        
        var_x = local_variance(x)
        var_x_hat = local_variance(x_hat)
        
        # Ratio of mean variances
        mean_ratio = np.mean(var_x_hat) / (np.mean(var_x) + 1e-10)
        
        # Correlation of variance maps
        corr, _ = pearsonr(var_x.flatten(), var_x_hat.flatten())
        
        return {
            'variance_ratio': float(mean_ratio),
            'variance_correlation': float(corr),
        }


class Evaluator:
    """
    Comprehensive evaluation pipeline.
    """
    
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
        self.metrics = SARMetrics()
    
    @torch.no_grad()
    def evaluate_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate a batch of images.
        
        Returns:
            reconstructions: Tensor of reconstructed images
            metrics: Dict of averaged metrics
        """
        x = batch.to(self.device)
        x_hat, z = self.model(x)
        
        # Move to numpy
        x_np = x.cpu().numpy()
        x_hat_np = x_hat.cpu().numpy()
        
        # Compute metrics for each image
        batch_metrics = {
            'mse': [], 'psnr': [], 'ssim': [], 'epi': [],
            'mae': [], 'hist_sim': [],
        }
        
        for i in range(len(batch)):
            orig = x_np[i, 0]
            recon = x_hat_np[i, 0]
            
            batch_metrics['mse'].append(self.metrics.mse(orig, recon))
            batch_metrics['psnr'].append(self.metrics.psnr(orig, recon))
            batch_metrics['ssim'].append(self.metrics.ssim(orig, recon))
            batch_metrics['epi'].append(self.metrics.edge_preservation_index(orig, recon))
            batch_metrics['mae'].append(self.metrics.mean_absolute_error(orig, recon))
            batch_metrics['hist_sim'].append(self.metrics.histogram_similarity(orig, recon))
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in batch_metrics.items()}
        
        return x_hat, avg_metrics
    
    def evaluate_dataset(self, dataloader) -> Dict:
        """
        Evaluate entire dataset.
        
        Returns:
            Comprehensive metrics dictionary with mean, std, min, max
        """
        all_metrics = {
            'mse': [], 'psnr': [], 'ssim': [], 'epi': [],
            'mae': [], 'hist_sim': [],
        }
        
        print("Evaluating dataset...")
        for batch in dataloader:
            _, batch_metrics = self.evaluate_batch(batch)
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
        
        # Compute statistics
        results = {}
        for key, values in all_metrics.items():
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return results
    
    @torch.no_grad()
    def analyze_latent_space(self, dataloader, n_batches: int = 10) -> Dict:
        """
        Analyze latent space statistics.
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


def print_evaluation_report(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    
    print("\nReconstruction Quality Metrics:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    
    for metric in ['psnr', 'ssim', 'mse', 'mae', 'epi', 'hist_sim']:
        if metric in results:
            r = results[metric]
            print(f"{metric.upper():<15} {r['mean']:>10.4f} {r['std']:>10.4f} "
                  f"{r['min']:>10.4f} {r['max']:>10.4f}")
    
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
            self.encoder = torch.nn.Conv2d(1, 64, 1)
            
        def forward(self, x):
            z = self.encoder(x)
            x_hat = x + self.noise_level * torch.randn_like(x)
            x_hat = x_hat.clamp(0, 1)
            return x_hat, z
        
        def encode(self, x):
            return self.encoder(x)
    
    model = DummyModel(noise_level=0.05)
    
    # Create test data
    test_data = torch.rand(20, 1, 64, 64)
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
    evaluator = Evaluator(model, device='cpu')
    
    # Test evaluate_dataset
    results = evaluator.evaluate_dataset(simple_loader)
    print_evaluation_report(results)
    
    # Test latent analysis
    latent_stats = evaluator.analyze_latent_space(simple_loader, n_batches=3)
    print("\nLatent Space Analysis:")
    for key, value in latent_stats.items():
        print(f"  {key}: {value}")
    
    # Test failure case finding
    failures = evaluator.find_failure_cases(simple_loader, n_worst=3)
    print(f"\nFound {len(failures)} worst cases")
    print(f"Worst MSE: {failures[0]['mse']:.6f}")
    
    print("\n✓ Evaluator test passed!")


if __name__ == "__main__":
    test_evaluator()