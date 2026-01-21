"""
Visualization Utilities for SAR Autoencoder

References:
    - Day 3, Section 3.3 of the learning guide
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple


class Visualizer:
    """
    Visualization toolkit for SAR autoencoder analysis.
    """
    
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Style settings
        plt.style.use('default')
        self.cmap = 'gray'
        self.diff_cmap = 'RdBu_r'  # Red-blue for differences
        
    def plot_reconstruction_grid(self, 
                                 originals: List[np.ndarray],
                                 reconstructions: List[np.ndarray],
                                 titles: Optional[List[str]] = None,
                                 n_cols: int = 4,
                                 save_name: Optional[str] = None,
                                 show: bool = True) -> plt.Figure:
        """
        Plot grid of original, reconstructed, and difference images.
        
        Args:
            originals: List of original images
            reconstructions: List of reconstructed images
            titles: Optional titles for each column
            n_cols: Number of image columns
            save_name: Filename to save (None = don't save)
            show: Whether to display the figure
        """
        n_images = min(len(originals), n_cols)
        
        fig, axes = plt.subplots(3, n_images, figsize=(3*n_images, 9))
        
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_images):
            orig = originals[i]
            recon = reconstructions[i]
            diff = orig - recon
            
            # Original
            axes[0, i].imshow(orig, cmap=self.cmap, vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=12)
            
            # Reconstructed
            axes[1, i].imshow(recon, cmap=self.cmap, vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=12)
            
            # Difference
            max_diff = max(abs(diff.min()), abs(diff.max()), 0.1)
            axes[2, i].imshow(diff, cmap=self.diff_cmap, 
                            vmin=-max_diff, vmax=max_diff)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Difference', fontsize=12)
            
            # Title with PSNR
            mse = np.mean((orig - recon) ** 2)
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            title = titles[i] if titles else f'PSNR: {psnr:.1f} dB'
            axes[0, i].set_title(title, fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_failure_analysis(self,
                              failures: List[Dict],
                              n_show: int = 8,
                              save_name: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        """
        Analyze and visualize failure cases.
        """
        n_show = min(n_show, len(failures))
        
        fig = plt.figure(figsize=(16, 4 * ((n_show + 3) // 4)))
        
        for i in range(n_show):
            sample = failures[i]
            orig = sample['original'].numpy().squeeze()
            recon = sample['reconstructed'].numpy().squeeze()
            diff = np.abs(orig - recon)
            
            # Create subplot
            ax1 = fig.add_subplot(((n_show + 3) // 4), 4, i + 1)
            
            # Show difference map (highlights problem areas)
            im = ax1.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            ax1.set_title(f'MSE: {sample["mse"]:.4f}', fontsize=10)
            ax1.axis('off')
        
        plt.suptitle('Failure Cases - Absolute Difference Maps', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_latent_analysis(self,
                            latent: np.ndarray,
                            n_channels: int = 16,
                            save_name: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
        """
        Visualize latent representation.
        
        Args:
            latent: Latent tensor of shape (C, H, W)
            n_channels: Number of channels to display
        """
        n_channels = min(n_channels, latent.shape[0])
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        axes = axes.flatten()
        
        for i in range(n_channels):
            channel = latent[i]
            vmax = max(abs(channel.min()), abs(channel.max()))
            
            im = axes[i].imshow(channel, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i].set_title(f'Ch {i}: μ={channel.mean():.2f}, σ={channel.std():.2f}',
                            fontsize=8)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        # Hide unused axes
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Latent Channel Activations', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_histogram_comparison(self,
                                  original: np.ndarray,
                                  reconstructed: np.ndarray,
                                  bins: int = 100,
                                  save_name: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Compare intensity histograms of original and reconstructed images.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original histogram
        axes[0].hist(original.flatten(), bins=bins, density=True, 
                    alpha=0.7, color='blue', label='Original')
        axes[0].set_xlabel('Intensity')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Original Distribution')
        axes[0].legend()
        
        # Reconstructed histogram
        axes[1].hist(reconstructed.flatten(), bins=bins, density=True,
                    alpha=0.7, color='orange', label='Reconstructed')
        axes[1].set_xlabel('Intensity')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Reconstructed Distribution')
        axes[1].legend()
        
        # Overlay comparison
        axes[2].hist(original.flatten(), bins=bins, density=True,
                    alpha=0.5, color='blue', label='Original')
        axes[2].hist(reconstructed.flatten(), bins=bins, density=True,
                    alpha=0.5, color='orange', label='Reconstructed')
        axes[2].set_xlabel('Intensity')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Distribution Comparison')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_metrics_over_dataset(self,
                                  metrics_list: List[Dict],
                                  save_name: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Plot metric distributions across dataset.
        """
        metrics_to_plot = ['psnr', 'ssim', 'epi']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 4))
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in metrics_list[0]:
                values = [m[metric] for m in metrics_list]
                
                axes[i].hist(values, bins=30, density=True, alpha=0.7)
                axes[i].axvline(np.mean(values), color='r', linestyle='--',
                               label=f'Mean: {np.mean(values):.3f}')
                axes[i].set_xlabel(metric.upper())
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'{metric.upper()} Distribution')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_training_curves(self,
                            history: List[Dict],
                            save_name: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
        """
        Plot training history curves.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(history) + 1)
        
        # Loss
        if 'train_loss' in history[0]:
            axes[0, 0].plot(epochs, [h['train_loss'] for h in history], 
                          'b-', label='Train')
        if 'val_loss' in history[0]:
            axes[0, 0].plot(epochs, [h['val_loss'] for h in history],
                          'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PSNR
        if 'train_psnr' in history[0]:
            axes[0, 1].plot(epochs, [h['train_psnr'] for h in history],
                          'b-', label='Train')
        if 'val_psnr' in history[0]:
            axes[0, 1].plot(epochs, [h['val_psnr'] for h in history],
                          'r-', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('PSNR Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # SSIM
        if 'train_ssim' in history[0]:
            axes[1, 0].plot(epochs, [h['train_ssim'] for h in history],
                          'b-', label='Train')
        if 'val_ssim' in history[0]:
            axes[1, 0].plot(epochs, [h['val_ssim'] for h in history],
                          'r-', label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('SSIM Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if 'learning_rate' in history[0]:
            axes[1, 1].plot(epochs, [h['learning_rate'] for h in history], 'g-')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_enl_comparison(self,
                           original: np.ndarray,
                           reconstructed: np.ndarray,
                           window_size: int = 32,
                           save_name: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
 
        enl_orig = SARMetrics.enl(original, window_size)
        enl_recon = SARMetrics.enl(reconstructed, window_size)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Original ENL
        im1 = axes[1].imshow(enl_orig, cmap='viridis', vmin=0, vmax=20)
        axes[1].set_title(f'Original ENL (mean: {np.mean(enl_orig):.1f})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Reconstructed image
        axes[2].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Reconstructed')
        axes[2].axis('off')
        
        # Reconstructed ENL
        im2 = axes[3].imshow(enl_recon, cmap='viridis', vmin=0, vmax=20)
        axes[3].set_title(f'Recon ENL (mean: {np.mean(enl_recon):.1f})')
        axes[3].axis('off')
        plt.colorbar(im2, ax=axes[3], fraction=0.046)
        
        plt.suptitle('Equivalent Number of Looks Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
