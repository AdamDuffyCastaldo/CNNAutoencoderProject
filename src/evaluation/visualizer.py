"""
Visualization Utilities for SAR Autoencoder

Comprehensive visualization toolkit for analyzing SAR image
reconstruction quality, including:
- Side-by-side comparison with zoomed crops
- Diverging colormap difference maps
- Rate-distortion curves
- Histogram overlays
- ENL mask visualization

References:
    - Day 3, Section 3.3 of the learning guide
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches


class Visualizer:
    """
    Visualization toolkit for SAR autoencoder analysis.

    Provides publication-quality visualizations for:
    - Reconstruction comparisons with zoomed crops
    - Rate-distortion curves
    - Histogram comparisons
    - ENL and texture analysis

    Attributes:
        save_dir: Directory to save visualizations
        cmap: Default colormap for grayscale images
        diff_cmap: Colormap for difference maps (diverging)
        dpi: Resolution for saved figures
    """

    def __init__(self, save_dir: str = 'visualizations', dpi: int = 150):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Style settings
        plt.style.use('default')
        self.cmap = 'gray'
        self.diff_cmap = 'RdBu_r'  # Red-blue for differences (centered at 0)
        self.dpi = dpi

    def plot_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        metrics: Dict,
        save_path: Optional[str] = None,
        zoom_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        auto_zoom: bool = True,
        n_zoom_regions: int = 2,
        zoom_size: int = 64,
        show: bool = True
    ) -> plt.Figure:
        """
        Comprehensive comparison visualization with zoomed crops.

        Creates a multi-row figure:
        - Row 1: Original, Reconstructed, Difference (diverging colormap)
        - Row 2: Zoomed crops from regions of interest

        Args:
            original: Original image (2D numpy array)
            reconstructed: Reconstructed image (2D numpy array)
            metrics: Dict with at least 'psnr' and 'ssim' keys
            save_path: Filename to save (relative to save_dir)
            zoom_regions: List of (y0, y1, x0, x1) tuples for manual regions
            auto_zoom: If True and no zoom_regions, auto-select high-error regions
            n_zoom_regions: Number of auto-zoom regions to find
            zoom_size: Size of zoom region (square)
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        # Compute difference
        diff = original - reconstructed
        max_abs_diff = max(abs(diff.min()), abs(diff.max()), 0.01)

        # Determine zoom regions
        if zoom_regions is None and auto_zoom:
            zoom_regions = self._find_interesting_regions(
                diff, n_regions=n_zoom_regions, region_size=zoom_size
            )

        # Create figure
        has_zoom = zoom_regions is not None and len(zoom_regions) > 0
        n_rows = 2 if has_zoom else 1
        n_zoom = len(zoom_regions) if has_zoom else 0

        if has_zoom:
            # Row 1: 3 main panels, Row 2: n_zoom*3 zoom panels (orig, recon, diff for each)
            fig = plt.figure(figsize=(12, 4 * n_rows))
            gs = GridSpec(n_rows, 3 + n_zoom, height_ratios=[1] * n_rows if n_rows > 1 else [1])

            # Main panels in row 1
            ax_orig = fig.add_subplot(gs[0, 0])
            ax_recon = fig.add_subplot(gs[0, 1])
            ax_diff = fig.add_subplot(gs[0, 2])

        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            ax_orig, ax_recon, ax_diff = axes

        # Plot original
        ax_orig.imshow(original, cmap=self.cmap, vmin=0, vmax=1)
        ax_orig.set_title('Original', fontsize=12)
        ax_orig.axis('off')

        # Plot reconstructed
        ax_recon.imshow(reconstructed, cmap=self.cmap, vmin=0, vmax=1)
        ax_recon.set_title('Reconstructed', fontsize=12)
        ax_recon.axis('off')

        # Plot difference with diverging colormap
        im_diff = ax_diff.imshow(diff, cmap=self.diff_cmap,
                                  vmin=-max_abs_diff, vmax=max_abs_diff)
        ax_diff.set_title('Difference', fontsize=12)
        ax_diff.axis('off')

        # Add colorbar for difference
        cbar = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar.set_label('Difference\n(blue=under, red=over)', fontsize=8)

        # Draw zoom region rectangles on main images
        if has_zoom:
            colors = plt.cm.Set1(np.linspace(0, 1, n_zoom))

            for i, (y0, y1, x0, x1) in enumerate(zoom_regions):
                color = colors[i]
                # Add rectangle to original
                rect_orig = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax_orig.add_patch(rect_orig)

                # Add rectangle to reconstructed
                rect_recon = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax_recon.add_patch(rect_recon)

            # Row 2: Zoomed regions
            for i, (y0, y1, x0, x1) in enumerate(zoom_regions):
                # Zoomed original
                ax_zoom_orig = fig.add_subplot(gs[1, i * 3 // n_zoom] if n_zoom <= 3
                                               else fig.add_subplot(2, n_zoom * 2, n_zoom * 2 + i * 2 + 1))
                crop_orig = original[y0:y1, x0:x1]
                ax_zoom_orig.imshow(crop_orig, cmap=self.cmap, vmin=0, vmax=1)
                ax_zoom_orig.set_title(f'Zoom {i+1}: Orig', fontsize=10)
                ax_zoom_orig.axis('off')
                # Add colored border
                for spine in ax_zoom_orig.spines.values():
                    spine.set_edgecolor(colors[i])
                    spine.set_linewidth(2)
                    spine.set_visible(True)

                # For simplicity, create a new figure layout for zooms
                # This is cleaner approach:

            # Recreate figure with better layout for zooms
            plt.close(fig)
            fig = self._create_comparison_with_zooms(
                original, reconstructed, diff, max_abs_diff,
                zoom_regions, metrics
            )

        # Add metrics to title
        psnr = metrics.get('psnr', np.nan)
        ssim = metrics.get('ssim', np.nan)
        epi = metrics.get('epi', np.nan)

        title_parts = []
        if not np.isnan(psnr):
            title_parts.append(f'PSNR: {psnr:.2f} dB')
        if not np.isnan(ssim):
            title_parts.append(f'SSIM: {ssim:.4f}')
        if not np.isnan(epi):
            title_parts.append(f'EPI: {epi:.4f}')

        if title_parts and not has_zoom:  # Title already set in zooms version
            fig.suptitle(' | '.join(title_parts), fontsize=12, y=1.02)

        plt.tight_layout()

        if save_path:
            full_path = self.save_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {full_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _create_comparison_with_zooms(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        diff: np.ndarray,
        max_abs_diff: float,
        zoom_regions: List[Tuple[int, int, int, int]],
        metrics: Dict
    ) -> plt.Figure:
        """Create comparison figure with proper zoom layout."""
        n_zoom = len(zoom_regions)

        # Create figure: row 1 = main images, row 2 = zoomed triplets
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, max(3, n_zoom * 3), height_ratios=[1, 0.8],
                      hspace=0.3, wspace=0.1)

        # Row 1: Main images
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_recon = fig.add_subplot(gs[0, 1])
        ax_diff = fig.add_subplot(gs[0, 2])

        colors = plt.cm.Set1(np.linspace(0, 1, max(n_zoom, 1)))

        # Main original
        ax_orig.imshow(original, cmap=self.cmap, vmin=0, vmax=1)
        ax_orig.set_title('Original', fontsize=12)
        ax_orig.axis('off')

        # Main reconstructed
        ax_recon.imshow(reconstructed, cmap=self.cmap, vmin=0, vmax=1)
        ax_recon.set_title('Reconstructed', fontsize=12)
        ax_recon.axis('off')

        # Main difference
        im_diff = ax_diff.imshow(diff, cmap=self.diff_cmap,
                                  vmin=-max_abs_diff, vmax=max_abs_diff)
        ax_diff.set_title('Difference', fontsize=12)
        ax_diff.axis('off')
        cbar = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar.set_label('blue=under, red=over', fontsize=8)

        # Draw rectangles on main images
        for i, (y0, y1, x0, x1) in enumerate(zoom_regions):
            for ax in [ax_orig, ax_recon]:
                rect = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor=colors[i], facecolor='none'
                )
                ax.add_patch(rect)

        # Row 2: Zoomed crops (original, reconstructed, difference for each region)
        for i, (y0, y1, x0, x1) in enumerate(zoom_regions):
            # Column indices for this zoom
            col_base = i * 3

            # Zoomed original
            ax_z_orig = fig.add_subplot(gs[1, col_base])
            crop_orig = original[y0:y1, x0:x1]
            ax_z_orig.imshow(crop_orig, cmap=self.cmap, vmin=0, vmax=1)
            ax_z_orig.set_title(f'Zoom {i+1}', fontsize=10)
            ax_z_orig.axis('off')
            for spine in ax_z_orig.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(3)
                spine.set_visible(True)

            # Zoomed reconstructed
            ax_z_recon = fig.add_subplot(gs[1, col_base + 1])
            crop_recon = reconstructed[y0:y1, x0:x1]
            ax_z_recon.imshow(crop_recon, cmap=self.cmap, vmin=0, vmax=1)
            ax_z_recon.set_title('Recon', fontsize=10)
            ax_z_recon.axis('off')
            for spine in ax_z_recon.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(3)
                spine.set_visible(True)

            # Zoomed difference
            ax_z_diff = fig.add_subplot(gs[1, col_base + 2])
            crop_diff = diff[y0:y1, x0:x1]
            ax_z_diff.imshow(crop_diff, cmap=self.diff_cmap,
                            vmin=-max_abs_diff, vmax=max_abs_diff)
            ax_z_diff.set_title('Diff', fontsize=10)
            ax_z_diff.axis('off')
            for spine in ax_z_diff.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Title with metrics
        psnr = metrics.get('psnr', np.nan)
        ssim = metrics.get('ssim', np.nan)
        epi = metrics.get('epi', np.nan)

        title_parts = []
        if not np.isnan(psnr):
            title_parts.append(f'PSNR: {psnr:.2f} dB')
        if not np.isnan(ssim):
            title_parts.append(f'SSIM: {ssim:.4f}')
        if not np.isnan(epi):
            title_parts.append(f'EPI: {epi:.4f}')

        if title_parts:
            fig.suptitle(' | '.join(title_parts), fontsize=14, y=0.98)

        return fig

    def _find_interesting_regions(
        self,
        diff: np.ndarray,
        n_regions: int = 2,
        region_size: int = 64,
        grid_step: int = 32
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find regions with highest absolute error for auto-zoom.

        Args:
            diff: Difference image (original - reconstructed)
            n_regions: Number of regions to find
            region_size: Size of each region (square)
            grid_step: Step size for grid search

        Returns:
            List of (y0, y1, x0, x1) tuples defining region bounds
        """
        h, w = diff.shape
        abs_diff = np.abs(diff)

        # Slide over image and compute mean absolute error in each cell
        regions = []
        half = region_size // 2

        for y in range(half, h - half, grid_step):
            for x in range(half, w - half, grid_step):
                y0 = max(0, y - half)
                y1 = min(h, y + half)
                x0 = max(0, x - half)
                x1 = min(w, x + half)

                mean_error = np.mean(abs_diff[y0:y1, x0:x1])
                regions.append((mean_error, y0, y1, x0, x1))

        # Sort by error (highest first)
        regions.sort(key=lambda x: x[0], reverse=True)

        # Take top n regions, ensuring they don't overlap too much
        selected = []
        for error, y0, y1, x0, x1 in regions:
            # Check overlap with already selected
            overlap = False
            for _, sy0, sy1, sx0, sx1 in selected:
                # Simple overlap check
                if not (x1 <= sx0 or x0 >= sx1 or y1 <= sy0 or y0 >= sy1):
                    # Check overlap ratio
                    inter_x = max(0, min(x1, sx1) - max(x0, sx0))
                    inter_y = max(0, min(y1, sy1) - max(y0, sy0))
                    inter_area = inter_x * inter_y
                    region_area = (x1 - x0) * (y1 - y0)
                    if inter_area / region_area > 0.3:
                        overlap = True
                        break

            if not overlap:
                selected.append((error, y0, y1, x0, x1))
                if len(selected) >= n_regions:
                    break

        return [(y0, y1, x0, x1) for _, y0, y1, x0, x1 in selected]

    def plot_rate_distortion(
        self,
        results: List[Dict],
        output_path: Optional[str] = None,
        title: str = "Rate-Distortion Comparison",
        show: bool = True
    ) -> plt.Figure:
        """
        Plot PSNR vs BPP and SSIM vs BPP curves.

        Creates a two-panel figure comparing multiple models/codecs
        in rate-distortion space.

        Args:
            results: List of dicts with keys: name, bpp, psnr, ssim
            output_path: Filename to save (relative to save_dir)
            title: Figure title
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        # Group results by name
        grouped = {}
        for r in results:
            name = r.get('name', 'Unknown')
            if name not in grouped:
                grouped[name] = {'bpp': [], 'psnr': [], 'ssim': []}
            if r.get('bpp') is not None:
                grouped[name]['bpp'].append(r['bpp'])
                grouped[name]['psnr'].append(r.get('psnr', np.nan))
                grouped[name]['ssim'].append(r.get('ssim', np.nan))

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Markers and colors for different methods
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
        colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))

        # Plot PSNR vs BPP
        ax1 = axes[0]
        for i, (name, data) in enumerate(grouped.items()):
            bpp = np.array(data['bpp'])
            psnr = np.array(data['psnr'])

            # Sort by BPP for line plot
            sort_idx = np.argsort(bpp)
            bpp_sorted = bpp[sort_idx]
            psnr_sorted = psnr[sort_idx]

            ax1.plot(bpp_sorted, psnr_sorted,
                    marker=markers[i % len(markers)],
                    color=colors[i],
                    label=name,
                    linewidth=2, markersize=8)

        ax1.set_xlabel('Bits Per Pixel (BPP)', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('PSNR vs BPP', fontsize=12)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Plot SSIM vs BPP
        ax2 = axes[1]
        for i, (name, data) in enumerate(grouped.items()):
            bpp = np.array(data['bpp'])
            ssim = np.array(data['ssim'])

            sort_idx = np.argsort(bpp)
            bpp_sorted = bpp[sort_idx]
            ssim_sorted = ssim[sort_idx]

            ax2.plot(bpp_sorted, ssim_sorted,
                    marker=markers[i % len(markers)],
                    color=colors[i],
                    label=name,
                    linewidth=2, markersize=8)

        ax2.set_xlabel('Bits Per Pixel (BPP)', fontsize=12)
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.set_title('SSIM vs BPP', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if output_path:
            full_path = self.save_dir / output_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {full_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_histogram_overlay(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        bins: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot overlaid histograms for intensity distribution comparison.

        Args:
            original: Original image
            reconstructed: Reconstructed image
            bins: Number of histogram bins
            save_path: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute histograms
        hist_orig, bin_edges = np.histogram(original.flatten(), bins=bins,
                                             range=(0, 1), density=True)
        hist_recon, _ = np.histogram(reconstructed.flatten(), bins=bins,
                                      range=(0, 1), density=True)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot with transparency
        ax.fill_between(bin_centers, hist_orig, alpha=0.5,
                       label='Original', color='blue')
        ax.fill_between(bin_centers, hist_recon, alpha=0.5,
                       label='Reconstructed', color='orange')

        # Compute histogram intersection for title
        intersection = np.sum(np.minimum(hist_orig, hist_recon)) / np.sum(hist_orig)

        ax.set_xlabel('Intensity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Intensity Distribution (Intersection: {intersection:.4f})',
                    fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.save_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {full_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def save_enl_mask(
        self,
        image: np.ndarray,
        save_path: str,
        window_size: int = 15,
        cv_threshold: float = 0.3,
        show: bool = False
    ) -> plt.Figure:
        """
        Save homogeneous region mask as PNG for inspection.

        Visualizes the regions used for ENL computation.

        Args:
            image: Input image
            save_path: Filename to save
            window_size: Window size for CV computation
            cv_threshold: Threshold for homogeneous classification
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        from scipy.ndimage import uniform_filter

        # Compute CV
        image = np.asarray(image, dtype=np.float64)
        local_mean = uniform_filter(image, size=window_size, mode='reflect')
        local_sq_mean = uniform_filter(image ** 2, size=window_size, mode='reflect')
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.maximum(local_var, 0)
        local_std = np.sqrt(local_var)
        cv = local_std / (local_mean + 1e-10)

        mask = cv < cv_threshold

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # CV map
        im_cv = axes[1].imshow(cv, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Coefficient of Variation', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im_cv, ax=axes[1], fraction=0.046)

        # Homogeneous mask
        axes[2].imshow(mask, cmap='gray')
        homog_frac = mask.mean() * 100
        axes[2].set_title(f'Homogeneous Mask ({homog_frac:.1f}%)', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            full_path = self.save_dir / save_path
            fig.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {full_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_reconstruction_grid(
        self,
        originals: List[np.ndarray],
        reconstructions: List[np.ndarray],
        titles: Optional[List[str]] = None,
        n_cols: int = 4,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot grid of original, reconstructed, and difference images.

        Args:
            originals: List of original images
            reconstructions: List of reconstructed images
            titles: Optional titles for each column
            n_cols: Number of image columns
            save_name: Filename to save (None = don't save)
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        n_images = min(len(originals), n_cols)

        fig, axes = plt.subplots(3, n_images, figsize=(3 * n_images, 9))

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
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_failure_analysis(
        self,
        failures: List[Dict],
        n_show: int = 8,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Analyze and visualize failure cases.

        Args:
            failures: List of dicts with 'original', 'reconstructed', 'mse'
            n_show: Number of failures to show
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        n_show = min(n_show, len(failures))

        fig = plt.figure(figsize=(16, 4 * ((n_show + 3) // 4)))

        for i in range(n_show):
            sample = failures[i]
            orig = sample['original'].numpy().squeeze()
            recon = sample['reconstructed'].numpy().squeeze()
            diff = np.abs(orig - recon)

            ax1 = fig.add_subplot(((n_show + 3) // 4), 4, i + 1)
            im = ax1.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            ax1.set_title(f'MSE: {sample["mse"]:.4f}', fontsize=10)
            ax1.axis('off')

        plt.suptitle('Failure Cases - Absolute Difference Maps', fontsize=14)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_latent_analysis(
        self,
        latent: np.ndarray,
        n_channels: int = 16,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize latent representation.

        Args:
            latent: Latent tensor of shape (C, H, W)
            n_channels: Number of channels to display
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        n_channels = min(n_channels, latent.shape[0])
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = axes.flatten()

        for i in range(n_channels):
            channel = latent[i]
            vmax = max(abs(channel.min()), abs(channel.max()))

            im = axes[i].imshow(channel, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i].set_title(f'Ch {i}: u={channel.mean():.2f}, s={channel.std():.2f}',
                             fontsize=8)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        # Hide unused axes
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Latent Channel Activations', fontsize=14)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_histogram_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        bins: int = 100,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Compare intensity histograms of original and reconstructed images.

        Args:
            original: Original image
            reconstructed: Reconstructed image
            bins: Number of histogram bins
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
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
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_metrics_over_dataset(
        self,
        metrics_list: List[Dict],
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot metric distributions across dataset.

        Args:
            metrics_list: List of metric dicts per sample
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        metrics_to_plot = ['psnr', 'ssim', 'epi']

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 4))

        for i, metric in enumerate(metrics_to_plot):
            if metric in metrics_list[0]:
                values = [m[metric] for m in metrics_list if metric in m]

                axes[i].hist(values, bins=30, density=True, alpha=0.7)
                axes[i].axvline(np.mean(values), color='r', linestyle='--',
                               label=f'Mean: {np.mean(values):.3f}')
                axes[i].set_xlabel(metric.upper())
                axes[i].set_ylabel('Density')
                axes[i].set_title(f'{metric.upper()} Distribution')
                axes[i].legend()

        plt.tight_layout()

        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_training_curves(
        self,
        history: List[Dict],
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot training history curves.

        Args:
            history: List of dicts with train/val metrics per epoch
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
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
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_enl_comparison(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        window_size: int = 32,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Compare ENL maps between original and reconstructed.

        Args:
            original: Original image
            reconstructed: Reconstructed image
            window_size: Window size for ENL computation
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        from scipy.ndimage import uniform_filter

        def compute_enl_map(image):
            image = np.asarray(image, dtype=np.float64)
            local_mean = uniform_filter(image, size=window_size, mode='reflect')
            local_sq_mean = uniform_filter(image ** 2, size=window_size, mode='reflect')
            local_var = local_sq_mean - local_mean ** 2
            local_var = np.maximum(local_var, 1e-10)
            return (local_mean ** 2) / local_var

        enl_orig = compute_enl_map(original)
        enl_recon = compute_enl_map(reconstructed)

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
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def test_visualizer():
    """Test visualization functions."""
    import tempfile
    import shutil

    print("Testing Visualizer...")

    # Create temp directory
    tmpdir = tempfile.mkdtemp()

    try:
        viz = Visualizer(save_dir=tmpdir)

        # Test data
        np.random.seed(42)
        orig = np.random.rand(256, 256).astype(np.float32)
        recon = np.clip(orig + 0.05 * np.random.randn(256, 256), 0, 1).astype(np.float32)

        # Test comprehensive comparison
        metrics = {'psnr': 26.5, 'ssim': 0.92, 'epi': 0.95}
        fig = viz.plot_comparison(orig, recon, metrics, 'test_comparison.png',
                                  auto_zoom=True, show=False)
        print(f"Comparison figure created with {len(fig.axes)} axes")

        # Test rate-distortion plot
        rd_data = [
            {'name': 'Autoencoder', 'bpp': 0.5, 'psnr': 28, 'ssim': 0.92},
            {'name': 'Autoencoder', 'bpp': 0.25, 'psnr': 25, 'ssim': 0.85},
            {'name': 'JPEG-2000', 'bpp': 0.5, 'psnr': 26, 'ssim': 0.88},
            {'name': 'JPEG-2000', 'bpp': 0.25, 'psnr': 22, 'ssim': 0.78},
        ]
        viz.plot_rate_distortion(rd_data, 'test_rd.png', show=False)
        print("Rate-distortion plot created")

        # Test histogram overlay
        viz.plot_histogram_overlay(orig, recon, save_path='test_hist.png', show=False)
        print("Histogram overlay created")

        # Test ENL mask
        viz.save_enl_mask(orig, 'test_enl_mask.png', show=False)
        print("ENL mask saved")

        print("All visualizer tests passed!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_visualizer()
