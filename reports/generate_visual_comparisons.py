"""Generate visual comparison gallery for the final report."""

import sys
sys.path.insert(0, '..')

import json
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')

# Publication-quality figure settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def main():
    # Ensure output directory exists
    Path('figures').mkdir(exist_ok=True)

    # Try to import models
    try:
        from src.data.datamodule import SARDataModule
        from src.models.autoencoder import SARAutoencoder
        from src.models.resnet_autoencoder import ResNetAutoencoder
        from src.evaluation.codec_baselines import JPEG2000Codec
        print("Models imported successfully")
    except ImportError as e:
        print(f"Could not import models: {e}")
        print("Generating placeholder visual comparisons...")
        generate_placeholder_visuals()
        return

    # Load test data
    print("Loading test data...")
    datamodule = SARDataModule(
        patches_path='../data/patches/metadata.npy',
        batch_size=16, val_fraction=0.1, num_workers=0, lazy=True
    )

    # Get validation loader
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

    print(f"Loaded {len(sample_images)} sample images")

    # Setup device and codec
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codec = JPEG2000Codec()

    def load_best_ae_for_ratio(ratio: int):
        """Load best autoencoder (ResNet if available, else Baseline) for given ratio."""
        lc_map = {4: 64, 8: 32, 16: 16}
        lc = lc_map[ratio]

        # Try ResNet first
        resnet_pattern = f'../notebooks/checkpoints/resnet_c{lc}_b64_cr{ratio}x_*'
        matches = sorted(glob.glob(resnet_pattern))
        if matches:
            ckpt_path = matches[-1] + '/best.pth'
            if Path(ckpt_path).exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                # ResNetAutoencoder(in_channels, base_channels, latent_channels)
                model = ResNetAutoencoder(in_channels=1, base_channels=64, latent_channels=lc)
                model.load_state_dict(ckpt['model_state_dict'])
                return model.to(device).eval(), 'ResNet'

        # Fallback to Baseline
        baseline_pattern = f'../notebooks/checkpoints/baseline_c{lc}_b64_cr{ratio}x_*'
        matches = sorted(glob.glob(baseline_pattern))
        if matches:
            ckpt_path = matches[-1] + '/best.pth'
            if Path(ckpt_path).exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                # SARAutoencoder(latent_channels, base_channels) - base_channels=64 by default
                model = SARAutoencoder(latent_channels=lc, base_channels=64)
                model.load_state_dict(ckpt['model_state_dict'])
                return model.to(device).eval(), 'Baseline'

        return None, None

    # Generate visual comparisons for each ratio
    for ratio in [4, 8, 16]:
        print(f"\nGenerating visual comparison for {ratio}x compression...")

        model, model_type = load_best_ae_for_ratio(ratio)
        if model is None:
            print(f"  No model found for {ratio}x, skipping")
            continue

        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        fig.suptitle(f'{ratio}x Compression: Original | {model_type} | JPEG-2000 | {model_type} Error\n(Note: JPEG-2000 on 8-bit data, Autoencoder on float32)', fontsize=14)

        for row, img in enumerate(sample_images):
            # Original
            axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row, 0].set_title('Original' if row == 0 else '')
            axes[row, 0].axis('off')

            # Autoencoder reconstruction
            with torch.no_grad():
                img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
                output = model(img_t)
                # Handle tuple output (output, latent) from ResNet
                if isinstance(output, tuple):
                    output = output[0]
                ae_recon = output.cpu().numpy()[0, 0]
            axes[row, 1].imshow(ae_recon, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title(f'{model_type}' if row == 0 else '')
            axes[row, 1].axis('off')

            # JPEG-2000 reconstruction
            # Calibrate quality for target ratio
            quality = codec.calibrate_quality(ratio, img)
            encoded = codec.encode(img, quality)
            jp2_recon = codec.decode(encoded)
            axes[row, 2].imshow(jp2_recon, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].set_title('JPEG-2000' if row == 0 else '')
            axes[row, 2].axis('off')

            # Autoencoder error map
            ae_error = np.abs(ae_recon - img)
            max_err = max(ae_error.max(), 0.1)
            axes[row, 3].imshow(ae_error, cmap='hot', vmin=0, vmax=max_err)
            axes[row, 3].set_title(f'{model_type} Error' if row == 0 else '')
            axes[row, 3].axis('off')

        plt.tight_layout()
        save_path = f'figures/visual_comparison_{ratio}x.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        del model
        torch.cuda.empty_cache()

    print("\nVisual comparison generation complete!")


def generate_placeholder_visuals():
    """Generate placeholder images when models are not available."""
    for ratio in [4, 8, 16]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Visual Comparison for {ratio}x\n\n(Models not available for generation)\n\nSee rate_distortion_*.png for quantitative results',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')

        save_path = f'figures/visual_comparison_{ratio}x.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved placeholder: {save_path}")


if __name__ == '__main__':
    main()
