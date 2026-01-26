"""
Production Inference Pipeline for SAR Autoencoder

Handles full image compression/decompression with:
- Intelligent tiling with overlap
- Seamless patch blending
- Preprocessing/postprocessing
- File I/O

References:
    - Day 6, Section 6.1 of the learning guide
"""

import torch
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import json

from .tiling import create_cosine_ramp_weights, extract_tiles, reconstruct_from_tiles


class SARCompressor:
    """
    Complete compression/decompression pipeline for SAR images.

    Handles full-size images through intelligent tiling with
    overlapping patches and smooth blending.

    Args:
        model_path: Path to saved model checkpoint
        device: Device to use (None = auto-detect)
        patch_size: Size of patches for processing
        overlap: Overlap between patches for seamless reconstruction
        batch_size: Batch size for inference (None = auto-detect)

    Example:
        >>> compressor = SARCompressor('checkpoints/best.pth')
        >>>
        >>> # Compress
        >>> latent, metadata = compressor.compress(raw_image)
        >>>
        >>> # Decompress
        >>> reconstructed = compressor.decompress(latent, metadata)
        >>>
        >>> # Get stats
        >>> stats = compressor.get_compression_stats(raw_image, latent)
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        patch_size: int = 256,
        overlap: int = 64,
        batch_size: Optional[int] = None
    ):
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Store tiling parameters
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

        # Load model and preprocessing params
        self._load_model(model_path)

        # Set batch size (auto-detect if not provided)
        if batch_size is None:
            self.batch_size = self._auto_detect_batch_size()
        else:
            self.batch_size = batch_size

        # Create blend weights for reconstruction
        self.blend_weights = create_cosine_ramp_weights(patch_size, overlap)

    def _load_model(self, model_path: str) -> None:
        """
        Load model and preprocessing parameters from checkpoint.

        Args:
            model_path: Path to saved checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract config
        self.config = checkpoint.get('config', {})

        # Extract preprocessing params (may be in config or at top level)
        if 'preprocessing_params' in self.config:
            self.preprocess_params = self.config['preprocessing_params']
        elif 'preprocess_params' in checkpoint:
            self.preprocess_params = checkpoint['preprocess_params']
        else:
            # Fallback to default values
            self.preprocess_params = {
                'vmin': 14.7688,
                'vmax': 24.5407
            }

        # Ensure vmin/vmax are floats (may be numpy scalars)
        self.preprocess_params['vmin'] = float(self.preprocess_params['vmin'])
        self.preprocess_params['vmax'] = float(self.preprocess_params['vmax'])

        # Get model parameters
        latent_channels = self.config.get('latent_channels', 16)
        base_channels = self.config.get('base_channels', 32)

        # Instantiate model (ResNetAutoencoder is the default for our checkpoints)
        from src.models.resnet_autoencoder import ResNetAutoencoder
        self.model = ResNetAutoencoder(
            in_channels=1,
            base_channels=base_channels,
            latent_channels=latent_channels
        )

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

    def _auto_detect_batch_size(self) -> int:
        """
        Auto-detect optimal batch size based on available GPU memory.

        Returns:
            Recommended batch size
        """
        if self.device.type != 'cuda':
            return 1

        try:
            # Get GPU properties
            props = torch.cuda.get_device_properties(self.device)
            total_vram_mb = props.total_memory / (1024 * 1024)

            # Estimate ~3MB per 256x256 tile (conservative estimate)
            # This accounts for input, latent, output, and intermediate activations
            mem_per_tile_mb = 3.0

            # Use 70% of total VRAM for inference
            usable_vram = total_vram_mb * 0.7

            # Calculate batch size
            batch_size = int(usable_vram / mem_per_tile_mb)

            # Clamp to reasonable range
            batch_size = max(1, min(64, batch_size))

            return batch_size
        except Exception:
            # Fallback to safe default
            return 8

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess raw SAR image for model input.

        Steps:
        1. Handle invalid values (<=0, NaN, Inf)
        2. Convert to dB: 10 * log10(image)
        3. Clip to [vmin, vmax]
        4. Normalize to [0, 1]

        Args:
            image: Raw SAR intensity image (linear scale)

        Returns:
            Preprocessed image normalized to [0, 1] range
        """
        vmin = self.preprocess_params['vmin']
        vmax = self.preprocess_params['vmax']

        # Handle invalid values with noise floor
        noise_floor = 1e-10
        image_clean = np.where(
            (image > 0) & np.isfinite(image),
            image,
            noise_floor
        )

        # Convert to dB
        image_db = 10.0 * np.log10(image_clean)

        # Clip to range
        image_clipped = np.clip(image_db, vmin, vmax)

        # Normalize to [0, 1]
        normalized = (image_clipped - vmin) / (vmax - vmin)

        return normalized.astype(np.float32)

    def inverse_preprocess(self, normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized image back to linear SAR values.

        Args:
            normalized: Normalized image in [0, 1] range

        Returns:
            Linear SAR intensity values
        """
        vmin = self.preprocess_params['vmin']
        vmax = self.preprocess_params['vmax']

        # Denormalize to dB
        image_db = normalized * (vmax - vmin) + vmin

        # Convert to linear
        image_linear = np.power(10.0, image_db / 10.0)

        return image_linear.astype(np.float32)

    def _process_tiles_batched(
        self,
        tiles: np.ndarray,
        encode: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Process tiles through encoder or decoder in batches.

        Args:
            tiles: Numpy array of tiles (N, H, W) for encode or (N, C, H, W) for decode
            encode: If True, encode tiles; if False, decode tiles
            progress_callback: Optional callback(current_batch, total_batches)

        Returns:
            Processed tiles as numpy array
        """
        n_tiles = tiles.shape[0]
        results = []

        n_batches = (n_tiles + self.batch_size - 1) // self.batch_size

        with torch.inference_mode():
            # Use AMP on CUDA for faster inference
            use_amp = self.device.type == 'cuda'

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_tiles)

                # Get batch
                batch = tiles[start:end]

                # Prepare tensor
                if encode:
                    # Add channel dimension for encoding: (N, H, W) -> (N, 1, H, W)
                    batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(self.device)
                else:
                    # Already has channels for decoding: (N, C, H, W)
                    batch_tensor = torch.from_numpy(batch).to(self.device)

                # Forward pass with optional AMP
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        if encode:
                            output = self.model.encode(batch_tensor)
                        else:
                            output = self.model.decode(batch_tensor)
                else:
                    if encode:
                        output = self.model.encode(batch_tensor)
                    else:
                        output = self.model.decode(batch_tensor)

                # Convert to numpy
                output_np = output.cpu().numpy()

                # For decode, remove channel dimension: (N, 1, H, W) -> (N, H, W)
                if not encode:
                    output_np = output_np.squeeze(1)

                results.append(output_np)

                # Progress callback
                if progress_callback is not None:
                    progress_callback(batch_idx + 1, n_batches)

        return np.concatenate(results, axis=0)

    def compress(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compress SAR image to latent representation.

        Args:
            image: Raw SAR intensity image (linear scale, 2D array)
            progress_callback: Optional callback(current_batch, total_batches)

        Returns:
            Tuple of:
            - latent_patches: Compressed latent patches (N, C, 16, 16)
            - metadata: Information needed for decompression
        """
        # Preprocess
        normalized = self.preprocess(image)

        # Extract tiles
        tiles, tile_metadata = extract_tiles(
            normalized,
            tile_size=self.patch_size,
            overlap=self.overlap
        )

        # Process through encoder
        latent_patches = self._process_tiles_batched(
            tiles,
            encode=True,
            progress_callback=progress_callback
        )

        # Build metadata for decompression
        metadata = {
            **tile_metadata,
            'preprocess_params': self.preprocess_params,
            'latent_channels': self.config.get('latent_channels', 16)
        }

        return latent_patches, metadata

    def decompress(
        self,
        latent_patches: np.ndarray,
        metadata: Dict,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Decompress latent representation back to SAR image.

        Args:
            latent_patches: Compressed latent patches (N, C, 16, 16)
            metadata: Decompression metadata from compress()
            progress_callback: Optional callback(current_batch, total_batches)

        Returns:
            Reconstructed SAR image (linear intensity)
        """
        # Process through decoder
        decoded_tiles = self._process_tiles_batched(
            latent_patches,
            encode=False,
            progress_callback=progress_callback
        )

        # Reconstruct full image with blending
        reconstructed_normalized = reconstruct_from_tiles(
            decoded_tiles,
            metadata,
            self.blend_weights
        )

        # Inverse preprocess to get linear SAR values
        reconstructed = self.inverse_preprocess(reconstructed_normalized)

        return reconstructed

    def get_compression_stats(
        self,
        image: np.ndarray,
        latent_patches: np.ndarray
    ) -> Dict:
        """
        Calculate compression statistics.

        Args:
            image: Original image
            latent_patches: Compressed latent patches

        Returns:
            Dictionary with compression statistics
        """
        h, w = image.shape
        input_elements = h * w

        # Latent is (N, C, 16, 16)
        latent_elements = latent_patches.size

        compression_ratio = input_elements / latent_elements

        # Bits per pixel assuming 8-bit quantized latents
        bpp = (latent_elements * 8) / input_elements

        return {
            'compression_ratio': float(compression_ratio),
            'bpp': float(bpp),
            'original_shape': image.shape,
            'latent_shape': latent_patches.shape,
            'n_tiles': latent_patches.shape[0],
            'latent_channels': latent_patches.shape[1] if latent_patches.ndim > 1 else 1
        }

    def save_compressed(
        self,
        latent: np.ndarray,
        metadata: Dict,
        output_path: str
    ) -> None:
        """
        Save compressed representation to file.

        Args:
            latent: Compressed latent patches
            metadata: Compression metadata
            output_path: Path to save compressed file
        """
        # Convert metadata to JSON-serializable format
        metadata_json = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                metadata_json[k] = float(v)
            elif isinstance(v, tuple):
                metadata_json[k] = list(v)
            else:
                metadata_json[k] = v

        np.savez_compressed(
            output_path,
            latent=latent,
            metadata=json.dumps(metadata_json)
        )

    def load_compressed(self, input_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load compressed representation from file.

        Args:
            input_path: Path to compressed file

        Returns:
            Tuple of (latent_patches, metadata)
        """
        data = np.load(input_path, allow_pickle=True)
        latent = data['latent']
        metadata = json.loads(str(data['metadata']))

        # Convert lists back to tuples where needed
        if 'grid_shape' in metadata:
            metadata['grid_shape'] = tuple(metadata['grid_shape'])
        if 'original_shape' in metadata:
            metadata['original_shape'] = tuple(metadata['original_shape'])
        if 'padded_shape' in metadata:
            metadata['padded_shape'] = tuple(metadata['padded_shape'])
        if 'padding' in metadata:
            metadata['padding'] = tuple(tuple(p) for p in metadata['padding'])

        return latent, metadata


def test_compressor():
    """
    Test the SARCompressor with synthetic data.

    This test:
    1. Loads the best checkpoint
    2. Creates synthetic SAR-like data (smooth regions + speckle noise)
    3. Tests compress/decompress round-trip via full API
    4. Tests progress callback functionality
    5. Verifies output shape and reasonable PSNR
    """
    import os

    print("Testing SARCompressor...")

    # Find checkpoint
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'notebooks', 'checkpoints', 'resnet_lite_v2_c16', 'best.pth'
    )

    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found at: {checkpoint_path}")
        print("  Skipping test.")
        return

    print(f"  Loading checkpoint: {checkpoint_path}")

    # Create compressor
    compressor = SARCompressor(
        model_path=checkpoint_path,
        patch_size=256,
        overlap=64
    )

    print(f"  Device: {compressor.device}")
    print(f"  Batch size: {compressor.batch_size}")
    print(f"  Preprocess params: vmin={compressor.preprocess_params['vmin']:.4f}, vmax={compressor.preprocess_params['vmax']:.4f}")

    # Create SAR-like test image (512x512)
    # SAR images have smooth regions with multiplicative speckle noise
    np.random.seed(42)

    # Create smooth base image (gradient + some structure)
    y, x = np.mgrid[0:512, 0:512].astype(np.float32)
    base = 0.3 + 0.4 * (np.sin(x / 50) * np.sin(y / 50) + 1) / 2

    # Add some larger-scale structure
    base += 0.2 * np.exp(-((x - 256)**2 + (y - 256)**2) / (2 * 100**2))

    # Normalized to [0, 1] range (like preprocessed SAR data)
    test_image_normalized = np.clip(base, 0, 1).astype(np.float32)

    print(f"  Test image shape: {test_image_normalized.shape}")
    print(f"  Test image range: [{test_image_normalized.min():.3f}, {test_image_normalized.max():.3f}]")

    # Test 1: Direct tile processing (bypasses preprocess for testing)
    print("\n  Test 1: Direct tile processing...")
    tiles, tile_metadata = extract_tiles(test_image_normalized, tile_size=256, overlap=64)
    print(f"    Extracted {tiles.shape[0]} tiles")

    # Process through model with progress callback
    progress_count = [0]
    def progress_callback(current, total):
        progress_count[0] += 1

    latent_patches = compressor._process_tiles_batched(tiles, encode=True, progress_callback=progress_callback)
    print(f"    Latent shape: {latent_patches.shape}")
    print(f"    Progress callbacks received: {progress_count[0]}")

    # Decode
    decoded_tiles = compressor._process_tiles_batched(latent_patches, encode=False)

    # Reconstruct
    reconstructed = reconstruct_from_tiles(decoded_tiles, tile_metadata, compressor.blend_weights)

    # Verify shape
    assert reconstructed.shape == test_image_normalized.shape, \
        f"Shape mismatch: {reconstructed.shape} vs {test_image_normalized.shape}"
    print(f"    Reconstructed shape: {reconstructed.shape} (matches input)")

    # Compute PSNR
    mse = np.mean((reconstructed - test_image_normalized) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    print(f"    Round-trip PSNR: {psnr:.2f} dB")

    # Get compression stats
    stats = compressor.get_compression_stats(test_image_normalized, latent_patches)
    print(f"    Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"    Bits per pixel: {stats['bpp']:.2f}")

    # Test 2: Full compress/decompress API with simulated linear SAR data
    print("\n  Test 2: Full compress/decompress API...")

    # Create simulated linear SAR intensity (in range that will map to our dB range)
    # vmin=14.77 dB -> 10^1.477 = 30, vmax=24.54 dB -> 10^2.454 = 284
    linear_sar = 30 + 200 * base + 10 * np.random.rand(512, 512).astype(np.float32)

    # Full compress
    latent, metadata = compressor.compress(linear_sar)
    print(f"    Compressed: {linear_sar.shape} -> {latent.shape}")

    # Full decompress
    reconstructed_sar = compressor.decompress(latent, metadata)
    print(f"    Decompressed: {reconstructed_sar.shape}")

    # Verify shape matches
    assert reconstructed_sar.shape == linear_sar.shape, \
        f"Shape mismatch: {reconstructed_sar.shape} vs {linear_sar.shape}"

    # Compute PSNR on linear values (should be reasonable for SAR-like data)
    mse_linear = np.mean((reconstructed_sar - linear_sar) ** 2)
    max_val = max(linear_sar.max(), reconstructed_sar.max())
    psnr_linear = 10 * np.log10(max_val**2 / mse_linear) if mse_linear > 0 else float('inf')
    print(f"    Linear PSNR: {psnr_linear:.2f} dB")

    # Test 3: Verify progress callbacks work in compress/decompress
    print("\n  Test 3: Progress callbacks in full API...")
    compress_progress = [0]
    decompress_progress = [0]

    def compress_cb(current, total):
        compress_progress[0] = current

    def decompress_cb(current, total):
        decompress_progress[0] = current

    latent2, _ = compressor.compress(linear_sar, progress_callback=compress_cb)
    _ = compressor.decompress(latent2, metadata, progress_callback=decompress_cb)

    print(f"    Compress progress final: {compress_progress[0]}")
    print(f"    Decompress progress final: {decompress_progress[0]}")
    assert compress_progress[0] > 0, "Compress progress callback not called"
    assert decompress_progress[0] > 0, "Decompress progress callback not called"

    # Minimum quality check (lower threshold since test data is synthetic)
    # The model was trained on real SAR data, so synthetic data performance is expected to be lower
    min_psnr = 10.0  # Very conservative threshold for synthetic data
    assert psnr > min_psnr, f"PSNR too low: {psnr:.2f} dB (expected > {min_psnr} dB)"

    print("\n  All tests PASSED!")


if __name__ == "__main__":
    test_compressor()
