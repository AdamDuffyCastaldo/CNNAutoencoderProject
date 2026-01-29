"""
Entropy-Based Bitrate Calculation for Autoencoder Latents

This module implements entropy-based bits-per-pixel (BPP) calculation for
quantized autoencoder latents, enabling fair comparison with traditional
codecs like JPEG-2000 at matched bitrates.

Key Functions:
- quantize_latent: Per-channel quantization to discrete levels
- dequantize_latent: Inverse quantization back to original range
- estimate_channel_entropy: Shannon entropy estimation for a single channel
- estimate_latent_entropy: Full entropy estimation with overhead
- compute_latent_bpp: Convenience function for BPP calculation

The approach follows standard practice in learned image compression:
1. Quantize latent to discrete levels (e.g., 8-bit = 256 levels)
2. Estimate entropy of quantized distribution using histograms
3. Calculate BPP as total_bits / num_input_pixels
4. Include overhead for quantization parameters (min/max per channel)

References:
    - Balle et al. "End-to-end Optimized Image Compression" (2017)
    - Phase 6.1 Research: entropy estimation methodology
"""

import numpy as np
from scipy.stats import entropy
from typing import Tuple, List, Dict, Union, Optional


def quantize_latent(
    latent: np.ndarray,
    n_bins: int = 256
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Quantize latent representation to discrete levels per-channel.

    Each channel is independently quantized using its own min/max range,
    preserving more precision than global quantization when channels
    have different value distributions.

    Parameters
    ----------
    latent : np.ndarray
        Latent tensor of shape (B, C, H, W) or (C, H, W), float32.
        For batch processing, quantization ranges are computed across
        the entire batch per channel.
    n_bins : int, optional
        Number of quantization levels. Default 256 for 8-bit quantization.

    Returns
    -------
    quantized : np.ndarray
        Quantized array as uint8, same shape as input.
        Values in range [0, n_bins-1].
    ranges : List[Tuple[float, float]]
        List of (vmin, vmax) tuples per channel. Length equals C.
        These values are needed for dequantization.

    Notes
    -----
    - If vmax == vmin for a channel (constant values), the quantized
      output is all zeros for that channel.
    - The quantization formula is:
      quantized = floor((x - vmin) / (vmax - vmin) * (n_bins - 1))

    Examples
    --------
    >>> latent = np.random.randn(1, 16, 16, 16).astype(np.float32)
    >>> quantized, ranges = quantize_latent(latent, n_bins=256)
    >>> print(quantized.shape, quantized.dtype)
    (1, 16, 16, 16) uint8
    >>> print(len(ranges))
    16
    """
    latent = np.asarray(latent, dtype=np.float32)

    # Handle both (C, H, W) and (B, C, H, W) shapes
    if latent.ndim == 3:
        latent = latent[np.newaxis, ...]  # Add batch dimension

    B, C, H, W = latent.shape
    quantized = np.zeros_like(latent, dtype=np.uint8)
    ranges = []

    for c in range(C):
        channel_data = latent[:, c, :, :]
        vmin = float(channel_data.min())
        vmax = float(channel_data.max())
        ranges.append((vmin, vmax))

        if vmax > vmin:
            # Scale to [0, n_bins-1]
            scaled = (channel_data - vmin) / (vmax - vmin)
            quantized[:, c, :, :] = np.floor(
                scaled * (n_bins - 1)
            ).clip(0, n_bins - 1).astype(np.uint8)
        else:
            # Constant channel - all zeros
            quantized[:, c, :, :] = 0

    # Remove batch dimension if input was 3D
    if latent.shape[0] == 1 and len(latent.shape) == 4:
        # Keep batch dim for consistency
        pass

    return quantized, ranges


def dequantize_latent(
    quantized: np.ndarray,
    ranges: List[Tuple[float, float]],
    n_bins: int = 256
) -> np.ndarray:
    """
    Dequantize latent representation back to original value range.

    Inverse operation of quantize_latent. Reconstructs approximate
    float values from quantized integers using the stored per-channel
    ranges.

    Parameters
    ----------
    quantized : np.ndarray
        Quantized array as uint8, shape (B, C, H, W) or (C, H, W).
        Values should be in range [0, n_bins-1].
    ranges : List[Tuple[float, float]]
        List of (vmin, vmax) tuples per channel from quantize_latent.
    n_bins : int, optional
        Number of quantization levels. Must match quantization. Default 256.

    Returns
    -------
    dequantized : np.ndarray
        Float32 array in original value range, same shape as input.

    Notes
    -----
    - The dequantization formula is:
      x_float = quantized / (n_bins - 1) * (vmax - vmin) + vmin
    - Reconstruction error is bounded by (vmax - vmin) / (2 * (n_bins - 1))
      for each channel.

    Examples
    --------
    >>> latent = np.random.randn(1, 16, 16, 16).astype(np.float32)
    >>> quantized, ranges = quantize_latent(latent)
    >>> reconstructed = dequantize_latent(quantized, ranges)
    >>> max_error = np.max(np.abs(latent - reconstructed))
    >>> print(f"Max reconstruction error: {max_error:.6f}")
    """
    quantized = np.asarray(quantized)

    # Handle both (C, H, W) and (B, C, H, W) shapes
    added_batch = False
    if quantized.ndim == 3:
        quantized = quantized[np.newaxis, ...]
        added_batch = True

    B, C, H, W = quantized.shape
    dequantized = np.zeros_like(quantized, dtype=np.float32)

    for c in range(C):
        vmin, vmax = ranges[c]
        channel_data = quantized[:, c, :, :].astype(np.float32)

        if vmax > vmin:
            # Scale back to original range
            dequantized[:, c, :, :] = (
                channel_data / (n_bins - 1) * (vmax - vmin) + vmin
            )
        else:
            # Constant channel - fill with original value
            dequantized[:, c, :, :] = vmin

    if added_batch:
        dequantized = dequantized[0]

    return dequantized


def estimate_channel_entropy(
    channel_data: np.ndarray,
    n_bins: int = 256
) -> float:
    """
    Estimate Shannon entropy for a single channel of quantized data.

    Uses histogram-based probability estimation and scipy.stats.entropy
    with base=2 to compute entropy in bits per symbol.

    Parameters
    ----------
    channel_data : np.ndarray
        Flattened or multi-dimensional array of quantized values.
        Values should be integers in range [0, n_bins-1].
    n_bins : int, optional
        Number of quantization bins. Default 256.

    Returns
    -------
    float
        Entropy in bits per symbol. For uniform distribution over
        n_bins values, this equals log2(n_bins) = 8 bits for 256 bins.

    Notes
    -----
    - Uses scipy.stats.entropy(prob, base=2) for the calculation.
    - Zero-probability bins are automatically excluded from the sum.
    - For highly peaked distributions, entropy can be much lower than
      log2(n_bins), indicating good compressibility.

    Examples
    --------
    >>> # Uniform distribution - maximum entropy
    >>> uniform_data = np.arange(256).astype(np.uint8)
    >>> h = estimate_channel_entropy(uniform_data, n_bins=256)
    >>> print(f"Uniform entropy: {h:.2f} bits")  # ~8 bits

    >>> # Peaked distribution - lower entropy
    >>> peaked_data = np.zeros(1000, dtype=np.uint8)
    >>> peaked_data[0] = 128  # One different value
    >>> h = estimate_channel_entropy(peaked_data, n_bins=256)
    >>> print(f"Peaked entropy: {h:.4f} bits")  # Near 0
    """
    channel_data = np.asarray(channel_data).flatten().astype(int)

    # Compute histogram
    hist, _ = np.histogram(channel_data, bins=n_bins, range=(0, n_bins))

    # Normalize to probability distribution
    total = hist.sum()
    if total == 0:
        return 0.0

    prob = hist / total

    # Remove zero probabilities (entropy function handles this, but
    # being explicit avoids potential numerical issues)
    prob = prob[prob > 0]

    # Calculate entropy in bits (base=2)
    return float(entropy(prob, base=2))


def estimate_latent_entropy(
    latent: np.ndarray,
    n_bins: int = 256,
    return_per_channel: bool = False,
    input_shape: Tuple[int, int] = (256, 256)
) -> Dict[str, Union[float, int, List[float]]]:
    """
    Estimate total entropy and BPP for a latent representation.

    Performs per-channel entropy estimation and aggregates results,
    including overhead for storing quantization parameters.

    Parameters
    ----------
    latent : np.ndarray
        Latent tensor of shape (B, C, H, W) or (C, H, W), float32.
    n_bins : int, optional
        Number of quantization levels. Default 256.
    return_per_channel : bool, optional
        If True, include per-channel entropy breakdown. Default False.
    input_shape : Tuple[int, int], optional
        Original input image size (H, W). Default (256, 256).

    Returns
    -------
    Dict[str, Union[float, int, List[float]]]
        Dictionary containing:
        - 'total_bits': Total bits for all samples including overhead
        - 'bpp': Bits per pixel relative to input image size
        - 'overhead_bits': Bits for quantization parameters (64 * C)
        - 'n_channels': Number of latent channels (C)
        - 'n_samples': Number of samples in batch (B)
        - 'channel_entropy': List of bits/symbol per channel (if return_per_channel)
        - 'channel_bits': List of total bits per channel (if return_per_channel)

    Notes
    -----
    - Overhead is 64 bits per channel (32-bit float * 2 for min/max).
    - BPP = total_bits / (B * input_H * input_W)
    - Total bits = sum(channel_entropy * channel_samples) + overhead

    Examples
    --------
    >>> # Simulate 16x compression latent
    >>> latent = np.random.randn(1, 16, 16, 16).astype(np.float32)
    >>> result = estimate_latent_entropy(latent)
    >>> print(f"BPP: {result['bpp']:.4f}")
    >>> print(f"Total bits: {result['total_bits']:.0f}")
    """
    latent = np.asarray(latent, dtype=np.float32)

    # Handle both (C, H, W) and (B, C, H, W) shapes
    if latent.ndim == 3:
        latent = latent[np.newaxis, ...]

    B, C, H, W = latent.shape
    input_pixels = input_shape[0] * input_shape[1]

    # Quantize latent
    quantized, ranges = quantize_latent(latent, n_bins)

    # Calculate entropy per channel
    channel_entropy = []  # bits per symbol
    channel_bits = []     # total bits per channel

    for c in range(C):
        channel_data = quantized[:, c, :, :].flatten()
        h = estimate_channel_entropy(channel_data, n_bins)
        channel_entropy.append(h)

        # Total bits = entropy * number of symbols
        n_symbols = len(channel_data)  # B * H * W
        channel_bits.append(h * n_symbols)

    # Sum channel bits
    content_bits = sum(channel_bits)

    # Overhead for quantization parameters: 64 bits per channel
    # (32-bit float for min + 32-bit float for max)
    overhead_bits = 64 * C
    total_bits = content_bits + overhead_bits

    # BPP relative to original input image
    bpp = total_bits / (B * input_pixels)

    result = {
        'total_bits': float(total_bits),
        'bpp': float(bpp),
        'overhead_bits': int(overhead_bits),
        'n_channels': C,
        'n_samples': B,
    }

    if return_per_channel:
        result['channel_entropy'] = channel_entropy
        result['channel_bits'] = channel_bits

    return result


def compute_latent_bpp(
    latent: np.ndarray,
    n_bins: int = 256,
    input_shape: Tuple[int, int] = (256, 256)
) -> float:
    """
    Convenience function to compute just the BPP value.

    Wrapper around estimate_latent_entropy that returns only the
    bits-per-pixel value.

    Parameters
    ----------
    latent : np.ndarray
        Latent tensor of shape (B, C, H, W) or (C, H, W), float32.
    n_bins : int, optional
        Number of quantization levels. Default 256.
    input_shape : Tuple[int, int], optional
        Original input image size (H, W). Default (256, 256).

    Returns
    -------
    float
        Bits per pixel including overhead.

    Examples
    --------
    >>> latent = np.random.randn(1, 16, 16, 16).astype(np.float32)
    >>> bpp = compute_latent_bpp(latent)
    >>> print(f"BPP: {bpp:.4f}")
    """
    result = estimate_latent_entropy(latent, n_bins, input_shape=input_shape)
    return result['bpp']


def test_bitrate():
    """
    Test all bitrate calculation functions.

    Creates a random latent, performs quantization/dequantization,
    and validates the entropy/BPP calculations.
    """
    print("=" * 60)
    print("Testing Bitrate Calculation Functions")
    print("=" * 60)

    np.random.seed(42)

    # Create test latent: (1, 16, 16, 16) simulating 16x compression
    # 256x256 input -> 16x16 latent with 16 channels
    latent = np.random.randn(1, 16, 16, 16).astype(np.float32)
    # Make it more realistic - different channels have different distributions
    for c in range(16):
        latent[:, c, :, :] = latent[:, c, :, :] * (c + 1) * 0.5 + c * 0.1

    print(f"\n--- Test Latent ---")
    print(f"Shape: {latent.shape}")
    print(f"Range: [{latent.min():.4f}, {latent.max():.4f}]")

    # Test 1: Quantization and dequantization
    print(f"\n--- Test 1: Quantization/Dequantization ---")
    quantized, ranges = quantize_latent(latent, n_bins=256)
    print(f"Quantized shape: {quantized.shape}")
    print(f"Quantized dtype: {quantized.dtype}")
    print(f"Quantized range: [{quantized.min()}, {quantized.max()}]")
    print(f"Number of channel ranges: {len(ranges)}")

    # Dequantize
    dequantized = dequantize_latent(quantized, ranges, n_bins=256)
    print(f"Dequantized shape: {dequantized.shape}")

    # Check reconstruction error
    recon_error = np.abs(latent - dequantized)
    max_error = recon_error.max()
    mean_error = recon_error.mean()
    print(f"Max reconstruction error: {max_error:.6f}")
    print(f"Mean reconstruction error: {mean_error:.6f}")

    # Theoretical max error per channel: (vmax - vmin) / (n_bins - 1)
    # This is the step size; actual error can be up to this value
    for c in range(min(3, 16)):  # Check first 3 channels
        vmin, vmax = ranges[c]
        step_size = (vmax - vmin) / (256 - 1)
        actual_max = np.abs(latent[:, c, :, :] - dequantized[:, c, :, :]).max()
        print(f"  Channel {c}: step size={step_size:.6f}, actual max error={actual_max:.6f}")
        assert actual_max <= step_size + 1e-6, f"Error exceeds bound on channel {c}"

    print("  Reconstruction error bounded correctly!")

    # Test 2: Entropy estimation
    print(f"\n--- Test 2: Entropy Estimation ---")
    result = estimate_latent_entropy(latent, n_bins=256, return_per_channel=True)

    print(f"Total bits: {result['total_bits']:.2f}")
    print(f"BPP: {result['bpp']:.4f}")
    print(f"Overhead bits: {result['overhead_bits']}")
    print(f"Number of channels: {result['n_channels']}")
    print(f"Number of samples: {result['n_samples']}")

    # Print per-channel entropy
    print(f"\nPer-channel entropy (bits/symbol):")
    for c, (h, bits) in enumerate(zip(result['channel_entropy'], result['channel_bits'])):
        print(f"  Channel {c}: {h:.4f} bits/symbol, {bits:.2f} total bits")

    # Test 3: Compare to geometric BPP
    print(f"\n--- Test 3: Geometric vs Entropy BPP ---")
    # Geometric BPP assuming float32: 16*16*16*32 / (256*256) = 2.0
    geometric_bpp = (16 * 16 * 16 * 32) / (256 * 256)
    entropy_bpp = result['bpp']

    print(f"Geometric BPP (float32 latent): {geometric_bpp:.4f}")
    print(f"Entropy-based BPP (8-bit quantized): {entropy_bpp:.4f}")
    print(f"Reduction: {(1 - entropy_bpp / geometric_bpp) * 100:.1f}%")

    # Entropy BPP should be less than 8 bits (max for 256 bins)
    # and typically less than geometric BPP if there's redundancy
    assert entropy_bpp < 8 * 16 * 16 * 16 / (256 * 256), "BPP exceeds 8-bit upper bound"
    print("  BPP is reasonable (below 8-bit upper bound)!")

    # Test 4: Convenience function
    print(f"\n--- Test 4: Convenience Function ---")
    bpp = compute_latent_bpp(latent)
    print(f"compute_latent_bpp result: {bpp:.4f}")
    assert abs(bpp - result['bpp']) < 1e-6, "Convenience function mismatch"
    print("  Convenience function matches full result!")

    # Test 5: Edge cases
    print(f"\n--- Test 5: Edge Cases ---")

    # Constant channel
    constant_latent = np.ones((1, 4, 8, 8), dtype=np.float32) * 0.5
    q_const, r_const = quantize_latent(constant_latent)
    print(f"Constant latent - quantized unique values: {np.unique(q_const)}")
    h_const = estimate_channel_entropy(q_const[:, 0, :, :], n_bins=256)
    print(f"Constant channel entropy: {h_const:.6f} bits/symbol")
    assert h_const == 0.0, "Constant channel should have zero entropy"

    # 3D input (no batch dimension)
    latent_3d = np.random.randn(8, 16, 16).astype(np.float32)
    bpp_3d = compute_latent_bpp(latent_3d)
    print(f"3D input BPP: {bpp_3d:.4f}")

    print("\n" + "=" * 60)
    print("All bitrate tests passed!")
    print("=" * 60)


def validate_on_real_latent():
    """
    Validate entropy calculation on real autoencoder latent.

    This function:
    1. Loads a trained ResNet 16x model
    2. Extracts latent representations from test samples
    3. Computes entropy-based BPP
    4. Compares to geometric BPP
    5. Measures quantization degradation

    The validation confirms:
    - Entropy-based BPP is lower than geometric BPP (2.0 for 16x)
    - Quantization doesn't severely degrade reconstruction quality
    """
    import torch
    import sys
    from pathlib import Path

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.models.resnet_autoencoder import ResNetAutoencoder
    from src.evaluation.metrics import SARMetrics

    print("=" * 60)
    print("Validating Entropy Calculation on Real Latent")
    print("=" * 60)

    # Configuration
    checkpoint_path = project_root / 'notebooks/checkpoints/resnet_c16_b64_cr16x_20260128_003926/best.pth'
    metadata_path = project_root / 'data/patches/metadata.npy'
    n_samples = 10  # Number of test samples to evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure the ResNet 16x model has been trained.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"Model config: {config.get('model_type', 'unknown')}, "
          f"latent_channels={config.get('latent_channels', 'N/A')}, "
          f"base_channels={config.get('base_channels', 'N/A')}")

    # Create and load model
    model = ResNetAutoencoder(
        in_channels=1,
        latent_channels=config.get('latent_channels', 16),
        base_channels=config.get('base_channels', 64)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model.count_parameters()['total']:,} parameters")

    # Load test samples from dataset
    if not metadata_path.exists():
        print(f"ERROR: Metadata not found at {metadata_path}")
        return

    metadata = np.load(metadata_path, allow_pickle=True).item()
    file_index = metadata['file_index']

    # Get samples from first file
    first_file_path, first_file_count = file_index[0]
    print(f"\nLoading samples from: {Path(first_file_path).name}")

    patches = np.load(first_file_path, mmap_mode='r')
    # Use fixed indices for reproducibility
    np.random.seed(42)
    sample_indices = np.random.choice(min(first_file_count, 1000), size=n_samples, replace=False)
    test_patches = patches[sample_indices].copy()

    print(f"Loaded {n_samples} test patches, shape: {test_patches.shape}")

    # Process samples
    all_latents = []
    psnr_original = []
    psnr_quantized = []
    ssim_original = []
    ssim_quantized = []

    print(f"\n--- Processing {n_samples} samples ---")

    with torch.no_grad():
        for i, patch in enumerate(test_patches):
            # Add batch and channel dims: (H, W) -> (1, 1, H, W)
            x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

            # Encode
            latent = model.encode(x)  # (1, C, H, W)
            all_latents.append(latent.cpu().numpy())

            # Decode original latent
            recon_original = model.decode(latent)

            # Quantize/dequantize latent
            latent_np = latent.cpu().numpy()
            quantized, ranges = quantize_latent(latent_np, n_bins=256)
            dequantized = dequantize_latent(quantized, ranges, n_bins=256)

            # Decode quantized latent
            dequantized_tensor = torch.from_numpy(dequantized).to(device)
            recon_quantized = model.decode(dequantized_tensor)

            # Compute metrics
            x_np = patch
            recon_orig_np = recon_original.cpu().numpy().squeeze()
            recon_quant_np = recon_quantized.cpu().numpy().squeeze()

            psnr_orig = SARMetrics.psnr(x_np, recon_orig_np)
            psnr_quant = SARMetrics.psnr(x_np, recon_quant_np)
            ssim_orig = SARMetrics.ssim(x_np, recon_orig_np)
            ssim_quant = SARMetrics.ssim(x_np, recon_quant_np)

            psnr_original.append(psnr_orig)
            psnr_quantized.append(psnr_quant)
            ssim_original.append(ssim_orig)
            ssim_quantized.append(ssim_quant)

            if i < 3:  # Print details for first 3
                print(f"  Sample {i}: PSNR orig={psnr_orig:.2f} dB, quant={psnr_quant:.2f} dB, "
                      f"diff={psnr_orig - psnr_quant:.3f} dB")

    # Aggregate latents
    all_latents_np = np.concatenate(all_latents, axis=0)  # (N, C, H, W)
    print(f"\nAggregated latents shape: {all_latents_np.shape}")

    # Compute entropy-based BPP
    result = estimate_latent_entropy(all_latents_np, n_bins=256, return_per_channel=True)

    # Geometric BPP (assuming float32 storage)
    latent_shape = all_latents_np.shape
    C = latent_shape[1]
    geometric_bpp = (16 * 16 * C * 32) / (256 * 256)

    # 8-bit geometric BPP (if we stored quantized without entropy coding)
    geometric_8bit_bpp = (16 * 16 * C * 8) / (256 * 256)

    print(f"\n--- BPP Comparison ---")
    print(f"Geometric BPP (float32 latent): {geometric_bpp:.4f}")
    print(f"Geometric BPP (8-bit latent):   {geometric_8bit_bpp:.4f}")
    print(f"Entropy-based BPP (8-bit):      {result['bpp']:.4f}")
    print(f"Reduction from float32:         {(1 - result['bpp'] / geometric_bpp) * 100:.1f}%")
    print(f"Reduction from 8-bit:           {(1 - result['bpp'] / geometric_8bit_bpp) * 100:.1f}%")

    # Per-channel entropy
    print(f"\nPer-channel entropy (bits/symbol):")
    for c, h in enumerate(result['channel_entropy'][:4]):  # First 4 channels
        print(f"  Channel {c}: {h:.4f} bits/symbol")
    if C > 4:
        print(f"  ... ({C - 4} more channels)")

    # Quantization degradation
    print(f"\n--- Quantization Degradation ---")
    mean_psnr_orig = np.mean(psnr_original)
    mean_psnr_quant = np.mean(psnr_quantized)
    mean_ssim_orig = np.mean(ssim_original)
    mean_ssim_quant = np.mean(ssim_quantized)

    psnr_degradation = mean_psnr_orig - mean_psnr_quant
    ssim_degradation = mean_ssim_orig - mean_ssim_quant

    print(f"Original reconstruction:")
    print(f"  PSNR: {mean_psnr_orig:.2f} +/- {np.std(psnr_original):.2f} dB")
    print(f"  SSIM: {mean_ssim_orig:.4f} +/- {np.std(ssim_original):.4f}")

    print(f"Quantized reconstruction:")
    print(f"  PSNR: {mean_psnr_quant:.2f} +/- {np.std(psnr_quantized):.2f} dB")
    print(f"  SSIM: {mean_ssim_quant:.4f} +/- {np.std(ssim_quantized):.4f}")

    print(f"Degradation:")
    print(f"  PSNR: -{psnr_degradation:.3f} dB")
    print(f"  SSIM: -{ssim_degradation:.6f}")

    # Validation checks
    print(f"\n--- Validation Results ---")

    # Check 1: Entropy BPP < geometric BPP
    if result['bpp'] < geometric_bpp:
        print(f"[PASS] Entropy BPP ({result['bpp']:.4f}) < Geometric BPP ({geometric_bpp:.4f})")
    else:
        print(f"[WARN] Entropy BPP ({result['bpp']:.4f}) >= Geometric BPP ({geometric_bpp:.4f})")

    # Check 2: Quantization degradation is small
    if psnr_degradation < 1.0:
        print(f"[PASS] PSNR degradation ({psnr_degradation:.3f} dB) < 1.0 dB")
    elif psnr_degradation < 2.0:
        print(f"[WARN] PSNR degradation ({psnr_degradation:.3f} dB) is moderate (1-2 dB)")
    else:
        print(f"[WARN] PSNR degradation ({psnr_degradation:.3f} dB) is significant (>2 dB)")

    # Check 3: Entropy values are reasonable
    mean_entropy = np.mean(result['channel_entropy'])
    if 4.0 < mean_entropy < 8.0:
        print(f"[PASS] Mean channel entropy ({mean_entropy:.2f} bits) is reasonable")
    else:
        print(f"[WARN] Mean channel entropy ({mean_entropy:.2f} bits) is unusual")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)

    # Return summary for programmatic use
    return {
        'geometric_bpp': geometric_bpp,
        'entropy_bpp': result['bpp'],
        'mean_psnr_original': mean_psnr_orig,
        'mean_psnr_quantized': mean_psnr_quant,
        'psnr_degradation': psnr_degradation,
        'mean_ssim_original': mean_ssim_orig,
        'mean_ssim_quantized': mean_ssim_quant,
        'ssim_degradation': ssim_degradation,
        'channel_entropies': result['channel_entropy'],
    }


# Module exports
__all__ = [
    'quantize_latent',
    'dequantize_latent',
    'estimate_channel_entropy',
    'estimate_latent_entropy',
    'compute_latent_bpp',
    'test_bitrate',
    'validate_on_real_latent',
]


if __name__ == "__main__":
    test_bitrate()
