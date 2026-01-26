"""
Tiling Infrastructure for Large SAR Image Processing

This module provides functions for splitting large images into overlapping tiles,
processing them individually, and reconstructing with seamless blending using
cosine-squared ramp weights.

Key functions:
- create_cosine_ramp_weights: Generate 2D blending weights for tile overlap
- extract_tiles: Split image into overlapping tiles with padding
- reconstruct_from_tiles: Reassemble tiles with weighted blending
"""

from typing import Dict, Optional, Tuple

import numpy as np


def create_cosine_ramp_weights(tile_size: int, overlap: int) -> np.ndarray:
    """
    Create 2D blending weights with cosine-squared ramp for tile overlap regions.

    The weights are 1.0 in the center of the tile and smoothly ramp in overlap
    regions using a cosine-squared function. This ensures that when adjacent
    tiles are overlapped, their combined weights sum to 1.0 in the overlap region.

    For boundary tiles (edges of the image), the weights at the extreme edges
    will still be positive, ensuring valid reconstruction even without a
    neighboring tile.

    Args:
        tile_size: Size of square tiles (pixels)
        overlap: Number of pixels overlap between adjacent tiles

    Returns:
        2D numpy array of shape (tile_size, tile_size) with float32 weights

    Raises:
        ValueError: If tile_size <= 0 or overlap < 0 or overlap > tile_size // 2
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap > tile_size // 2:
        raise ValueError(f"overlap ({overlap}) cannot exceed tile_size // 2 ({tile_size // 2})")

    # Handle zero overlap case - uniform weights
    if overlap == 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    # Create distance-based weight map
    # Each pixel's weight depends on its distance from the tile edges
    # Pixels in the center (not in any overlap region) get weight 1.0
    # Pixels in overlap regions get a cosine-squared ramp

    weights_2d = np.ones((tile_size, tile_size), dtype=np.float32)

    # Create 1D ramp for edges: sin^2 from 0 to pi/2 gives 0 to 1 transition
    ramp = np.sin(np.linspace(0, np.pi / 2, overlap)) ** 2

    # Apply ramps to each edge
    # Left edge: multiply by ramp (0 to 1 going right)
    for i in range(overlap):
        weights_2d[:, i] *= ramp[i]

    # Right edge: multiply by ramp (1 to 0 going right)
    for i in range(overlap):
        weights_2d[:, tile_size - overlap + i] *= ramp[overlap - 1 - i]

    # Top edge: multiply by ramp (0 to 1 going down)
    for i in range(overlap):
        weights_2d[i, :] *= ramp[i]

    # Bottom edge: multiply by ramp (1 to 0 going down)
    for i in range(overlap):
        weights_2d[tile_size - overlap + i, :] *= ramp[overlap - 1 - i]

    return weights_2d


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64
) -> Tuple[np.ndarray, Dict]:
    """
    Extract overlapping tiles from an image with reflection padding.

    The image is padded using reflection mode to ensure all tiles are complete
    and edge artifacts are minimized. Padding is added on all sides to ensure
    that the overlap regions of boundary tiles extend into padding, allowing
    proper blending weight coverage of the original image region.

    Tiles are extracted in row-major order.

    Args:
        image: 2D numpy array of shape (H, W)
        tile_size: Size of square tiles to extract (default: 256)
        overlap: Number of pixels overlap between adjacent tiles (default: 64)

    Returns:
        Tuple of:
        - tiles: 3D numpy array of shape (N, tile_size, tile_size)
        - metadata: Dict with keys:
            - grid_shape: (rows, cols) number of tiles in each dimension
            - original_shape: (H, W) original image dimensions
            - padded_shape: (H_pad, W_pad) padded image dimensions
            - padding: ((top, bottom), (left, right)) padding applied
            - tile_size: tile size used
            - overlap: overlap used
            - stride: effective stride between tiles

    Raises:
        ValueError: If image is empty or invalid parameters
    """
    if image.size == 0:
        raise ValueError("Cannot extract tiles from empty image")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got {image.ndim}D array")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    original_shape = image.shape
    H, W = original_shape

    # Compute stride
    stride = tile_size - overlap

    # Add overlap padding on all sides to ensure the first/last tiles' overlap
    # regions extend into padding (not into the original image edge)
    # This ensures proper blending weight coverage of the original image
    top_pad = overlap
    left_pad = overlap

    # Handle images smaller than the non-overlap portion of a tile
    effective_tile = tile_size - 2 * overlap  # Non-overlapping center
    if effective_tile <= 0:
        # Overlap is more than half tile size, special handling
        effective_tile = stride

    # Calculate tiles needed to cover the original image
    # With offset padding, original image starts at (overlap, overlap)
    n_rows = max(1, int(np.ceil(H / stride)))
    n_cols = max(1, int(np.ceil(W / stride)))

    # Calculate required padded dimensions
    # The last tile ends at: overlap + (n_tiles - 1) * stride + tile_size
    # We need this to be at least: overlap + original_dim + overlap (to allow bottom/right ramps)
    required_h = top_pad + (n_rows - 1) * stride + tile_size
    required_w = left_pad + (n_cols - 1) * stride + tile_size

    # Calculate bottom/right padding needed
    min_h = top_pad + H + overlap  # Need overlap on bottom for last tile's ramp
    min_w = left_pad + W + overlap  # Need overlap on right for last tile's ramp

    bottom_pad = max(required_h, min_h) - (top_pad + H)
    right_pad = max(required_w, min_w) - (left_pad + W)

    # Apply padding
    image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='reflect')
    padded_shape = image.shape

    # Extract tiles
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * stride
            x = col * stride
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

    tiles = np.stack(tiles, axis=0)

    metadata = {
        'grid_shape': (n_rows, n_cols),
        'original_shape': original_shape,
        'padded_shape': padded_shape,
        'padding': ((top_pad, bottom_pad), (left_pad, right_pad)),
        'tile_size': tile_size,
        'overlap': overlap,
        'stride': stride
    }

    return tiles, metadata


def reconstruct_from_tiles(
    tiles: np.ndarray,
    metadata: Dict,
    blend_weights: np.ndarray
) -> np.ndarray:
    """
    Reconstruct an image from overlapping tiles using weighted blending.

    Each tile is multiplied by the blend weights and accumulated into the output.
    A separate weight accumulator tracks the total weight at each pixel, and the
    final output is normalized by dividing by the accumulated weights.

    Args:
        tiles: 3D numpy array of shape (N, tile_size, tile_size)
        metadata: Dict returned by extract_tiles containing reconstruction info
        blend_weights: 2D numpy array of shape (tile_size, tile_size)

    Returns:
        Reconstructed image with original dimensions (H, W)

    Raises:
        ValueError: If tiles/metadata/weights are inconsistent
    """
    n_rows, n_cols = metadata['grid_shape']
    padded_shape = metadata['padded_shape']
    original_shape = metadata['original_shape']
    tile_size = metadata['tile_size']
    stride = metadata['stride']
    padding = metadata['padding']

    expected_n_tiles = n_rows * n_cols
    if tiles.shape[0] != expected_n_tiles:
        raise ValueError(f"Expected {expected_n_tiles} tiles, got {tiles.shape[0]}")
    if tiles.shape[1:] != (tile_size, tile_size):
        raise ValueError(f"Tile shape mismatch: expected ({tile_size}, {tile_size}), got {tiles.shape[1:]}")
    if blend_weights.shape != (tile_size, tile_size):
        raise ValueError(f"Weight shape mismatch: expected ({tile_size}, {tile_size}), got {blend_weights.shape}")

    # Initialize accumulators
    output = np.zeros(padded_shape, dtype=np.float32)
    weight_sum = np.zeros(padded_shape, dtype=np.float32)

    # Accumulate weighted tiles
    tile_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * stride
            x = col * stride

            tile = tiles[tile_idx].astype(np.float32)
            output[y:y + tile_size, x:x + tile_size] += tile * blend_weights
            weight_sum[y:y + tile_size, x:x + tile_size] += blend_weights

            tile_idx += 1

    # Normalize by accumulated weights (avoid division by zero)
    output = np.divide(output, weight_sum, where=weight_sum > 0, out=output)

    # Remove padding to get original size
    (top_pad, bottom_pad), (left_pad, right_pad) = padding
    orig_H, orig_W = original_shape

    # Always crop to original dimensions
    # Start from top_pad offset (for small images) and take original size
    output = output[top_pad:top_pad + orig_H, left_pad:left_pad + orig_W]

    return output


def visualize_blend_weights(
    tile_size: int,
    overlap: int,
    output_path: Optional[str] = None
) -> None:
    """
    Visualize the blending weights used for tile reconstruction.

    Creates a figure showing:
    1. 2D heatmap of the weights
    2. 1D cross-section through the horizontal center

    This is useful for debugging blending and verifying smooth transitions.

    Args:
        tile_size: Size of square tiles (pixels)
        overlap: Number of pixels overlap between adjacent tiles
        output_path: If provided, save figure to this path; otherwise display

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    weights = create_cosine_ramp_weights(tile_size, overlap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2D heatmap
    im = axes[0].imshow(weights, cmap='viridis', aspect='equal')
    axes[0].set_title(f'Blend Weights (tile={tile_size}, overlap={overlap})')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=axes[0], label='Weight')

    # 1D cross-section through center
    center = tile_size // 2
    cross_section = weights[center, :]
    axes[1].plot(cross_section, 'b-', linewidth=2)
    axes[1].axvline(x=overlap, color='r', linestyle='--', alpha=0.7, label=f'Overlap boundary ({overlap})')
    axes[1].axvline(x=tile_size - overlap, color='r', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlim(0, tile_size)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel('Position (pixels)')
    axes[1].set_ylabel('Weight')
    axes[1].set_title(f'Horizontal Cross-Section (y={center})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved blend weights visualization to: {output_path}")
    else:
        plt.show()


def test_tiling():
    """
    Test the tiling functions to verify correctness.

    Tests:
    1. Cosine weights properties (center=1.0, corners<0.5)
    2. Weight summation in overlap regions
    3. Round-trip reconstruction accuracy
    4. Various image sizes
    5. Zero overlap
    6. Small images
    7. Visualization output
    """
    print("Testing tiling module...")

    # Test 1: Cosine weights properties
    print("\n1. Testing cosine ramp weights...")
    tile_size = 256
    overlap = 64
    weights = create_cosine_ramp_weights(tile_size, overlap)

    assert weights.shape == (tile_size, tile_size), f"Weight shape wrong: {weights.shape}"
    assert weights.dtype == np.float32, f"Weight dtype wrong: {weights.dtype}"

    # Center should be 1.0
    center = tile_size // 2
    assert np.isclose(weights[center, center], 1.0), f"Center weight should be 1.0, got {weights[center, center]}"
    print(f"   Center weight: {weights[center, center]:.4f} (expected 1.0)")

    # Corners (overlap region) should be < 0.5
    corner_weight = weights[0, 0]
    assert corner_weight < 0.5, f"Corner weight should be < 0.5, got {corner_weight}"
    print(f"   Corner weight: {corner_weight:.4f} (expected < 0.5)")

    # Test that weights at overlap boundaries are roughly 0.5 (where two tiles meet)
    edge_weight = weights[overlap, center]  # Just inside the overlap region
    print(f"   Edge weight at overlap boundary: {edge_weight:.4f}")

    print("   PASSED: Cosine weights have correct properties")

    # Test 2: Weight summation in overlap regions
    print("\n2. Testing weight summation in overlaps...")
    # When two tiles overlap, their weights should sum to 1.0
    # Create two adjacent weight arrays and check their sum
    stride = tile_size - overlap

    # Simulate two horizontally adjacent tiles
    combined_width = tile_size + stride
    combined = np.zeros(combined_width)
    combined[:tile_size] += weights[center, :]  # First tile's horizontal weights
    combined[stride:stride + tile_size] += weights[center, :]  # Second tile (shifted by stride)

    # In the overlap region, sum should be approximately 1.0
    overlap_sum = combined[stride:tile_size]
    assert np.allclose(overlap_sum, 1.0, atol=1e-5), f"Overlap sum should be 1.0, got {overlap_sum.mean():.4f}"
    print(f"   Overlap region sum: {overlap_sum.mean():.6f} (expected 1.0)")
    print("   PASSED: Overlapping weights sum to 1.0")

    # Test 3: Round-trip reconstruction
    print("\n3. Testing round-trip reconstruction...")

    # Create test image
    np.random.seed(42)
    test_image = np.random.rand(512, 512).astype(np.float32)

    # Extract tiles
    tiles, metadata = extract_tiles(test_image, tile_size=256, overlap=64)
    print(f"   Extracted {tiles.shape[0]} tiles from {test_image.shape} image")
    print(f"   Grid shape: {metadata['grid_shape']}")

    # Reconstruct with identity transform (tiles unchanged)
    blend_weights = create_cosine_ramp_weights(256, 64)
    reconstructed = reconstruct_from_tiles(tiles, metadata, blend_weights)

    # Check reconstruction matches original
    assert reconstructed.shape == test_image.shape, f"Shape mismatch: {reconstructed.shape} vs {test_image.shape}"

    max_error = np.abs(reconstructed - test_image).max()
    mean_error = np.abs(reconstructed - test_image).mean()
    print(f"   Max reconstruction error: {max_error:.2e}")
    print(f"   Mean reconstruction error: {mean_error:.2e}")

    assert max_error < 1e-5, f"Reconstruction error too high: {max_error}"
    print("   PASSED: Round-trip reconstruction within tolerance")

    # Test 4: Various image sizes
    print("\n4. Testing various image sizes...")
    test_sizes = [(512, 512), (500, 500), (256, 256), (128, 128), (1000, 800)]

    for H, W in test_sizes:
        test_img = np.random.rand(H, W).astype(np.float32)
        tiles, meta = extract_tiles(test_img, tile_size=256, overlap=64)
        weights = create_cosine_ramp_weights(256, 64)
        recon = reconstruct_from_tiles(tiles, meta, weights)

        error = np.abs(recon - test_img).max()
        status = "OK" if error < 1e-5 else "FAIL"
        print(f"   {H}x{W}: {tiles.shape[0]} tiles, error={error:.2e} [{status}]")
        assert error < 1e-5, f"Failed for size {H}x{W}"

    print("   PASSED: All image sizes work correctly")

    # Test 5: Zero overlap (should work)
    print("\n5. Testing zero overlap...")
    weights_no_overlap = create_cosine_ramp_weights(256, 0)
    assert np.all(weights_no_overlap == 1.0), "Zero overlap should give uniform weights"
    print("   PASSED: Zero overlap produces uniform weights")

    # Test 6: Small images (smaller than tile size)
    print("\n6. Testing small images...")
    small_img = np.random.rand(100, 100).astype(np.float32)
    tiles_small, meta_small = extract_tiles(small_img, tile_size=256, overlap=64)
    weights = create_cosine_ramp_weights(256, 64)
    recon_small = reconstruct_from_tiles(tiles_small, meta_small, weights)

    error_small = np.abs(recon_small - small_img).max()
    print(f"   100x100 image: {tiles_small.shape[0]} tiles, error={error_small:.2e}")
    assert error_small < 1e-5, f"Failed for small image"
    print("   PASSED: Small images handled correctly")

    # Test 7: Additional weight property assertions
    print("\n7. Testing additional weight properties...")
    weights = create_cosine_ramp_weights(256, 64)

    # Weights at tile center should be 1.0
    center = 256 // 2
    assert np.isclose(weights[center, center], 1.0, atol=1e-6), \
        f"Center weight should be 1.0, got {weights[center, center]}"
    print(f"   Weight at center ({center}, {center}): {weights[center, center]:.6f}")

    # Weights at corners (overlap regions from both edges) should be < 0.5
    corner_weight = weights[0, 0]
    assert corner_weight < 0.5, f"Corner weight should be < 0.5, got {corner_weight}"
    print(f"   Weight at corner (0, 0): {corner_weight:.6f}")

    # Reconstruction error should be < 1e-5 for identity transform
    np.random.seed(123)
    test_img = np.random.rand(512, 512).astype(np.float32)
    tiles, meta = extract_tiles(test_img, tile_size=256, overlap=64)
    recon = reconstruct_from_tiles(tiles, meta, weights)
    identity_error = np.abs(recon - test_img).max()
    assert identity_error < 1e-5, f"Identity reconstruction error should be < 1e-5, got {identity_error}"
    print(f"   Identity reconstruction error: {identity_error:.2e} (< 1e-5 required)")

    print("   PASSED: Additional weight properties verified")

    # Test 8: Save visualization
    print("\n8. Saving blend weights visualization...")
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              'notebooks', 'evaluations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'blend_weights_sample.png')
    visualize_blend_weights(256, 64, output_path=output_path)
    assert os.path.exists(output_path), f"Visualization file not created at {output_path}"
    print(f"   Saved to: {output_path}")
    print("   PASSED: Visualization saved successfully")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_tiling()
