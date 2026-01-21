"""
PyTorch Dataset Classes for SAR Autoencoder

This module provides Dataset classes for loading SAR patches.

References:
    - Day 2, Section 2.6 of the learning guide
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import random


class SARPatchDataset(Dataset):
    """
    Dataset for SAR patches.
    
    Handles:
    - Numpy array to tensor conversion
    - Optional data augmentation (flips, 90° rotations)
    - Channel dimension addition
    
    Args:
        patches: Numpy array of shape (N, H, W), values in [0, 1]
        augment: Whether to apply random augmentation
        transform: Optional additional transform function
    
    Note:
        For SAR, safe augmentations are:
        - Horizontal flip
        - Vertical flip
        - 90° rotations
        
        NOT safe (changes SAR physics):
        - Arbitrary rotations
        - Intensity jittering
        - Elastic deformations
    
    Example:
        >>> patches = np.load('patches.npy')
        >>> dataset = SARPatchDataset(patches, augment=True)
        >>> sample = dataset[0]
        >>> print(sample.shape)  # torch.Size([1, 256, 256])
    """
    
    def __init__(
        self,
        patches: np.ndarray,
        augment: bool = True,
        transform: Optional[Callable] = None
    ):
        # Store patches as float32
        self.patches = patches.astype(np.float32)
        self.augment = augment
        self.transform = transform

        # Validate input shape: should be (N, H, W)
        if len(self.patches.shape) != 3:
            raise ValueError(f"Expected patches shape (N, H, W), got {self.patches.shape}")

        # Validate values in [0, 1]
        if self.patches.min() < 0 or self.patches.max() > 1:
            raise ValueError(
                f"Expected patch values in [0, 1], got [{self.patches.min():.4f}, {self.patches.max():.4f}]"
            )

    def __len__(self) -> int:
        """Return number of patches."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get patch as tensor.

        Args:
            idx: Patch index

        Returns:
            Tensor of shape (1, H, W)
        """
        # Copy patch (critical for augmentation on mmap arrays)
        patch = self.patches[idx].copy()

        if self.augment:
            patch = self._augment(patch)

        if self.transform is not None:
            patch = self.transform(patch)

        # Add channel dimension: (H, W) -> (1, H, W)
        patch = torch.from_numpy(patch).unsqueeze(0)

        return patch

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations.

        Augmentations:
        - 50% chance horizontal flip
        - 50% chance vertical flip
        - Random 90 degree rotation (0, 90, 180, or 270 degrees)

        Args:
            patch: Input patch (H, W)

        Returns:
            Augmented patch
        """
        # Random horizontal flip
        if random.random() > 0.5:
            patch = np.fliplr(patch).copy()

        # Random vertical flip
        if random.random() > 0.5:
            patch = np.flipud(patch).copy()

        # Random 90 degree rotation
        k = random.randint(0, 3)
        if k > 0:
            patch = np.rot90(patch, k).copy()

        return patch


class LazyPatchDataset(Dataset):
    """
    Memory-efficient dataset loading patches from multiple .npy files.

    Uses memory mapping to avoid loading all 182GB into RAM.

    Args:
        metadata_path: Path to metadata.npy containing file_index
        augment: Whether to apply augmentation
        shuffle_seed: Seed for deterministic shuffling (None = no shuffle)
        transform: Optional additional transform function

    Example:
        >>> ds = LazyPatchDataset('data/patches/metadata.npy', augment=True)
        >>> print(len(ds))  # Total patches across all files
        >>> sample = ds[0]
        >>> print(sample.shape)  # torch.Size([1, 256, 256])
    """

    def __init__(
        self,
        metadata_path: str,
        augment: bool = True,
        shuffle_seed: Optional[int] = 42,
        transform: Optional[Callable] = None
    ):
        metadata_path = Path(metadata_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.file_index = metadata['file_index']  # [(path, count), ...]
        self.augment = augment
        self.transform = transform

        # Build cumulative sum for O(log n) file lookup
        self.cumsum = [0]
        for _, count in self.file_index:
            self.cumsum.append(self.cumsum[-1] + count)
        self.total = self.cumsum[-1]

        # Create shuffle index if seed provided
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            self.shuffle_idx = rng.permutation(self.total)
        else:
            self.shuffle_idx = np.arange(self.total)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = self.shuffle_idx[idx]

        # Binary search for file
        file_idx = np.searchsorted(self.cumsum[1:], real_idx, side='right')
        local_idx = real_idx - self.cumsum[file_idx]

        fpath, _ = self.file_index[file_idx]
        patch = np.load(fpath, mmap_mode='r')[local_idx].copy()  # .copy() critical!

        if self.augment:
            patch = self._augment(patch)

        if self.transform is not None:
            patch = self.transform(patch)

        return torch.from_numpy(patch).unsqueeze(0).float()

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentations (same as SARPatchDataset)."""
        # Random horizontal flip
        if random.random() > 0.5:
            patch = np.fliplr(patch).copy()

        # Random vertical flip
        if random.random() > 0.5:
            patch = np.flipud(patch).copy()

        # Random 90 degree rotation
        k = random.randint(0, 3)
        if k > 0:
            patch = np.rot90(patch, k).copy()

        return patch


class SARImageDataset(Dataset):
    """
    Dataset for full SAR images (for evaluation).

    Unlike SARPatchDataset, this loads full images without patching.
    Useful for evaluation where you want to reconstruct complete images.

    Args:
        image_paths: List of paths to preprocessed images
        transform: Optional transform function
    """

    def __init__(
        self,
        image_paths: list,
        transform: Optional[Callable] = None
    ):
        # TODO: Implement initialization
        raise NotImplementedError("TODO: Implement SARImageDataset")

    def __len__(self) -> int:
        raise NotImplementedError("TODO: Implement __len__")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get image and metadata.

        Returns:
            image: Tensor of shape (1, H, W)
            metadata: Dict with path, shape, etc.
        """
        raise NotImplementedError("TODO: Implement __getitem__")


def verify_patch_files(metadata_path: str) -> list:
    """
    Verify all patch files can be loaded.

    Args:
        metadata_path: Path to metadata.npy

    Returns:
        List of (path, error) tuples for any failures. Empty list = all OK.
    """
    metadata = np.load(metadata_path, allow_pickle=True).item()
    errors = []
    for fpath, expected_count in metadata['file_index']:
        try:
            data = np.load(fpath, mmap_mode='r')
            if len(data) != expected_count:
                errors.append((fpath, f"Expected {expected_count} patches, got {len(data)}"))
        except Exception as e:
            errors.append((fpath, str(e)))
    return errors


def test_dataset():
    """Test dataset classes."""
    print("Testing SARPatchDataset...")
    
    # Create synthetic patches
    np.random.seed(42)
    test_patches = np.random.rand(100, 256, 256).astype(np.float32)
    
    # Test without augmentation
    dataset = SARPatchDataset(test_patches, augment=False)
    sample = dataset[0]
    
    assert sample.shape == (1, 256, 256), f"Wrong shape: {sample.shape}"
    print(f"✓ Shape correct: {sample.shape}")
    
    assert sample.dtype == torch.float32, f"Wrong dtype: {sample.dtype}"
    print(f"✓ Dtype correct: {sample.dtype}")
    
    # Test augmentation creates variety
    dataset_aug = SARPatchDataset(test_patches, augment=True)
    sample1 = dataset_aug[0]
    sample2 = dataset_aug[0]  # Same index
    
    if not torch.allclose(sample1, sample2):
        print("✓ Augmentation creates variety")
    else:
        print("⚠ Warning: Augmentation may not be working")
    
    print("All dataset tests passed!")


if __name__ == "__main__":
    test_dataset()
