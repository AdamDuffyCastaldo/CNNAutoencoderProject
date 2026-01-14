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
from typing import Optional, Tuple, Callable
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
        # TODO: Implement dataset initialization
        #
        # self.patches = patches.astype(np.float32)
        # self.augment = augment
        # self.transform = transform
        #
        # Validate input:
        # - Check shape is (N, H, W)
        # - Check values in [0, 1]
        
        raise NotImplementedError("TODO: Implement dataset initialization")
    
    def __len__(self) -> int:
        """Return number of patches."""
        # TODO: Return length
        raise NotImplementedError("TODO: Implement __len__")
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get patch as tensor.
        
        Args:
            idx: Patch index
        
        Returns:
            Tensor of shape (1, H, W)
        """
        # TODO: Implement __getitem__
        #
        # patch = self.patches[idx].copy()
        # 
        # if self.augment:
        #     patch = self._augment(patch)
        # 
        # if self.transform is not None:
        #     patch = self.transform(patch)
        # 
        # # Add channel dimension: (H, W) → (1, H, W)
        # patch = torch.from_numpy(patch).unsqueeze(0)
        # 
        # return patch
        
        raise NotImplementedError("TODO: Implement __getitem__")
    
    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations.
        
        Augmentations:
        - 50% chance horizontal flip
        - 50% chance vertical flip
        - Random 90° rotation (0, 90, 180, or 270 degrees)
        
        Args:
            patch: Input patch (H, W)
        
        Returns:
            Augmented patch
        """
        # TODO: Implement augmentation
        #
        # # Random horizontal flip
        # if random.random() > 0.5:
        #     patch = np.fliplr(patch).copy()
        # 
        # # Random vertical flip
        # if random.random() > 0.5:
        #     patch = np.flipud(patch).copy()
        # 
        # # Random 90° rotation
        # k = random.randint(0, 3)
        # if k > 0:
        #     patch = np.rot90(patch, k).copy()
        # 
        # return patch
        
        raise NotImplementedError("TODO: Implement augmentation")


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
