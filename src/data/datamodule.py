"""
Data Module for SAR Autoencoder

Manages train/validation splits and DataLoader creation.

References:
    - Day 2, Section 2.6 of the learning guide
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .dataset import SARPatchDataset


class SARDataModule:
    """
    Data module for managing train/validation data.
    
    Handles:
    - Loading patches from disk
    - Splitting into train/validation
    - Creating DataLoaders
    
    Args:
        patches_path: Path to .npy file containing patches
        val_fraction: Fraction of data for validation
        batch_size: Batch size for DataLoaders
        num_workers: Number of DataLoader workers
        augment_train: Whether to augment training data
        seed: Random seed for reproducible splits
    
    Example:
        >>> data = SARDataModule('patches.npy', val_fraction=0.1)
        >>> train_loader = data.train_dataloader()
        >>> val_loader = data.val_dataloader()
    """
    
    def __init__(
        self,
        patches_path: str,
        val_fraction: float = 0.1,
        batch_size: int = 16,
        num_workers: int = 4,
        augment_train: bool = True,
        seed: int = 42
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # TODO: Implement data loading and splitting
        #
        # 1. Load patches from file
        # 2. Split into train/val using seed
        # 3. Create SARPatchDataset instances
        #
        # print(f"Loading patches from {patches_path}")
        # all_patches = np.load(patches_path)
        # 
        # # Split
        # np.random.seed(seed)
        # indices = np.random.permutation(len(all_patches))
        # val_size = int(len(all_patches) * val_fraction)
        # 
        # self.train_patches = all_patches[indices[val_size:]]
        # self.val_patches = all_patches[indices[:val_size]]
        # 
        # # Create datasets
        # self.train_dataset = SARPatchDataset(self.train_patches, augment=augment_train)
        # self.val_dataset = SARPatchDataset(self.val_patches, augment=False)
        
        raise NotImplementedError("TODO: Implement data loading")
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        # TODO: Implement train dataloader
        #
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     drop_last=True,  # For stable BatchNorm
        # )
        
        raise NotImplementedError("TODO: Implement train dataloader")
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        # TODO: Implement val dataloader
        raise NotImplementedError("TODO: Implement val dataloader")
    
    def get_sample_batch(self, split: str = 'train') -> torch.Tensor:
        """Get a sample batch for visualization."""
        loader = self.train_dataloader() if split == 'train' else self.val_dataloader()
        return next(iter(loader))
    
    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset)
    
    @property
    def val_size(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset)


def test_datamodule():
    """Test data module."""
    print("Testing SARDataModule...")
    
    # Create test patches
    np.random.seed(42)
    test_patches = np.random.rand(100, 256, 256).astype(np.float32)
    np.save('test_patches.npy', test_patches)
    
    # Test data module
    data = SARDataModule(
        'test_patches.npy',
        val_fraction=0.2,
        batch_size=8,
        num_workers=0,
    )
    
    print(f"✓ Train size: {data.train_size}")
    print(f"✓ Val size: {data.val_size}")
    
    # Test loaders
    train_batch = data.get_sample_batch('train')
    val_batch = data.get_sample_batch('val')
    
    print(f"✓ Train batch shape: {train_batch.shape}")
    print(f"✓ Val batch shape: {val_batch.shape}")
    
    # Cleanup
    import os
    os.remove('test_patches.npy')
    
    print("All datamodule tests passed!")


if __name__ == "__main__":
    test_datamodule()
