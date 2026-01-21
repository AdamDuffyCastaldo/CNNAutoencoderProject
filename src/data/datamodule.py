"""
Data Module for SAR Autoencoder

Manages train/validation splits and DataLoader creation.

References:
    - Day 2, Section 2.6 of the learning guide
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from .dataset import SARPatchDataset, LazyPatchDataset


class SARDataModule:
    """
    Data module for managing train/validation data.

    Handles:
    - Loading patches from disk (single file or multi-file via metadata)
    - Splitting into train/validation
    - Creating DataLoaders

    Supports two modes:
    - lazy=True (default): Memory-efficient loading from multiple .npy files
    - lazy=False: Load all patches into memory

    Args:
        patches_path: Path to .npy file or metadata.npy for lazy loading
        val_fraction: Fraction of data for validation
        batch_size: Batch size for DataLoaders (default: 8 for 8GB VRAM)
        num_workers: Number of DataLoader workers (default: 0 for Windows)
        augment_train: Whether to augment training data
        seed: Random seed for reproducible splits
        lazy: Use LazyPatchDataset (default: True)
        max_samples: Optional limit on total samples (for debugging)

    Example:
        >>> # Lazy loading from multi-file dataset
        >>> data = SARDataModule('data/patches/metadata.npy', lazy=True)
        >>> train_loader = data.train_dataloader()
        >>> val_loader = data.val_dataloader()

        >>> # In-memory loading from single file
        >>> data = SARDataModule('patches.npy', lazy=False)
    """

    def __init__(
        self,
        patches_path: str,
        val_fraction: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 0,
        augment_train: bool = True,
        seed: int = 42,
        lazy: bool = True,
        max_samples: Optional[int] = None
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self._metadata = None

        patches_path = Path(patches_path)

        # Determine loading mode
        is_metadata = patches_path.name == 'metadata.npy'

        if lazy and is_metadata:
            # Lazy loading from multiple .npy files
            print(f"Loading metadata from {patches_path}")
            self._metadata = np.load(patches_path, allow_pickle=True).item()
            total = sum(count for _, count in self._metadata['file_index'])

            if max_samples is not None:
                total = min(total, max_samples)

            # Split by index ranges (first 90% train, last 10% val)
            val_size = int(total * val_fraction)
            train_size = total - val_size

            print(f"Total patches: {total}")
            print(f"Train: {train_size}, Val: {val_size}")

            # Create train dataset (uses internal shuffling)
            self.train_dataset = _LazySubsetDataset(
                metadata_path=str(patches_path),
                start_idx=0,
                end_idx=train_size,
                augment=augment_train,
                shuffle_seed=seed
            )

            # Create val dataset (no shuffle for reproducible validation)
            self.val_dataset = _LazySubsetDataset(
                metadata_path=str(patches_path),
                start_idx=train_size,
                end_idx=total,
                augment=False,
                shuffle_seed=None  # No shuffle for validation
            )

        else:
            # In-memory loading from single .npy file
            print(f"Loading patches from {patches_path}")
            all_patches = np.load(patches_path)

            if max_samples is not None:
                all_patches = all_patches[:max_samples]

            # Split
            np.random.seed(seed)
            indices = np.random.permutation(len(all_patches))
            val_size = int(len(all_patches) * val_fraction)

            train_patches = all_patches[indices[val_size:]]
            val_patches = all_patches[indices[:val_size]]

            print(f"Total patches: {len(all_patches)}")
            print(f"Train: {len(train_patches)}, Val: {len(val_patches)}")

            # Create datasets
            self.train_dataset = SARPatchDataset(train_patches, augment=augment_train)
            self.val_dataset = SARPatchDataset(val_patches, augment=False)

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # For stable BatchNorm
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0)
        )

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

    @property
    def preprocessing_params(self) -> Optional[Dict]:
        """Return preprocessing parameters for inverse transform."""
        if self._metadata is not None and 'vmin' in self._metadata and 'vmax' in self._metadata:
            return {'vmin': self._metadata['vmin'], 'vmax': self._metadata['vmax']}
        return None


class _LazySubsetDataset(torch.utils.data.Dataset):
    """
    Subset of LazyPatchDataset for train/val splitting.

    Instead of loading all indices, this directly maps to a contiguous
    range within the full dataset.
    """

    def __init__(
        self,
        metadata_path: str,
        start_idx: int,
        end_idx: int,
        augment: bool = True,
        shuffle_seed: Optional[int] = None
    ):
        import random

        metadata_path = Path(metadata_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.file_index = metadata['file_index']
        self.augment = augment
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.length = end_idx - start_idx

        # Build cumulative sum for O(log n) file lookup
        self.cumsum = [0]
        for _, count in self.file_index:
            self.cumsum.append(self.cumsum[-1] + count)
        self.total = self.cumsum[-1]

        # Create mapping from subset index to global index
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            all_indices = rng.permutation(self.total)
            self.indices = all_indices[start_idx:end_idx]
        else:
            self.indices = np.arange(start_idx, end_idx)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        import random

        real_idx = self.indices[idx]

        # Binary search for file
        file_idx = np.searchsorted(self.cumsum[1:], real_idx, side='right')
        local_idx = real_idx - self.cumsum[file_idx]

        fpath, _ = self.file_index[file_idx]
        patch = np.load(fpath, mmap_mode='r')[local_idx].copy()  # .copy() critical!

        if self.augment:
            patch = self._augment(patch)

        return torch.from_numpy(patch).unsqueeze(0).float()

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        import random

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


def test_datamodule():
    """Test data module."""
    print("Testing SARDataModule...")

    # Create test patches
    np.random.seed(42)
    test_patches = np.random.rand(100, 256, 256).astype(np.float32)
    np.save('test_patches.npy', test_patches)

    # Test data module (in-memory mode)
    data = SARDataModule(
        'test_patches.npy',
        val_fraction=0.2,
        batch_size=8,
        num_workers=0,
        lazy=False,  # Test in-memory mode
    )

    print(f"OK: Train size: {data.train_size}")
    print(f"OK: Val size: {data.val_size}")

    # Test loaders
    train_batch = data.get_sample_batch('train')
    val_batch = data.get_sample_batch('val')

    print(f"OK: Train batch shape: {train_batch.shape}")
    print(f"OK: Val batch shape: {val_batch.shape}")

    # Cleanup
    import os
    os.remove('test_patches.npy')

    print("All datamodule tests passed!")


if __name__ == "__main__":
    test_datamodule()
