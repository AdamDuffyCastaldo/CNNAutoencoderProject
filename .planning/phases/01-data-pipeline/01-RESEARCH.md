# Phase 1: Data Pipeline - Research

**Researched:** 2026-01-21
**Domain:** SAR Image Preprocessing, PyTorch Data Loading
**Confidence:** HIGH

## Summary

Phase 1 implements a complete data pipeline for SAR autoencoder training. The codebase already contains substantial working code in `preprocessing.py` (core preprocessing functions complete) and stub implementations in `dataset.py` and `datamodule.py`. The project has successfully processed 720,953 patches from 44 Sentinel-1 GeoTIFF files using the existing preprocessing functions.

The standard approach for SAR preprocessing is well-documented in the project's knowledge base (`05_SAR_PREPROCESSING.md`) and has been validated through the learning notebooks. The preprocessing pipeline converts linear intensity to dB scale, handles invalid values with noise floor substitution, clips dynamic range using percentile bounds, and normalizes to [0,1].

Key gaps are the Dataset and DataModule implementations, which exist as stubs with detailed comments showing the intended implementation. The patch extraction and preprocessing infrastructure is already working.

**Primary recommendation:** Implement the stub functions in `dataset.py` and `datamodule.py` following the commented-out reference implementations already present in those files.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rasterio | >=1.3.0 | GeoTIFF I/O for Sentinel-1 data | Industry standard for geospatial raster data, handles CRS/metadata |
| numpy | latest | Array operations, patch manipulation | Universal scientific computing |
| torch | >=2.0.0 | Dataset/DataLoader, tensor operations | PyTorch ecosystem standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| psutil | latest | Memory monitoring during processing | Large dataset handling |
| tqdm | >=4.65.0 | Progress bars for long operations | User feedback during preprocessing |
| gc | stdlib | Garbage collection for memory management | Processing large SAR images |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rasterio | xarray-sentinel | More features for SAFE format, but heavier dependency |
| numpy patches | torchvision transforms | Less control over SAR-specific filtering |
| Single .npy file | Multiple .npy files | Already implemented - multiple files work around memory limits |

**Installation:**
```bash
pip install rasterio>=1.3.0 numpy torch>=2.0.0 psutil tqdm
```

## Architecture Patterns

### Existing Project Structure
```
src/data/
  preprocessing.py    # WORKING: preprocess_sar_complete, extract_patches, analyze_sar_statistics
  dataset.py          # STUB: SARPatchDataset, SARImageDataset
  datamodule.py       # STUB: SARDataModule
src/utils/
  io.py               # WORKING: load_sar_image, find_all_sar_files, save/load_checkpoint
data/
  raw/                # 22 .SAFE folders with 46 TIFF files
  patches/            # 720,953 patches in 44 .npy files + metadata.npy
checkpoints/
  global_bounds.npy   # Preprocessing parameters: vmin=14.77, vmax=24.54 dB
```

### Pattern 1: Multi-File Lazy Loading Dataset
**What:** Load patches from multiple .npy files on-demand rather than loading all into memory
**When to use:** Dataset larger than RAM (current: 182.5 GB total)
**Example:**
```python
# Source: processingalldata.ipynb (existing implementation)
class LazyPatchDataset(Dataset):
    def __init__(self, metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.file_index = metadata['file_index']  # [(path, count), ...]
        self.shuffle_idx = np.load(metadata_path.parent / "shuffle_idx.npy")

        # Build cumsum for file lookup
        self.cumsum = [0]
        for _, count in self.file_index:
            self.cumsum.append(self.cumsum[-1] + count)

    def __getitem__(self, idx):
        real_idx = self.shuffle_idx[idx]
        # Find which file contains this index
        for i, (start, end) in enumerate(zip(self.cumsum[:-1], self.cumsum[1:])):
            if start <= real_idx < end:
                fpath, _ = self.file_index[i]
                local_idx = real_idx - start
                patch = np.load(fpath, mmap_mode='r')[local_idx]
                return torch.from_numpy(patch.copy()).unsqueeze(0).float()
```

### Pattern 2: In-Memory Dataset with Pre-loaded Patches
**What:** Load all patches into memory for faster training
**When to use:** Subset of data that fits in RAM, or GPU with enough VRAM
**Example:**
```python
# Source: dataset.py stub (reference implementation)
class SARPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, augment: bool = True):
        self.patches = patches.astype(np.float32)
        self.augment = augment

        # Validate
        assert patches.ndim == 3, f"Expected (N, H, W), got {patches.shape}"
        assert patches.min() >= 0 and patches.max() <= 1, "Values must be in [0,1]"

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].copy()
        if self.augment:
            patch = self._augment(patch)
        return torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)
```

### Anti-Patterns to Avoid
- **Loading full .npy files for each sample:** Use memory mapping (`mmap_mode='r'`) or keep file handles open
- **Per-image normalization:** Always use dataset-wide vmin/vmax from training set
- **Re-computing preprocessing for each epoch:** Preprocess once, save patches to disk

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GeoTIFF reading | Custom TIFF parser | rasterio | Handles CRS, metadata, chunked reading |
| dB conversion with invalid handling | Manual checks | Existing `preprocess_sar_complete` | Already validates and handles edge cases |
| Memory-efficient large array loading | Custom chunking | `np.load(mmap_mode='r')` | NumPy's memory mapping is optimized |
| Batch collation | Custom batch building | `DataLoader(collate_fn=None)` | Default collation handles tensors correctly |
| Random augmentation | Manual random calls | `torch.randint` / existing `_augment` | PyTorch RNG plays well with DataLoader workers |

**Key insight:** The preprocessing pipeline is already implemented and tested. The remaining work is implementing the Dataset/DataLoader wrapper around existing `.npy` patch files.

## Common Pitfalls

### Pitfall 1: Windows DataLoader num_workers Issues
**What goes wrong:** Setting `num_workers > 0` on Windows can cause crashes or slowdowns due to spawn-based multiprocessing
**Why it happens:** Windows uses spawn instead of fork for multiprocessing, requiring all code to be picklable and wrapped in `if __name__ == '__main__'`
**How to avoid:** Start with `num_workers=0` on Windows, test incrementally (try 2, then 4). Use `persistent_workers=True` if using workers.
**Warning signs:** Freezing at epoch start, "can't pickle" errors, memory explosion

### Pitfall 2: Memory Mapping with Augmentation
**What goes wrong:** Trying to modify memory-mapped arrays in-place causes errors or corruption
**Why it happens:** Memory-mapped arrays are read-only by default
**How to avoid:** Always `.copy()` before augmentation: `patch = data[idx].copy()`
**Warning signs:** "assignment destination is read-only" errors

### Pitfall 3: Inconsistent Preprocessing Parameters
**What goes wrong:** Using different vmin/vmax for training vs inference produces mismatched distributions
**Why it happens:** Computing percentiles per-image or forgetting to save parameters
**How to avoid:** Load from `checkpoints/global_bounds.npy` or `data/patches/metadata.npy`
**Warning signs:** Validation loss much higher than training, reconstructions look shifted

### Pitfall 4: BatchNorm with Small Batches
**What goes wrong:** BatchNorm statistics become unstable with batch_size < 8
**Why it happens:** Running mean/var estimated from too few samples
**How to avoid:** Keep `batch_size >= 8` or use `drop_last=True` in DataLoader
**Warning signs:** Loss variance increases at end of epoch, inference differs from training

### Pitfall 5: Corrupted .npy Files from Interrupted Saves
**What goes wrong:** Some patch files are truncated (as seen in notebook errors)
**Why it happens:** Process killed or disk full during np.save
**How to avoid:** Verify files can be loaded before training, delete and regenerate corrupted files
**Warning signs:** "could only read 0 elements" errors, file size doesn't match expected

## Code Examples

Verified patterns from existing codebase:

### Loading SAR Image with Rasterio
```python
# Source: src/utils/io.py (existing, working)
def load_sar_image(tiff_path):
    """Load Sentinel-1 GeoTIFF."""
    with rasterio.open(tiff_path) as src:
        image = src.read(1).astype(np.float32)
    return image
```

### Complete Preprocessing Pipeline
```python
# Source: src/data/preprocessing.py (existing, working)
def preprocess_sar_complete(image, vmin=None, vmax=None, clip_percentiles=(1, 99)):
    """Complete preprocessing: invalid handling -> dB -> clip -> normalize"""
    # Step 1: Handle invalid values
    invalid_mask = (image <= 0) | np.isnan(image) | np.isinf(image)
    image_clean = np.copy(image)
    noise_floor = 1e-10
    image_clean = np.where(image <= 0, noise_floor, image_clean)
    image_clean = np.where(np.isnan(image_clean), noise_floor, image_clean)
    image_clean = np.where(np.isinf(image_clean), noise_floor, image_clean)

    # Step 2: Convert to dB
    image_db = 10 * np.log10(image_clean)

    # Step 3: Determine clip bounds (from training set, not per-image)
    if vmin is None or vmax is None:
        vmin = np.percentile(image_db, clip_percentiles[0])
        vmax = np.percentile(image_db, clip_percentiles[1])

    # Step 4: Clip and normalize to [0, 1]
    image_clipped = np.clip(image_db, vmin, vmax)
    normalized = (image_clipped - vmin) / (vmax - vmin)

    params = {'vmin': vmin, 'vmax': vmax, 'invalid_mask': invalid_mask}
    return normalized, params
```

### Patch Extraction with Quality Filtering
```python
# Source: src/data/preprocessing.py (existing, working)
def extract_patches(image, patch_size=256, stride=128, min_valid=0.9):
    """Extract patches, filter by quality (not too saturated)."""
    patches = []
    positions = []
    h, w = image.shape

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            # Check for valid pixels (not at clip boundaries)
            valid_frac = np.mean((patch > 0.01) & (patch < 0.99))
            if valid_frac >= min_valid:
                patches.append(patch)
                positions.append((i, j))

    return np.array(patches), positions
```

### SAR-Safe Augmentation
```python
# Source: dataset.py stub (to be implemented)
def _augment(self, patch: np.ndarray) -> np.ndarray:
    """SAR-safe augmentations: flips and 90-degree rotations only."""
    # Horizontal flip (50% chance)
    if random.random() > 0.5:
        patch = np.fliplr(patch).copy()

    # Vertical flip (50% chance) - valid for SAR
    if random.random() > 0.5:
        patch = np.flipud(patch).copy()

    # Random 90-degree rotation
    k = random.randint(0, 3)
    if k > 0:
        patch = np.rot90(patch, k).copy()

    return patch
```

### DataLoader Configuration for Training
```python
# Source: datamodule.py stub (to be implemented)
def train_dataloader(self) -> DataLoader:
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,  # Start with 0 on Windows
        pin_memory=True,               # Faster GPU transfer
        drop_last=True,                # Stable BatchNorm
        persistent_workers=(self.num_workers > 0)  # Avoid worker respawn overhead
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Load all patches into one array | Multi-file with lazy loading | Project current | Handles 182GB dataset |
| Per-image normalization | Dataset-wide normalization | Standard practice | Consistent distributions |
| `num_workers=4` default | Start with `num_workers=0` on Windows | PyTorch forums | Avoids Windows multiprocessing issues |

**Deprecated/outdated:**
- Single `all_patches.npy` file: Disk space and memory issues at scale

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal num_workers for Windows**
   - What we know: 0 is safe, higher may help
   - What's unclear: Best value for this specific hardware/data
   - Recommendation: Test incrementally during Phase 1 implementation

2. **VH vs VV Polarization Handling**
   - What we know: Both polarizations are extracted and processed
   - What's unclear: Should they be used separately or combined?
   - Recommendation: Treat as separate training data for now, experiment with fusion later

3. **Corrupted Patch Files**
   - What we know: At least one file (`s1c-iw-grd-vh-20260117t225204*`) failed to write completely
   - What's unclear: Full extent of corruption
   - Recommendation: Add verification step to data loading that validates file integrity

## Sources

### Primary (HIGH confidence)
- `src/data/preprocessing.py` - Working implementation with tests
- `src/data/dataset.py` - Stub with detailed reference implementation
- `src/data/datamodule.py` - Stub with detailed reference implementation
- `src/utils/io.py` - Working rasterio integration
- `learningnotebooks/phase4_sar_codec/processingalldata.ipynb` - End-to-end validation
- `.planning/knowledge/05_SAR_PREPROCESSING.md` - Domain knowledge
- `.planning/research/PITFALLS.md` - Catalogued failure modes

### Secondary (MEDIUM confidence)
- [PyTorch DataLoader Documentation](https://docs.pytorch.org/docs/stable/data.html) - Official guidance
- [PyTorch Custom Datasets Tutorial](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) - Pattern reference
- [Rasterio Reading Datasets](https://rasterio.readthedocs.io/en/stable/topics/reading.html) - GeoTIFF patterns

### Tertiary (LOW confidence)
- WebSearch results for Windows DataLoader performance - Community experience varies

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Project already using these libraries successfully
- Architecture: HIGH - Patterns validated in existing notebooks
- Pitfalls: HIGH - Documented in project knowledge base and observed in notebooks

**Research date:** 2026-01-21
**Valid until:** 2026-02-21 (stable domain, existing infrastructure)

## Implementation Gap Analysis

### Already Implemented (WORKING)
| Component | File | Status |
|-----------|------|--------|
| GeoTIFF loading | `utils/io.py:load_sar_image` | Complete |
| File discovery | `utils/io.py:find_all_sar_files` | Complete |
| Invalid value handling | `preprocessing.py:preprocess_sar_complete` | Complete |
| dB conversion | `preprocessing.py:preprocess_sar_complete` | Complete |
| Percentile clipping | `preprocessing.py:preprocess_sar_complete` | Complete |
| Normalization | `preprocessing.py:preprocess_sar_complete` | Complete |
| Patch extraction | `preprocessing.py:extract_patches` | Complete |
| Quality filtering | `preprocessing.py:extract_patches` | Complete |
| Global bounds computation | `processingalldata.ipynb` | Complete (saved) |
| Patch files | `data/patches/*.npy` | 720,953 patches |

### Needs Implementation (STUBS)
| Component | File | Reference Available |
|-----------|------|---------------------|
| SARPatchDataset.__init__ | `dataset.py` | Commented code present |
| SARPatchDataset.__len__ | `dataset.py` | Trivial |
| SARPatchDataset.__getitem__ | `dataset.py` | Commented code present |
| SARPatchDataset._augment | `dataset.py` | Commented code present |
| SARDataModule.__init__ | `datamodule.py` | Commented code present |
| SARDataModule.train_dataloader | `datamodule.py` | Commented code present |
| SARDataModule.val_dataloader | `datamodule.py` | Trivial (copy of train) |

### Needs Implementation (NEW - for multi-file support)
| Component | Purpose | Complexity |
|-----------|---------|------------|
| LazyPatchDataset | Load from multiple .npy files | Medium |
| File integrity check | Detect corrupted .npy files | Low |
| Subset sampler | Sample manageable training subset | Low |

### Stub Functions to Complete (preprocessing.py)
| Function | Status | Notes |
|----------|--------|-------|
| `handle_invalid_values` | STUB | Logic exists in `preprocess_sar_complete` |
| `from_db` | STUB | Simple: `10 ** (db / 10)` |
| `compute_clip_bounds` | STUB | Logic exists in `preprocess_sar_complete` |

**Summary:** ~80% of Phase 1 functionality is already implemented. Primary work is completing Dataset/DataLoader stubs and adding multi-file support for the existing 182GB patch dataset.
