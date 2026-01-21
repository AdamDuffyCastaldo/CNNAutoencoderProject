---
phase: 01-data-pipeline
plan: 01
subsystem: data
tags: [pytorch, dataset, dataloader, numpy, mmap, augmentation]

# Dependency graph
requires: []
provides:
  - SARPatchDataset for in-memory patch loading
  - LazyPatchDataset for memory-efficient 182GB multi-file loading
  - SARDataModule for train/val DataLoader creation
  - Preprocessing utilities (handle_invalid_values, from_db, compute_clip_bounds)
affects: [02-baseline-model, training, evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Memory mapping with np.load(mmap_mode='r') for large datasets
    - Binary search for O(log n) file lookup in multi-file datasets
    - Deterministic shuffling with seed for reproducibility

key-files:
  created: []
  modified:
    - src/data/preprocessing.py
    - src/data/dataset.py
    - src/data/datamodule.py

key-decisions:
  - "batch_size=8 default for 8GB VRAM constraint"
  - "num_workers=0 default for Windows compatibility"
  - "Always .copy() patches before augmentation for mmap compatibility"

patterns-established:
  - "Lazy loading: Use LazyPatchDataset with metadata.npy for large datasets"
  - "Train/val split: 90% train / 10% val with deterministic seed"
  - "Augmentation: horizontal flip, vertical flip, 90-degree rotations only"

# Metrics
duration: 7min
completed: 2026-01-21
---

# Phase 1 Plan 01: Data Pipeline Summary

**PyTorch DataLoader infrastructure delivering (8, 1, 256, 256) batches from 696,277 SAR patches to GPU with memory-efficient lazy loading**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-21T20:14:46Z
- **Completed:** 2026-01-21T20:21:16Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Completed preprocessing utility functions (handle_invalid_values, from_db, compute_clip_bounds)
- Implemented SARPatchDataset for in-memory loading with augmentation support
- Implemented LazyPatchDataset for memory-efficient loading of 182GB dataset across 44 .npy files
- Implemented SARDataModule supporting both lazy and in-memory modes
- Verified end-to-end: batches of shape (8, 1, 256, 256) delivered to GPU with values in [0,1]

## Task Commits

Each task was committed atomically:

1. **Task 1: Complete preprocessing.py stub functions** - `511ac5b` (feat)
2. **Task 2: Implement SARPatchDataset and LazyPatchDataset** - `93d5418` (feat)
3. **Task 3: Implement SARDataModule with lazy loading support** - `ebdb243` (feat)

## Files Created/Modified
- `src/data/preprocessing.py` - handle_invalid_values, from_db, compute_clip_bounds implementations
- `src/data/dataset.py` - SARPatchDataset, LazyPatchDataset, verify_patch_files
- `src/data/datamodule.py` - SARDataModule with lazy/in-memory modes, _LazySubsetDataset

## Decisions Made
- **batch_size=8**: Conservative default for 8GB VRAM constraint (RTX 3070)
- **num_workers=0**: Windows compatibility - avoids multiprocessing issues
- **Lazy loading as default**: Uses mmap to avoid loading 182GB into RAM
- **Binary search for file lookup**: O(log n) performance for locating patches in multi-file dataset
- **Always .copy() before augmentation**: Required for mmap arrays to avoid read-only errors

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect dB clamping in preprocess_sar_complete()**
- **Found during:** Task 1 (preprocessing tests)
- **Issue:** Line `image_db = np.maximum(image_db, noise_floor)` incorrectly clamped dB values to 1e-10, replacing all negative dB values (typical SAR range is -30 to +5 dB)
- **Fix:** Removed the incorrect clamping; dB values can be any real number after log transform
- **Files modified:** src/data/preprocessing.py
- **Verification:** Test preprocessing now passes; normalized values correctly in [0, 1]
- **Committed in:** 511ac5b (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Bug fix was essential for correct preprocessing. No scope creep.

## Issues Encountered
- Windows console encoding issue with Unicode checkmark characters in test output - resolved by using ASCII 'OK' instead
- This is cosmetic only, not a functional issue

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Data pipeline complete and verified
- Ready for Phase 2 (Baseline Model) - can create training loop using SARDataModule
- All 696,277 patches accessible via lazy loading without memory overflow
- Preprocessing parameters (vmin, vmax) accessible for inverse transform during evaluation

---
*Phase: 01-data-pipeline*
*Completed: 2026-01-21*
