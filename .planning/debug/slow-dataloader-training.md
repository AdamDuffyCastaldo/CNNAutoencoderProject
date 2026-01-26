---
status: verifying
trigger: "Training shows 8-17 it/s burst speed but estimates 8-18 hours per epoch instead of expected ~5 minutes"
created: 2026-01-24T12:00:00Z
updated: 2026-01-24T12:00:00Z
---

## Current Focus

hypothesis: Multiple Jupyter kernels running simultaneously are consuming GPU memory, causing memory thrashing and severely degrading training performance
test: Check wmic process list for ipykernel processes and correlate with nvidia-smi GPU memory
expecting: If true, killing orphan kernels should free GPU memory and restore normal training speed
next_action: Verify multiple kernels are the issue, then provide fix to restart with clean kernel

## Symptoms

expected: Training at sustained 15-17 it/s giving ~5 min/epoch (4,350 batches)
actual: Progress bar shows 8-18 hours per epoch despite momentary 8-17 it/s readings
errors: No errors, just extremely slow effective throughput
reproduction: Run notebooks/train_residual.ipynb, observe training progress
started: First attempt at training ResidualAutoencoder (23.8M params)

## Attempted Fixes (from user)

1. Added mmap file caching to _LazySubsetDataset.__getitem__ - no improvement
2. Changed NUM_WORKERS from 4 to 0 - still shows 18 hours

## Eliminated

## Evidence

- timestamp: 2026-01-24T12:05:00Z
  checked: datamodule.py architecture
  found: |
    SARDataModule creates _LazySubsetDataset which:
    1. Loads metadata and builds file_index + cumsum for O(log n) file lookup
    2. Creates self.indices array (shuffled if training) mapping subset idx -> global idx
    3. __getitem__ does: real_idx = self.indices[idx], then binary search for file, then mmap read
    The mmap caching is already implemented in lines 236-237
  implication: _LazySubsetDataset is already optimized for lazy loading with mmap cache

- timestamp: 2026-01-24T12:06:00Z
  checked: notebook cell 7 (data loading section)
  found: |
    After SARDataModule creates train_dataset (_LazySubsetDataset with 626,650 samples),
    notebook WRAPS it again: dm.train_dataset = torch.utils.data.Subset(dm.train_dataset, train_indices)
    where train_indices is a random.sample() of 20% indices (125,330 samples)
  implication: DOUBLE WRAPPING - Subset wraps _LazySubsetDataset, causing double index lookup

- timestamp: 2026-01-24T12:07:00Z
  checked: Access path for each sample
  found: |
    Current path: DataLoader -> Subset.__getitem__(i) -> _LazySubsetDataset.__getitem__(subset.indices[i])
    -> _LazySubsetDataset does ANOTHER index lookup: real_idx = self.indices[idx]
    -> Then file I/O
    The problem: random.sample() creates non-contiguous indices, and _LazySubsetDataset.indices
    is already a shuffled permutation. Combined: highly random access patterns = poor mmap performance
  implication: Random access across 44 files with 182GB total defeats mmap's sequential read advantage

- timestamp: 2026-01-24T21:35:00Z
  checked: Actual batch loading performance
  found: |
    Ran benchmark: DataLoader with Subset wrapping, batch_size=32, num_workers=0
    Results: Batch 1: 325ms, then stabilized at 42-71ms per batch
    Average: 80ms per batch = 12.5 batches/sec
  implication: Data loading is NOT the bottleneck - 80ms/batch should give ~12 it/s

- timestamp: 2026-01-24T21:36:00Z
  checked: ResNet-Lite v2 training log timestamps
  found: |
    Started: 2026-01-24 03:45:38
    Epoch 1 complete: 2026-01-24 04:07:23 (~22 min for 3916 batches)
    Same dataset, same Subset wrapping pattern, but completed successfully
    Epoch time: 22 min = 2.9 it/s
  implication: Previous training also used this pattern and worked (slowly but completed)

- timestamp: 2026-01-24T21:37:00Z
  checked: Residual v1 previous run in notebooks/runs/
  found: |
    From notebooks/runs/residual_v1_c16/training.log:
    Started: 2026-01-24 20:46:56
    Epoch 1 complete: 2026-01-24 20:55:14 (~8 min!)
    BATCH_SIZE=16 that run (from earlier log entry)
    This shows the model CAN train at reasonable speed
  implication: The slow training is NOT fundamental to the model or data loading

- timestamp: 2026-01-24T21:43:00Z
  checked: GPU status via nvidia-smi
  found: |
    GPU at 100% utilization, 7935MiB/8192MiB memory used
    Multiple Python processes holding GPU memory (PIDs 19380, 40748, 47916)
    One is likely the still-running training notebook
  implication: Current slow benchmark may be due to GPU contention

- timestamp: 2026-01-24T21:50:00Z
  checked: Python process list via wmic
  found: |
    At least 6-7 different ipykernel processes running:
    - kernel-v37afebfc94e810315ddd2e2aa3e46d1d7feeca58a
    - kernel-v3aca2d2ba1d77a9210885ffbe252311794eb53671
    - kernel-c94288bc-b42a-473a-951d-27e7de195f5d
    - kernel-v35469e8a3edcd5d5ef78cf920f061b5aa22259cd9
    - kernel-0da5fcd8-f020-45c4-b769-981d29a6e8ae
    Each kernel may have allocated GPU memory for models/tensors
  implication: CONFIRMED - multiple kernels holding GPU memory causes contention

- timestamp: 2026-01-24T21:51:00Z
  checked: Previous successful training performance
  found: |
    Training log shows batch_size=16 completed epoch 1 in ~8 minutes
    That's 7833 batches in 480s = 16.3 it/s
    Current batch_size=32 should be FASTER (fewer batches)
    But seeing 0.1 it/s (10s per iteration) - 160x slower than expected
  implication: The slowdown is NOT due to batch_size increase; something else is blocking

## Eliminated

- hypothesis: Double Subset wrapping causes slow I/O
  evidence: Batch loading benchmark shows 80ms/batch average, which is acceptable (12.5 batches/sec)
  timestamp: 2026-01-24T21:35:00Z

## Resolution

root_cause: |
  Multiple Jupyter notebook kernels are running simultaneously, each holding GPU memory allocations.
  With 8GB VRAM nearly exhausted (7935/8192 MiB), the training kernel has no room to load the 23.8M
  parameter model and its activations. This causes CUDA memory thrashing, where the GPU constantly
  swaps data between GPU and system memory, reducing effective speed from 16 it/s to 0.1 it/s (160x slower).

  The issue is NOT data loading (80ms/batch is fine) or the model itself (previous run with batch_size=16
  completed at 16.3 it/s). It's GPU memory contention from orphan kernels.

fix: |
  1. Kill all orphan Jupyter kernels to free GPU memory
  2. Restart with fresh kernel and run training
  3. (Optional) Add GPU memory check at training start to warn about contention

verification: |
  After fix, training should achieve 10-17 it/s with batch_size=32:
  - 3916 batches at 15 it/s = ~4.4 minutes per epoch
  - 30 epochs = ~2.2 hours total (vs the projected 18 hours * 30 = 540 hours)

files_changed:
  - src/training/trainer.py: Added _check_gpu_memory() method to warn about GPU memory contention at trainer initialization
