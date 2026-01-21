# External Integrations

**Analysis Date:** 2026-01-21

## APIs & External Services

**Geospatial Data:**
- Sentinel-1 GeoTIFF files - SAR image data source
  - Format: .tiff files packaged in .SAFE directories
  - SDK/Client: rasterio>=1.3.0
  - Loading: `src/utils/io.py` - `load_sar_image()` and `find_all_sar_files()`

## Data Storage

**Databases:**
- None detected - Project uses local filesystem only

**File Storage:**
- Local filesystem only
  - Raw SAR data: Configured path in YAML (raw .SAFE directories)
  - Processed patches: `data/patches/patches.npy` (numpy arrays)
  - Checkpoints: `checkpoints/` directory
  - Logs: `runs/` directory

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- None detected - No authentication required for local development
- Sentinel-1 data requires ESA Copernicus access credentials for actual download (not integrated)

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, Rollbar, etc.)

**Logs:**
- TensorBoard logging: `src/training/trainer.py` uses `torch.utils.tensorboard.SummaryWriter`
  - Log directory: `runs/` (configurable in YAML)
  - Metrics logged: training/validation loss, SSIM, PSNR, learning rate
  - Image logging every N epochs (configurable: `log_images_every`)

**Model Monitoring:**
- TensorBoard dashboards for training curves
- Checkpoint-based model tracking (best and latest models)

## CI/CD & Deployment

**Hosting:**
- Not deployed - Research/learning project

**CI Pipeline:**
- None detected

## Environment Configuration

**Required env vars:**
- None detected - All configuration via `configs/default.yaml`

**Secrets location:**
- No secrets in codebase
- Sentinel-1 ESA credentials would be user-provided if downloading real data

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Data Pipeline Dependencies

**Input Data Sources:**
- Sentinel-1 L2A GeoTIFF files (user-provided)
  - Location: configurable raw data directory
  - Resolution: Typically 10m × 10m pixels
  - Polarizations: VV and VH channels

**Preprocessing Pipeline:**
- `src/data/preprocessing.py` - SAR-specific preprocessing
  - Invalid value handling
  - dB scale conversion
  - Percentile/sigma-based clipping
  - Normalization to [0, 1] range
  - Parameters configured in YAML

**Training Data:**
- Patches extracted via `src/data/preprocessing.py` - `extract_patches()`
  - Patch size: 256×256 (configurable)
  - Format: `.npy` (NumPy arrays)
  - Split: Training/validation split ratio in YAML

## Checkpoint & Model Management

**Model Persistence:**
- PyTorch `.pth` format checkpoint files
  - Save location: `checkpoints/` directory
  - Contents: model state dict, optimizer state, epoch, config
  - Functions: `src/utils/io.py` - `save_checkpoint()`, `load_checkpoint()`

**Inference Artifacts:**
- Compressed image data (.npy) from `src/inference/compressor.py`
- Metrics JSON from evaluation runs

---

*Integration audit: 2026-01-21*
