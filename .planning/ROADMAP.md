# Project Roadmap

## Overview

This roadmap guides the implementation of a CNN-based autoencoder for compressing Sentinel-1 SAR satellite imagery. The project follows a 7-phase structure derived from the requirements, progressing from data infrastructure through baseline model, evaluation framework, architecture enhancements, full-image inference, comprehensive comparison study, and production deployment. Each phase builds on the previous, with clear success criteria that can be verified before proceeding.

**Target outcome:** A trained compression system achieving >30 dB PSNR at 16x compression while preserving SAR-specific characteristics (speckle texture, edge detail), with complete rate-distortion analysis across multiple architecture variants.

---

## Phase 1: Data Pipeline

**Goal:** Establish a complete, verified preprocessing pipeline that transforms raw Sentinel-1 GeoTIFF images into normalized 256x256 patches suitable for neural network training.

**Dependencies:** None

**Estimated Complexity:** Medium

**Plans:** 1 plan

Plans:
- [x] 01-01-PLAN.md - Complete PyTorch data loading infrastructure (SARPatchDataset, LazyPatchDataset, SARDataModule)

### Success Criteria

1. Raw GeoTIFF files load successfully with rasterio and invalid values (zeros, NaN, negative) are handled without errors
2. Preprocessing produces normalized [0,1] patches where visual inspection confirms SAR structures are visible and not dominated by outliers
3. Patch extraction generates a training set of 1000+ quality-filtered patches with <1% invalid pixels
4. Preprocessing parameters (vmin, vmax, noise_floor) are saved alongside data and can reproduce identical normalization
5. DataLoader delivers batches of shape (N, 1, 256, 256) to GPU without memory errors at batch_size=8

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR1.1 | Load Sentinel-1 GeoTIFF images using rasterio |
| FR1.2 | Convert linear intensity to dB scale |
| FR1.3 | Handle invalid values with noise floor substitution |
| FR1.4 | Clip dynamic range using percentile-based bounds |
| FR1.5 | Normalize to [0,1] range using training-set bounds |
| FR1.6 | Extract 256x256 patches with configurable overlap |
| FR1.7 | Filter patches by quality (variance, invalid ratio) |
| FR1.8 | Save preprocessing parameters with checkpoint |
| FR1.9 | Support data augmentation (flip, crop only) |
| FR1.10 | Create PyTorch Dataset and DataLoader |

### Deliverables

- `src/data/preprocessing.py` - Complete implementation (partially exists as stubs)
- `src/data/dataset.py` - SARPatchDataset with augmentation support
- `src/data/datamodule.py` - DataModule for train/val/test splits
- Saved preprocessing artifacts (parameters JSON, processed patches)
- Verification notebook or script demonstrating pipeline correctness

### Key Tasks (High-Level)

- [ ] Implement remaining stub functions in preprocessing.py (handle_invalid_values, from_db, compute_clip_bounds)
- [ ] Implement GeoTIFF loading with rasterio
- [ ] Implement SARPatchDataset class with augmentation
- [ ] Implement geographic train/val/test splitting strategy
- [ ] Create data preparation script to process raw Sentinel-1 scenes
- [ ] Verify roundtrip preprocessing (original -> normalized -> reconstructed)

---

## Phase 2: Baseline Model

**Goal:** Implement and train a working plain 4-layer encoder-decoder autoencoder that achieves convergent training and produces recognizable reconstructions.

**Dependencies:** Phase 1 (Data Pipeline)

**Estimated Complexity:** Medium

**Plans:** 4 plans

Plans:
- [x] 02-01-PLAN.md - Implement ConvBlock, DeconvBlock, SSIMLoss, CombinedLoss (foundational components)
- [x] 02-02-PLAN.md - Implement SAREncoder, SARDecoder, SARAutoencoder (model architecture)
- [x] 02-03-PLAN.md - Implement Trainer with TensorBoard, checkpointing, early stopping (training infrastructure)
- [x] 02-04-PLAN.md - Create training script and train baseline model (integration and training)

### Success Criteria

1. Model forward pass completes without error: input (N,1,256,256) produces output (N,1,256,256) and latent (N,C,16,16)
2. Training loss decreases over epochs (no NaN, no explosion) and validation loss follows training loss
3. Checkpoints save and load correctly, enabling training resumption
4. TensorBoard shows logged scalars (loss, PSNR, SSIM) and sample reconstruction images
5. Baseline model at 16x compression (16 latent channels) achieves PSNR >25 dB on validation set

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR2.1 | Implement ConvBlock: Conv2d + BatchNorm + LeakyReLU(0.2) |
| FR2.2 | Implement DeconvBlock: ConvTranspose2d + BatchNorm + ReLU |
| FR2.4 | Implement 4-layer encoder: 256x256x1 -> 16x16xC |
| FR2.5 | Implement 4-layer decoder: 16x16xC -> 256x256x1 |
| FR2.6 | Use 5x5 kernels for improved receptive field |
| FR2.7 | Configurable latent channels for compression ratio control |
| FR2.8 | Sigmoid output activation for bounded [0,1] output |
| FR2.9 | No activation on final encoder layer |
| FR3.1 | Combined loss function: configurable MSE + SSIM weights |
| FR3.2 | Default loss weights: 0.5 MSE + 0.5 SSIM |
| FR3.3 | Adam optimizer with configurable learning rate |
| FR3.4 | Learning rate scheduler (ReduceLROnPlateau) |
| FR3.5 | Gradient clipping (max norm 1.0) |
| FR3.6 | Training loop with configurable epochs, early stopping |
| FR3.7 | Checkpoint saving (model, optimizer, config, preprocessing) |
| FR3.8 | Best model tracking based on validation loss |
| FR3.9 | Resume training from checkpoint |
| FR3.10 | TensorBoard logging (scalars, images) |
| FR4.1 | PSNR metric computation |
| FR4.2 | SSIM metric computation |

### Deliverables

- `src/models/blocks.py` - ConvBlock, DeconvBlock implementations
- `src/models/encoder.py` - 4-layer encoder
- `src/models/decoder.py` - 4-layer decoder
- `src/models/autoencoder.py` - Complete autoencoder wrapper
- `src/losses/combined.py` - MSE + SSIM combined loss
- `src/training/trainer.py` - Complete training loop
- Trained baseline checkpoint at 16x compression
- TensorBoard logs demonstrating training convergence

### Key Tasks (High-Level)

- [x] Implement ConvBlock and DeconvBlock in blocks.py
- [x] Implement Encoder class with configurable latent channels
- [x] Implement Decoder class mirroring encoder
- [x] Implement SARAutoencoder wrapper combining encoder/decoder
- [x] Implement combined loss function (MSE + SSIM)
- [x] Implement Trainer class with all training features
- [x] Train baseline model and verify convergence
- [x] Validate checkpoint save/load functionality

---

## Phase 3: SAR Evaluation Framework

**Goal:** Implement SAR-specific quality metrics (ENL ratio, EPI) and evaluation tools that enable informed architecture decisions beyond standard PSNR/SSIM. Establish traditional codec baselines (JPEG-2000, etc.) for meaningful comparison.

**Dependencies:** Phase 2 (Baseline Model)

**Estimated Complexity:** Medium

**Plans:** 3 plans

Plans:
- [x] 03-01-PLAN.md - Implement core SAR metrics (ENL ratio, EPI, MS-SSIM, histogram similarity)
- [x] 03-02-PLAN.md - Implement traditional codec baselines (JPEG-2000, JPEG with calibration)
- [x] 03-03-PLAN.md - Build evaluation infrastructure (Evaluator updates, visualizer enhancements, CLI script)

### Success Criteria

1. ENL ratio computes correctly on homogeneous regions, returning values between 0.5-2.0 for typical SAR images
2. Edge Preservation Index (EPI) computes correctly, returning values near 1.0 for good reconstructions
3. Evaluation script generates a complete metrics report (PSNR, SSIM, MS-SSIM, ENL ratio, EPI) for any trained model
4. Visual comparison tool produces side-by-side images (original, reconstructed, difference) that reveal quality issues
5. Rate-distortion curve generation works for multiple checkpoints at different compression ratios
6. Traditional codec baselines (JPEG-2000 at minimum) evaluated at matching compression ratios (8x, 16x, 32x) with same metrics

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR4.3 | MS-SSIM metric computation |
| FR4.4 | Compression ratio calculation |
| FR4.5 | Bits per pixel (BPP) calculation |
| FR4.6 | ENL ratio metric for speckle preservation |
| FR4.7 | Edge Preservation Index (EPI) metric |
| FR4.8 | Visual comparison generation |
| FR4.9 | Rate-distortion curve generation |
| FR4.10 | Batch evaluation across test set with statistics |
| FR4.11 | Traditional codec baseline comparison (JPEG-2000, optionally JPEG, WebP) |

### Deliverables

- `src/evaluation/metrics.py` - Complete ENL, EPI, histogram similarity implementations
- `src/evaluation/evaluator.py` - Batch evaluation with statistics
- `src/evaluation/visualizer.py` - Comparison image generation
- `src/evaluation/codec_baselines.py` - Traditional codec compression/evaluation
- Evaluation script for running complete assessment on checkpoints
- Baseline model evaluation report with all metrics
- Traditional codec benchmark results at 8x, 16x, 32x compression

### Key Tasks (High-Level)

- [x] Implement ENL ratio computation (requires homogeneous region detection)
- [x] Implement Edge Preservation Index (EPI)
- [x] Implement histogram similarity metric
- [x] Implement MS-SSIM using pytorch-msssim
- [x] Implement compression ratio and BPP calculation
- [x] Build Evaluator class for batch evaluation with statistics
- [x] Build Visualizer class for comparison images
- [x] Implement JPEG-2000 baseline using OpenCV or Pillow (openjpeg backend)
- [x] Optionally implement JPEG, WebP baselines for additional comparison
- [x] Evaluate traditional codecs on test set with same metrics as autoencoder
- [x] Create evaluation script and run on baseline model

---

## Phase 4: Architecture Enhancement

**Goal:** Implement residual blocks and CBAM attention, then train enhanced architecture variants to demonstrate quality improvements over baseline.

**Dependencies:** Phase 3 (SAR Evaluation Framework)

**Estimated Complexity:** High

**Plans:** 6 plans

Plans:
- [ ] 04-01-PLAN.md - Implement PreActResidualBlock and CBAM attention modules (building blocks)
- [ ] 04-02-PLAN.md - Create ResidualAutoencoder (Variant B architecture)
- [ ] 04-03-PLAN.md - Create AttentionAutoencoder (Variant C architecture)
- [ ] 04-04-PLAN.md - Train Variant B (Residual) at 16x compression
- [ ] 04-05-PLAN.md - Train Variant C (Attention) at 16x compression
- [ ] 04-06-PLAN.md - Compare architectures and assess phase success

### Success Criteria

1. ResidualBlock forward pass preserves spatial dimensions and enables skip connections
2. CBAM attention module applies channel and spatial attention without errors
3. Residual architecture (Variant B) achieves PSNR at least 1.5 dB higher than baseline at same compression ratio
4. Residual+CBAM architecture (Variant C) achieves PSNR at least 0.5 dB higher than Variant B
5. All architecture variants maintain ENL ratio between 0.8-1.2 (no over-smoothing)

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR2.3 | Implement ResidualBlock with projection shortcut |
| FR2.10 | Implement CBAM attention module |
| FR6.2 | Train residual architecture at multiple compression ratios |
| FR6.3 | Train residual+CBAM architecture at multiple compression ratios |

### Deliverables

- `src/models/blocks.py` - PreActResidualBlock, CBAM implementations
- `src/models/residual_autoencoder.py` - Variant B architecture
- `src/models/attention_autoencoder.py` - Variant C architecture
- Trained checkpoints for Variant B (Residual) at 16x compression
- Trained checkpoints for Variant C (Res+CBAM) at 16x compression
- Comparison metrics showing improvement over baseline

### Key Tasks (High-Level)

- [ ] Implement PreActResidualBlock with pre-activation ordering
- [ ] Implement PreActResidualBlockDown and PreActResidualBlockUp
- [ ] Implement ChannelAttention, SpatialAttention, CBAM modules
- [ ] Create ResidualAutoencoder (Variant B)
- [ ] Create AttentionAutoencoder (Variant C)
- [ ] Train Variant B (Residual) at 16x compression
- [ ] Train Variant C (Res+CBAM) at 16x compression
- [ ] Evaluate all variants and compare metrics

---

## Phase 5: Full Image Inference

**Goal:** Implement tiled inference with blending that processes complete Sentinel-1 scenes without visible seams or memory issues. Support raw GeoTIFF input for end-to-end compression pipeline.

**Dependencies:** Phase 4 (Architecture Enhancement)

**Estimated Complexity:** Medium-High

### Success Criteria

1. Tiled inference processes a 10000x10000 pixel image without GPU memory overflow
2. Cosine ramp blending produces seamless reconstructions with no visible tile boundaries
3. Processing time for full scene is under 5 minutes on RTX 3070
4. Inverse preprocessing correctly restores linear SAR intensity values
5. Round-trip full image compression maintains PSNR within 0.5 dB of patch-level metrics
6. Raw Sentinel-1 GeoTIFF can be compressed and decompressed end-to-end via CLI or API
7. Output GeoTIFF preserves geospatial metadata (CRS, transform, nodata) from input

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR5.1 | Single patch encode/decode |
| FR5.2 | Full image tiled inference with configurable overlap |
| FR5.3 | Cosine ramp blending for tile boundaries |
| FR5.4 | Memory-efficient processing for large images |
| FR5.5 | Inverse preprocessing (restore linear SAR values) |
| FR5.6 | Preserve GeoTIFF metadata in output |
| FR5.7 | Accept raw Sentinel-1 GeoTIFF as input (end-to-end pipeline) |
| FR5.8 | Automatic preprocessing using saved model parameters |
| FR5.9 | CLI interface for compress/decompress operations |

### Deliverables

- `src/inference/compressor.py` - Complete tiled compression/decompression
- `src/inference/pipeline.py` - End-to-end raw GeoTIFF -> compressed -> GeoTIFF pipeline
- `compress.py` / `decompress.py` - CLI scripts for end-to-end usage
- Full scene compression script
- Blending weight visualization demonstrating smooth transitions
- Performance benchmarks (time, memory) for full scene processing

### Key Tasks (High-Level)

- [ ] Implement single patch encode/decode in compressor
- [ ] Implement tiled processing with configurable overlap
- [ ] Implement cosine ramp blending weights
- [ ] Implement memory-efficient batch processing
- [ ] Implement inverse preprocessing for output
- [ ] Implement raw GeoTIFF loading with rasterio (reuse Phase 1 preprocessing)
- [ ] Implement automatic preprocessing using checkpoint's saved parameters
- [ ] Implement GeoTIFF output with preserved metadata (CRS, transform, nodata)
- [ ] Create CLI interface: `python compress.py input.tif -o compressed.bin --model best.pth`
- [ ] Create CLI interface: `python decompress.py compressed.bin -o output.tif`
- [ ] Test on full Sentinel-1 scene and verify seamlessness
- [ ] Benchmark processing time and memory usage

---

## Phase 6: Final Experiments

**Goal:** Execute the complete experiment matrix (3 architectures x 3 compression ratios = 9 models) and produce a comprehensive comparison study with rate-distortion analysis.

**Dependencies:** Phase 5 (Full Image Inference)

**Estimated Complexity:** High

### Success Criteria

1. All 9 experiment configurations (Plain/Residual/Res+CBAM at 8x/16x/32x) complete training successfully
2. Rate-distortion curves are generated showing PSNR vs BPP for all architectures AND traditional codecs
3. At least one configuration achieves PSNR >30 dB at 16x compression (primary target)
4. Best autoencoder variant outperforms JPEG-2000 at equivalent compression ratio
5. Statistical analysis includes mean and standard deviation across test set for all metrics
6. Final documentation includes visual examples at each compression level showing quality differences vs traditional codecs

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR6.1 | Train plain architecture at multiple compression ratios (8x, 16x, 32x) |
| FR6.2 | Train residual architecture at multiple compression ratios |
| FR6.3 | Train residual+CBAM architecture at multiple compression ratios |
| FR6.4 | Generate rate-distortion curves comparing all variants |
| FR6.5 | Statistical analysis of results |
| FR6.6 | Document findings with visual examples |

### Deliverables

- 9 trained model checkpoints (3 architectures x 3 compression ratios)
- Rate-distortion curves (PSNR vs BPP, SSIM vs BPP) including JPEG-2000 baseline
- Comprehensive metrics table with statistics (autoencoders + traditional codecs)
- Visual comparison gallery: autoencoder vs JPEG-2000 at each compression level
- Final analysis document summarizing findings and codec comparison

### Key Tasks (High-Level)

- [ ] Train Plain architecture at 8x, 16x, 32x compression
- [ ] Train Residual architecture at 8x, 16x, 32x compression
- [ ] Train Res+CBAM architecture at 8x, 16x, 32x compression
- [ ] Evaluate all models on test set with full metrics suite
- [ ] Include JPEG-2000 results in rate-distortion curves
- [ ] Generate rate-distortion curves (autoencoders + traditional codecs)
- [ ] Create visual comparison gallery (include JPEG-2000 comparison)
- [ ] Document findings: autoencoder vs traditional codec performance

---

## Phase 7: Deployment

**Goal:** Package the best-performing model for production deployment with multiple export formats, containerization, and optional API serving.

**Dependencies:** Phase 6 (Final Experiments)

**Estimated Complexity:** Medium

### Success Criteria

1. Model exports to ONNX format and runs inference correctly outside PyTorch
2. TorchScript export works for deployment without Python dependencies
3. Docker container runs inference on CPU and GPU with single command
4. REST API endpoint accepts GeoTIFF upload and returns compressed/decompressed result
5. Documentation covers all deployment options with examples
6. Inference latency meets targets: <1s per patch, <5 min full scene

### Requirements Mapped

| ID | Requirement |
|----|-------------|
| FR7.1 | Export model to ONNX format |
| FR7.2 | Export model to TorchScript |
| FR7.3 | Docker container with GPU support |
| FR7.4 | REST API for compression/decompression |
| FR7.5 | Deployment documentation |
| FR7.6 | Model versioning and checksum verification |

### Deliverables

- `scripts/export_onnx.py` - ONNX export with validation
- `scripts/export_torchscript.py` - TorchScript export
- `Dockerfile` and `docker-compose.yml` - Container setup (CPU + GPU variants)
- `src/api/` - FastAPI application for REST serving
- `docs/deployment.md` - Deployment guide with examples
- Published Docker image (optional: Docker Hub / GitHub Container Registry)

### Key Tasks (High-Level)

- [ ] Implement ONNX export with dynamic batch size
- [ ] Validate ONNX inference matches PyTorch output
- [ ] Implement TorchScript export (trace or script)
- [ ] Create Dockerfile with CUDA support
- [ ] Create docker-compose for easy local deployment
- [ ] Implement FastAPI endpoints: `/compress`, `/decompress`, `/health`
- [ ] Add file upload/download handling for GeoTIFF
- [ ] Write deployment documentation
- [ ] Benchmark inference latency in container
- [ ] Optional: CI/CD pipeline for automated builds

---

## Progress Overview

| Phase | Status | Success Criteria Met |
|-------|--------|---------------------|
| 1 - Data Pipeline | Complete | 5/5 |
| 2 - Baseline Model | Complete | 5/5 |
| 3 - SAR Evaluation | Complete | 6/6 |
| 4 - Architecture Enhancement | In Progress | 0/5 |
| 5 - Full Image Inference | Not Started | 0/7 |
| 6 - Final Experiments | Not Started | 0/6 |
| 7 - Deployment | Not Started | 0/6 |

---

## Non-Functional Requirements (Cross-Cutting)

These NFRs apply across all phases:

| ID | Requirement | Target |
|----|-------------|--------|
| NFR1.1 | Training fits in 8GB VRAM | Batch size 8 minimum |
| NFR1.2 | Single experiment completes in <8 hours | Per architecture/CR combo |
| NFR1.3 | Inference latency <1s per patch | Real-time preview |
| NFR1.4 | Full image inference <5 minutes | 10000x10000 pixels |
| NFR2.1 | PSNR at 16x compression | >30 dB |
| NFR2.2 | SSIM at 16x compression | >0.85 |
| NFR2.3 | ENL ratio | 0.8-1.2 |
| NFR2.4 | EPI | >0.85 |
| NFR3.1 | Follow existing code conventions | Type hints, docstrings |
| NFR3.2 | Configurable via YAML/JSON | Reproducibility |
| NFR3.3 | Deterministic training with seed | Reproducibility |
| NFR3.4 | Clear separation of concerns | Maintainability |

---

## Codebase State Summary

The project has an established skeleton with most functionality as stubs (`NotImplementedError`). Key observations:

**Partially Implemented:**
- `src/data/preprocessing.py` - `preprocess_sar_complete()`, `inverse_preprocess()`, `extract_patches()`, `analyze_sar_statistics()` are working; stub functions remain
- `src/evaluation/metrics.py` - Basic PSNR, SSIM, MSE, MAE, correlation working; ENL, EPI, histogram similarity are stubs

**Pure Stubs:**
- `src/models/blocks.py` - All blocks (ConvBlock, DeconvBlock, ResidualBlock, CBAM)
- `src/models/encoder.py`, `decoder.py`, `autoencoder.py`
- `src/training/trainer.py`
- `src/inference/compressor.py`
- `src/losses/` - Loss functions

**Ready to Use:**
- Project structure and module organization
- Configuration system
- TensorBoard integration points
- Type hints and docstrings throughout

---

*Roadmap created: 2026-01-21*
*Phase 2 planned: 2026-01-21*
*Phase 3 planned: 2026-01-24*
*Phase 3 complete: 2026-01-24*
*Phase 4 planned: 2026-01-24*
*Derived from: PROJECT.md, REQUIREMENTS.md, research/SUMMARY.md*
