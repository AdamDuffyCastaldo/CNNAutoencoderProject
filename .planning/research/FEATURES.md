# Feature Landscape: SAR Image Compression Autoencoders

**Domain:** Learned compression for Sentinel-1 SAR satellite imagery
**Researched:** 2026-01-21
**Context:** Brownfield project adding autoencoder compression to existing SAR processing pipeline

---

## Table Stakes

Features users expect from any SAR compression system. Missing = product feels incomplete or unusable.

### Preprocessing Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| dB transform (linear to log) | SAR dynamic range (40-60 dB) incompatible with neural networks; multiplicative speckle becomes additive | Low | Reversible transform, must store params |
| Invalid value handling | SAR contains zeros, NaN, negatives from shadows/calibration | Low | Replace with noise floor before log |
| Dynamic range clipping | Outliers (corner reflectors, deep shadows) destabilize training | Low | Percentile or fixed (-25 to +5 dB typical) |
| Normalization to [0,1] | Neural network weight initialization assumes bounded inputs | Low | Min-max using dataset-wide bounds |
| Patch extraction | Full SAR images (25K x 16K pixels) exceed GPU memory | Low | 256x256 patches standard for 10m resolution |
| Quality filtering | Remove no-data, homogeneous, corrupted patches | Low | Min variance threshold, max invalid ratio |

**Dependencies:** Each preprocessing step requires the previous. Order is fixed.

### Core Autoencoder Architecture

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| CNN encoder (spatial downsampling) | Extract compressed representation from image | Medium | 4-5 layers for 256x256 to 16x16 |
| CNN decoder (spatial upsampling) | Reconstruct image from latent | Medium | Mirror encoder architecture |
| Configurable latent dimensions | Control compression ratio | Low | Latent channels = primary tuning knob |
| Skip/residual connections | Prevent vanishing gradients, preserve information | Medium | ResNet-style blocks standard |
| Multiple compression ratios | Compare quality/size tradeoffs | Low | Train variants with different latent sizes |

**Dependencies:** Encoder/decoder are coupled architecturally.

### Training Infrastructure

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| MSE loss | Pixel-level fidelity baseline | Low | Simple, always include |
| SSIM loss | Structural preservation | Low | Perceptually better than MSE alone |
| Combined loss (MSE + SSIM) | Balance pixel accuracy and structure | Low | Weighted sum, tune weights |
| Learning rate scheduling | Convergence optimization | Low | Cosine annealing or step decay |
| Checkpointing | Resume training, save best models | Low | Save model + optimizer + epoch |
| Early stopping | Prevent overfitting | Low | Monitor validation loss |
| TensorBoard logging | Training visibility | Low | Loss curves, sample reconstructions |

### Evaluation Metrics

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| PSNR | Standard image quality metric | Low | Target: >30 dB good, >35 dB excellent |
| SSIM | Structural similarity | Low | Target: >0.9 good, >0.95 excellent |
| ENL ratio | SAR-specific speckle preservation | Medium | Ratio of ENL(recon)/ENL(original), target ~1.0 |
| EPI | Edge preservation index | Medium | Gradient correlation, target >0.9 |
| Compression ratio | Space savings | Low | Input size / latent size |
| Bits per pixel (BPP) | Industry standard metric | Low | Enables comparison with JPEG, etc. |

### Inference Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Full image tiling | Process images larger than patch size | Medium | Overlapping patches for seamless output |
| Patch blending | Avoid visible seams at patch boundaries | Medium | Cosine ramp weights in overlap regions |
| Inverse preprocessing | Restore linear SAR values | Low | Must preserve radiometric accuracy |
| Metadata preservation | Store params needed for reconstruction | Low | Normalization bounds, model config |

---

## Differentiators

Features that set a great SAR compressor apart. Not expected, but highly valued.

### Architecture Enhancements

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Attention mechanisms | Allocate capacity to edges/features, less to flat regions | Medium | Self-attention in bottleneck, 2-5 dB improvement possible |
| Multi-scale processing | Capture both fine detail and global context | High | U-Net style or pyramid architectures |
| Learned entropy model | Estimate actual bitrate, not just latent size | High | Hyperprior networks (Balle 2018 style) |
| Variable-rate compression | Single model for multiple compression levels | High | Gain vectors or channel dropout |
| Quantization-aware training | Robust to post-training quantization | Medium | Add noise during training to simulate int8 |

### SAR-Specific Optimizations

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Speckle-aware loss | Better handle multiplicative noise statistics | Medium | Log-domain loss, ratio-based metrics |
| Edge-weighted loss | Prioritize structure over texture | Medium | Gradient magnitude weighting |
| Terrain-adaptive compression | Different strategies for water/urban/vegetation | High | Requires segmentation or learned adaptation |
| Multi-polarization support | Compress VV+VH jointly, exploit correlation | High | Multi-channel input, shared encoder |
| Phase preservation | Maintain coherence for InSAR applications | Very High | Complex-valued networks, specialized loss |

### Deployment Features

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| ONNX export | Cross-platform deployment | Medium | Standard export workflow |
| TensorRT optimization | GPU inference speedup | High | Requires careful layer support verification |
| Entropy coding stage | Actual bitstream, not just latent arrays | High | Arithmetic coding on quantized latents |
| Progressive decoding | View low-res preview before full decode | Very High | Hierarchical latent structure |

### Evaluation Enhancements

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Rate-distortion curves | Compare architectures at same bitrate | Medium | Multiple models, sweep compression levels |
| Downstream task evaluation | Verify utility for real applications | High | Ship detection, change detection accuracy |
| Perceptual quality metrics | LPIPS, FID for SAR-like images | Medium | May need fine-tuning on SAR data |
| Statistical distribution tests | Verify speckle statistics preserved | Medium | K-S test, ENL histograms |

---

## Anti-Features

Features to deliberately NOT build. Common mistakes in this domain.

### Over-Engineering for V1

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| End-to-end entropy coding | Adds massive complexity, marginal gains for research | Evaluate latent size, add entropy coding later |
| Real-time streaming | Completely different architecture requirements | Focus on batch processing quality first |
| On-device optimization | Premature optimization, unclear deployment target | Train quality models first, optimize later |
| Multi-polarization from start | Doubles complexity, single-channel sufficient for v1 | Prove concept with VV only, add VH later |
| Complex-valued processing | Phase preservation rarely needed, huge complexity | Use amplitude/intensity only unless InSAR required |

### Architecture Mistakes

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Too-deep networks | Diminishing returns, training instability, GPU memory | 4-5 encoder layers is sufficient for 256x256 |
| Global attention everywhere | O(n^2) complexity, memory explosion | Use attention only at bottleneck or sparse attention |
| Very high compression (>64x) | Quality cliff, need extreme architecture tuning | Start with 8-16x, push higher gradually |
| Per-image normalization | Inconsistent meaning across images, can't compare | Use dataset-wide normalization bounds |
| Training without validation | No overfitting detection, no early stopping | Always use train/val split |

### Loss Function Mistakes

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| MSE only | Blurry reconstructions, poor perceptual quality | Combine MSE + SSIM, possibly add perceptual loss |
| Heavy perceptual loss | VGG features not trained on SAR, may be wrong | Light perceptual weight or SAR-specific features |
| L1 only in linear domain | Speckle-sensitive, unstable gradients | Work in dB domain where speckle is additive |
| Adversarial loss from start | Training instability, mode collapse risk | Get autoencoder working first, add GAN later |

### Evaluation Mistakes

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| PSNR/SSIM only | Misses SAR-specific degradation (speckle, edges) | Include ENL ratio and EPI |
| Qualitative only | Subjective, not reproducible | Always include quantitative metrics |
| Single test image | Not representative, cherry-picking risk | Evaluate on held-out test set |
| Ignore radiometric accuracy | Intensity relationships must be preserved | Check slope/intercept of original vs reconstructed |

---

## Feature Dependencies

```
Preprocessing
    |
    v
dB Transform --> Clipping --> Normalization --> Patch Extraction
                                                       |
                                                       v
                                              Core Autoencoder
                                              (Encoder + Decoder)
                                                       |
                              +------------------------+------------------------+
                              |                        |                        |
                              v                        v                        v
                        Basic Training           Residual Blocks          Attention
                        (MSE + SSIM)             (improves quality)    (improves edges)
                              |                        |                        |
                              +------------------------+------------------------+
                                                       |
                                                       v
                                              Evaluation Suite
                                              (PSNR, SSIM, ENL, EPI)
                                                       |
                                                       v
                                              Inference Pipeline
                                              (Tiling + Blending)
                                                       |
                                                       v
                                         === V1 COMPLETE ===
                                                       |
                                                       v
                          +----------------------------+----------------------------+
                          |                            |                            |
                          v                            v                            v
                    Entropy Coding              Variable Rate             Multi-Polarization
                    (actual bitstream)         (single model)            (VV + VH joint)
                          |                            |                            |
                          +----------------------------+----------------------------+
                                                       |
                                                       v
                                              Production Deployment
                                              (ONNX, TensorRT, etc.)
```

---

## MVP Recommendation

For MVP, prioritize in order:

### Must Have (Table Stakes)

1. **Preprocessing pipeline** - Cannot train without this
2. **Basic autoencoder** (encoder + decoder, configurable latent) - Core functionality
3. **MSE + SSIM loss** - Adequate quality for research
4. **Basic metrics** (PSNR, SSIM, compression ratio) - Must measure what we're optimizing
5. **Training loop with checkpointing** - Can't iterate without this
6. **Patch-based inference** - Process full images

### Should Have (Quality Gate)

7. **Residual connections** - Significant quality improvement, moderate effort
8. **ENL ratio metric** - SAR-specific validation
9. **EPI metric** - Edge quality validation
10. **TensorBoard logging** - Training visibility

### Nice to Have (Differentiators for V1)

11. **Attention at bottleneck** - Edge quality boost
12. **Multiple compression ratio variants** - Comparison study
13. **Rate-distortion curves** - Professional evaluation

### Defer to Post-MVP

- Entropy coding / actual bitstream
- Variable-rate single model
- Multi-polarization
- Deployment optimization
- Progressive decoding

---

## Complexity Estimates Summary

| Complexity | Count | Examples |
|------------|-------|----------|
| Low | 18 | Preprocessing steps, basic metrics, loss functions |
| Medium | 12 | Encoder/decoder, residual blocks, attention, tiling |
| High | 8 | Entropy models, variable-rate, multi-polarization, downstream evaluation |
| Very High | 2 | Phase preservation, progressive decoding |

**Estimated effort for MVP (table stakes + should have):** 2-3 weeks focused development

**Estimated effort for full V1 (including nice-to-have):** 4-6 weeks

---

## SAR-Specific Requirements Summary

| SAR Characteristic | Impact on System | How Addressed |
|--------------------|------------------|---------------|
| High dynamic range (40-60 dB) | Network can't handle raw values | dB transform + clipping |
| Multiplicative speckle | Unstable training in linear domain | Work in dB (makes speckle additive) |
| Invalid pixels (zeros, NaN) | log(0) = -inf, training crash | Replace with noise floor |
| Large image sizes | GPU memory limits | Patch-based processing with overlap |
| Edge importance | Edges carry key information (structures, boundaries) | EPI metric, optional edge-weighted loss |
| Speckle texture | Random but statistically meaningful | ENL ratio to verify preservation |
| Radiometric accuracy | Downstream applications need correct intensities | Validate slope ~1, intercept ~0 |
| 10m resolution | 256x256 patch = 2.56km x 2.56km | Good context size for terrain features |

---

## Sources

**Project-internal (HIGH confidence):**
- `.planning/knowledge/05_SAR_PREPROCESSING.md` - Comprehensive preprocessing guide
- `.planning/knowledge/06_SAR_QUALITY_METRICS.md` - Metric definitions and targets
- `.planning/knowledge/07_COMPRESSION_TRADEOFFS.md` - Latent space design principles
- `.planning/PROJECT.md` - Project requirements and constraints
- `src/evaluation/metrics.py` - Existing metric implementations (partially complete)
- `src/inference/compressor.py` - Inference pipeline structure (stubs)

**Domain knowledge (MEDIUM confidence):**
- Learned image compression follows established patterns (encoder-decoder, MSE+perceptual loss)
- SAR preprocessing to dB domain is standard practice in remote sensing
- ENL and EPI are established SAR quality metrics

**Note:** WebSearch unavailable for this research session. Features based on project documentation and domain expertise. State-of-the-art comparison with published learned compression methods (Balle 2018, Minnen 2018, Cheng 2020) would strengthen differentiator recommendations.
