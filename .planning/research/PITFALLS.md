# Domain Pitfalls: SAR Compression Autoencoders

**Domain:** CNN-based compression for Sentinel-1 SAR imagery
**Researched:** 2026-01-21
**Confidence:** MEDIUM (based on domain knowledge and project analysis; WebSearch unavailable for verification)

---

## Critical Pitfalls

Mistakes that cause rewrites, fundamentally broken models, or wasted months of effort.

---

### Pitfall 1: Training on Linear Intensity Values

**What goes wrong:**
Training the autoencoder directly on raw SAR linear intensity values causes the network to focus almost entirely on bright pixels (strong backscatter) while treating most of the image as essentially zero. This results in a model that reconstructs bright features poorly and dark features not at all.

**Why it happens:**
SAR intensity values span 4-6 orders of magnitude (0.00001 to 100+). Without log/dB transformation:
- 95% of pixels have values < 0.1
- The top 1% of pixels dominate MSE loss by 10,000x or more
- Gradients are essentially zero for dark regions

**Warning signs:**
- Loss decreases rapidly then plateaus at a high value
- Reconstructions show only the brightest features (urban areas, ships)
- Dark regions (water, smooth surfaces) appear as flat gray
- PSNR looks reasonable but visual quality is terrible
- Model outputs converge to the mean for most pixels

**Consequences:**
- Model learns nothing useful for most terrain types
- Complete failure on agricultural, vegetated, or oceanic scenes
- Results are not publishable or usable

**Prevention:**
1. **Always convert to dB before training:** `dB = 10 * log10(intensity + noise_floor)`
2. Apply the transformation in the preprocessing pipeline, not as a network layer
3. Verify value distribution is approximately Gaussian after transformation
4. Include histogram visualization in your data verification step

**Detection test:**
```python
# Before training, check input distribution
import matplotlib.pyplot as plt
plt.hist(training_data.flatten(), bins=100)
# Should show approximately normal distribution, not exponential/power-law
```

**Phase to address:** Data preprocessing (before any model training)

**Severity:** CRITICAL - Model will not work without this

---

### Pitfall 2: Inconsistent Preprocessing Parameters Across Train/Val/Test

**What goes wrong:**
Computing normalization bounds (vmin/vmax for clipping) separately on training, validation, and test sets causes the network to see different value distributions at each stage. The model optimizes for training distribution but fails on val/test.

**Why it happens:**
- Each set computed percentile clipping independently
- Different geographic regions have different backscatter distributions
- Test set from new sensor acquisition may have calibration differences

**Warning signs:**
- Validation loss significantly higher than training loss from epoch 1
- Test metrics much worse than validation metrics
- Reconstructions look color-shifted or contrast-stretched relative to inputs
- Model works well on some images but terribly on others

**Consequences:**
- Cannot reliably evaluate model quality
- Model performance varies unpredictably
- Deployment fails on new data

**Prevention:**
1. **Compute vmin/vmax ONLY from training set**
2. Apply the same fixed parameters to val/test/inference
3. Save preprocessing parameters alongside model checkpoint
4. Validate at inference time that input range matches training assumptions

**Implementation:**
```python
# CORRECT: Compute from training only
vmin, vmax = compute_bounds_from_training(train_data)
save_to_checkpoint({'vmin': vmin, 'vmax': vmax, 'model_state': ...})

# At inference
params = load_checkpoint()
assert input.min() >= params['vmin'] - tolerance, "Input out of expected range"
```

**Phase to address:** Data preprocessing, checkpoint saving/loading

**Severity:** CRITICAL - Silent failure mode that corrupts all metrics

---

### Pitfall 3: Ignoring Speckle Statistics in Evaluation

**What goes wrong:**
Evaluating SAR compression using only PSNR/SSIM misses whether the reconstruction preserves the statistical properties that SAR applications depend on. A model might achieve good PSNR by over-smoothing, which destroys radiometric calibration.

**Why it happens:**
- PSNR/SSIM are standard metrics, easy to report
- Over-smoothing actually improves PSNR in noisy images
- SAR-specific metrics (ENL, EPI) are less familiar

**Warning signs:**
- Very high PSNR (>35 dB) but reconstructions look "painted" or artificial
- Homogeneous regions (water, fields) look perfectly smooth in reconstruction
- ENL ratio >> 1.0 (reconstruction is smoother than original)
- Edge Preservation Index < 0.85

**Consequences:**
- Radiometric calibration lost (cannot do quantitative analysis)
- Change detection fails (smoothed speckle doesn't match)
- Classifier accuracy drops on reconstructed data
- Published results don't reflect actual usability

**Prevention:**
1. **Always report ENL ratio** for homogeneous regions (target: 0.9-1.1)
2. **Always report Edge Preservation Index** (target: > 0.9)
3. Include visual comparison showing speckle texture preservation
4. Test on downstream tasks (classification, segmentation) not just reconstruction metrics

**Evaluation checklist:**
```python
metrics = {
    'psnr': compute_psnr(orig, recon),
    'ssim': compute_ssim(orig, recon),
    'enl_ratio': compute_enl_ratio(orig, recon, homogeneous_mask),  # REQUIRED
    'epi': compute_edge_preservation(orig, recon),  # REQUIRED
}
# All four metrics must be acceptable, not just PSNR/SSIM
```

**Phase to address:** Evaluation framework setup

**Severity:** CRITICAL - Misleading evaluation leads to wrong architecture choices

---

### Pitfall 4: Loss Function Encouraging Over-Smoothing

**What goes wrong:**
Using MSE loss alone (or MSE-dominant combined loss) causes the network to learn the conditional mean, which for noisy SAR data is a smoothed version of the true signal. The model becomes a denoiser rather than a compressor.

**Why it happens:**
- MSE minimization mathematically yields the conditional expectation
- For multiplicative speckle noise, this is the underlying signal without noise
- Network "solves" the reconstruction problem by predicting smooth images

**Warning signs:**
- Training converges quickly and loss reaches a low value
- Reconstructions look denoised/filtered
- Fine textures (vegetation, sea surface) disappear
- Model performance looks great on PSNR but users complain about quality
- ENL of reconstruction >> ENL of original

**Consequences:**
- Lost information that cannot be recovered
- Changed statistical properties break downstream applications
- Model is effectively a lossy denoiser, not a compressor

**Prevention:**
1. **Use perceptual or structural loss components** (SSIM, MS-SSIM)
2. Weight SSIM loss heavily relative to MSE: start with 0.5 MSE + 0.5 SSIM
3. Add edge-aware loss terms if edge preservation is critical
4. Monitor ENL ratio during training, not just aggregate loss
5. Consider adversarial training for texture preservation (advanced)

**Loss configuration:**
```python
# BAD: MSE-dominant
loss = 0.9 * mse_loss + 0.1 * ssim_loss  # Will over-smooth

# BETTER: Balanced
loss = 0.5 * mse_loss + 0.5 * ssim_loss  # Starting point

# BEST: Tune based on ENL ratio
# If ENL_ratio > 1.1: reduce MSE weight
# If ENL_ratio < 0.9: reduce SSIM weight
```

**Phase to address:** Loss function design and training hyperparameters

**Severity:** CRITICAL - Determines fundamental model behavior

---

### Pitfall 5: Not Handling Invalid Values Before dB Conversion

**What goes wrong:**
Taking log10 of zero or negative values produces -inf or NaN, which propagates through the entire training batch and corrupts all gradients. A single invalid pixel can break training.

**Why it happens:**
- SAR data contains zeros (no-data regions, radar shadows, calibration artifacts)
- Some processing pipelines produce small negative values (numerical precision)
- Standard log operation is undefined for non-positive numbers

**Warning signs:**
- Loss becomes NaN during training
- Model outputs are all NaN or constant
- Some batches work, others crash
- Training works on some images but not others

**Consequences:**
- Training crashes or produces useless model
- Debugging is difficult (NaN source is unclear)
- Wasted GPU hours on corrupted training

**Prevention:**
1. **Always apply noise floor before dB conversion:** `max(intensity, 1e-10)`
2. Validate data after loading: check for NaN, Inf, negative values
3. Filter patches that are mostly invalid (>1% invalid pixels)
4. Add assertions in data pipeline to catch invalid values early

**Data validation:**
```python
def validate_sar_data(data):
    assert not np.any(np.isnan(data)), "NaN values found"
    assert not np.any(np.isinf(data)), "Inf values found"
    assert np.all(data >= 0), "Negative values found"
    # After dB conversion
    data_db = 10 * np.log10(np.maximum(data, 1e-10))
    assert not np.any(np.isnan(data_db)), "dB conversion produced NaN"
```

**Phase to address:** Data preprocessing (earliest stage)

**Severity:** CRITICAL - Causes immediate training failure

---

## Major Pitfalls

Mistakes that cause significant delays, poor performance, or substantial rework.

---

### Pitfall 6: Latent Space Too Small for SAR Complexity

**What goes wrong:**
Choosing aggressive compression (32x or 64x) because it works for natural images fails for SAR. SAR's high-frequency speckle texture and edge sharpness require more latent capacity than typical photographs.

**Why it happens:**
- Copying compression ratios from natural image papers
- Underestimating information content in speckle
- GPU memory pressure encouraging smaller latents

**Warning signs:**
- Edges in reconstruction appear soft or wavy
- Fine structures (roads, field boundaries) merge together
- "Ringing" artifacts around high-contrast features
- Quality acceptable for smooth regions but bad for textured areas

**Consequences:**
- Model cannot represent important SAR features
- Architecture design wasted on insufficient capacity
- Need to retrain with larger latent

**Prevention:**
1. Start conservative: **16x compression (16x16x16 latent for 256x256 input)**
2. Only increase compression after validating quality at conservative setting
3. Use residual blocks to increase effective capacity without adding latent dimensions
4. Compare edge preservation across compression levels before committing

**Recommended starting points for SAR:**
| Quality Need | Compression | Latent (256x256 input) | Expected PSNR |
|--------------|-------------|------------------------|---------------|
| High fidelity | 8x | 16x16x32 | >35 dB |
| Balanced | 16x | 16x16x16 | >30 dB |
| Bandwidth-constrained | 32x | 16x16x8 | >27 dB |

**Phase to address:** Architecture design

**Severity:** MAJOR - Requires retraining with different architecture

---

### Pitfall 7: Batch Normalization Issues at Inference

**What goes wrong:**
Batch normalization layers behave differently in training vs. evaluation mode. If not properly switched, or if batch size is 1 during inference, the model produces inconsistent or degraded results.

**Why it happens:**
- Forgetting `model.eval()` before inference
- Running inference with batch_size=1 (running statistics estimated from single sample)
- BN running statistics not properly updated during training

**Warning signs:**
- Model performs well during training validation but poorly in standalone inference
- Results vary depending on batch composition
- Single-image inference produces different results than batch inference
- Tile boundaries visible when reconstructing large images from patches

**Consequences:**
- Inconsistent deployment behavior
- Cannot reliably compress individual patches
- Tile seams in full-image reconstruction

**Prevention:**
1. Always call `model.eval()` before inference
2. Consider using **Instance Normalization** or **Group Normalization** instead of BatchNorm
3. If using BatchNorm, ensure batch_size >= 8 during inference (or pad batch)
4. Test inference with single-sample batches before deployment

**Architecture consideration:**
```python
# Option 1: Use Group Normalization (batch-size independent)
self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

# Option 2: Make BatchNorm optional
self.norm = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
```

**Phase to address:** Architecture design, inference pipeline

**Severity:** MAJOR - Causes deployment failures

---

### Pitfall 8: Ignoring Dynamic Range in Output Activation

**What goes wrong:**
Using sigmoid activation on the decoder output when the data isn't normalized to [0, 1], or using no activation when the data should be bounded. This causes clipping or unbounded outputs that don't match the input distribution.

**Why it happens:**
- Mismatch between preprocessing normalization and network output
- Copying architecture from papers that used different data ranges
- Not thinking through the full data flow

**Warning signs:**
- Reconstruction values pile up at 0 or 1 (sigmoid clipping)
- Reconstruction values go outside valid range (no activation)
- Bimodal output distribution when input is unimodal
- Systematic bias in bright or dark regions

**Consequences:**
- Lost information at dynamic range extremes
- Metrics don't reflect true quality
- Reconstruction artifacts in extreme intensity regions

**Prevention:**
1. **Match output activation to input normalization:**
   - Input [0, 1] -> sigmoid output
   - Input [-1, 1] -> tanh output
   - Input unbounded -> no activation (but add clipping post-hoc)
2. Verify reconstruction value range matches input range
3. Check histogram of outputs vs inputs

**Verification:**
```python
# After forward pass
assert recon.min() >= 0 and recon.max() <= 1, "Output outside [0,1]"
# Or check distribution match
assert abs(recon.mean() - input.mean()) < 0.1, "Systematic bias in output"
```

**Phase to address:** Architecture design, preprocessing pipeline

**Severity:** MAJOR - Causes systematic reconstruction errors

---

### Pitfall 9: Memory Overflow on Full-Size SAR Images

**What goes wrong:**
Training works fine on patches, but inference crashes or produces artifacts when processing full Sentinel-1 scenes (25000 x 16000 pixels).

**Why it happens:**
- Full scenes don't fit in GPU memory
- Naive patch-based processing creates visible tile seams
- Overlap-and-blend requires careful implementation
- Memory allocation for intermediate tensors not considered

**Warning signs:**
- Out-of-memory errors at inference time
- Grid-like artifacts in full-image reconstructions
- Seams visible at patch boundaries
- Different quality at tile edges vs centers

**Consequences:**
- Cannot process real satellite data
- Deployment-blocking issue discovered late
- Need to implement complex tiling logic

**Prevention:**
1. Design inference pipeline with tiling from the start
2. Use **overlapping patches with cosine/Gaussian blend**
3. Test full-image inference early (even with untrained model)
4. Budget memory: `full_image_memory = H * W * intermediate_channels * 4 bytes`

**Tiling strategy:**
```python
# Overlap should be >= receptive field of network
overlap = 64  # pixels
# Blend weights: cosine ramp at edges
blend_weights = create_cosine_ramp_weights(patch_size, overlap)
# Process patches
for patch, coords in sliding_window(image, patch_size, stride=patch_size-overlap):
    reconstructed = model(patch)
    accumulator[coords] += reconstructed * blend_weights
    weight_sum[coords] += blend_weights
final = accumulator / weight_sum
```

**Phase to address:** Inference pipeline design (plan early, implement before final evaluation)

**Severity:** MAJOR - Blocks real-world deployment

---

### Pitfall 10: Checkpoint Incompatibility Across Experiments

**What goes wrong:**
Changing architecture (channels, layers, block types) invalidates all previous checkpoints. Cannot resume training or compare fairly across experiments.

**Why it happens:**
- Architecture defined by code, not saved with checkpoint
- Hyperparameters not logged with weights
- No versioning of model configurations

**Warning signs:**
- "Size mismatch" errors when loading checkpoints
- Cannot reproduce previous results
- Unclear which config produced which checkpoint
- Experiments not comparable due to unknown configuration differences

**Consequences:**
- Lost training progress when making changes
- Cannot fairly compare architectures
- Results not reproducible

**Prevention:**
1. **Save full config dict in checkpoint:**
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'config': config_dict,
       'preprocessing_params': {'vmin': vmin, 'vmax': vmax},
       'epoch': epoch,
       'optimizer_state': optimizer.state_dict()
   }, checkpoint_path)
   ```
2. Use configuration-based model construction:
   ```python
   model = SARAutoencoder.from_config(checkpoint['config'])
   model.load_state_dict(checkpoint['model_state_dict'])
   ```
3. Log experiments with tools like MLflow, Weights & Biases, or simple JSON

**Phase to address:** Training infrastructure (implement before extensive experimentation)

**Severity:** MAJOR - Wastes significant compute resources

---

## Moderate Pitfalls

Mistakes that cause delays or technical debt but are recoverable.

---

### Pitfall 11: Not Monitoring Latent Space Statistics

**What goes wrong:**
Latent space collapses to narrow range, constant values, or has dead channels. The network isn't using its full capacity, limiting compression efficiency.

**Why it happens:**
- No monitoring of latent activations
- Learning rate too high causes collapse
- Initialization issues
- Bottleneck too narrow for the task

**Warning signs:**
- Latent values cluster near zero or have very small variance
- Some latent channels are always zero (dead channels)
- Reconstruction quality plateaus despite training
- Adding latent capacity doesn't improve quality

**Prevention:**
1. Log latent statistics every N epochs: mean, std, min, max, histogram
2. Track per-channel activation statistics
3. Use appropriate initialization (He or Xavier)
4. Consider adding latent regularization if collapse occurs

**Monitoring code:**
```python
def analyze_latent(z):
    return {
        'mean': z.mean().item(),
        'std': z.std().item(),
        'min': z.min().item(),
        'max': z.max().item(),
        'dead_channels': (z.abs().mean(dim=[0,2,3]) < 1e-6).sum().item()
    }
```

**Phase to address:** Training monitoring

**Severity:** MODERATE - Reduces model efficiency

---

### Pitfall 12: Inadequate Data Augmentation for SAR

**What goes wrong:**
Applying augmentations designed for natural images (color jitter, random crops from any location) that don't make sense for SAR or miss SAR-specific opportunities.

**Why it happens:**
- Copying augmentation pipelines from natural image projects
- Not understanding SAR imaging geometry
- Missing domain-specific augmentation opportunities

**Warning signs:**
- Model performs well on training geography but poorly on new regions
- Overfitting despite large dataset
- Model sensitive to patch location within scene

**Appropriate SAR augmentations:**
- **Random horizontal flip:** Valid (SAR is not directional in azimuth)
- **Random vertical flip:** CAUTION - may not be valid due to radar look direction
- **Random rotation (90, 180, 270):** CAUTION - may affect speckle statistics
- **Additive noise:** NOT RECOMMENDED - speckle is multiplicative
- **Multiplicative noise:** Can work to simulate speckle variation
- **Random crop:** Good
- **Elastic deformation:** Generally not appropriate for SAR

**Prevention:**
1. Research SAR imaging geometry before choosing augmentations
2. Start with conservative augmentations: flip, crop
3. Validate that augmentation doesn't change speckle statistics
4. Test on geographically diverse data

**Phase to address:** Data pipeline design

**Severity:** MODERATE - Affects generalization

---

### Pitfall 13: Learning Rate Too High for Deep Autoencoders

**What goes wrong:**
Deep autoencoders (4+ downsampling layers) are sensitive to learning rate. Too high causes training instability, mode collapse, or oscillation.

**Why it happens:**
- Using default learning rates from simpler models
- Long path from input to output amplifies gradient issues
- Reconstruction task is ill-conditioned at high compression

**Warning signs:**
- Loss spikes during training
- Loss oscillates without decreasing
- Reconstruction quality varies wildly between epochs
- Model produces constant output (mode collapse)

**Prevention:**
1. Start with **lr = 1e-4** for Adam (not 1e-3)
2. Use learning rate warmup for first few epochs
3. Use gradient clipping (max_norm=1.0)
4. Consider cosine annealing or reduce-on-plateau scheduler

**Training configuration:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Phase to address:** Training hyperparameters

**Severity:** MODERATE - Fixable without architecture changes

---

### Pitfall 14: Evaluating on Training-Like Data Only

**What goes wrong:**
Model achieves great metrics on validation set that comes from the same geographic region/acquisition as training, but fails on operationally relevant test data.

**Why it happens:**
- Train/val/test splits are random patches from same scenes
- Model memorizes scene-specific characteristics
- No geographic diversity in evaluation

**Warning signs:**
- Validation metrics closely track training metrics
- Model performs very differently on data from new regions
- Metrics don't match operational user experience

**Prevention:**
1. **Geographic split:** Train on some scenes, validate on entirely different scenes
2. **Temporal split:** Train on older acquisitions, test on newer ones
3. **Cross-sensor validation:** If possible, test on data from different Sentinel-1 satellite (S1A vs S1B)
4. Report metrics separately for seen vs unseen geographic regions

**Data organization:**
```
data/
  train/  # Scenes from regions A, B, C
  val/    # Scenes from regions D, E (never seen in training)
  test/   # Scenes from regions F, G (different from train AND val)
```

**Phase to address:** Data organization, evaluation design

**Severity:** MODERATE - Affects real-world performance assessment

---

## Minor Pitfalls

Issues that cause annoyance but are quickly fixable.

---

### Pitfall 15: Forgetting to Save Best Model vs Latest Model

**What goes wrong:**
Training completes but only the last epoch is saved. If training overfit or had a bad final epoch, the best-performing weights are lost.

**Prevention:**
- Save checkpoint when validation loss improves
- Keep both "best" and "latest" checkpoints
- Log which epoch produced the best model

**Phase to address:** Training infrastructure

**Severity:** MINOR - Annoying but training can be rerun

---

### Pitfall 16: TensorBoard Logs Not Including Images

**What goes wrong:**
Monitoring only scalar metrics misses visual artifacts that metrics don't capture. Model looks good on metrics but has visible problems.

**Prevention:**
- Log sample reconstructions every N epochs
- Include difference images (original - reconstruction)
- Log histogram of latent values

**Phase to address:** Training monitoring

**Severity:** MINOR - Easy to add retrospectively

---

### Pitfall 17: Inconsistent Random Seeds

**What goes wrong:**
Cannot reproduce results because random state wasn't fixed. Different runs give different results without explanation.

**Prevention:**
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

**Phase to address:** Training infrastructure

**Severity:** MINOR - Easy to fix

---

## Phase-Specific Warning Summary

| Phase | Likely Pitfalls | Mitigation |
|-------|-----------------|------------|
| **Data Preprocessing** | #1 (Linear values), #2 (Inconsistent params), #5 (Invalid values), #12 (Bad augmentation) | Validate dB conversion, save params, check for NaN |
| **Architecture Design** | #6 (Small latent), #7 (BatchNorm), #8 (Output activation) | Start conservative (16x), use GroupNorm, match activation to data range |
| **Loss Function** | #3 (Missing SAR metrics), #4 (Over-smoothing) | Include ENL/EPI in eval, balance MSE/SSIM |
| **Training Loop** | #10 (Checkpoint compat), #11 (Latent collapse), #13 (High LR), #15 (Best model) | Save config in checkpoint, monitor latent stats, use warmup |
| **Evaluation** | #3 (Missing metrics), #14 (Same-region eval) | Geographic splits, SAR-specific metrics |
| **Inference Pipeline** | #7 (BatchNorm eval), #9 (Memory/tiling) | model.eval(), implement tiled inference early |

---

## Quick Checklist: Before You Start Training

- [ ] Data converted to dB (not linear intensity)
- [ ] Invalid values handled (noise floor applied)
- [ ] Preprocessing params computed from training set only
- [ ] Preprocessing params saved for inference
- [ ] Output activation matches data normalization
- [ ] ENL ratio and EPI included in evaluation metrics
- [ ] Loss function balances MSE and SSIM (not MSE-only)
- [ ] Learning rate is conservative (1e-4)
- [ ] Latent space monitoring implemented
- [ ] Checkpoint saves config, params, and model state
- [ ] Train/val/test are geographically separated
- [ ] Tiled inference considered for full-scene processing

---

## Sources

- Project knowledge base: `.planning/knowledge/05_SAR_PREPROCESSING.md`
- Project knowledge base: `.planning/knowledge/06_SAR_QUALITY_METRICS.md`
- Project knowledge base: `.planning/knowledge/07_COMPRESSION_TRADEOFFS.md`
- Project concerns audit: `.planning/codebase/CONCERNS.md`
- Domain expertise on SAR imaging and neural compression

**Confidence note:** This document is based on established SAR processing principles and common failure modes in learned compression. WebSearch verification was unavailable. Individual pitfalls should be validated against published literature when implementing prevention strategies.

---

*Last updated: 2026-01-21*
