# Deep Dive: Compression Tradeoffs and Latent Space Design

This document provides a comprehensive explanation of the tradeoffs involved in image compression autoencoders, how to design the latent space, and how to systematically explore the quality-compression frontier.

---

## Table of Contents

1. [The Fundamental Tradeoff](#the-fundamental-tradeoff)
2. [Information Theory Perspective](#information-theory-perspective)
3. [Latent Space Design](#latent-space-design)
4. [Architecture Impact on Compression](#architecture-impact-on-compression)
5. [Exploring the Tradeoff Space](#exploring-the-tradeoff-space)
6. [Quantization Considerations](#quantization-considerations)
7. [Practical Guidelines](#practical-guidelines)

---

## The Fundamental Tradeoff

### The Core Tension

Compression is fundamentally about removing information. More compression = less information retained.

```
Original: 256×256×1 = 65,536 values
Latent: H×W×C values

The smaller the latent, the more information must be discarded.
```

**Rate-Distortion Theory:**
For any given compression rate (bits), there's a minimum achievable distortion.
For any given distortion level, there's a minimum required rate.

```
Rate ↓ → Distortion ↑ (more compression = worse quality)
Rate ↑ → Distortion ↓ (less compression = better quality)
```

### What Gets Lost?

When compressing images, networks typically discard:

**First to go (easy to lose):**
- High-frequency noise
- Fine textures
- Subtle intensity variations

**Last to go (hard to compress):**
- Major structures and edges
- Object boundaries
- Large-scale patterns

**For SAR specifically:**
- Speckle can be compressed (it's random noise)
- Edges and structures must be preserved
- This is why SAR compresses better than you might expect!

### The Quality Cliff

Quality doesn't degrade linearly with compression:

```
Compression Ratio vs PSNR (typical behavior):

CR:     4×    8×    16×   32×   64×   128×
PSNR:   45    40    35    30    25    20    (dB)
        ↑     ↑     ↑     ↑     ↑     ↑
        Near  Very  Good  Accept Visible  Poor
        less  good        able   artifacts
```

There's often a "knee" in the curve where quality drops rapidly.

---

## Information Theory Perspective

### Entropy and Compression

**Entropy** measures the inherent information content:

```
H(X) = -Σ p(x) × log₂(p(x))
```

- High entropy: Many equally-likely values (random noise) → hard to compress
- Low entropy: Few common values (structured patterns) → easy to compress

**SAR images:**
- Speckle: High entropy (random)
- Structures: Low entropy (predictable)
- Overall: Moderate entropy, good compression potential

### Rate-Distortion Bound

For a source with distortion measure D and rate R:

```
R(D) = min_{p(x̂|x)} I(X; X̂)
subject to: E[d(X, X̂)] ≤ D
```

This defines the theoretical minimum rate for any desired distortion.

**Practical implication:**
No compression algorithm can beat this bound. Neural networks can approach it but not exceed it.

### Latent Entropy

The latent space has its own entropy:

```
H(Z) = -Σ p(z) × log₂(p(z))
```

**Well-trained autoencoder:**
- Latent values spread across the representable range
- No "dead" regions (wasted capacity)
- Entropy close to theoretical maximum for the dimensionality

---

## Latent Space Design

### Dimensions of Design

The latent space has three key dimensions:

**1. Spatial dimensions (H_lat × W_lat):**
```
256×256 input → ?×? latent

4× spatial reduction: 64×64 latent (aggressive)
16× spatial reduction: 16×16 latent (moderate)
256× spatial reduction: 1×1 latent (extreme - usually too aggressive)
```

**2. Channel dimensions (C_lat):**
```
More channels = more capacity at each spatial location
Fewer channels = higher compression
```

**3. Bit precision:**
```
Float32: 32 bits per value (training default)
Float16: 16 bits per value (inference optimization)
Int8: 8 bits per value (quantized deployment)
```

### Compression Ratio Calculation

```python
def compression_ratio(input_shape, latent_shape, input_bits=32, latent_bits=32):
    """
    input_shape: (H, W, C) or (H, W) for grayscale
    latent_shape: (H_lat, W_lat, C_lat)
    """
    input_size = np.prod(input_shape) * input_bits
    latent_size = np.prod(latent_shape) * latent_bits
    return input_size / latent_size
```

**Examples:**
```
Input: 256×256×1, 32-bit
Latent: 16×16×64, 32-bit → CR = 65536 / 16384 = 4×
Latent: 16×16×32, 32-bit → CR = 65536 / 8192 = 8×
Latent: 16×16×16, 32-bit → CR = 65536 / 4096 = 16×
Latent: 16×16×8, 32-bit → CR = 65536 / 2048 = 32×
Latent: 8×8×16, 32-bit → CR = 65536 / 1024 = 64×
```

### Spatial vs Channel Tradeoff

Two ways to achieve the same compression ratio:

**Option A: Large spatial, few channels**
```
16×16×16 = 4096 values
More spatial resolution, less feature diversity
Good for: Images with fine spatial details
```

**Option B: Small spatial, many channels**
```
8×8×64 = 4096 values
Less spatial resolution, more feature diversity
Good for: Images with complex textures/patterns
```

**For SAR:**
- Edges need spatial resolution
- Speckle is random (doesn't need spatial detail)
- Balance: 16×16 spatial with moderate channels (16-64) often works well

### Bottleneck Architecture

The encoder progressively reduces spatial size while increasing channels:

```
256×256×1 → input
128×128×64 → layer 1 (4× spatial reduction, 64× channel increase)
64×64×128 → layer 2
32×32×256 → layer 3
16×16×C → bottleneck (C is the key design choice)
```

**The bottleneck is the information choke point.**

Everything the decoder reconstructs must pass through this bottleneck.

---

## Architecture Impact on Compression

### Encoder Depth

Deeper encoders can learn more complex compressions:

```
Shallow (3 layers):
- Can't learn complex patterns
- May waste latent capacity
- Fast training

Deep (6+ layers):
- Better feature extraction
- More efficient compression
- Slower training, risk of vanishing gradients
```

**Recommendation:** 4-5 layers for 256×256 input (reaches 16×16 spatial).

### Residual Connections

Residual blocks help preserve information:

```
Without residual: Information must be re-learned at each layer
With residual: Information can pass through, network learns refinements
```

**Impact on compression:**
- Better reconstruction quality at same compression ratio
- Or same quality at higher compression ratio
- Adds parameters but worth it for quality

### Attention Mechanisms

Attention helps the network focus capacity where needed:

```
Without attention: Equal capacity for edges and flat regions
With attention: More capacity for edges, less for flat regions
```

**Impact on compression:**
- Better edge preservation
- Potentially higher effective compression
- Adds computation but not much to latent size

### Width vs Depth

**Wider networks (more channels per layer):**
```
+ More features learned at each resolution
+ More parallelism (faster on GPU)
- More parameters
- May overfit with limited data
```

**Deeper networks (more layers):**
```
+ More abstraction levels
+ Can learn more complex transforms
- Harder to train
- Diminishing returns after some depth
```

**For your 8GB GPU:** Moderate width (64-128 channels) with 4-5 layers is a good starting point.

---

## Exploring the Tradeoff Space

### Systematic Experimentation

To understand the tradeoff for your SAR data, run experiments varying:

**1. Latent channels (primary knob):**
```python
latent_channels = [8, 16, 32, 64]  # 32×, 16×, 8×, 4× compression
```

**2. Architecture (secondary):**
```python
architectures = ['plain', 'residual', 'residual+attention']
```

**3. Fixed hyperparameters:**
- Same training epochs, batch size, learning rate
- Same loss function
- Same evaluation metrics

### Experiment Design

```python
experiments = []

for latent_ch in [8, 16, 32, 64]:
    for arch in ['plain', 'residual']:
        experiments.append({
            'name': f'{arch}_latent{latent_ch}',
            'latent_channels': latent_ch,
            'architecture': arch,
            'compression_ratio': 65536 / (16 * 16 * latent_ch)
        })

# Run each experiment
results = []
for exp in experiments:
    model = create_model(exp)
    metrics = train_and_evaluate(model)
    results.append({**exp, **metrics})

# Analyze
df = pd.DataFrame(results)
```

### Plotting Rate-Distortion Curves

```python
import matplotlib.pyplot as plt

def plot_rate_distortion(results_df):
    """Plot quality vs compression."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for arch in results_df['architecture'].unique():
        data = results_df[results_df['architecture'] == arch]
        data = data.sort_values('compression_ratio')

        # PSNR vs CR
        axes[0].plot(data['compression_ratio'], data['psnr'],
                     'o-', label=arch)
        axes[0].set_xlabel('Compression Ratio')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].legend()

        # SSIM vs CR
        axes[1].plot(data['compression_ratio'], data['ssim'],
                     'o-', label=arch)
        axes[1].set_xlabel('Compression Ratio')
        axes[1].set_ylabel('SSIM')
        axes[1].legend()

        # PSNR vs BPP
        axes[2].plot(data['bpp'], data['psnr'],
                     'o-', label=arch)
        axes[2].set_xlabel('Bits Per Pixel')
        axes[2].set_ylabel('PSNR (dB)')
        axes[2].legend()

    plt.tight_layout()
    return fig
```

### Analyzing Results

**What to look for:**

1. **Diminishing returns:** Where does adding capacity stop helping?
```
If CR 8× → 16×: PSNR drops 5 dB (significant)
If CR 4× → 8×: PSNR drops 1 dB (diminishing returns)
→ CR 8× may be the sweet spot
```

2. **Architecture impact:** How much does residual/attention help?
```
Plain at CR 16×: PSNR 30 dB
Residual at CR 16×: PSNR 32 dB
→ Residual gives 2 dB improvement (worth it!)
```

3. **Quality threshold:** What's the minimum acceptable quality?
```
If PSNR > 30 dB is required:
- Plain: CR ≤ 8× achievable
- Residual: CR ≤ 16× achievable
```

---

## Quantization Considerations

### Why Quantization Matters

Training uses float32 (32 bits per value).
Deployment can use fewer bits:

```
Float32: Full precision, no loss
Float16: Half precision, minimal loss, 2× compression
Int8: 8-bit integers, some loss, 4× compression
```

### Post-Training Quantization

After training, convert latent to lower precision:

```python
def quantize_latent(latent, bits=8):
    """Quantize latent to fixed-point."""
    # Find range
    vmin, vmax = latent.min(), latent.max()

    # Scale to [0, 2^bits - 1]
    scale = (2 ** bits - 1) / (vmax - vmin + 1e-10)
    quantized = ((latent - vmin) * scale).round().astype(np.uint8)

    return quantized, vmin, vmax, scale


def dequantize_latent(quantized, vmin, vmax, scale):
    """Restore from quantized values."""
    return quantized.astype(np.float32) / scale + vmin
```

### Impact on Quality

Quantization adds noise to the latent:

```
Float32 latent: 32 bits → decoder → reconstruction
Int8 latent: 8 bits → dequantize → decoder → reconstruction
                ↑
        Quantization noise added here
```

**Typical impact:**
```
Float32 → Float16: < 0.1 dB PSNR loss
Float32 → Int8: 0.5-2 dB PSNR loss
Float32 → Int4: 2-5 dB PSNR loss
```

### Quantization-Aware Training

Train the network to be robust to quantization:

```python
class QuantizationNoise(nn.Module):
    """Add simulated quantization noise during training."""

    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits

    def forward(self, x):
        if self.training:
            # Simulate quantization noise
            scale = 2 ** self.bits - 1
            noise_magnitude = 1.0 / scale
            noise = torch.rand_like(x) * noise_magnitude - noise_magnitude / 2
            return x + noise
        return x
```

### Total Compression with Quantization

```
Example:
Input: 256×256×1 × 32 bits = 2,097,152 bits

Latent: 16×16×16 × 8 bits (quantized) = 32,768 bits

Total CR = 2,097,152 / 32,768 = 64× compression!
```

---

## Practical Guidelines

### Choosing Your Operating Point

**Step 1: Define quality threshold**
```
"I need PSNR > 30 dB for my application"
or
"I need SSIM > 0.9"
```

**Step 2: Run experiments across compression ratios**
```python
for cr in [4, 8, 16, 32, 64]:
    train_and_evaluate(compression_ratio=cr)
```

**Step 3: Find the highest CR meeting your threshold**
```
Results:
CR 64×: PSNR 25 dB (below threshold)
CR 32×: PSNR 29 dB (below threshold)
CR 16×: PSNR 33 dB (meets threshold) ← Choose this!
CR 8×: PSNR 37 dB (exceeds threshold)
```

**Step 4: Refine with architecture changes**
```
Can we achieve CR 32× with better architecture?
- Add residual blocks: CR 32× → PSNR 31 dB (meets threshold!)
```

### Recommended Starting Points

**For SAR satellite imagery:**

| Use Case | Suggested CR | Latent Config | Expected Quality |
|----------|--------------|---------------|------------------|
| High quality archive | 8× | 16×16×32 | PSNR > 35 dB |
| Bandwidth constrained | 16× | 16×16×16 | PSNR > 30 dB |
| Extreme constraint | 32× | 16×16×8 | PSNR > 27 dB |
| Preview/thumbnail | 64× | 8×8×16 | PSNR > 24 dB |

### Memory Budget

For your 8GB GPU, estimate memory:

```python
def estimate_memory(batch_size, input_size, model_params, bits=32):
    """Rough memory estimate in GB."""
    # Input batch
    input_mem = batch_size * np.prod(input_size) * bits / 8 / 1e9

    # Model parameters
    param_mem = model_params * bits / 8 / 1e9

    # Activations (roughly 2x model params for forward + backward)
    activation_mem = batch_size * model_params * bits / 8 / 1e9 * 2

    # Optimizer states (2x params for Adam)
    optimizer_mem = model_params * bits / 8 / 1e9 * 2

    total = input_mem + param_mem + activation_mem + optimizer_mem
    return total
```

**Typical values:**
```
256×256 input, 4-layer encoder/decoder, 64 base channels:
- ~5M parameters
- Batch 16: ~4 GB
- Batch 8: ~2.5 GB
- Batch 4: ~1.5 GB
```

You have room to experiment with batch size 8-16.

### Training Stability

Higher compression can make training less stable:

```
CR 8×: Easy to train, converges smoothly
CR 16×: May need learning rate tuning
CR 32×: May need warmup, careful initialization
CR 64×: May collapse, need extensive tuning
```

**Tips for high compression:**
1. Use learning rate warmup
2. Start with lower compression, fine-tune to higher
3. Use residual connections
4. Monitor for mode collapse (network outputs constant)

### Summary Recommendations

**For your SAR comparison study:**

1. **Baseline experiments:**
   - CR = [8×, 16×, 32×] with plain architecture
   - Establish baseline quality curve

2. **Architecture experiments:**
   - Add residual blocks to best baseline
   - Add attention to bottleneck
   - Compare improvement vs overhead

3. **Final selection:**
   - Choose operating point based on quality needs
   - Document tradeoffs for different use cases

4. **Reporting:**
   - Rate-distortion curves
   - Visual examples at each compression level
   - SAR-specific metrics (ENL, EPI)
