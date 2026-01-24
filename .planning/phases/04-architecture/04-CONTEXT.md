# Phase 4: Architecture Enhancement - Context

**Gathered:** 2026-01-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement residual blocks and CBAM attention modules, then train two enhanced architecture variants (Variant B: Residual, Variant C: Res+CBAM) to demonstrate quality improvements over baseline. Target: +1.5 dB PSNR for residual, +0.5 dB more for CBAM, while maintaining ENL ratio within acceptable range.

</domain>

<decisions>
## Implementation Decisions

### Residual Block Design
- Use **basic block** structure (two 3x3 convs with skip connection), not bottleneck
- **2 residual blocks per encoder/decoder stage** (4 stages = 8 blocks per encoder/decoder)
- Skip connections **within stage only** — no spanning across downsample/upsample operations
- **Pre-activation** ordering (BN → ReLU → Conv), ResNet-v2 style
- **Decoder mirrors encoder exactly** — same number of residual blocks, reversed
- **ReLU activation** (not LeakyReLU) for residual blocks
- **No dropout** — rely on BatchNorm for regularization
- **3x3 kernel size** for convolutions inside residual blocks

### CBAM Integration
- Apply CBAM **after every residual block** (maximum attention coverage)
- **Reduction ratio 16** for channel attention (standard CBAM paper value)
- Apply CBAM in **both encoder and decoder**
- **7x7 spatial attention kernel** (standard CBAM paper value)
- CBAM positioned **after residual block** (not inside), cleaner separation
- **No BatchNorm inside CBAM's MLP** — follow original CBAM paper (FC → ReLU → FC)

### Training Strategy
- **Train from scratch** — fair comparison, all variants start with random initialization
- **Loss weights: 0.7 MSE + 0.3 SSIM** — emphasize pixel accuracy for PSNR improvement
- **Same 20% data subset** as baseline — fair comparison, only architecture differs
- Train Variants **sequentially** — Residual first, then Res+CBAM

### Claude's Discretion
- Learning rate adjustment for deeper networks (baseline was 1e-3)
- Number of epochs (baseline was 30) — may train to convergence with early stopping
- Batch size adjustment based on model size and VRAM usage (baseline was 32 with AMP)

### Variant Comparison
- Train **both variants** (Variant B: Residual, Variant C: Res+CBAM) in this phase
- Compare against **both** original baseline (20.47 dB) and ResNet-Lite v2 (21.2 dB)
- **Full analysis** reporting: metrics table, visual comparison, plus per-region analysis
- **Primary success metric: PSNR** — match success criteria targets
- **Minor ENL deviation acceptable** (0.7-1.3) if PSNR gain is substantial

</decisions>

<specifics>
## Specific Ideas

- Pre-activation residual blocks (BN-ReLU-Conv) for better gradient flow
- CBAM after every block provides maximum attention coverage — want to see if this helps SAR feature extraction
- Loss weight shift toward MSE (0.7/0.3) to push PSNR higher, since that's the primary target metric
- Sequential training validates residual benefit before adding CBAM complexity

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-architecture*
*Context gathered: 2026-01-24*
