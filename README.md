# CNN Autoencoder for SAR Image Compression

A neural compression codec for Sentinel-1 SAR (Synthetic Aperture Radar) imagery using CNN autoencoders.

## Project Structure

```
├── notebooks/                    # Training notebooks
│   ├── train_baseline.ipynb      # Baseline autoencoder training
│   ├── train_resnet.ipynb        # ResNet-Lite autoencoder training
│   ├── checkpoints/              # Trained model weights
│   └── runs/                     # TensorBoard logs
│
├── learningnotebooks/            # Learning exercises
│   ├── phase1_compression_fundamentals/
│   ├── phase2_learned_compression/
│   └── phase4_sar_codec/
│
├── src/                          # Source code
│   ├── models/                   # Neural network architectures
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Training loops
│   ├── losses/                   # Loss functions (MSE + SSIM)
│   ├── evaluation/               # SAR metrics and evaluation tools
│   └── compression/              # Entropy coding utilities
│
├── scripts/                      # CLI scripts
│   └── evaluate_model.py         # Model evaluation script
│
├── data/                         # SAR data (gitignored)
│   └── patches/                  # Preprocessed 256x256 patches
│
├── checkpoints/                  # Legacy checkpoints (gitignored)
└── configs/                      # Configuration files
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Results

| Model | Parameters | PSNR | SSIM | Compression |
|-------|------------|------|------|-------------|
| Baseline | 2.3M | 20.5 dB | 0.646 | 16x |
| ResNet-Lite | 5.6M | 21.2 dB | 0.726 | 16x |

## Evaluation

SAR-specific metrics:
- **ENL Ratio**: Equivalent Number of Looks preservation (0.851)
- **EPI**: Edge Preservation Index (0.876)

Run evaluation:
```bash
python scripts/evaluate_model.py --checkpoint notebooks/checkpoints/resnet_lite_v2_c16/best.pth
```

## TensorBoard

```bash
tensorboard --logdir=notebooks/runs
```

## Learning Progress

### Phase 1: Compression Fundamentals
- Information Theory
- Lossless Compression
- Quantization
- Transform Coding

### Phase 2: Learned Compression
- Autoencoder Foundations
- Quantization Problem
- Entropy Models

### Phase 4: SAR Codec
- SAR-Specific Optimizations
- Final Evaluation
