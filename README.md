# CNN Autoencoder-Based Lossy Codec for SAR Data

A learning project building toward a neural compression codec for Synthetic Aperture Radar imagery.

## Project Structure

```
├── notebooks/                    # Learning exercises and experiments
│   ├── phase1_compression_fundamentals/
│   ├── phase2_learned_compression/
│   ├── phase3_sar_fundamentals/
│   └── phase4_sar_codec/
│
├── src/                          # Reusable production code
│   ├── models/                   # Neural network architectures
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Training loops and losses
│   ├── compression/              # Entropy coding and codec utilities
│   └── evaluation/               # Metrics and evaluation tools
│
├── configs/                      # Training and experiment configs
├── scripts/                      # CLI scripts for train/compress/evaluate
├── experiments/                  # Experiment logs and results
├── data/                         # Local data (gitignored)
└── checkpoints/                  # Model weights (gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Progress

### Phase 1: Compression Fundamentals
- [ ] 1.1 Information Theory
- [ ] 1.2 Lossless Compression
- [ ] 1.3 Quantization
- [ ] 1.4 Transform Coding
- [ ] 1.5 Complete Codecs

### Phase 2: Learned Compression
- [ ] 2.1 Autoencoder Foundations
- [ ] 2.2 Quantization Problem
- [ ] 2.3 Entropy Models
- [ ] 2.4 Full Model

### Phase 3: SAR Fundamentals
- [ ] SAR Physics and Data
- [ ] SAR Preprocessing

### Phase 4: SAR Codec
- [ ] SAR-Specific Optimizations
- [ ] Final Evaluation
