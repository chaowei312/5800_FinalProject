# Recurrent vs. Standard Transformers: Parameter-Efficient Classification Across Sentiment and Domains

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **ANLY-5800 Final Project** | Georgetown University | Fall 2025  
> **Team**: Chenxi Guo, Jiayi Peng, Chaowei Wang, Junchen Han

---

## Overview

This project compares **Recurrent Transformer** and **Standard Transformer** architectures for parameter-efficient text classification. Our recurrent model achieves comparable performance with **64% fewer parameters**.

---

## Project Structure

```
5800_FinalProject/
├── app/                      # Web app & CLI inference
│   ├── web_app.py           # Flask web interface
│   ├── cli.py               # Command-line interface
│   ├── inference.py         # Inference classes
│   └── SwiGLU_demo/         # Interactive activation demo
├── models/                   # Model implementations
│   ├── baseline/            # Standard Transformer
│   ├── recurrent/           # Recurrent Transformer
│   └── modules/             # Custom components (Flash Attention, SwiGLU, RoPE, RMSNorm)
├── training/                 # Training scripts
│   ├── train_baseline.py
│   └── train_recurrent.py
├── evaluation/               # Evaluation scripts
├── data/                     # Datasets (SST-2, Yelp, Multi-domain)
├── notebooks/                # Analysis notebooks
├── configs/                  # Model configurations
└── requirements.txt
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/chaowei312/5800_FinalProject.git
cd 5800_FinalProject

# Install dependencies
pip install -r requirements.txt

# Verify installation
python app/test_installation.py
```

---

## Models

### Baseline Transformer
- Architecture: 6 layers × 384 hidden dim × 6 heads
- Parameters: ~9.8M
- Features: Flash Attention, SwiGLU, RoPE, RMSNorm

### Recurrent Transformer
- Architecture: 3 layers × 2 iterations × 256 hidden dim × 4 heads
- Parameters: ~3.4M (64% reduction)
- Cross-iteration residuals: $h^{(i+1)} = \text{TransformerLayers}(h^{(i)}) + 0.5 \cdot h^{(i)}$

**Key Innovation**: Achieves effective depth of 6 layers by iterating through 3 layers twice, dramatically reducing parameters.

---

## Training

### Basic Training

```bash
# Train baseline model
python training/train_baseline.py \
    --data_dir data/processed \
    --output_dir checkpoints/baseline \
    --config configs/bert_small_config.json \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 10

# Train recurrent model
python training/train_recurrent.py \
    --data_dir data/processed \
    --output_dir checkpoints/recurrent \
    --config configs/recurrent_config_template.json \
    --recurrent_depth 2
```

### Key Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--batch_size` | Training batch size | 32 |
| `--learning_rate` | Initial learning rate | 2e-5 |
| `--num_epochs` | Maximum epochs | 10 |
| `--early_stopping_patience` | Early stopping patience | 3 |
| `--mixed_precision` | Use FP16 training | False |

**Training Features**: Automatic checkpointing, early stopping, learning rate scheduling, gradient clipping

---

## Evaluation

```bash
# Evaluate model
python evaluation/eval.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --data_dir data/processed \
    --split test
```

**Metrics**: Accuracy, F1 Score, Precision, Recall, Inference Time, Model Size

---

## Experiments

We conducted five experiments to evaluate parameter efficiency and robustness:

1. **Data Size Sensitivity**: Train on 10%, 50%, 100% of SST-2
   - Recurrent model shows better sample efficiency at low data regimes

2. **Text Length Robustness**: Short vs long sequence performance
   - Recurrent model excels on longer sequences

3. **Cross-Domain Generalization**: SST-2 (movies) → Yelp (business)
   - Recurrent model: +2.4% better transfer accuracy

4. **Parameter-Matched Comparison**: Same parameter budget
   - Recurrent architecture: +1.5% accuracy at matched parameters

5. **Multi-Domain Classification**: 3-class domain identification
   - Both models achieve >87% accuracy

**Key Insight**: Recurrent transformers maintain performance with significantly fewer parameters, especially beneficial for deployment scenarios with memory constraints.

---

## Custom Implementations

We implement 5 state-of-the-art components from scratch:

1. **Recurrent Transformer**: Iterative layer processing with cross-iteration residuals
2. **Flash Attention**: Memory-efficient O(N) attention
3. **SwiGLU Activation**: Swish-Gated Linear Units
4. **RoPE**: Rotary Position Embeddings
5. **RMSNorm**: Efficient normalization

See [`proposal.md`](proposal.md) for mathematical formulations and [`app/ARCHITECTURE.md`](app/ARCHITECTURE.md) for system design.

---

## Interactive Demo

### SwiGLU Visualization

```bash
cd app/SwiGLU_demo
python -m http.server 8080
```

Open: [http://localhost:8080](http://localhost:8080)

Real-time visualization of Swish-Gated Linear Unit activation with adjustable parameters.



