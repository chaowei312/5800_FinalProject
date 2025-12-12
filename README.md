# Recurrent vs. Standard Transformers: Parameter and Compute-Efficient Classification Across Sentiment and Domains

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **ANLY-5800 Final Project** | Georgetown University | Fall 2025  
> **Team**: Chenxi Guo, Jiayi Peng, Chaowei Wang, Juncheng Han

---

## Overview

This project compares **Recurrent Transformer** and **Standard Transformer** architectures for parameter-efficient text classification. Our recurrent model achieves comparable performance with **64% fewer parameters**.

---

## Project Structure

```
5800_FinalProject/
├── app/                  # Web app & CLI
│   ├── web_app.py
│   ├── cli.py
│   └── SwiGLU_demo/
├── models/               # Baseline, recurrent & custom modules
│   ├── baseline/
│   ├── recurrent/
│   └── modules/ (Flash Attention, SwiGLU, RoPE, RMSNorm)
├── training/             # Training pipeline and utilities
├── evaluation/           # Evaluation and metrics
├── data/                 # Preprocessed datasets
├── configs/              # Model configs
├── notebooks/            # Analysis & result notebooks
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
    --num_epochs 5

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


We conducted five core experiments to systematically evaluate parameter efficiency and robustness of recurrent Transformers under realistic constraints:

1. **Data Size Sensitivity**
   Models are trained on 10%, 50%, and 100% of the SST-2 training set.
   The recurrent architecture demonstrates improved sample efficiency and more stable performance in low-data regimes.

2. **Text Length Sensitivity**
   Performance is evaluated separately on short and long input sequences derived from SST-2.
   The recurrent model consistently performs better on longer sequences, suggesting that iterative refinement effectively captures extended contextual dependencies.

3. **Cross-Domain Generalization**
   Models trained on SST-2 (movie reviews) are evaluated on Yelp Review Polarity (business reviews).
   The recurrent model achieves stronger transfer performance, indicating improved robustness to domain shift.


4. **Multi-Domain Classification**
   Models are evaluated on a unified three-class review dataset (movies, local business, online shopping).
   Both architectures achieve over 87% accuracy, demonstrating the practicality of recurrent Transformers beyond binary sentiment classification.

5. **Supplementary Deployment Analysis**
   As an auxiliary experiment, we evaluate FP16 quantization to assess deployment efficiency.
   Results show stable accuracy under half-precision inference with significant reductions in memory footprint and inference latency.


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


### Text Classification Web Application

```bash
python web_app.py
````

Then open the browser at: [http://localhost:5000](http://localhost:5000)

### SwiGLU Visualization

```bash
cd app/SwiGLU_demo
python -m http.server 8080
```

Open: [http://localhost:8080](http://localhost:8080)

Real-time visualization of Swish-Gated Linear Unit activation with adjustable parameters.



