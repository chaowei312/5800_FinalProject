# ANLY-5800 Final Project – Proposal Template

Use this as a guide for your **2-page max** project proposal (one per group). You can write it directly in this file or a separate PDF in your repo.

---

## 1. Team

- **Project title**: Recurrent vs Standard Transformers: A Comparative Analysis of Parameter Efficiency in Sentiment Classification
- **Team members**: [Names & NetIDs]
- **Preferred track**: **(D) Analysis** - Architectural comparison and efficiency study

---

## 2. Problem statement & motivation

- **Task**: What are you trying to do? (e.g., sentiment classification, QA, code generation, tool-using agent)
  - Sentiment classification using a recurrent Transformer architecture for better parameter efficiency.

### Mathematical Formulations

#### Baseline Transformer
The baseline model implements a standard multi-layer transformer with pre-norm architecture:

$$\mathbf{f}: \mathcal{X} \rightarrow \mathcal{Y}, \quad \mathbf{f}(\mathbf{x}) = \text{softmax}(\mathbf{W}_c \cdot \text{TransformerStack}(\text{Embed}(\mathbf{x})))$$

**Input Embedding:**
$$\mathbf{h}_0 = \text{LayerNorm}(\mathbf{W}_e[\mathbf{x}] + \mathbf{P}(\text{pos}) + \mathbf{T}(\text{type}))$$

**Transformer Layer** (pre-norm residual):
$$\begin{align}
\mathbf{h}'_l &= \mathbf{h}_{l-1} + \text{Dropout}(\text{MultiHeadAttn}(\text{Norm}_1(\mathbf{h}_{l-1}))) \\
\mathbf{h}_l &= \mathbf{h}'_l + \text{Dropout}(\text{FFN}(\text{Norm}_2(\mathbf{h}'_l)))
\end{align}$$

where $\text{Norm} \in \{\text{LayerNorm}, \text{RMSNorm}\}$ based on configuration.

**Multi-Head Attention:**
$$\text{MultiHeadAttn}(\mathbf{x}) = \mathbf{W}_O \cdot \text{Concat}(\text{head}_1, ..., \text{head}_h)$$

where each attention head is computed as:
$$\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax}\left(\frac{\mathbf{Q}_i\mathbf{K}_i^T}{\sqrt{d_k}}\right)\mathbf{V}_i$$

with projections:
$$\mathbf{Q}_i = \mathbf{x}\mathbf{W}^Q_i, \quad \mathbf{K}_i = \mathbf{x}\mathbf{W}^K_i, \quad \mathbf{V}_i = \mathbf{x}\mathbf{W}^V_i$$

**Feed-Forward Network** (with optional SwiGLU):
$$\text{FFN}(\mathbf{x}) = \begin{cases}
\text{SwiGLU}(\mathbf{x}) & \text{if use\_swiglu} \\
\mathbf{W}_2 \cdot \text{GELU}(\mathbf{x}\mathbf{W}_1) + \mathbf{b}_2 & \text{otherwise}
\end{cases}$$

where $\text{GELU}(x) = x \cdot \Phi(x)$ and $\Phi$ is the standard Gaussian CDF.

**Classification:**
$$\mathbf{y} = \arg\max(\mathbf{W}_c \cdot \mathbf{h}_L[0, :])$$

#### Recurrent Transformer
The recurrent model iterates through fewer layers multiple times with cross-iteration residuals:

$$\mathbf{g}: \mathcal{X} \rightarrow \mathcal{X}, \quad \mathbf{g}^{(t)}(\mathbf{x}) = \mathbf{x}^{(t+1)}$$

**Recurrent Processing:**
$$\mathbf{h}^{(r+1)} = \text{TransformerLayers}(\mathbf{h}^{(r)}) + \alpha \cdot \mathbf{h}^{(r)}$$

where $r \in [1, R_{\text{depth}}]$ is the recurrence iteration and $\alpha = 0.5$ is the residual scale.

**Effective Depth:**
$$D_{\text{eff}} = N_{\text{layers}} \times R_{\text{depth}}$$

**Parameter Efficiency:**
$$|\theta_{\text{recurrent}}| = \frac{|\theta_{\text{baseline}}|}{R_{\text{depth}}} + \epsilon$$

where $\epsilon$ accounts for additional residual connections.

### Custom Modules Implementation

#### 1. **Rotary Position Embeddings (RoPE)**
Encodes position information through rotation matrices in complex space:

$$\text{RoPE}(\mathbf{x}_m, m) = \mathbf{x}_m \odot e^{im\theta}$$

where the rotation frequencies are:
$$\theta_j = 10000^{-2j/d}, \quad j \in [0, d/2]$$

Applied to query-key pairs:
$$\mathbf{q}_m' = \text{RoPE}(\mathbf{q}_m, m), \quad \mathbf{k}_n' = \text{RoPE}(\mathbf{k}_n, n)$$

#### 2. **Flash Attention**
Memory-efficient attention using tiled computation with recomputation:

$$\text{FlashAttn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

- Memory complexity: $O(N)$ instead of $O(N^2)$
- Uses `F.scaled_dot_product_attention` when available
- Automatic fallback to standard attention with attention weight caching

#### 3. **SwiGLU Activation**
Gated linear unit with Swish activation for improved gradient flow:

$$\text{SwiGLU}(\mathbf{x}) = (\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \odot \text{Swish}(\mathbf{x} \mathbf{W}_2 + \mathbf{b}_2)$$

where Swish activation is:
$$\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$

#### 4. **RMS Normalization**
Root Mean Square normalization for training stability:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma$$

where:
$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

Compare with standard LayerNorm:
$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu(\mathbf{x})}{\sigma(\mathbf{x})} \cdot \gamma + \beta$$

- RMSNorm is more efficient: $O(d)$ vs $O(2d)$ operations
- No mean centering, only variance normalization
- Empirically similar performance with lower computational cost

### Tokenization & Embedding Methods

#### WordPiece Tokenization
Text is tokenized using BERT's WordPiece algorithm:
$$\text{text} \xrightarrow{\text{WordPiece}} [\text{CLS}, t_1, t_2, ..., t_n, \text{SEP}]$$

where $t_i \in \mathcal{V}$ and $|\mathcal{V}| = 30,522$.

#### Embedding Composition
The input embedding combines three components:
$$\mathbf{E}(\mathbf{x}) = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{position}} + \mathbf{E}_{\text{segment}}$$

where:
- $\mathbf{E}_{\text{token}} \in \mathbb{R}^{d_{\text{model}}}$: Token embeddings
- $\mathbf{E}_{\text{position}} \in \mathbb{R}^{d_{\text{model}}}$: Position embeddings (or RoPE)
- $\mathbf{E}_{\text{segment}} \in \mathbb{R}^{d_{\text{model}}}$: Segment/type embeddings

#### Model Dimensions
- **Baseline**: $d_{\text{model}} = 384$, $d_{\text{ff}} = 1536$
- **Recurrent**: $d_{\text{model}} = 256$, $d_{\text{ff}} = 1024$
- **Max Sequence Length**: $L_{\text{max}} = 128$ (configurable up to 512)
  
- **Why it matters**: Briefly explain why this problem is interesting or important (scientifically or practically).
  - Sentiment signals enable downstream industrial pipelines, such as education grading and evaluation workflows.
  - A recurrent Transformer can generalize better while supporting reasoning-style, stepwise conditioning in a compact model.
- **Desired outcome**: What will success look like in 3 weeks?
  - Benchmark against an encoder baseline, demonstrating measurable performance gains while keeping the recurrent model smaller and within a comparable training budget.
---

## 3. Datasets

- **Primary dataset(s)**:
  - *SST-2 (Stanford Sentiment Treebank v2)* – [GLUE benchmark](https://huggingface.co/datasets/glue/viewer/sst2/train) with 67k English movie-review sentences labeled positive/negative.
  - *Yelp Review Polarity* (optional scale-up) – [Hugging Face](https://huggingface.co/datasets/yelp_polarity) release with 560k English business reviews (binary sentiment).
- **Preprocessing**:
  - **Normalization**: Standardize text by lowercasing and normalizing whitespace. Truncate sequences to 256 tokens to optimize GPU memory usage.
  - **Tokenization**: Use Hugging Face `AutoTokenizer` (initialized from the baseline checkpoint) or the `tokenizers` library/SentencePiece to map text to token IDs.
  - **Label Mapping (Yelp)**: If using Yelp, map 4–5 stars → positive and 1–2 stars → negative; filter out reviews >512 tokens.
- **Train/val/test split**:
  - SST-2: follow GLUE splits (≈67k train, 872 dev, 1,821 test) and carve out 5% of the train set as an internal validation set for hyperparameters.
  - Yelp: use official 560k train / 38k test split and reserve 10k samples from the train portion as validation when needed.

---

## 4. Baseline

- **Baseline model/system**: What is the simplest reasonable model you will implement in Week 1?
  - BERT-style encoder-only transformer with 6 layers, 384 hidden dimensions, 6 attention heads
  - Total parameters: ~9.8M (baseline) vs ~3.4M (recurrent with same effective depth)
  
- **Baseline metrics**: What metric(s) will you report (accuracy, F1, BLEU/ROUGE, perplexity, etc.)?
  - **Primary**: Accuracy, F1-score
  - **Secondary**: Precision, Recall, Matthews Correlation Coefficient (MCC)
  - **Efficiency**: Inference time (ms/sample), Model size (MB), FLOPs per forward pass
  - **Training**: Loss curves, convergence speed, early stopping behavior

---

## 5. Approach (beyond baseline)

### Core Improvements Implemented

1. **Recurrent Transformer Architecture**
   - Iterate through fewer transformer layers multiple times (e.g., 3 layers × 2 iterations)
   - Residual connections across iterations: **h^{(i+1)} = TransformerLayers(h^{(i)}) + 0.5 · h^{(i)}**
   - Achieves 64% parameter reduction while maintaining comparable performance

2. **Advanced Optimization Techniques**
   - Early stopping with patience=3 and validation loss tracking
   - Learning rate scheduling with ReduceLROnPlateau
   - Gradient clipping (max_norm=1.0) for stable training
   - Mixed precision training support for faster computation

3. **Hyperparameter Sensitivity Analysis**
   - Grid search over architectural parameters (hidden_size, num_layers, recurrent_depth)
   - Bubble plot visualizations: Accuracy vs Model Size vs Inference Time
   - Pareto frontier analysis for optimal configuration selection

4. **Width vs Depth Trade-off Study**
   - Compare wide-shallow (512×6) vs narrow-deep (384×3×2) architectures
   - Analyze parameter efficiency: accuracy per million parameters
   - Measure inference latency differences at similar computational budgets

---

## 6. Compute & resources

- **Will you use Jetstream2?** No
- **Rough plan**: 
  - Model sizes: 3-10M parameters
  - Batch size: 16-32 samples
  - Training time: ~30 minutes per model on GPU
- **Other resources**: Local NVIDIA GPU (CUDA-enabled), CPU fallback supported

---

## 7. Risks & scope

- **What could go wrong?** (e.g., data too noisy, model too big to train, evaluation too hard)
- **Plan B**: If your original idea is too ambitious, what scaled-down version will you execute?

---

## 8. Milestones

Very briefly, list what you plan to achieve by:

- **End of Week 1**: Implement baseline transformer and core modules (RoPE, Flash Attention, SwiGLU, RMSNorm)
- **End of Week 2**: Develop recurrent transformer, training pipeline with early stopping, evaluation framework
- **End of Week 3**: Hyperparameter tuning, comparative analysis, visualization dashboard, final report

These align with the course-wide milestones in `project/README.md`.
