# ANLY-5800 Final Project – Proposal

---

## 1. Team

- **Project title**: Recurrent vs. Standard Transformers: Parameter-Efficient Classification Across Sentiment and Domains
- **Team members**: 
  - Chenxi Guo & cg1372
  - Jiayi Peng & jp2132
  - Chaowei Wang & cw1278
  - Junchen Han & jh2732
- **Preferred track**: **(D) Analysis** - Architectural comparison and efficiency study

---

## 2. Problem statement & motivation

- **Task**: What are you trying to do? (e.g., sentiment classification, QA, code generation, tool-using agent)
  We aim to implement and evaluate a Recurrent Transformer architecture for parameter-efficient text classification.

  Primary Task: Binary Sentiment Classification (positive/negative).

  Generalization Task: Domain Classification (identifying the source of the review: movies, business, or shopping) to test the architecture's capacity to encode semantic topic information beyond sentiment.

### Mathematical Formulations

#### Baseline Transformer
The baseline model implements a standard multi-layer transformer with pre-norm architecture:

$$\mathbf{f}: \mathcal{X} \rightarrow \mathcal{Y}, \quad \mathbf{f}(\mathbf{x}) = \text{softmax}(\mathbf{W}_c \cdot \text{TransformerStack}(\text{Embed}(\mathbf{x})))$$

**Input Embedding:**

$$\mathbf{h}_0 = \text{LayerNorm}(\mathbf{W}_e[\mathbf{x}] + \mathbf{P}(\text{pos}) + \mathbf{T}(\text{type}))$$

**Transformer Layer** (pre-norm residual):

$$
\begin{align}
\mathbf{h}'_l &= \mathbf{h}_{l-1} + \text{Dropout}(\text{MultiHeadAttn}(\text{Norm}_1(\mathbf{h}_{l-1}))) \\
\mathbf{h}_l &= \mathbf{h}'_l + \text{Dropout}(\text{FFN}(\text{Norm}_2(\mathbf{h}'_l)))
\end{align}
$$

where $\text{Norm} \in \{\text{LayerNorm}, \text{RMSNorm}\}$ based on configuration.

**Multi-Head Attention:**

$$\text{MultiHeadAttn}(\mathbf{x}) = \mathbf{W}_O \cdot \text{Concat}(\text{head}_1, ..., \text{head}_h)$$

where each attention head is computed as:

$$\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax}\left(\frac{\mathbf{Q}_i\mathbf{K}_i^T}{\sqrt{d_k}}\right)\mathbf{V}_i$$

with projections:

$$\mathbf{Q}_i = \mathbf{x}\mathbf{W}^Q_i, \quad \mathbf{K}_i = \mathbf{x}\mathbf{W}^K_i, \quad \mathbf{V}_i = \mathbf{x}\mathbf{W}^V_i$$

**Feed-Forward Network (with optional SwiGLU)**


$$
\mathrm{FFN}(\mathbf{x}) =
\begin{cases}
\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) & \text{if } \mathrm{use\_swiglu}=1, \\[6pt]
\mathbf{W}_2\, \mathrm{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) + \mathbf{b}_2 
& \text{if } \mathrm{use\_swiglu}=0.
\end{cases}
$$

$$
\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x})
= 
\mathbf{W}_2 \Big[
(\mathbf{x}\mathbf{W}_1^{(1)} + \mathbf{b}_1^{(1)})
\odot 
\mathrm{SiLU}(\mathbf{x}\mathbf{W}_1^{(2)} + \mathbf{b}_1^{(2)})
\Big]
+ \mathbf{b}_2.
$$


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

where each component corresponds to:

- **Token embeddings**: $\mathbf{E}_{\text{token}} \in \mathbb{R}^{d_{\text{model}}}$
- **Position embeddings**: $\mathbf{E}_{\text{position}} \in \mathbb{R}^{d_{\text{model}}}$ (or RoPE)
- **Segment embeddings**: $\mathbf{E}_{\text{segment}} \in \mathbb{R}^{d_{\text{model}}}$

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
  - *SST-2 (Stanford Sentiment Treebank v2)* – [GLUE benchmark](https://huggingface.co/datasets/glue/viewer/sst2/train) containing ~67k English movie-review sentences labeled as positive or negative.
  - *Yelp Review Polarity (subset)* – We extract a 67k subset from the [Hugging Face](https://huggingface.co/datasets/yelp_polarity) release after filtering out all 3-star reviews. Labels are remapped such that 4–5 stars → positive and 1–2 stars → negative.

- **Auxiliary dataset(s)**:
  - *Amazon Polarity (subset)* – From the MTEB Amazon Polarity corpus[MTEB](https://huggingface.co/datasets/mteb/amazon_polarity).  
    We construct a balanced ~22k subset by filtering out 2–3 star reviews and mapping 1-star → negative, 4–5 stars → positive. Domain tag: **“online shopping”**.

  - *Unified Multi-Domain Review Dataset (67k)* – A composite dataset built by sampling one-third from each source:
      - SST-2 (domain: **“movies”**),
      - Yelp Review Polarity (domain: **“local business”**),
      - Amazon Polarity subset (domain: **“online shopping”**).  
    This dataset enables domain-aware evaluation, supporting both binary sentiment prediction and **three-class domain classification**.

  These auxiliary resources allow us to test how recurrent vs. standard Transformers generalize across review types and to explore broader **review monitoring** applications.



- **Preprocessing**:
  - **Normalization**: Lowercasing, whitespace normalization, HTML/URL removal, and truncation to 256 tokens to maintain consistent GPU memory usage.
  - **Tokenization**: Apply Hugging Face `AutoTokenizer` (initialized from the baseline model checkpoint) or a SentencePiece-based tokenizer to convert text into token IDs.
  - **Yelp Filter & Label Mapping**: Remove 3-star reviews entirely; convert 1–2 stars to negative and 4–5 stars to positive.

- **Train/val/test split**:
  - **SST-2**: Follow GLUE splits (~67k train, 872 dev, 1,821 test) and create an additional 5% internal validation split from the train portion for hyperparameter tuning.
  - **Yelp subset (67k)**: After filtering and subsampling to 67k reviews, perform an **80/10/10** split into train/validation/test to mirror the SST-2 scale.


---

## 4. Baseline

- **Baseline model/system**: What is the simplest reasonable model you will implement in Week 1?
  - BERT-style encoder-only transformer with 6 layers, 384 hidden dimensions, 6 attention heads
  - Total parameters: ~9.8M (baseline) vs ~3.4M (recurrent with same effective depth)
  
- **Baseline metrics**: What metric(s) will you report (accuracy, F1, BLEU/ROUGE, perplexity, etc.)?
  - **Primary**: Accuracy, F1-score
  - **Secondary**: Precision, Recall
  - **Efficiency**: Inference time (ms/sample), Model size (MB)
  - **Training**: Loss curves, convergence speed, early stopping behavior

---

## 5. Approach (beyond baseline)

### Core Improvements Implemented

1. **Recurrent Transformer Architecture**
   - Iterate through fewer transformer layers multiple times (e.g., $3$ layers $\times$ $2$ iterations)
   - Residual connections across iterations: $\mathbf{h}^{(i+1)} = \text{TransformerLayers}(\mathbf{h}^{(i)}) + 0.5 \cdot \mathbf{h}^{(i)}$
   - Achieves 64% parameter reduction while maintaining comparable performance

2. **Advanced Optimization Techniques**
   - Early stopping with patience=3 and validation loss tracking
   - Learning rate scheduling with ReduceLROnPlateau
   - Gradient clipping ($\|\nabla\|_{\text{max}} = 1.0$) for stable training
   - Mixed precision training support for faster computation

3. **Hyperparameter Sensitivity Analysis**
   - Grid search over architectural parameters ($d_{\text{model}}$, $N_{\text{layers}}$, $R_{\text{depth}}$)
   - Bubble plot visualizations: Accuracy vs Model Size vs Inference Time
   - Pareto frontier analysis for optimal configuration selection

4. **Width vs Depth Trade-off Study**
   - Compare wide-shallow ($512 \times 6$) vs narrow-deep ($384 \times 3 \times 2$) architectures
   - Analyze parameter efficiency: accuracy per million parameters
   - Measure inference latency differences at similar computational budgets

### Experiments

We conduct a series of controlled experiments to evaluate the parameter efficiency and robustness of recurrent Transformers compared to standard Transformer baselines.

1. **Data Size Sensitivity (50% and 10% SST-2 subsamples)**  
   To examine how model architectures behave under limited data, we create two random SST-2 subsamples containing **50%** and **10%** of the original training set.  
   This experiment isolates the effect of dataset scale on:
   - convergence stability,  
   - sample efficiency,  
   - and whether the recurrent Transformer maintains accuracy under scarce supervision. 

2. **Short vs. Long Text Sensitivity (SST-2 length-based subsets)**  
   To examine how parameter sharing interacts with input length, we construct two subsets of SST-2 based on tokenized sequence length:  
   - **Short-text subset** (lowest 30%)  
   - **Long-text subset** (highest 30%)  
   We train baseline and recurrent models separately on each subset and analyze differences in convergence behavior, accuracy, and inference speed.  
   This allows us to assess whether recurrent refinement offers advantages for longer sequences under fixed model capacity.

3. **Cross-Domain Robustness (Yelp Review Polarity subset)**  
   To evaluate generalization outside the movie-review domain, we train both architectures on a 67k Yelp subset (after filtering and balancing).  
   By comparing performance across SST-2 → Yelp, we analyze domain shift effects and determine whether recurrent Transformers retain competitive accuracy under distributional changes.

4. **Same-Parameter Comparison (Recurrent vs. Standard with matched parameter budget)**  
   Beyond the default architecture comparison, we construct a standard Transformer whose parameter count is matched to the recurrent model.  
   This experiment isolates the architectural effect by ensuring both models operate under identical parameter budgets, enabling a clean evaluation of parameter efficiency.

5. **Extension: Three-Class Domain Classification (Unified 67k review dataset)**  
   Finally, we use our unified multi-domain review dataset (movies, local business, online shopping) to test the models on a **three-class domain classification** task.  
   This serves as an auxiliary application demonstrating how far parameter-efficient recurrent Transformers can generalize beyond binary sentiment classification.  
   It also provides insight into the practicality of such architectures for broader **review monitoring** scenarios.

Together, these experiments provide a comprehensive evaluation across task difficulty, text length, domain shift, and auxiliary classification settings, allowing us to characterize the conditions under which recurrent Transformers deliver meaningful parameter savings.


---

## 6. Compute & resources

- **Will you use Jetstream2?** No
- **Rough plan**: 
  - Model sizes: 3-10M parameters
  - Batch size: 16-32 samples
  - Training time: ~30 minutes per model on GPU
- **Other resources**: NVIDIA GPU 1 RTX 5090; Colab

---

## 7. Risks & scope

- **What could go wrong?** 
  - **Convergence Issues**: Recurrent model might fail to converge when reusing same parameters across deeper iterations
    - Risk of gradient vanishing/exploding through repeated parameter application
    - Potential representation collapse when $R_{\text{depth}} > 3$
  - **Training Instability**: Cross-iteration residuals ($\alpha \cdot \mathbf{h}^{(r)}$) may cause optimization challenges
  - **Performance Degradation**: Deeper recurrence might not improve or even hurt performance vs baseline

- **Plan B**: 
  - **Hierarchical Conditioning**: Introduce a separate lightweight transformer between recurrence steps:

    $$\mathbf{h}^{(r+1)} = \text{PersistentBlock}(\text{ConditioningLayer}_r(\mathbf{h}^{(r)}))$$
    
    where $\text{ConditioningLayer}_r$ is a small transformer that adds inter-iteration conditioning
  - **Adaptive Residual Scaling**: Learn $\alpha_r$ per iteration instead of fixed $\alpha = 0.5$
  - **Fallback to Standard Architecture**: If recurrent approach fails, compare different depth/width trade-offs in standard transformers

---

## 8. Milestones

Very briefly, list what you plan to achieve by:

- **End of Week 1**: Implement baseline transformer and core modules (RoPE, Flash Attention, SwiGLU, RMSNorm)
- **End of Week 2**: Develop recurrent transformer, training pipeline with early stopping, evaluation framework
- **End of Week 3**: comparative analysis, final report

These align with the course-wide milestones in `project/README.md`.
