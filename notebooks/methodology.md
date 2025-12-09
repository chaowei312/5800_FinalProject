# Methodology: Transformer Architecture Implementation Details

This document describes the implementation details of the Standard Transformer and Recurrent Transformer models used in this project, including modern architectural enhancements.

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [SwiGLU Activation](#swiglu-activation)
3. [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
4. [RMSNorm](#rmsnorm)
5. [Standard vs Recurrent Transformer](#standard-vs-recurrent-transformer)
6. [CLS Token Usage](#cls-token-usage)
7. [Configuration Options](#configuration-options)

---

## Model Overview

Both models follow a BERT-like encoder architecture with the following enhancements:

| Component | Traditional BERT | Our Implementation |
|-----------|------------------|-------------------|
| Position Encoding | Learned Absolute | **RoPE** (Rotary) |
| FFN Activation | GELU | **SwiGLU** |
| Normalization | LayerNorm | **RMSNorm** |
| Normalization Position | Post-Norm | **Pre-Norm** |
| Attention | Standard | Flash Attention (optional) |

---

## SwiGLU Activation

### What is SwiGLU?

SwiGLU (Swish-Gated Linear Unit) is a gated activation function that combines the Swish activation with a gating mechanism. It was introduced in [Shazeer, 2020](https://arxiv.org/abs/2002.05202) and is used in modern transformers like LLaMA and PaLM.

### Mathematical Formulation

$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_2 x)$$

Where:
- $\text{Swish}(x) = x \cdot \sigma(x) = x \cdot \text{sigmoid}(x)$ (also called SiLU)
- $\odot$ denotes element-wise (Hadamard) multiplication
- $W_1, W_2, W_3$ are learnable projection matrices

### Our Implementation

```python
class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, bias=True, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim or int(2 * input_dim * 4 / 3)
        
        # Three linear projections
        self.w1 = nn.Linear(input_dim, self.hidden_dim, bias=bias)  # Gate path
        self.w2 = nn.Linear(input_dim, self.hidden_dim, bias=bias)  # Linear path
        self.w3 = nn.Linear(self.hidden_dim, input_dim, bias=bias)  # Output projection
        
    def forward(self, x):
        gate = self.w1(x)
        activation = F.silu(gate)           # Swish activation
        gated = activation * self.w2(x)     # Gating mechanism
        output = self.w3(gated)             # Project back
        return output
```

### Key Characteristics

1. **Three-matrix design**: Unlike standard FFN with 2 matrices, SwiGLU uses 3 ($W_1$, $W_2$, $W_3$)
2. **Hidden dimension**: Default expansion is $\frac{8}{3} \times d_{\text{input}}$ (compensates for extra matrix)
3. **Smooth gradients**: Swish provides non-zero gradients for negative inputs (unlike ReLU)
4. **Self-gating**: The input controls its own gating through the Swish function

### Why SwiGLU?

| Aspect | ReLU/GELU | SwiGLU |
|--------|-----------|--------|
| Gradient flow | Dead neurons (ReLU) | Always non-zero |
| Expressiveness | Single path | Dual path with gating |
| Performance | Baseline | +1-2% on benchmarks |

---

## Rotary Position Embedding (RoPE)

### What is RoPE?

RoPE encodes position information by rotating query and key vectors in the attention mechanism. Introduced in [Su et al., 2021](https://arxiv.org/abs/2104.09864), it provides relative position awareness without explicit position IDs.

### Intuition: Rotation on the Complex Plane

**Simple case ($d = 2$):** Consider each token embedding as a point $(x_0, x_1)$ on the 2D plane, which can be viewed as a complex number $z = x_0 + ix_1$. RoPE rotates this point by an angle $\theta$ proportional to its position $m$:

$$z' = z \cdot e^{im\theta} = (x_0 + ix_1)(\cos m\theta + i\sin m\theta)$$

Expanding:

$$\begin{aligned}
x'_0 &= x_0 \cos(m\theta) - x_1 \sin(m\theta) \\
x'_1 &= x_0 \sin(m\theta) + x_1 \cos(m\theta)
\end{aligned}$$

In rotation matrix form:

$$\mathbf{x}' = R_m \mathbf{x} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$$

This is simply a **2D rotation matrix** applied to the embedding!

**General case ($d = 2K$):** For higher dimensions, we partition the $d$-dimensional embedding into $K$ pairs of dimensions. Each pair $(x_{2i}, x_{2i+1})$ is treated as a separate complex number and rotated with a **different frequency** $\theta_i$:

$$\theta_i = 10000^{-2i/d}, \quad i \in \{0, 1, \ldots, \tfrac{d}{2}-1\}$$

| Pair Index $i$ | Dimensions | Frequency $\theta_i$ | Rotation Period |
|----------------|------------|---------------------|-----------------|
| 0 | $(x_0, x_1)$ | $\theta_0 = 1$ | Short (high freq) |
| 1 | $(x_2, x_3)$ | $\theta_1 \approx 0.01$ | Medium |
| ... | ... | ... | ... |
| $K-1$ | $(x_{d-2}, x_{d-1})$ | $\theta_{K-1} \approx 10^{-4}$ | Long (low freq) |

**Why different frequencies?** Using varied frequencies ensures that the combined rotation pattern is **unique for each position**, minimizing collision rates. This is analogous to how different digit places in a number system represent different magnitudes—low-frequency pairs encode "coarse" position information, while high-frequency pairs encode "fine" details.

### Mathematical Formulation

For a position $m$ and dimension pair $(d_{2i}, d_{2i+1})$:

$$\text{RoPE}(x, m) = \begin{bmatrix} x_0 \cos(m\theta_0) - x_1 \sin(m\theta_0) \\ x_0 \sin(m\theta_0) + x_1 \cos(m\theta_0) \\ x_2 \cos(m\theta_1) - x_3 \sin(m\theta_1) \\ \vdots \end{bmatrix}$$

Where the frequency for pair index $i$ is:

$$\theta_i = 10000^{-2i/d}, \quad i \in \{0, 1, \ldots, \tfrac{d}{2}-1\}$$

### Key Characteristics

1. **Relative position awareness**: When computing $q_m^T k_n$, the rotation angles combine as $e^{i(m-n)\theta}$, naturally encoding **relative distance** $(m-n)$
2. **Collision-resistant**: With $K$ pairs using exponentially decaying frequencies, the probability of two positions having identical rotations is negligible for sequences up to millions of tokens
3. **Extrapolation**: Can handle sequences longer than training length due to continuous rotation
4. **No learnable parameters**: Pure mathematical transformation (deterministic)
5. **Applied to Q/K only**: Values remain unchanged, preserving the original information

### Our Implementation

RoPE is applied **inside the attention layer** to Q and K vectors (not to token embeddings):

```python
class MultiHeadFlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rope=None, ...):
        self.rope = rope  # Rotary Position Embedding module
        
    def forward(self, x, ...):
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        # Apply RoPE to Q and K (not V!)
        if self.rope is not None:
            query = query.view(batch, seq, num_heads, head_dim)
            key = key.view(batch, seq, num_heads, head_dim)
            query, key = self.rope(query, key)  # Rotate Q and K
            query = query.view(batch, seq, embed_dim)
            key = key.view(batch, seq, embed_dim)
        
        # Compute attention with rotated Q, K
        attn_output = attention(query, key, value)
```

**Key points:**
- Token embeddings: $e = e_{\text{token}} + e_{\text{type}}$ (no position added)
- Q, K in attention: RoPE rotation applied
- V in attention: No rotation (preserves original information)

### RoPE vs Other Position Encodings

| Type | Relative | Learnable | Extrapolation | Computation |
|------|----------|-----------|---------------|-------------|
| Absolute (BERT) | ❌ | ✅ | Poor | $\mathcal{O}(1)$ |
| Sinusoidal | ❌ | ❌ | Limited | $\mathcal{O}(1)$ |
| ALiBi | ✅ | ❌ | Good | $\mathcal{O}(L^2)$ |
| **RoPE** | ✅ | ❌ | **Good** | $\mathcal{O}(L \cdot d)$ |

### Why RoPE Over ALiBi? Preserving Dot-Product Structure

**ALiBi** adds a position penalty **after** the dot product:

$$\text{Attention}_{ij}^{\text{ALiBi}} = \text{softmax}\left( q_i^T k_j - m \cdot |i-j| \right)$$

The penalty $-m \cdot |i-j|$ is **independent of content** — distant tokens are penalized regardless of semantic relevance.

**RoPE** applies rotation **before** the dot product:

$$\text{Attention}_{ij}^{\text{RoPE}} = \text{softmax}\left( (R_i q_i)^T (R_j k_j) \right) = \text{softmax}\left( q_i^T R_{j-i} k_j \right)$$

Position is encoded **within** the similarity computation, not added on top.

#### Comparison Example

Consider two semantically similar tokens at different distances:

| Method | Similar & Close | Similar & Far |
|--------|-----------------|---------------|
| **ALiBi** | $+5 - 1 = +4$ ✅ | $+5 - 10 = -5$ ❌ |
| **RoPE** | High (rotated similarity) | Still high if semantically aligned ✅ |

ALiBi's linear penalty can **override** semantic similarity, while RoPE preserves it in rotated space.

#### Why This Matters

For tasks requiring **long-range semantic connections** (coreference, long documents), RoPE allows semantically related tokens to attend to each other even at distance, while ALiBi may suppress these connections.

| Aspect | ALiBi | RoPE |
|--------|-------|------|
| Formula | $q^T k + \text{bias}(i,j)$ | $(R_i q)^T (R_j k)$ |
| Position encoding | Additive (post dot-product) | Multiplicative (pre dot-product) |
| Semantic preservation | ❌ Can be overridden | ✅ Preserved in rotated space |
| Long-range attention | Penalized uniformly | Based on rotated similarity |

---

## RMSNorm

### What is RMSNorm?

RMSNorm (Root Mean Square Layer Normalization) is a simplified normalization that only uses the root mean square statistic, without mean centering.

### Mathematical Formulation

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

where:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

### Our Implementation

```python
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return x_normalized * self.weight
```

### RMSNorm vs LayerNorm

| Aspect | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Statistics | Mean $\mu$ + Variance $\sigma^2$ | RMS only |
| Parameters | $\gamma, \beta$ (scale, shift) | $\gamma$ only (scale) |
| Computation | More expensive | ~10-15% faster |
| Performance | Baseline | Similar or better |

---

## Standard vs Recurrent Transformer

### Standard Transformer (Baseline)

The standard transformer processes input through $N$ layers **once**:

$$\text{Input} \rightarrow \text{Embedding} \rightarrow [\text{Layer}_1 \rightarrow \text{Layer}_2 \rightarrow \cdots \rightarrow \text{Layer}_N] \rightarrow \text{Classifier} \rightarrow \text{Output}$$

**Implementation:**
```python
class BaselineModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.get_input_embeddings(input_ids)
        
        # Single pass through all layers
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask)
        
        # Classification using [CLS] token
        pooled_output = hidden_states[:, 0]
        logits = self.classifier(pooled_output)
        return logits
```

### Recurrent Transformer

The recurrent transformer processes input through layers **multiple times** (iterative refinement):

$$\text{Input} \rightarrow \text{Embedding} \rightarrow \underbrace{[\text{Layer}_1 \rightarrow \cdots \rightarrow \text{Layer}_N]}_{\times D \text{ iterations}} \rightarrow \text{Classifier} \rightarrow \text{Output}$$

Where $D$ = `recurrent_depth` (number of iterations)

**Implementation:**
```python
class RecurrentModel(nn.Module):
    def forward_recurrent(self, hidden_states, attention_mask=None):
        for depth_iteration in range(self.config.recurrent_depth):
            iteration_input = hidden_states  # Save for residual
            
            # Process through all layers
            for layer in self.layers:
                hidden_states, _ = layer(hidden_states, attention_mask)
            
            # Residual across iterations (helps gradient flow)
            if depth_iteration > 0:
                hidden_states = hidden_states + iteration_input * self.config.residual_scale
        
        return hidden_states
```

### Key Differences

| Aspect | Standard Transformer | Recurrent Transformer |
|--------|---------------------|----------------------|
| Layer passes | 1 | $D$ (recurrent_depth) |
| Effective depth | $N$ layers | $N \times D$ layers |
| Weight sharing | No | Optional across iterations |
| Residual | Within layers | Within + across iterations |
| Compute | $\mathcal{O}(N)$ | $\mathcal{O}(N \times D)$ |
| Parameters | Full | Same or shared |

### Recurrent Transformer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $D$ (recurrent_depth) | 1 | Number of iterations through all layers |
| share_weights_across_depth | False | Whether to share weights across iterations |
| $\alpha$ (residual_scale) | 0.5 | Scale factor for cross-iteration residual: $h^{(t)} + \alpha \cdot h^{(t-1)}$ |

### Gradient Update Analysis: Standard vs Recurrent Transformer

This section analyzes how parameter gradients are computed and accumulated in each architecture.

#### Standard Transformer: Parameter Gradients

In a standard $N$-layer transformer, each layer $l$ has unique parameters $\theta_l = \{W_l^Q, W_l^K, W_l^V, W_l^O, W_l^{FFN}\}$.

The gradient for layer $l$'s parameters:

$$\frac{\partial \mathcal{L}}{\partial \theta_l} = \frac{\partial \mathcal{L}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial \theta_l}$$

Each parameter receives gradients from **one forward pass** through its layer:

$$\Delta \theta_l = -\eta \cdot \frac{\partial \mathcal{L}}{\partial \theta_l}$$

| Layer | Parameters | Gradient Updates per Step |
|-------|------------|---------------------------|
| Layer 1 | $\theta_1$ | 1 |
| Layer 2 | $\theta_2$ | 1 |
| $\vdots$ | $\vdots$ | $\vdots$ |
| Layer $N$ | $\theta_N$ | 1 |
| **Total** | $N \times |\theta|$ | $N$ updates |

#### Recurrent Transformer: Parameter Gradients

##### Case 1: Weight Sharing Across Iterations (`share_weights = True`)

With weight sharing, the **same parameters** $\theta$ are reused $L$ times. The gradient accumulates contributions from all iterations:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=0}^{L-1} \frac{\partial \mathcal{L}}{\partial H^{(t)}} \cdot \frac{\partial H^{(t)}}{\partial \theta}$$

This is analogous to **Backpropagation Through Time (BPTT)** in RNNs:

$$\Delta \theta = -\eta \cdot \sum_{t=1}^{D} \frac{\partial \mathcal{L}}{\partial \theta}\bigg|_{t}$$

**Effect:** Each parameter sees $D$ times more gradient signal per training step!

| Component | Parameters | Gradient Contributions |
|-----------|------------|------------------------|
| Shared Layers | $\theta$ | $D$ (one per iteration) |
| **Total** | $N \times |\theta|$ | $N \times D$ contributions |

##### Case 2: No Weight Sharing (`share_weights = False`, our default)

Each iteration uses **independent parameters** $\theta^{(t)}$, but with cross-iteration residuals:

$$H^{(t)} = f_{\theta^{(t)}}(H^{(t-1)}) + \alpha \cdot H^{(t-1)}$$

The gradient for iteration $t$'s parameters:

$$\frac{\partial \mathcal{L}}{\partial \theta^{(t)}} = \frac{\partial \mathcal{L}}{\partial H^{(t)}} \cdot \frac{\partial H^{(t)}}{\partial \theta^{(t)}}$$

The cross-iteration residual affects gradient magnitude:

$$\frac{\partial \mathcal{L}}{\partial H^{(t-1)}} = \frac{\partial \mathcal{L}}{\partial H^{(t)}} \cdot \left( \frac{\partial f}{\partial H^{(t-1)}} + \alpha I \right)$$

| Iteration | Parameters | Gradient Source |
|-----------|------------|-----------------|
| $t = D$ | $\theta^{(D)}$ | Direct from loss |
| $t = D-1$ | $\theta^{(D-1)}$ | Through $H^{(D)}$ + $\alpha$ skip |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $t = 1$ | $\theta^{(1)}$ | Through all + $D-1$ skips |
| **Total** | $N \times D \times |\theta|$ | Rich gradient paths |

#### Gradient Magnitude Comparison

For a parameter in layer $l$ at iteration $t$, the gradient magnitude depends on the path length to the loss:

**Standard Transformer:**
$$\left\| \frac{\partial \mathcal{L}}{\partial \theta_l} \right\| \propto \prod_{k=l+1}^{N} \|J_k\|$$

where $J_k$ is the Jacobian of layer $k$.

**Recurrent Transformer (no sharing):**
$$\left\| \frac{\partial \mathcal{L}}{\partial \theta^{(t)}_l} \right\| \propto \prod_{k=l+1}^{N} \|J_k\| \cdot \prod_{s=t+1}^{D} (\|J_s\| + \alpha)$$

The $+\alpha$ term ensures gradients don't vanish across iterations:

$$\prod_{s=t+1}^{D} (\|J_s\| + \alpha) \geq \alpha^{D-t}$$

#### Summary: Parameter Update Characteristics

| Aspect | Standard | Recurrent (shared) | Recurrent (no share) |
|--------|----------|-------------------|---------------------|
| Parameters | $N \times |\theta|$ | $N \times |\theta|$ | $N \times D \times |\theta|$ |
| Gradients per param | 1 | $D$ (accumulated) | 1 |
| Gradient paths | $N$ residuals | $N$ + BPTT | $N \times D$ + $\alpha$ skips |
| Update frequency | Once | Once (but $D\times$ signal) | Once |
| Vanishing risk | Low | Medium (like RNN) | **Lowest** ($\alpha$ preserves) |

#### Intuition

- **Standard:** Each layer learns independently, gradients flow through residual connections
- **Recurrent (shared):** Same weights refined $D$ times, like an RNN unrolled in depth
- **Recurrent (no share):** More parameters, but $\alpha$-residuals create gradient highways across iterations, ensuring even early-iteration parameters receive strong gradients

---

## CLS Token Usage

### ✅ Yes, Both Models Use CLS Token

Both the Standard and Recurrent Transformer use the **[CLS] token** (first position) for classification:

$$\text{logits} = W_{\text{cls}} \cdot \text{Dropout}(h_{[:,0]}) + b_{\text{cls}}$$

Where:
- $h_{[:,0]} \in \mathbb{R}^{B \times d}$ is the hidden state at position 0 (CLS token)
- $W_{\text{cls}} \in \mathbb{R}^{d \times C}$ is the classifier weight matrix
- $C$ is the number of classes (2 for sentiment, 3 for domain)

### Why CLS Token?

1. **BERT convention**: Compatible with BERT tokenizers that prepend [CLS]
2. **Global representation**: [CLS] attends to all tokens, aggregating sequence information
3. **No pooling needed**: Direct use of single position representation
4. **Consistent**: Same approach for both sentiment (2-class) and domain (3-class) classification

### Input Format

$$\underbrace{[\text{CLS}]}_{\text{Position 0}} \text{ This movie was great ! } [\text{SEP}] [\text{PAD}] [\text{PAD}] \cdots$$

The $[\text{CLS}]$ token at position 0 is used for classification.

---

## Configuration Options

### BaselineConfig

```python
@dataclass
class BaselineConfig:
    # Architecture
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    
    # Enhancements (all enabled by default)
    use_flash_attention: bool = True
    use_swiglu: bool = True
    use_rope: bool = True
    use_rms_norm: bool = True
    
    # Task
    num_labels: int = 2  # 2 for sentiment, 3 for domain
```

### RecurrentConfig

```python
@dataclass
class RecurrentConfig:
    # Same base architecture as BaselineConfig, plus:
    
    # Recurrent-specific
    recurrent_depth: int = 1              # Iterations through layers
    share_weights_across_depth: bool = False
    residual_scale: float = 0.5
```

---

## Summary

| Component | Implementation | Reference |
|-----------|----------------|-----------|
| **SwiGLU** | $\text{Swish}(W_1 x) \odot W_2 x$, then $W_3$ | [Shazeer 2020](https://arxiv.org/abs/2002.05202) |
| **RoPE** | Rotate $Q, K$ by position-dependent angles | [Su et al. 2021](https://arxiv.org/abs/2104.09864) |
| **RMSNorm** | $x / \text{RMS}(x) \cdot \gamma$ | [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) |
| **Pre-Norm** | $x + \text{sublayer}(\text{norm}(x))$ | [Xiong et al. 2020](https://arxiv.org/abs/2002.04745) |
| **CLS Token** | $h_{[:,0]}$ for classification | BERT |
| **Recurrent** | $N$ layers $\times D$ iterations | Universal Transformers |

---

## Interactive Demo

Try the **SwiGLU Interactive Demo** to visualize how parameters affect the activation:

```bash
cd app/SwiGLU_demo
python -m http.server 8080
# Open http://localhost:8080
```

