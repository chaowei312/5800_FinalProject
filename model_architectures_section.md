## 2.2 Model Architectures

We compare two transformer architectures with fixed configurations designed to isolate the effects of recurrent processing versus standard stacking. Both models use identical custom components (Flash Attention, SwiGLU, RoPE, RMSNorm) to ensure a fair comparison.

### 2.2.1 Baseline Transformer

The baseline model follows a standard BERT-style encoder architecture with the following fixed configuration:

| Configuration | Value |
|---------------|-------|
| Number of layers | 6 |
| Hidden dimensions | 384 |
| Attention heads | 6 |
| Intermediate size | 1536 (4× hidden) |
| **Total parameters** | **9.8M** |

**Architecture**: Each layer applies multi-head self-attention followed by a feed-forward network with pre-norm residual connections:

$$
\begin{align}
h'_l &= h_{l-1} + \text{Dropout}(\text{MultiHeadAttn}(\text{Norm}_1(h_{l-1}))) \\
h_l &= h'_l + \text{Dropout}(\text{FFN}(\text{Norm}_2(h'_l)))
\end{align}
$$

Classification is performed on the `[CLS]` token representation from the final layer:

$$
y = \arg\max(\text{softmax}(W_c \cdot h_L[0, :]))
$$

### 2.2.2 Recurrent Transformer

The recurrent model achieves comparable effective depth through iterative processing with significantly fewer parameters. We fix the following configuration:

| Configuration | Value |
|---------------|-------|
| Number of layers | 3 |
| Recurrent iterations | 2 |
| Hidden dimensions | 256 |
| Attention heads | 4 |
| Intermediate size | 1024 (4× hidden) |
| **Total parameters** | **3.4M (65% reduction)** |

**Key Innovation**: The same 3 transformer layers are applied twice with cross-iteration residual connections:

$$
h^{(r+1)} = \text{TransformerLayers}(h^{(r)}) + \alpha \cdot h^{(r)}
$$

where $r \in \{1, 2\}$ denotes the iteration and $\alpha = 0.5$ is the residual scaling factor.

**Effective Depth**: Despite having only 3 physical layers, the model achieves an effective depth of 6 layers through 2 iterations:

$$
D_{\text{effective}} = N_{\text{layers}} \times N_{\text{iterations}} = 3 \times 2 = 6
$$

**Processing Flow**:
```
Input → Embeddings → [Layer₁ → Layer₂ → Layer₃] × iteration 1
                      ↓ (residual α=0.5)
                     [Layer₁ → Layer₂ → Layer₃] × iteration 2
                      ↓
                    Classifier
```

### 2.2.3 Fixed Configuration Rationale

We deliberately fix these architectures to create a controlled comparison:

- **Matched Effective Depth**: Both models have 6 effective layers to ensure comparable representational capacity
- **Parameter Budget**: The recurrent model uses ~1/3 of baseline parameters (3.4M vs 9.8M) to test parameter efficiency
- **Proportional Scaling**: Both use 4× expansion ratio (intermediate size / hidden size) and maintain standard attention head ratios

This fixed design isolates the architectural contribution of recurrent processing from confounding factors like model capacity or training configurations.

### 2.2.4 Shared Custom Components

Both architectures incorporate the same modern components to ensure fair comparison:

- **Flash Attention**: Memory-efficient scaled dot-product attention with O(N) complexity
- **SwiGLU Activation**: Gated activation in FFN layers for improved gradient flow
- **RoPE**: Rotary position embeddings for better length generalization
- **RMSNorm**: Root Mean Square normalization for efficient and stable training

These shared components eliminate implementation variance and focus evaluation on the core architectural difference: standard layer stacking versus recurrent iterative processing.

