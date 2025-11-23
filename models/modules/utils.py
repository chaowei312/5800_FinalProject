"""
Utility functions for model modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Name of activation function
        
    Returns:
        Activation module
    """
    activation = activation.lower()
    
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'mish': nn.Mish(),
        'softplus': nn.Softplus(),
        'identity': nn.Identity()
    }
    
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    
    return activations[activation]


def init_weights(module: nn.Module, init_type: str = 'xavier_uniform'):
    """
    Initialize weights of a module.
    
    Args:
        module: Module to initialize
        init_type: Type of initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif init_type == 'normal':
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif init_type == 'uniform':
            nn.init.uniform_(module.weight, a=-0.1, b=0.1)
        
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
            
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
            
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Causal mask tensor
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1
    )
    return mask


def create_padding_mask(
    input_ids: torch.Tensor,
    pad_token_id: int,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create padding mask from input IDs.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        pad_token_id: Padding token ID
        dtype: Data type for mask
        
    Returns:
        Padding mask [batch_size, 1, 1, seq_len]
    """
    padding_mask = (input_ids == pad_token_id).to(dtype)
    padding_mask = padding_mask.masked_fill(padding_mask == 1, float('-inf'))
    padding_mask = padding_mask.masked_fill(padding_mask == 0, 0.0)
    
    # Add dimensions for broadcasting
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
    
    return padding_mask


def compute_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Compute scaled attention scores.
    
    Args:
        query: Query tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor [batch_size, num_heads, seq_len, head_dim]
        scale: Scale factor (defaults to 1/sqrt(head_dim))
        
    Returns:
        Attention scores [batch_size, num_heads, seq_len, seq_len]
    """
    if scale is None:
        scale = query.size(-1) ** -0.5
    
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    return scores


def apply_attention(
    scores: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> torch.Tensor:
    """
    Apply attention weights to values.
    
    Args:
        scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
        value: Value tensor [batch_size, num_heads, seq_len, head_dim]
        mask: Optional attention mask
        dropout: Optional dropout layer
        
    Returns:
        Attention output [batch_size, num_heads, seq_len, head_dim]
    """
    if mask is not None:
        scores = scores + mask
    
    weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        weights = dropout(weights)
    
    output = torch.matmul(weights, value)
    return output


def get_sinusoidal_embeddings(
    num_positions: int,
    embedding_dim: int,
    base: float = 10000.0
) -> torch.Tensor:
    """
    Generate sinusoidal positional embeddings.
    
    Args:
        num_positions: Number of positions
        embedding_dim: Embedding dimension
        base: Base for frequency computation
        
    Returns:
        Sinusoidal embeddings [num_positions, embedding_dim]
    """
    positions = torch.arange(num_positions).float().unsqueeze(1)
    dim_indices = torch.arange(0, embedding_dim, 2).float()
    
    frequencies = 1.0 / (base ** (dim_indices / embedding_dim))
    angles = positions * frequencies
    
    embeddings = torch.zeros(num_positions, embedding_dim)
    embeddings[:, 0::2] = torch.sin(angles)
    embeddings[:, 1::2] = torch.cos(angles)
    
    return embeddings


def gradient_clipping(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default: 2.0 for L2)
        
    Returns:
        Total gradient norm before clipping
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return total_norm.item()


class ModelCheckpoint:
    """
    Utility class for model checkpointing.
    """
    
    def __init__(self, save_dir: str, monitor: str = 'val_loss', mode: str = 'min'):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def should_save(self, score: float) -> bool:
        """Check if model should be saved based on score."""
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def update(self, score: float):
        """Update best score."""
        self.best_score = score
