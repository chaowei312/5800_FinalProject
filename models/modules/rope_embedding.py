"""
Rotary Position Embedding (RoPE) implementation.
Based on: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding for transformers.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequency tensor
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq.to(device if device else torch.device('cpu')))
        
        # Cache for positional embeddings
        self._cached_embeddings = None
        self._cached_seq_len = 0
    
    def _compute_embeddings(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sin and cos embeddings for the given sequence length."""
        if self._cached_embeddings is not None and self._cached_seq_len >= seq_len:
            return self._cached_embeddings[0][:seq_len], self._cached_embeddings[1][:seq_len]
        
        position_ids = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(position_ids, self.inv_freq)
        
        # Create sin and cos embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Cache the embeddings
        self._cached_embeddings = (cos_emb, sin_emb)
        self._cached_seq_len = seq_len
        
        return cos_emb, sin_emb
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids: Optional position IDs
            
        Returns:
            Rotated query and key tensors
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Get position embeddings
        cos_emb, sin_emb = self._compute_embeddings(seq_len)
        cos_emb = cos_emb.unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, dim]
        sin_emb = sin_emb.unsqueeze(1).unsqueeze(0)
        
        # Apply rotation
        q_rotated = apply_rotary_pos_emb(q, cos_emb, sin_emb)
        k_rotated = apply_rotary_pos_emb(k, cos_emb, sin_emb)
        
        return q_rotated, k_rotated


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.
    
    Args:
        x: Input tensor
        cos: Cosine embeddings
        sin: Sine embeddings
        
    Returns:
        Rotated tensor
    """
    return (x * cos) + (rotate_half(x) * sin)


class LearnedPositionalEmbedding(nn.Module):
    """
    Standard learned positional embeddings as a fallback.
    """
    
    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(max_seq_len, embedding_dim)
        
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(position_ids)
