"""
Flash Attention implementation for memory-efficient attention computation.
Includes both standard and flash attention variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FlashAttention(nn.Module):
    """
    Memory-efficient attention mechanism with optional flash attention.
    Falls back to standard attention if flash attention is not available.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True,
        attention_scale: Optional[float] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_flash = use_flash
        
        if attention_scale is None:
            self.scale = self.head_dim ** -0.5
        else:
            self.scale = attention_scale
            
        # Check if flash attention is available
        self.flash_available = self._check_flash_attention()
        
        if not self.flash_available and use_flash:
            print("Flash attention not available, falling back to standard attention")
            self.use_flash = False
            
        self.dropout_layer = nn.Dropout(dropout)
        
    def _check_flash_attention(self) -> bool:
        """Check if flash attention is available in the environment."""
        try:
            # Check for flash attention support
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return True
            return False
        except:
            return False
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: Optional[float] = None,
        is_causal: bool = False,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention with optional flash attention.
        
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            dropout_p: Dropout probability (overrides self.dropout if provided)
            is_causal: Whether to use causal attention
            return_attention_weights: Whether to return attention weights
            
        Returns:
            attention output and optional attention weights
        """
        batch_size, seq_len, _ = query.shape
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        dropout_p = dropout_p if dropout_p is not None else self.dropout
        
        if self.use_flash and self.flash_available and not return_attention_weights:
            # Use PyTorch's scaled_dot_product_attention for efficiency
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=dropout_p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale
            )
            attn_weights = None
        else:
            # Standard attention computation
            attn_output, attn_weights = self._standard_attention(
                query, key, value,
                attention_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal
            )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        if return_attention_weights and attn_weights is not None:
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard attention computation fallback.
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if is_causal:
            seq_len = query.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device) * float('-inf'),
                diagonal=1
            )
            scores = scores + causal_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights


class MultiHeadFlashAttention(nn.Module):
    """
    Multi-head attention with flash attention support.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Flash attention layer
        self.attention = FlashAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash=use_flash
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal attention
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optional attention weights
        """
        # Project to query, key, value
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            query, key, value,
            attention_mask=attention_mask,
            is_causal=is_causal,
            return_attention_weights=return_attention_weights
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights
