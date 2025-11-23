"""
SwiGLU activation function implementation.
Based on: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: Swish-Gated Linear Unit.
    More effective than standard ReLU or GELU for transformers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or int(2 * input_dim * 4 / 3)  # Standard expansion
        
        # Gating mechanism
        self.w1 = nn.Linear(input_dim, self.hidden_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, self.hidden_dim, bias=bias)
        self.w3 = nn.Linear(self.hidden_dim, input_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Activated tensor [batch_size, seq_len, input_dim]
        """
        # Gate and activation
        gate = self.w1(x)
        activation = F.silu(gate)  # Swish activation
        
        # Gated linear unit
        gated = activation * self.w2(x)
        
        # Apply dropout
        gated = self.dropout(gated)
        
        # Project back
        output = self.w3(gated)
        
        return output
    
    def reset_parameters(self):
        """Initialize parameters."""
        for module in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class GeGLU(nn.Module):
    """
    GeGLU activation function: GELU-Gated Linear Unit.
    Alternative to SwiGLU using GELU activation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or int(2 * input_dim * 4 / 3)
        
        self.w1 = nn.Linear(input_dim, self.hidden_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, self.hidden_dim, bias=bias)
        self.w3 = nn.Linear(self.hidden_dim, input_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeGLU activation."""
        gate = F.gelu(self.w1(x))
        gated = gate * self.w2(x)
        gated = self.dropout(gated)
        output = self.w3(gated)
        return output


class FFNWithSwiGLU(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    Drop-in replacement for standard FFN in transformers.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'swiglu',
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or 4 * embed_dim
        
        if activation.lower() == 'swiglu':
            self.activation_layer = SwiGLU(
                input_dim=embed_dim,
                hidden_dim=self.hidden_dim,
                bias=bias,
                dropout=dropout
            )
        elif activation.lower() == 'geglu':
            self.activation_layer = GeGLU(
                input_dim=embed_dim,
                hidden_dim=self.hidden_dim,
                bias=bias,
                dropout=dropout
            )
        else:
            # Standard FFN fallback
            self.activation_layer = nn.Sequential(
                nn.Linear(embed_dim, self.hidden_dim, bias=bias),
                nn.GELU() if activation.lower() == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, embed_dim, bias=bias)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFN with SwiGLU.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        return self.activation_layer(x)


class GLU(nn.Module):
    """
    Basic Gated Linear Unit.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)
