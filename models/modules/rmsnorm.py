"""
RMSNorm (Root Mean Square Layer Normalization) implementation.
More stable and efficient than LayerNorm for certain architectures.
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm normalizes the inputs by the root mean square,
    providing similar benefits to LayerNorm but with improved efficiency.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of shape [..., normalized_shape]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x_normalized = x / rms
        
        # Apply affine transformation if enabled
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
            if self.bias is not None:
                x_normalized = x_normalized + self.bias
        
        return x_normalized
    
    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, ' \
               f'elementwise_affine={self.elementwise_affine}'


class LayerNormWithRMS(nn.Module):
    """
    Adaptive normalization that can switch between LayerNorm and RMSNorm.
    Useful for experimentation and comparison.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        use_rms: bool = True,
        bias: bool = True
    ):
        super().__init__()
        self.use_rms = use_rms
        
        if use_rms:
            self.norm = RMSNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                bias=bias
            )
        else:
            self.norm = nn.LayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                bias=bias
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class PreNormalization(nn.Module):
    """
    Pre-normalization wrapper for transformer layers.
    Applies normalization before the sublayer.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        sublayer: nn.Module,
        use_rms: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm = LayerNormWithRMS(
            normalized_shape=normalized_shape,
            use_rms=use_rms
        )
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply pre-normalization pattern: x + sublayer(norm(x))
        """
        residual = x
        x = self.norm(x)
        x = self.sublayer(x, *args, **kwargs)
        if isinstance(x, tuple):
            x = x[0]  # Handle multi-output sublayers
        x = self.dropout(x)
        return residual + x


class PostNormalization(nn.Module):
    """
    Post-normalization wrapper for transformer layers.
    Applies normalization after the sublayer.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        sublayer: nn.Module,
        use_rms: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm = LayerNormWithRMS(
            normalized_shape=normalized_shape,
            use_rms=use_rms
        )
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply post-normalization pattern: norm(x + sublayer(x))
        """
        residual = x
        x = self.sublayer(x, *args, **kwargs)
        if isinstance(x, tuple):
            x = x[0]  # Handle multi-output sublayers
        x = self.dropout(x)
        x = self.norm(residual + x)
        return x
