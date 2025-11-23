"""
Recurrent transformer model - pure iterative refinement without extra state.
Implements a transformer that processes hidden states through layers multiple times.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import math

from ..modules import (
    MultiHeadFlashAttention,
    FFNWithSwiGLU,
    RMSNorm,
    RotaryPositionalEmbedding,
    get_activation
)


@dataclass
class RecurrentConfig:
    """Configuration for recurrent transformer model."""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    num_labels: int = 2
    
    # Recurrent-specific parameters (simplified)
    recurrent_depth: int = 1  # Number of iterations through transformer layers
    share_weights_across_depth: bool = False  # Share weights across recurrent iterations
    residual_scale: float = 0.5  # Scale for residual connections across iterations
    
    # Legacy parameters (kept for compatibility but unused)
    state_size: int = 512  # DEPRECATED - no longer used
    chunk_size: int = 64   # DEPRECATED - no longer used
    use_gating: bool = False  # DEPRECATED - no longer used
    state_dropout: float = 0.1  # DEPRECATED - no longer used
    memory_efficient: bool = True
    
    # Custom enhancements
    use_flash_attention: bool = True
    use_swiglu: bool = True
    use_rope: bool = True
    use_rms_norm: bool = True
    activation: str = "gelu"
    
    # Model type for loading pretrained
    model_type: str = "recurrent-bert"
    pretrained_path: Optional[str] = None


# RecurrentState and GatedStateUpdate classes removed - pure recurrent transformer doesn't need them
# The model now just iterates through transformer layers without extra state management


class RecurrentTransformerBlock(nn.Module):
    """Pure transformer block for recurrent processing - no extra state."""
    
    def __init__(self, config: RecurrentConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Multi-head attention
        if config.use_flash_attention:
            self.attention = MultiHeadFlashAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                use_flash=True
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
        
        # Feed-forward network
        if config.use_swiglu:
            self.ffn = FFNWithSwiGLU(
                embed_dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation='swiglu'
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                get_activation(config.activation),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
        
        # Normalization layers
        if config.use_rms_norm:
            self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pure transformer forward pass - just process hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, optional attention weights)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        if isinstance(self.attention, MultiHeadFlashAttention):
            attn_output, attn_weights = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                return_attention_weights=return_attention
            )
        else:
            attn_output, attn_weights = self.attention(
                hidden_states, hidden_states, hidden_states,
                attn_mask=attention_mask,
                need_weights=return_attention
            )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        
        return hidden_states, attn_weights


class RecurrentModel(nn.Module):
    """
    Pure recurrent transformer model - iterates through layers multiple times.
    No extra state management, just refining hidden representations.
    """
    
    def __init__(self, config: RecurrentConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        if config.use_rope:
            self.position_embeddings = RotaryPositionalEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_seq_len=config.max_position_embeddings
            )
        else:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size
            )
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        
        # Initial normalization and dropout
        if config.use_rms_norm:
            self.embeddings_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.embeddings_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.embeddings_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks (will be reused recurrent_depth times)
        self.layers = nn.ModuleList([
            RecurrentTransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Classification head
        self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'weight'):
                module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
    
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get input embeddings."""
        seq_length = input_ids.size(1)
        
        # Token embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        if not self.config.use_rope:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        else:
            embeddings = inputs_embeds + token_type_embeddings
        
        # Apply normalization and dropout
        embeddings = self.embeddings_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings)
        
        return embeddings
    
    def forward_recurrent(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Process hidden states through layers multiple times (pure recurrent)."""
        all_attention_weights = [] if return_attention else None
        
        # Iterate through layers recurrent_depth times
        for depth_iteration in range(self.config.recurrent_depth):
            # Store input for residual connection across iterations
            iteration_input = hidden_states
            
            # Process through each transformer layer
            for layer_idx, layer in enumerate(self.layers):
                # Pure transformer forward pass
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_attention=return_attention and depth_iteration == self.config.recurrent_depth - 1
                )
                
                # Collect attention weights from last iteration only
                if return_attention and attn_weights is not None and depth_iteration == self.config.recurrent_depth - 1:
                    all_attention_weights.append(attn_weights)
            
            # Add residual connection across iterations (helps gradient flow)
            if depth_iteration > 0 and self.config.recurrent_depth > 1:
                hidden_states = hidden_states + iteration_input * self.config.residual_scale
        
        return hidden_states, all_attention_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the pure recurrent transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            labels: Labels for classification
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with logits, loss, and optional hidden states
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Get embeddings
        hidden_states = self.get_input_embeddings(
            input_ids, token_type_ids, position_ids
        )
        
        # Create attention mask if needed
        if attention_mask is not None:
            # Check if any layer uses standard MultiheadAttention
            uses_standard_attention = any(
                isinstance(layer.attention, nn.MultiheadAttention) 
                for layer in self.layers
            )
            
            if uses_standard_attention:
                # For standard PyTorch MultiheadAttention, create square mask
                attention_mask_expanded = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).float()
                attention_mask = (1.0 - attention_mask_expanded) * -10000.0
                # Take first batch item's mask since PyTorch expects 2D or 3D
                attention_mask = attention_mask[0]
            else:
                # For Flash attention format
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Process through layers with recurrent iterations
        hidden_states, all_attention_weights = self.forward_recurrent(
            hidden_states, attention_mask, return_attention
        )
        
        # Classification using [CLS] token (first position)
        pooled_output = hidden_states[:, 0]
        
        pooled_output = self.classifier_dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states
        }
        
        if loss is not None:
            outputs['loss'] = loss
            
        if return_attention:
            outputs['attentions'] = all_attention_weights
        
        return outputs
    
    def save_checkpoint(self, path: str, optimizer=None, epoch=None, best_score=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'best_score': best_score
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Recurrent model checkpoint saved to {path}")
