"""
Baseline transformer model with optional enhancements.
Wrapper for BERT-based models with custom modules.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import os

from ..modules import (
    MultiHeadFlashAttention,
    FFNWithSwiGLU,
    RMSNorm,
    LayerNormWithRMS,
    RotaryPositionalEmbedding
)


@dataclass
class BaselineConfig:
    """Configuration for baseline transformer model."""
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
    
    # Custom enhancements
    use_flash_attention: bool = True
    use_swiglu: bool = True
    use_rope: bool = True
    use_rms_norm: bool = True
    activation: str = "gelu"
    
    # Model type for loading pretrained
    model_type: str = "bert"  # Can be "bert-tiny", "bert-small", "bert-mini"
    pretrained_path: Optional[str] = None


class TransformerBlock(nn.Module):
    """Single transformer block with optional enhancements."""
    
    def __init__(self, config: BaselineConfig):
        super().__init__()
        
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
                nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
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
        """Forward pass through transformer block."""
        
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


class BaselineModel(nn.Module):
    """
    Baseline transformer model with optional enhancements.
    Can load pretrained BERT weights and add custom modules.
    """
    
    def __init__(self, config: BaselineConfig):
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
        
        # Token type embeddings (for BERT)
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
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Classification head
        self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Load pretrained weights if specified
        if config.pretrained_path:
            self.load_pretrained(config.pretrained_path)
    
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs for BERT
            position_ids: Position IDs
            labels: Labels for classification
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with logits, loss, and optional attention weights
        """
        # Get embeddings
        hidden_states = self.get_input_embeddings(
            input_ids, token_type_ids, position_ids
        )
        
        # Create attention mask if needed
        if attention_mask is not None:
            # For standard MultiheadAttention, create square mask
            # For Flash attention, convert to proper format
            if isinstance(self.layers[0].attention, nn.MultiheadAttention):
                # Standard PyTorch MultiheadAttention expects square mask for self-attention
                seq_len = attention_mask.size(1)
                # Create causal mask shape [seq_len, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).float()
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                # Take first batch item's mask since PyTorch expects 2D or 3D
                extended_attention_mask = extended_attention_mask[0]
            else:
                # Flash attention format
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Pass through transformer layers
        all_attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                return_attention=return_attention
            )
            if return_attention and attn_weights is not None:
                all_attention_weights.append(attn_weights)
        
        # Classification
        # Use [CLS] token representation (first token)
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
            'loss': loss,
            'hidden_states': hidden_states
        }
        
        if return_attention:
            outputs['attention_weights'] = all_attention_weights
        
        return outputs
    
    def load_pretrained(self, path: str):
        """Load pretrained weights from path."""
        if os.path.exists(path):
            print(f"Loading pretrained weights from {path}")
            state_dict = torch.load(path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load with strict=False to allow for architecture differences
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
            
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")
                
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
        print(f"Model checkpoint saved to {path}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load model from pretrained weights."""
        config = BaselineConfig(**kwargs)
        config.model_type = model_name
        
        # Map model names to paths
        model_paths = {
            'bert-tiny': 'bert-tiny/pytorch_model.bin',
            'bert-small': 'bert-small/pytorch_model.bin',
            'bert-mini': 'bert-mini/pytorch_model.bin'
        }
        
        if model_name in model_paths:
            config.pretrained_path = model_paths[model_name]
        
        return cls(config)
