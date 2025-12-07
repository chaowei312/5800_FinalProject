"""
Model loading utilities for both baseline and recurrent transformers.
"""

import os
import torch
from typing import Dict, Any, Optional
from dataclasses import asdict

# Import model classes
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.baseline.baseline_model import BaselineModel, BaselineConfig
from models.recurrent.recurrent_model import RecurrentModel, RecurrentConfig


def load_baseline_model(
    checkpoint_path: str,
    num_labels: int = 2,
    device: str = 'cpu',
    hidden_size: int = None,
    num_hidden_layers: int = None,
    num_attention_heads: int = None,
    intermediate_size: int = None
) -> BaselineModel:
    """
    Load a trained baseline transformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        num_labels: Number of output labels (2 for sentiment, 3 for multi-domain)
        device: Device to load model on ('cpu' or 'cuda')
        hidden_size: Hidden size (if None, inferred from checkpoint)
        num_hidden_layers: Number of layers (if None, inferred from checkpoint)
        num_attention_heads: Number of attention heads (if None, inferred from checkpoint)
        intermediate_size: Intermediate size (if None, inferred from checkpoint)
        
    Returns:
        Loaded baseline model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading baseline model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
        # Convert to BaselineConfig if it's a dict or dataclass
        if isinstance(config, dict):
            config['num_labels'] = num_labels
            # Override with provided parameters
            if hidden_size is not None:
                config['hidden_size'] = hidden_size
            if num_hidden_layers is not None:
                config['num_hidden_layers'] = num_hidden_layers
            if num_attention_heads is not None:
                config['num_attention_heads'] = num_attention_heads
            if intermediate_size is not None:
                config['intermediate_size'] = intermediate_size
            config = BaselineConfig(**config)
        else:
            config.num_labels = num_labels
            if hidden_size is not None:
                config.hidden_size = hidden_size
            if num_hidden_layers is not None:
                config.num_hidden_layers = num_hidden_layers
            if num_attention_heads is not None:
                config.num_attention_heads = num_attention_heads
            if intermediate_size is not None:
                config.intermediate_size = intermediate_size
    else:
        # Try to infer from checkpoint state_dict
        print("Warning: No config found in checkpoint, inferring from state_dict")
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        
        # Infer hidden_size from embeddings
        inferred_hidden_size = state_dict['word_embeddings.weight'].shape[1]
        inferred_num_layers = sum(1 for k in state_dict.keys() if k.startswith('layers.') and k.endswith('.norm1.weight'))
        
        # Use provided or inferred values
        config = BaselineConfig(
            num_labels=num_labels,
            hidden_size=hidden_size or inferred_hidden_size,
            num_hidden_layers=num_hidden_layers or inferred_num_layers,
            num_attention_heads=num_attention_heads or (inferred_hidden_size // 64),
            intermediate_size=intermediate_size or (inferred_hidden_size * 4),
            use_flash_attention=True,
            use_swiglu=True,
            use_rope=True,
            use_rms_norm=True
        )
        print(f"Inferred config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    # Create model
    model = BaselineModel(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # Checkpoint is directly the state dict
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Baseline model loaded successfully")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Number of labels: {config.num_labels}")
    
    return model


def load_recurrent_model(
    checkpoint_path: str,
    num_labels: int = 2,
    device: str = 'cpu',
    hidden_size: int = None,
    num_hidden_layers: int = None,
    num_attention_heads: int = None,
    intermediate_size: int = None,
    recurrent_depth: int = None
) -> RecurrentModel:
    """
    Load a trained recurrent transformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        num_labels: Number of output labels (2 for sentiment, 3 for multi-domain)
        device: Device to load model on ('cpu' or 'cuda')
        hidden_size: Hidden size (if None, inferred from checkpoint)
        num_hidden_layers: Number of layers (if None, inferred from checkpoint)
        num_attention_heads: Number of attention heads (if None, inferred from checkpoint)
        intermediate_size: Intermediate size (if None, inferred from checkpoint)
        recurrent_depth: Recurrent depth (if None, inferred from checkpoint)
        
    Returns:
        Loaded recurrent model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading recurrent model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
        # Convert to RecurrentConfig if it's a dict or dataclass
        if isinstance(config, dict):
            config['num_labels'] = num_labels
            # Override with provided parameters
            if hidden_size is not None:
                config['hidden_size'] = hidden_size
            if num_hidden_layers is not None:
                config['num_hidden_layers'] = num_hidden_layers
            if num_attention_heads is not None:
                config['num_attention_heads'] = num_attention_heads
            if intermediate_size is not None:
                config['intermediate_size'] = intermediate_size
            if recurrent_depth is not None:
                config['recurrent_depth'] = recurrent_depth
            config = RecurrentConfig(**config)
        else:
            config.num_labels = num_labels
            if hidden_size is not None:
                config.hidden_size = hidden_size
            if num_hidden_layers is not None:
                config.num_hidden_layers = num_hidden_layers
            if num_attention_heads is not None:
                config.num_attention_heads = num_attention_heads
            if intermediate_size is not None:
                config.intermediate_size = intermediate_size
            if recurrent_depth is not None:
                config.recurrent_depth = recurrent_depth
    else:
        # Try to infer from checkpoint state_dict
        print("Warning: No config found in checkpoint, inferring from state_dict")
        state_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        
        # Infer hidden_size from embeddings
        inferred_hidden_size = state_dict['word_embeddings.weight'].shape[1]
        inferred_num_layers = sum(1 for k in state_dict.keys() if k.startswith('layers.') and k.endswith('.norm1.weight'))
        
        # Use provided or inferred values
        config = RecurrentConfig(
            num_labels=num_labels,
            hidden_size=hidden_size or inferred_hidden_size,
            num_hidden_layers=num_hidden_layers or inferred_num_layers,
            num_attention_heads=num_attention_heads or (inferred_hidden_size // 64),
            intermediate_size=intermediate_size or (inferred_hidden_size * 4),
            recurrent_depth=recurrent_depth or 2,
            residual_scale=0.5,
            use_flash_attention=True,
            use_swiglu=True,
            use_rope=True,
            use_rms_norm=True
        )
        print(f"Inferred config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, recurrent_depth={config.recurrent_depth}")
    
    # Create model
    model = RecurrentModel(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # Checkpoint is directly the state dict
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Recurrent model loaded successfully")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Recurrent depth: {config.recurrent_depth}")
    print(f"  - Number of labels: {config.num_labels}")
    
    return model


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a saved model without fully loading it.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        
    Returns:
        Dictionary containing model metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'checkpoint_path': checkpoint_path,
        'has_config': 'config' in checkpoint,
        'has_optimizer': 'optimizer_state_dict' in checkpoint,
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'best_score': checkpoint.get('best_score', 'Unknown')
    }
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, '__dict__'):
            info['config'] = vars(config)
        elif isinstance(config, dict):
            info['config'] = config
    
    return info

