"""
Configuration loader for model architectures and training parameters.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class ModelConfiguration:
    """Unified configuration for model architecture and training."""
    
    # Model identification
    model_name: str
    model_type: str  # 'baseline' or 'recurrent'
    
    # Architecture parameters
    architecture: Dict[str, Any]
    
    # Training defaults
    training_defaults: Dict[str, Any]
    
    # Optional enhancements
    enhancements: Optional[Dict[str, Any]] = None
    
    # Paths
    pretrained_path: Optional[str] = None
    vocab_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'architecture': self.architecture,
            'training_defaults': self.training_defaults,
            'enhancements': self.enhancements,
            'pretrained_path': self.pretrained_path,
            'vocab_file': self.vocab_file
        }


class ConfigLoader:
    """Load and manage model configurations."""
    
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = config_dir
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files."""
        config_files = [
            'bert_tiny_config.json',
            'bert_small_config.json',
            'bert_mini_config.json',
            'recurrent_config_template.json'
        ]
        
        for config_file in config_files:
            path = os.path.join(self.config_dir, config_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config_name = config_file.replace('_config.json', '').replace('_', '-')
                    self.configs[config_name] = json.load(f)
    
    def get_model_config(
        self,
        model_name: str,
        model_type: str = 'baseline',
        recurrent_preset: Optional[str] = None
    ) -> ModelConfiguration:
        """
        Get model configuration.
        
        Args:
            model_name: Name of the model ('bert-tiny', 'bert-small', 'bert-mini')
            model_type: Type of model ('baseline' or 'recurrent')
            recurrent_preset: Preset for recurrent depth configuration
            
        Returns:
            ModelConfiguration object
        """
        # Load base configuration
        base_config = self.configs.get(model_name)
        if not base_config:
            raise ValueError(f"Configuration not found for {model_name}")
        
        # Filter and map architecture parameters
        architecture = base_config['architecture'].copy()
        
        # Map 'hidden_act' to 'activation' if present
        if 'hidden_act' in architecture:
            architecture['activation'] = architecture.pop('hidden_act')
        
        # Remove parameters that don't exist in our configs
        params_to_remove = [
            'position_embedding_type', 'use_cache', 'classifier_dropout'
        ]
        for param in params_to_remove:
            architecture.pop(param, None)
        
        # Create model configuration
        config = ModelConfiguration(
            model_name=model_name,
            model_type=model_type,
            architecture=architecture,
            training_defaults=base_config['training_defaults'].copy(),
            pretrained_path=base_config.get('pretrained_path'),
            vocab_file=base_config.get('vocab_file')
        )
        
        # Add recurrent enhancements if needed
        if model_type == 'recurrent':
            recurrent_template = self.configs.get('recurrent-config-template', {})
            
            # Add architectural enhancements
            config.enhancements = recurrent_template.get('architectural_enhancements', {}).copy()
            
            # Add recurrent-specific parameters
            recurrent_params = recurrent_template.get('recurrent_enhancements', {})
            state_size = int(config.architecture['hidden_size'] * 
                           recurrent_params.get('state_size_multiplier', 0.5))
            
            config.architecture.update({
                'state_size': state_size,
                'chunk_size': recurrent_params.get('chunk_size', 64),
                'recurrent_depth': recurrent_params.get('recurrent_depth', 1),
                'use_gating': recurrent_params.get('use_gating', True),
                'state_dropout': recurrent_params.get('state_dropout', 0.1),
                'memory_efficient': recurrent_params.get('memory_efficient', True),
                'share_weights_across_depth': False
            })
            
            # Apply recurrent preset if specified
            if recurrent_preset:
                presets = recurrent_template.get('recurrent_depth_presets', {})
                if recurrent_preset in presets:
                    preset_config = presets[recurrent_preset]
                    config.architecture['num_hidden_layers'] = preset_config['num_hidden_layers']
                    config.architecture['recurrent_depth'] = preset_config['recurrent_depth']
                    print(f"Applied recurrent preset '{recurrent_preset}': {preset_config['description']}")
        else:
            # Add enhancements for baseline
            config.enhancements = {
                'use_flash_attention': True,
                'use_swiglu': True,
                'use_rope': True,
                'use_rms_norm': True
            }
        
        return config
    
    def get_training_config(
        self,
        model_name: str,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get training configuration with optional overrides.
        
        Args:
            model_name: Name of the model
            override_params: Parameters to override defaults
            
        Returns:
            Training configuration dictionary
        """
        base_config = self.configs.get(model_name)
        if not base_config:
            raise ValueError(f"Configuration not found for {model_name}")
        
        training_config = base_config['training_defaults'].copy()
        
        # Apply overrides
        if override_params:
            training_config.update(override_params)
        
        return training_config
    
    def list_available_models(self) -> list:
        """List available model configurations."""
        return [name for name in self.configs.keys() if 'bert' in name]
    
    def list_recurrent_presets(self) -> Dict[str, str]:
        """List available recurrent depth presets."""
        recurrent_template = self.configs.get('recurrent-config-template', {})
        presets = recurrent_template.get('recurrent_depth_presets', {})
        return {name: config['description'] for name, config in presets.items()}
    
    def save_config(self, config: ModelConfiguration, filename: str):
        """Save configuration to file."""
        path = os.path.join(self.config_dir, filename)
        with open(path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"Configuration saved to {path}")


def load_model_from_config(config: ModelConfiguration):
    """
    Load model from configuration.
    
    Args:
        config: ModelConfiguration object
        
    Returns:
        Initialized model
    """
    if config.model_type == 'baseline':
        from models.baseline import BaselineModel, BaselineConfig
        
        model_config = BaselineConfig(**config.architecture)
        if config.enhancements:
            for key, value in config.enhancements.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        model = BaselineModel(model_config)
        
    elif config.model_type == 'recurrent':
        from models.recurrent import RecurrentModel, RecurrentConfig
        
        model_config = RecurrentConfig(**config.architecture)
        if config.enhancements:
            for key, value in config.enhancements.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
        
        model = RecurrentModel(model_config)
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Load pretrained weights if available
    if config.pretrained_path and os.path.exists(config.pretrained_path):
        try:
            state_dict = torch.load(config.pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {config.pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model


# Example usage
if __name__ == "__main__":
    # Initialize config loader
    loader = ConfigLoader()
    
    # List available models
    print("Available models:", loader.list_available_models())
    
    # List recurrent presets
    print("\nRecurrent presets:")
    for name, description in loader.list_recurrent_presets().items():
        print(f"  {name}: {description}")
    
    # Load baseline configuration
    baseline_config = loader.get_model_config('bert-tiny', model_type='baseline')
    print(f"\nBaseline config loaded: {baseline_config.model_name}")
    
    # Load recurrent configuration with preset
    recurrent_config = loader.get_model_config(
        'bert-tiny',
        model_type='recurrent',
        recurrent_preset='balanced'
    )
    print(f"Recurrent config loaded with preset")
