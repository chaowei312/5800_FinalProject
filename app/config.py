"""
Configuration file for the text classification app.

This file defines default paths and settings for the classifiers.
You can modify these to customize the app behavior.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Model checkpoint paths
MODEL_PATHS = {
    # Binary sentiment classification models
    'sentiment_baseline': os.path.join(PROJECT_ROOT, 'configs', 'Baseline_best_model.pt'),
    'sentiment_recurrent': os.path.join(PROJECT_ROOT, 'configs', 'Recurrent_best_model.pt'),
    
    # Multi-domain classification models
    'domain_baseline': os.path.join(PROJECT_ROOT, 'configs', 'Baseline_best_model_multi.pt'),
    'domain_recurrent': os.path.join(PROJECT_ROOT, 'configs', 'Recurrent_best_model_multi.pt'),
}

# Tokenizer configuration
TOKENIZER_NAME = 'bert-base-uncased'
MAX_LENGTH = 256

# Device configuration
# Set to 'cuda' to force GPU, 'cpu' to force CPU, or None for automatic detection
DEVICE = None  # Auto-detect by default

# Model configurations
MODEL_CONFIG = {
    'sentiment': {
        'num_labels': 2,
        'task_name': 'Binary Sentiment Classification',
        'labels': ['Negative', 'Positive']
    },
    'domain': {
        'num_labels': 3,
        'task_name': 'Multi-Domain Classification',
        'labels': ['movie_review', 'online_shopping', 'local_business_review']
    }
}

# Inference settings
INFERENCE_CONFIG = {
    'batch_size': 32,
    'show_progress': True,
    'temperature': 1.0,  # For softmax (1.0 = no change)
}

# Output formatting
OUTPUT_FORMAT = {
    'decimal_places': 4,
    'show_probabilities': False,
    'verbose': True
}

# CLI defaults
CLI_DEFAULTS = {
    'task': 'both',
    'sentiment_model': 'baseline',
    'domain_model': 'baseline',
    'show_probs': False
}


def get_model_path(model_key: str) -> str:
    """
    Get the path to a model checkpoint.
    
    Args:
        model_key: One of 'sentiment_baseline', 'sentiment_recurrent',
                   'domain_baseline', 'domain_recurrent'
    
    Returns:
        Path to the model checkpoint
    
    Raises:
        KeyError: If model_key is not recognized
        FileNotFoundError: If the model file doesn't exist
    """
    if model_key not in MODEL_PATHS:
        raise KeyError(f"Unknown model key: {model_key}. "
                      f"Available keys: {list(MODEL_PATHS.keys())}")
    
    path = MODEL_PATHS[model_key]
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    
    return path


def check_models_available() -> dict:
    """
    Check which model checkpoints are available.
    
    Returns:
        Dictionary mapping model keys to availability status
    """
    return {
        key: os.path.exists(path)
        for key, path in MODEL_PATHS.items()
    }


def print_config():
    """Print the current configuration."""
    print("="*70)
    print("TEXT CLASSIFICATION APP - CONFIGURATION")
    print("="*70)
    
    print("\n📁 Project Root:")
    print(f"   {PROJECT_ROOT}")
    
    print("\n🤖 Model Paths:")
    for key, path in MODEL_PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"   {exists} {key:25} {path}")
    
    print(f"\n🔤 Tokenizer: {TOKENIZER_NAME}")
    print(f"📏 Max Length: {MAX_LENGTH}")
    print(f"💻 Device: {DEVICE if DEVICE else 'Auto-detect'}")
    
    print("\n📊 Task Configurations:")
    for task, config in MODEL_CONFIG.items():
        print(f"   {task}:")
        print(f"      Labels: {config['num_labels']} ({', '.join(config['labels'])})")
    
    print("="*70)


if __name__ == '__main__':
    # Print configuration when run as a script
    print_config()
    
    # Check model availability
    print("\n🔍 Checking model availability...")
    availability = check_models_available()
    
    available_count = sum(availability.values())
    total_count = len(availability)
    
    if available_count == total_count:
        print(f"✅ All {total_count} models are available!")
    else:
        print(f"⚠️  {available_count}/{total_count} models are available")
        print("\nMissing models:")
        for key, available in availability.items():
            if not available:
                print(f"   ✗ {key}: {MODEL_PATHS[key]}")

