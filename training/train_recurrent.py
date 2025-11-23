"""
Training script for pure recurrent transformer model.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .trainer_base import BaseTrainer, TrainingConfig
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recurrent import RecurrentModel, RecurrentConfig


class RecurrentTrainer(BaseTrainer):
    """Trainer for pure recurrent transformer model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pure recurrent transformer doesn't need state tracking
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if self.config.mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids'),
                    labels=batch['labels']
                )
            loss = outputs['loss']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                labels=batch['labels']
            )
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch['labels']).float().mean().item()
        
        return loss.item(), accuracy
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                labels=batch['labels']
            )
            
            loss = outputs['loss'].item()
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch['labels']).float().mean().item()
        
        return loss, accuracy


def create_recurrent_model(config: TrainingConfig) -> RecurrentModel:
    """
    Create recurrent model with configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        RecurrentModel instance
    """
    # Create recurrent configuration
    recurrent_config = RecurrentConfig(
        vocab_size=30522,  # BERT vocabulary size
        hidden_size=config.hidden_size if hasattr(config, 'hidden_size') else 256,
        num_hidden_layers=config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 3,
        recurrent_depth=config.recurrent_depth if hasattr(config, 'recurrent_depth') else 2,
        num_attention_heads=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 4,
        intermediate_size=config.intermediate_size if hasattr(config, 'intermediate_size') else 1024,
        hidden_dropout_prob=config.dropout_prob if hasattr(config, 'dropout_prob') else 0.1,
        attention_probs_dropout_prob=config.dropout_prob if hasattr(config, 'dropout_prob') else 0.1,
        num_labels=config.num_labels if hasattr(config, 'num_labels') else 2,
        
        # Enhancements
        use_flash_attention=config.use_flash_attention if hasattr(config, 'use_flash_attention') else True,
        use_swiglu=config.use_swiglu if hasattr(config, 'use_swiglu') else True,
        use_rope=config.use_rope if hasattr(config, 'use_rope') else True,
        use_rms_norm=config.use_rms_norm if hasattr(config, 'use_rms_norm') else True,
        
        # Model loading
        model_type=config.model_type if hasattr(config, 'model_type') else 'recurrent-bert',
        pretrained_path=config.pretrained_path if hasattr(config, 'pretrained_path') else None
    )
    
    # Create model
    model = RecurrentModel(recurrent_config)
    
    return model


def train_recurrent_model():
    """Main training function for recurrent model."""
    import argparse
    from training.utils import prepare_sst2_data, load_tokenizer
    
    parser = argparse.ArgumentParser(description='Train Recurrent Transformer Model')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size of the model')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--recurrent_depth', type=int, default=2,
                        help='Number of iterations through layers')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=1024,
                        help='Size of feedforward layer')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of steps between evaluations')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Number of steps between logging')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Number of steps between checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='checkpoints/recurrent',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/recurrent',
                        help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use_realtime_plot', action='store_true',
                        help='Show realtime training plots')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Early stopping patience')
    
    # Model enhancements
    parser.add_argument('--use_flash_attention', action='store_true', default=True,
                        help='Use flash attention')
    parser.add_argument('--use_swiglu', action='store_true', default=True,
                        help='Use SwiGLU activation')
    parser.add_argument('--use_rope', action='store_true', default=True,
                        help='Use RoPE embeddings')
    parser.add_argument('--use_rms_norm', action='store_true', default=True,
                        help='Use RMS normalization')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    print("Loading data...")
    tokenizer = load_tokenizer('bert-base-uncased')
    train_loader, val_loader, test_loader = prepare_sst2_data(
        data_dir='data/processed',
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=128
    )
    
    # Create configuration
    config = TrainingConfig(
        model_name='recurrent-transformer',
        model_type='recurrent-bert',
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        recurrent_depth=args.recurrent_depth,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=args.mixed_precision,
        use_realtime_plot=args.use_realtime_plot,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        use_flash_attention=args.use_flash_attention,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rms_norm=args.use_rms_norm
    )
    
    # Create model
    print(f"\nCreating recurrent model:")
    print(f"  Architecture: {args.hidden_size}H × {args.num_hidden_layers}L × {args.recurrent_depth}R")
    print(f"  Effective depth: {args.num_hidden_layers * args.recurrent_depth} layers")
    model = create_recurrent_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Create trainer
    trainer = RecurrentTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Train model
    print(f"\nStarting training for {args.num_epochs} epochs...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train_recurrent_model()