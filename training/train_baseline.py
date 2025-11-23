"""
Training script for baseline transformer model.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from .trainer_base import BaseTrainer, TrainingConfig
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import BaselineModel, BaselineConfig


class BaselineTrainer(BaseTrainer):
    """Trainer for baseline transformer model."""
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data containing input_ids, attention_mask, labels
            
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
            
            # Backward pass with gradient scaling
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


def create_baseline_model(config: TrainingConfig) -> BaselineModel:
    """
    Create baseline model with configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Baseline model instance
    """
    model_config = BaselineConfig(
        model_type=config.model_type,
        num_labels=2,  # Binary classification for SST-2
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_flash_attention=True,
        use_swiglu=True,
        use_rope=True,
        use_rms_norm=True
    )
    
    # Set pretrained path based on model type
    if config.model_type == 'bert-tiny':
        model_config.pretrained_path = 'bert-tiny/pytorch_model.bin'
        model_config.hidden_size = 128
        model_config.num_hidden_layers = 2
        model_config.num_attention_heads = 2
        model_config.intermediate_size = 512
    elif config.model_type == 'bert-small':
        model_config.pretrained_path = 'bert-small/pytorch_model.bin'
        model_config.hidden_size = 512
        model_config.num_hidden_layers = 4
        model_config.num_attention_heads = 8
        model_config.intermediate_size = 2048
    elif config.model_type == 'bert-mini':
        model_config.pretrained_path = 'bert-mini/pytorch_model.bin'
        model_config.hidden_size = 256
        model_config.num_hidden_layers = 4
        model_config.num_attention_heads = 4
        model_config.intermediate_size = 1024
    
    return BaselineModel(model_config)


def main():
    """Main training function."""
    import argparse
    from torch.utils.data import DataLoader
    from .utils.data_loader import prepare_sst2_data, load_tokenizer
    
    parser = argparse.ArgumentParser(description='Train baseline transformer model')
    parser.add_argument('--model-type', type=str, default='bert-tiny',
                       choices=['bert-tiny', 'bert-small', 'bert-mini'],
                       help='Type of BERT model to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='checkpoints/baseline',
                       help='Output directory for checkpoints')
    parser.add_argument('--use-cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_name='baseline',
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device='cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu',
        mixed_precision=args.mixed_precision,
        use_realtime_plot=True,
        plot_update_frequency=10
    )
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.model_type)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_sst2_data(
        data_dir=config.data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=128
    )
    
    # Create model
    model = create_baseline_model(config)
    
    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Train model
    history = trainer.train()
    
    print("\\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    if 'test_accuracy' in history:
        print(f"Test accuracy: {history['test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
