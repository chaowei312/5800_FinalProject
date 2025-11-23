"""
Base trainer class for model training.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau
)
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import os
import json
from tqdm import tqdm
import time
from abc import ABC, abstractmethod

from .utils.visualization import RealtimePlotter, TrainingMetrics


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model parameters
    model_name: str = "baseline"
    model_type: str = "bert-tiny"
    
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    adam_epsilon: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, step, exponential, onecycle, plateau
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    
    # Training settings
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # Paths
    output_dir: str = "checkpoints"
    data_dir: str = "data/processed"
    log_dir: str = "logs"
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Visualization
    use_realtime_plot: bool = True
    plot_update_frequency: int = 10
    
    # Seed
    seed: int = 42


class BaseTrainer(ABC):
    """Abstract base class for model training."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize plotter
        self.plotter = None
        if config.use_realtime_plot:
            self.plotter = RealtimePlotter(
                update_frequency=config.plot_update_frequency,
                save_path=config.log_dir
            )
            self.plotter.init_plot()
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': []
        }
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        if self.config.optimizer_type.lower() == 'adamw':
            return AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_type.lower() == 'adam':
            return Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            return SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        if self.config.scheduler_type.lower() == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )
        elif self.config.scheduler_type.lower() == 'step':
            return StepLR(
                self.optimizer,
                step_size=len(self.train_loader),
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler_type.lower() == 'exponential':
            return ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif self.config.scheduler_type.lower() == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1
            )
        elif self.config.scheduler_type.lower() == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        else:
            return None
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        pass
    
    @abstractmethod
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        pass
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Training step
            loss, accuracy = self.train_step(batch)
            
            # Accumulate metrics
            total_loss += loss
            batch_size = batch['input_ids'].size(0)
            total_correct += accuracy * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{self.get_lr():.2e}'
            })
            
            # Update global step
            self.global_step += 1
            
            # Update learning rate (if not plateau scheduler)
            if self.scheduler and self.config.scheduler_type.lower() != 'plateau':
                self.scheduler.step()
            
            # Logging and visualization
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_accuracy = total_correct / total_samples
                
                if self.plotter:
                    self.plotter.update(
                        step=self.global_step,
                        train_loss=avg_loss,
                        train_acc=avg_accuracy,
                        learning_rate=self.get_lr(),
                        epoch=self.current_epoch + 1,
                        total_epochs=self.config.num_epochs
                    )
            
            # Evaluation
            if self.val_loader and self.global_step % self.config.eval_steps == 0:
                val_loss, val_accuracy = self.evaluate()
                self.model.train()
                
                if self.plotter:
                    self.plotter.update(
                        step=self.global_step,
                        val_loss=val_loss,
                        val_acc=val_accuracy,
                        learning_rate=self.get_lr(),
                        epoch=self.current_epoch + 1,
                        total_epochs=self.config.num_epochs
                    )
                
                # Early stopping check
                if self.config.early_stopping:
                    if self._check_early_stopping(val_loss):
                        print(f"Early stopping triggered at epoch {self.current_epoch + 1}")
                        return total_loss / len(self.train_loader), total_correct / total_samples
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """Evaluate model on validation/test set."""
        self.model.eval()
        loader = loader or self.val_loader
        
        if loader is None:
            return 0.0, 0.0
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, accuracy = self.eval_step(batch)
                
                batch_size = batch['input_ids'].size(0)
                total_loss += loss * batch_size
                total_correct += accuracy * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            val_loss, val_acc = self.evaluate()
            
            # Update scheduler (for plateau scheduler)
            if self.scheduler and self.config.scheduler_type.lower() == 'plateau':
                self.scheduler.step(val_loss)
            
            # Update history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['train_accuracies'].append(train_acc)
            self.training_history['val_accuracies'].append(val_acc)
            self.training_history['learning_rates'].append(self.get_lr())
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {self.get_lr():.2e}")
            
            # Update plotter
            if self.plotter:
                self.plotter.update(
                    step=self.global_step,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    learning_rate=self.get_lr(),
                    epoch=epoch + 1,
                    total_epochs=self.config.num_epochs
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self.config.early_stopping:
                if self._check_early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        training_time = time.time() - start_time
        print(f"\\nTraining completed in {training_time:.2f} seconds")
        
        # Final evaluation on test set
        if self.test_loader:
            test_loss, test_acc = self.evaluate(self.test_loader)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            self.training_history['test_loss'] = test_loss
            self.training_history['test_accuracy'] = test_acc
        
        # Save final plot
        if self.plotter:
            self.plotter.save_final_plot(
                os.path.join(self.config.log_dir, f"{self.config.model_name}_final_plot.png")
            )
        
        # Save training history
        self.save_history()
        
        return self.training_history
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        if val_loss < self.best_val_loss - self.config.early_stopping_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                return True
        return False
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_step_{self.global_step}.pt'
        path = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            print(f"Saved best model to {path}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config.log_dir, f"{self.config.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', {})
        print(f"Loaded checkpoint from {path}")
