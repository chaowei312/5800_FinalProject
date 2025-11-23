"""
Real-time visualization utilities for training progress.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import time
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')


class TrainingMetrics:
    """Container for training metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.train_losses = deque(maxlen=window_size)
        self.val_losses = deque(maxlen=window_size)
        self.train_accuracies = deque(maxlen=window_size)
        self.val_accuracies = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)
        
        # Full history for final plots
        self.full_train_losses = []
        self.full_val_losses = []
        self.full_train_accuracies = []
        self.full_val_accuracies = []
        self.full_learning_rates = []
        self.full_steps = []
        
    def add_train_metrics(self, step: int, loss: float, accuracy: float, lr: float):
        """Add training metrics."""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(lr)
        self.steps.append(step)
        
        self.full_train_losses.append(loss)
        self.full_train_accuracies.append(accuracy)
        self.full_learning_rates.append(lr)
        self.full_steps.append(step)
    
    def add_val_metrics(self, loss: float, accuracy: float):
        """Add validation metrics."""
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        
        self.full_val_losses.append(loss)
        self.full_val_accuracies.append(accuracy)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metrics."""
        metrics = {}
        if self.train_losses:
            metrics['train_loss'] = self.train_losses[-1]
        if self.train_accuracies:
            metrics['train_acc'] = self.train_accuracies[-1]
        if self.val_losses:
            metrics['val_loss'] = self.val_losses[-1]
        if self.val_accuracies:
            metrics['val_acc'] = self.val_accuracies[-1]
        if self.learning_rates:
            metrics['learning_rate'] = self.learning_rates[-1]
        return metrics


class RealtimePlotter:
    """
    Real-time plotting for training metrics with dynamic updates.
    Clears previous plots and shows current training progress.
    """
    
    def __init__(
        self,
        update_frequency: int = 10,
        figsize: Tuple[int, int] = (15, 10),
        style: str = 'seaborn-v0_8-darkgrid',
        save_path: Optional[str] = None
    ):
        self.update_frequency = update_frequency
        self.figsize = figsize
        self.save_path = save_path
        self.metrics = TrainingMetrics()
        self.step_counter = 0
        self.start_time = time.time()
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Initialize figure
        self.fig = None
        self.axes = None
        self.lines = {}
        self.is_notebook = self._check_notebook()
        
    def _check_notebook(self) -> bool:
        """Check if running in Jupyter notebook."""
        try:
            get_ipython()
            return True
        except:
            return False
    
    def init_plot(self):
        """Initialize the plot with subplots."""
        if self.is_notebook:
            # For Jupyter notebooks
            plt.ion()  # Interactive mode
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.fig.suptitle('Training Progress - Real-time Updates', fontsize=16, fontweight='bold')
        
        # Configure subplots
        self.axes[0, 0].set_title('Loss', fontsize=14)
        self.axes[0, 0].set_xlabel('Steps')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Accuracy', fontsize=14)
        self.axes[0, 1].set_xlabel('Steps')
        self.axes[0, 1].set_ylabel('Accuracy (%)')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('Learning Rate', fontsize=14)
        self.axes[1, 0].set_xlabel('Steps')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Training Statistics', fontsize=14)
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Initialize empty lines
        self.lines['train_loss'], = self.axes[0, 0].plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.lines['val_loss'], = self.axes[0, 0].plot([], [], 'r--', label='Val Loss', linewidth=2)
        self.axes[0, 0].legend(loc='upper right')
        
        self.lines['train_acc'], = self.axes[0, 1].plot([], [], 'g-', label='Train Acc', linewidth=2)
        self.lines['val_acc'], = self.axes[0, 1].plot([], [], 'orange', label='Val Acc', linewidth=2, linestyle='--')
        self.axes[0, 1].legend(loc='lower right')
        
        self.lines['lr'], = self.axes[1, 0].plot([], [], 'purple', linewidth=2)
        
    def update(
        self,
        step: int,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None
    ):
        """
        Update plot with new metrics.
        
        Args:
            step: Current training step
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Current learning rate
            epoch: Current epoch
            total_epochs: Total number of epochs
        """
        self.step_counter += 1
        
        # Add metrics
        if train_loss is not None and learning_rate is not None:
            self.metrics.add_train_metrics(
                step, train_loss,
                train_acc if train_acc is not None else 0,
                learning_rate
            )
        
        if val_loss is not None:
            self.metrics.add_val_metrics(
                val_loss,
                val_acc if val_acc is not None else 0
            )
        
        # Update plot at specified frequency
        if self.step_counter % self.update_frequency == 0:
            self._refresh_plot(epoch, total_epochs)
    
    def _refresh_plot(self, epoch: Optional[int] = None, total_epochs: Optional[int] = None):
        """Refresh the plot with current data."""
        if self.fig is None:
            self.init_plot()
        
        # Clear and update axes
        if self.is_notebook:
            clear_output(wait=True)
        
        # Update loss plot
        if self.metrics.full_steps:
            self.axes[0, 0].clear()
            self.axes[0, 0].plot(
                self.metrics.full_steps,
                self.metrics.full_train_losses,
                'b-', label='Train Loss', linewidth=2, alpha=0.8
            )
            if self.metrics.full_val_losses:
                val_steps = np.linspace(
                    min(self.metrics.full_steps),
                    max(self.metrics.full_steps),
                    len(self.metrics.full_val_losses)
                )
                self.axes[0, 0].plot(
                    val_steps,
                    self.metrics.full_val_losses,
                    'r--', label='Val Loss', linewidth=2, alpha=0.8
                )
            self.axes[0, 0].set_title('Loss', fontsize=14)
            self.axes[0, 0].set_xlabel('Steps')
            self.axes[0, 0].set_ylabel('Loss')
            self.axes[0, 0].legend(loc='upper right')
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Update accuracy plot
        if self.metrics.full_train_accuracies:
            self.axes[0, 1].clear()
            self.axes[0, 1].plot(
                self.metrics.full_steps,
                [acc * 100 for acc in self.metrics.full_train_accuracies],
                'g-', label='Train Acc', linewidth=2, alpha=0.8
            )
            if self.metrics.full_val_accuracies:
                val_steps = np.linspace(
                    min(self.metrics.full_steps),
                    max(self.metrics.full_steps),
                    len(self.metrics.full_val_accuracies)
                )
                self.axes[0, 1].plot(
                    val_steps,
                    [acc * 100 for acc in self.metrics.full_val_accuracies],
                    'orange', label='Val Acc', linewidth=2, linestyle='--', alpha=0.8
                )
            self.axes[0, 1].set_title('Accuracy', fontsize=14)
            self.axes[0, 1].set_xlabel('Steps')
            self.axes[0, 1].set_ylabel('Accuracy (%)')
            self.axes[0, 1].legend(loc='lower right')
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Update learning rate plot
        if self.metrics.full_learning_rates:
            self.axes[1, 0].clear()
            self.axes[1, 0].plot(
                self.metrics.full_steps,
                self.metrics.full_learning_rates,
                'purple', linewidth=2, alpha=0.8
            )
            self.axes[1, 0].set_title('Learning Rate', fontsize=14)
            self.axes[1, 0].set_xlabel('Steps')
            self.axes[1, 0].set_ylabel('Learning Rate')
            self.axes[1, 0].set_yscale('log')
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Update statistics
        self._update_statistics(epoch, total_epochs)
        
        plt.tight_layout()
        
        if self.is_notebook:
            display(self.fig)
        else:
            plt.pause(0.01)  # Small pause for update
        
        # Save figure if path provided
        if self.save_path:
            self.fig.savefig(f"{self.save_path}/training_progress.png", dpi=100, bbox_inches='tight')
    
    def _update_statistics(self, epoch: Optional[int] = None, total_epochs: Optional[int] = None):
        """Update statistics text panel."""
        self.axes[1, 1].clear()
        self.axes[1, 1].axis('off')
        
        # Calculate statistics
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        latest_metrics = self.metrics.get_latest_metrics()
        
        # Create statistics text
        stats_text = f"Training Statistics\\n" + "=" * 25 + "\\n\\n"
        
        if epoch is not None and total_epochs is not None:
            stats_text += f"Epoch: {epoch}/{total_epochs}\\n"
            progress = (epoch / total_epochs) * 100
            stats_text += f"Progress: {progress:.1f}%\\n\\n"
        
        stats_text += f"Latest Metrics:\\n"
        stats_text += "-" * 20 + "\\n"
        
        for key, value in latest_metrics.items():
            if 'loss' in key:
                stats_text += f"{key.replace('_', ' ').title()}: {value:.4f}\\n"
            elif 'acc' in key:
                stats_text += f"{key.replace('_', ' ').title()}: {value*100:.2f}%\\n"
            elif 'learning_rate' in key:
                stats_text += f"Learning Rate: {value:.2e}\\n"
        
        stats_text += f"\\nElapsed Time: {elapsed_str}\\n"
        
        if self.metrics.full_steps:
            steps_per_second = len(self.metrics.full_steps) / elapsed_time
            stats_text += f"Steps/sec: {steps_per_second:.2f}\\n"
        
        # Best metrics
        if self.metrics.full_val_losses:
            best_val_loss = min(self.metrics.full_val_losses)
            stats_text += f"\\nBest Val Loss: {best_val_loss:.4f}\\n"
        
        if self.metrics.full_val_accuracies:
            best_val_acc = max(self.metrics.full_val_accuracies)
            stats_text += f"Best Val Acc: {best_val_acc*100:.2f}%\\n"
        
        self.axes[1, 1].text(
            0.1, 0.5, stats_text,
            transform=self.axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment='center',
            fontfamily='monospace'
        )
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def close(self):
        """Close the plot."""
        if self.fig is not None:
            plt.close(self.fig)
    
    def save_final_plot(self, path: str):
        """Save the final plot."""
        if self.fig is not None:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Final plot saved to {path}")


class ComparisonPlotter:
    """Plot comparison between multiple models."""
    
    @staticmethod
    def plot_comparison(
        metrics_dict: Dict[str, TrainingMetrics],
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple models.
        
        Args:
            metrics_dict: Dictionary of model_name -> TrainingMetrics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
            color = colors[idx % len(colors)]
            
            # Plot losses
            if metrics.full_train_losses:
                axes[0, 0].plot(
                    metrics.full_steps,
                    metrics.full_train_losses,
                    color=color, label=f'{model_name} Train',
                    linewidth=2, alpha=0.7
                )
            
            if metrics.full_val_losses:
                val_steps = np.linspace(
                    min(metrics.full_steps),
                    max(metrics.full_steps),
                    len(metrics.full_val_losses)
                )
                axes[0, 0].plot(
                    val_steps,
                    metrics.full_val_losses,
                    color=color, label=f'{model_name} Val',
                    linewidth=2, linestyle='--', alpha=0.7
                )
            
            # Plot accuracies
            if metrics.full_train_accuracies:
                axes[0, 1].plot(
                    metrics.full_steps,
                    [acc * 100 for acc in metrics.full_train_accuracies],
                    color=color, label=f'{model_name} Train',
                    linewidth=2, alpha=0.7
                )
            
            if metrics.full_val_accuracies:
                val_steps = np.linspace(
                    min(metrics.full_steps),
                    max(metrics.full_steps),
                    len(metrics.full_val_accuracies)
                )
                axes[0, 1].plot(
                    val_steps,
                    [acc * 100 for acc in metrics.full_val_accuracies],
                    color=color, label=f'{model_name} Val',
                    linewidth=2, linestyle='--', alpha=0.7
                )
        
        # Configure axes
        axes[0, 0].set_title('Loss Comparison')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bar chart for final metrics
        model_names = list(metrics_dict.keys())
        final_train_losses = []
        final_val_losses = []
        final_val_accs = []
        
        for metrics in metrics_dict.values():
            if metrics.full_train_losses:
                final_train_losses.append(metrics.full_train_losses[-1])
            else:
                final_train_losses.append(0)
            
            if metrics.full_val_losses:
                final_val_losses.append(metrics.full_val_losses[-1])
            else:
                final_val_losses.append(0)
            
            if metrics.full_val_accuracies:
                final_val_accs.append(metrics.full_val_accuracies[-1] * 100)
            else:
                final_val_accs.append(0)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, final_train_losses, width, label='Train Loss', alpha=0.8)
        axes[1, 0].bar(x + width/2, final_val_losses, width, label='Val Loss', alpha=0.8)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Final Loss')
        axes[1, 0].set_title('Final Loss Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(x, final_val_accs, alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Validation Accuracy (%)')
        axes[1, 1].set_title('Final Validation Accuracy')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        return fig
