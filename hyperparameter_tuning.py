"""
Hyperparameter tuning for baseline and recurrent models with sensitivity analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import itertools
import time
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.baseline import BaselineModel, BaselineConfig
from models.recurrent import RecurrentModel, RecurrentConfig
from training.utils import prepare_sst2_data, load_tokenizer
from evaluation import ModelEvaluator


class HyperparameterTuner:
    """Hyperparameter tuning with performance analysis."""
    
    def __init__(
        self,
        model_type: str = 'baseline',
        device: str = 'cuda',
        subset_size: int = 1000  # Use subset for faster tuning
    ):
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.subset_size = subset_size
        self.results = []
        
        # Load data
        self.tokenizer = load_tokenizer('bert-base-uncased')
        self.train_loader, self.val_loader, _ = self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data loaders with subset for faster tuning."""
        train_loader, val_loader, test_loader = prepare_sst2_data(
            data_dir='data/processed',
            tokenizer=self.tokenizer,
            batch_size=32,
            max_length=128
        )
        
        # Create subset for faster experimentation
        if self.subset_size:
            train_indices = torch.randperm(len(train_loader.dataset))[:self.subset_size]
            val_indices = torch.randperm(len(val_loader.dataset))[:min(500, len(val_loader.dataset))]
            
            train_subset = Subset(train_loader.dataset, train_indices)
            val_subset = Subset(val_loader.dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def define_search_space(self) -> Dict[str, List[Any]]:
        """Define hyperparameter search space."""
        if self.model_type == 'baseline':
            # Baseline model focuses on traditional transformer hyperparameters
            return {
                'hidden_size': [128, 256, 512],
                'num_hidden_layers': [2, 4, 6, 8],  # More layers for baseline
                'num_attention_heads': [2, 4, 8],
                'intermediate_size': [256, 512, 1024, 2048],
                'dropout_prob': [0.1, 0.2, 0.3],
                'learning_rate': [1e-5, 3e-5, 5e-5],
                'batch_size': [16, 32, 64],
                # Architectural enhancements
                'use_flash_attention': [True, False],
                'use_swiglu': [True, False],
                'use_rope': [True, False],
                'use_rms_norm': [True, False]
            }
        else:  # recurrent
            # Recurrent model focuses on state management and depth iteration
            return {
                'hidden_size': [128, 256, 384],
                'num_hidden_layers': [1, 2, 3, 4],  # Fewer unique layers
                'recurrent_depth': [1, 2, 3, 4],    # Multiple iterations
                'num_attention_heads': [2, 4, 8],
                'dropout_prob': [0.1, 0.2],
                'learning_rate': [1e-5, 3e-5, 5e-5],
                'batch_size': [16, 32],
                'residual_scale': [0.5, 0.7, 1.0]   # Cross-iteration residual scale
            }
    
    def create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create model with given hyperparameters."""
        if self.model_type == 'baseline':
            # Baseline model configuration - standard transformer
            config = BaselineConfig(
                vocab_size=30522,
                hidden_size=hyperparams['hidden_size'],
                num_hidden_layers=hyperparams['num_hidden_layers'],
                num_attention_heads=hyperparams['num_attention_heads'],
                intermediate_size=hyperparams['intermediate_size'],
                hidden_dropout_prob=hyperparams['dropout_prob'],
                attention_probs_dropout_prob=hyperparams['dropout_prob'],
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                num_labels=2,
                # Baseline-specific enhancements
                use_flash_attention=hyperparams.get('use_flash_attention', True),
                use_swiglu=hyperparams.get('use_swiglu', True),
                use_rope=hyperparams.get('use_rope', True),
                use_rms_norm=hyperparams.get('use_rms_norm', True),
                activation='gelu',
                model_type='baseline'
            )
            return BaselineModel(config)
        else:
            # Recurrent model configuration - with state management
            config = RecurrentConfig(
                vocab_size=30522,
                hidden_size=hyperparams['hidden_size'],
                num_hidden_layers=hyperparams['num_hidden_layers'],
                num_attention_heads=hyperparams['num_attention_heads'],
                intermediate_size=hyperparams.get('intermediate_size', hyperparams['hidden_size'] * 4),
                hidden_dropout_prob=hyperparams['dropout_prob'],
                attention_probs_dropout_prob=hyperparams['dropout_prob'],
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
                num_labels=2,
                # Recurrent-specific parameters (Pure recurrent transformer)
                recurrent_depth=hyperparams['recurrent_depth'],
                residual_scale=hyperparams.get('residual_scale', 0.5),
                memory_efficient=True,
                share_weights_across_depth=False,
                # Recurrent model always uses these enhancements
                use_flash_attention=True,
                use_swiglu=True,
                use_rope=True,
                use_rms_norm=True,
                activation='gelu',
                model_type='recurrent'
            )
            return RecurrentModel(config)
    
    def train_and_evaluate(
        self,
        model: nn.Module,
        hyperparams: Dict[str, Any],
        num_epochs: int = 3
    ) -> Dict[str, float]:
        """Train model and evaluate performance."""
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(self.train_loader))
        
        # Evaluation
        model.eval()
        evaluator = ModelEvaluator(model, self.device)
        metrics = evaluator.evaluate(self.val_loader, verbose=False)
        
        # Measure inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(10):  # Average over 10 runs
                batch = next(iter(self.val_loader))
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                start_time = time.time()
                _ = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                inference_times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        return {
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'loss': metrics['loss'],
            'perplexity': metrics['perplexity'],
            'inference_time_ms': avg_inference_time,
            'model_size_mb': metrics['model_size_mb'],
            'total_parameters': metrics['total_parameters'],
            'final_train_loss': train_losses[-1]
        }
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]] = None,
        num_epochs: int = 3
    ) -> pd.DataFrame:
        """Perform grid search over hyperparameters."""
        if param_grid is None:
            param_grid = self.define_search_space()
        
        # Create parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} hyperparameter combinations")
        
        for i, params in enumerate(tqdm(param_combinations, desc="Grid Search")):
            hyperparams = dict(zip(param_names, params))
            
            try:
                # Create and train model
                model = self.create_model(hyperparams)
                metrics = self.train_and_evaluate(model, hyperparams, num_epochs)
                
                # Store results
                result = {**hyperparams, **metrics}
                self.results.append(result)
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Failed for params {hyperparams}: {e}")
                continue
        
        return pd.DataFrame(self.results)
    
    def random_search(
        self,
        param_grid: Dict[str, List[Any]] = None,
        n_iter: int = 20,
        num_epochs: int = 3
    ) -> pd.DataFrame:
        """Perform random search over hyperparameters."""
        if param_grid is None:
            param_grid = self.define_search_space()
        
        print(f"Testing {n_iter} random hyperparameter combinations")
        
        for i in tqdm(range(n_iter), desc="Random Search"):
            hyperparams = {
                key: np.random.choice(values)
                for key, values in param_grid.items()
            }
            
            try:
                # Create and train model
                model = self.create_model(hyperparams)
                metrics = self.train_and_evaluate(model, hyperparams, num_epochs)
                
                # Store results
                result = {**hyperparams, **metrics}
                self.results.append(result)
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Failed for params {hyperparams}: {e}")
                continue
        
        return pd.DataFrame(self.results)
    
    def save_results(self, filepath: str):
        """Save tuning results to file."""
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


class HyperparameterVisualizer:
    """Visualize hyperparameter sensitivity and performance trade-offs."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        
    def plot_bubble_chart(
        self,
        x_param: str = 'inference_time_ms',
        y_param: str = 'accuracy',
        size_param: str = 'model_size_mb',
        color_param: str = 'num_hidden_layers',
        save_path: str = None
    ):
        """
        Create bubble chart showing multi-dimensional trade-offs.
        
        Args:
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            size_param: Parameter for bubble size
            color_param: Parameter for bubble color
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Normalize size for better visualization
        sizes = self.df[size_param].values
        sizes_normalized = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 1000 + 100
        
        # Create scatter plot
        scatter = ax.scatter(
            self.df[x_param],
            self.df[y_param],
            s=sizes_normalized,
            c=self.df[color_param],
            alpha=0.6,
            cmap='viridis',
            edgecolors='black',
            linewidth=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_param.replace('_', ' ').title(), fontsize=12)
        
        # Labels and title
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Performance vs Efficiency Trade-off\n(Bubble size: {size_param})', 
                     fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add size legend
        for size_val in [sizes.min(), np.median(sizes), sizes.max()]:
            ax.scatter([], [], s=(size_val - sizes.min()) / (sizes.max() - sizes.min()) * 1000 + 100,
                      c='gray', alpha=0.6, label=f'{size_val:.1f} MB')
        ax.legend(scatterpoints=1, title='Model Size', loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bubble chart saved to {save_path}")
        
        plt.show()
    
    def plot_parallel_coordinates(
        self,
        params: List[str] = None,
        metric: str = 'accuracy',
        save_path: str = None
    ):
        """Create parallel coordinates plot for hyperparameter analysis."""
        from pandas.plotting import parallel_coordinates
        
        if params is None:
            params = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                     'dropout_prob', 'learning_rate']
        
        # Filter columns
        cols = params + [metric]
        df_filtered = self.df[cols].copy()
        
        # Normalize numerical columns
        for col in params:
            if df_filtered[col].dtype in ['float64', 'int64']:
                df_filtered[col] = (df_filtered[col] - df_filtered[col].min()) / \
                                  (df_filtered[col].max() - df_filtered[col].min())
        
        # Bin the metric for coloring
        df_filtered['performance'] = pd.qcut(df_filtered[metric], q=5, 
                                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        parallel_coordinates(df_filtered, 'performance', cols=params, 
                             colormap='RdYlGn', alpha=0.4, ax=ax)
        
        ax.set_title(f'Hyperparameter Sensitivity (colored by {metric})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.legend(loc='upper right', title='Performance')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parallel coordinates plot saved to {save_path}")
        
        plt.show()
    
    def plot_heatmap(
        self,
        param1: str,
        param2: str,
        metric: str = 'accuracy',
        save_path: str = None
    ):
        """Create heatmap showing interaction between two parameters."""
        # Pivot data for heatmap
        pivot_table = self.df.pivot_table(
            values=metric,
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=pivot_table.mean().mean(), ax=ax)
        
        ax.set_title(f'{metric.title()} by {param1} and {param2}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(param2.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(param1.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_sensitivity_analysis(self, save_path: str = None):
        """Create comprehensive sensitivity analysis plots."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Get numeric columns for analysis
        numeric_params = [col for col in self.df.columns 
                         if self.df[col].dtype in ['float64', 'int64'] 
                         and col not in ['accuracy', 'f1', 'loss', 'inference_time_ms', 
                                       'model_size_mb', 'total_parameters']]
        
        # 1. Box plots for each hyperparameter vs accuracy
        for i, param in enumerate(numeric_params[:6]):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            
            # Create bins for continuous parameters
            if self.df[param].nunique() > 5:
                bins = pd.qcut(self.df[param], q=3, duplicates='drop')
                self.df.boxplot(column='accuracy', by=bins, ax=ax)
                ax.set_xlabel(param.replace('_', ' ').title())
            else:
                self.df.boxplot(column='accuracy', by=param, ax=ax)
                ax.set_xlabel(param.replace('_', ' ').title())
            
            ax.set_ylabel('Accuracy')
            ax.set_title('')
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        # 2. Correlation matrix
        ax = fig.add_subplot(gs[2, :])
        corr_cols = ['accuracy', 'f1', 'inference_time_ms', 'model_size_mb'] + numeric_params[:4]
        corr_matrix = self.df[corr_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensitivity analysis saved to {save_path}")
        
        plt.show()
    
    def plot_pareto_frontier(self, save_path: str = None):
        """Plot Pareto frontier for multi-objective optimization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Accuracy vs Inference Time
        x1 = self.df['inference_time_ms'].values
        y1 = self.df['accuracy'].values
        
        # Find Pareto frontier
        pareto_idx1 = self._find_pareto_frontier(x1, y1, minimize_x=True, maximize_y=True)
        
        axes[0].scatter(x1, y1, alpha=0.5, label='All configurations')
        axes[0].scatter(x1[pareto_idx1], y1[pareto_idx1], 
                       color='red', s=100, label='Pareto optimal', zorder=5)
        axes[0].plot(np.sort(x1[pareto_idx1]), 
                    y1[pareto_idx1][np.argsort(x1[pareto_idx1])], 
                    'r--', alpha=0.5)
        
        axes[0].set_xlabel('Inference Time (ms)', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy vs Inference Time Trade-off', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Model Size
        x2 = self.df['model_size_mb'].values
        y2 = self.df['accuracy'].values
        
        pareto_idx2 = self._find_pareto_frontier(x2, y2, minimize_x=True, maximize_y=True)
        
        axes[1].scatter(x2, y2, alpha=0.5, label='All configurations')
        axes[1].scatter(x2[pareto_idx2], y2[pareto_idx2], 
                       color='red', s=100, label='Pareto optimal', zorder=5)
        axes[1].plot(np.sort(x2[pareto_idx2]), 
                    y2[pareto_idx2][np.argsort(x2[pareto_idx2])], 
                    'r--', alpha=0.5)
        
        axes[1].set_xlabel('Model Size (MB)', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy vs Model Size Trade-off', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Pareto Frontier Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto frontier plot saved to {save_path}")
        
        plt.show()
    
    def _find_pareto_frontier(self, x, y, minimize_x=True, maximize_y=True):
        """Find Pareto frontier indices."""
        pareto_idx = []
        
        for i in range(len(x)):
            is_pareto = True
            for j in range(len(x)):
                if i == j:
                    continue
                
                if minimize_x and maximize_y:
                    if x[j] <= x[i] and y[j] >= y[i]:
                        if x[j] < x[i] or y[j] > y[i]:
                            is_pareto = False
                            break
            
            if is_pareto:
                pareto_idx.append(i)
        
        return np.array(pareto_idx)


def run_hyperparameter_tuning(
    model_type: str = 'baseline',
    search_type: str = 'random',
    n_iter: int = 20,
    save_dir: str = 'tuning_results'
):
    """Run complete hyperparameter tuning pipeline."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize tuner
    tuner = HyperparameterTuner(model_type=model_type, subset_size=1000)
    
    # Perform search
    if search_type == 'grid':
        # Define smaller grid for demonstration
        param_grid = {
            'hidden_size': [128, 256],
            'num_hidden_layers': [2, 4],
            'num_attention_heads': [2, 4],
            'intermediate_size': [256, 512],
            'dropout_prob': [0.1, 0.2],
            'learning_rate': [3e-5, 5e-5],
            'batch_size': [32],
            'use_flash_attention': [True],
            'use_swiglu': [True, False],
            'use_rope': [True],
            'use_rms_norm': [True]
        }
        results_df = tuner.grid_search(param_grid, num_epochs=3)
    else:
        results_df = tuner.random_search(n_iter=n_iter, num_epochs=3)
    
    # Save results
    results_path = os.path.join(save_dir, f'{model_type}_tuning_results.csv')
    tuner.save_results(results_path)
    
    # Visualize results
    visualizer = HyperparameterVisualizer(results_df)
    
    # Create all visualizations
    visualizer.plot_bubble_chart(
        save_path=os.path.join(save_dir, f'{model_type}_bubble_chart.png')
    )
    
    visualizer.plot_parallel_coordinates(
        save_path=os.path.join(save_dir, f'{model_type}_parallel_coords.png')
    )
    
    visualizer.plot_heatmap(
        'num_hidden_layers', 'hidden_size',
        save_path=os.path.join(save_dir, f'{model_type}_heatmap.png')
    )
    
    visualizer.plot_sensitivity_analysis(
        save_path=os.path.join(save_dir, f'{model_type}_sensitivity.png')
    )
    
    visualizer.plot_pareto_frontier(
        save_path=os.path.join(save_dir, f'{model_type}_pareto.png')
    )
    
    # Print best configurations
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)
    
    # Best for accuracy
    best_acc = results_df.loc[results_df['accuracy'].idxmax()]
    print("\nBest Accuracy Configuration:")
    print(f"  Accuracy: {best_acc['accuracy']:.4f}")
    print(f"  F1 Score: {best_acc['f1']:.4f}")
    print(f"  Inference Time: {best_acc['inference_time_ms']:.2f} ms")
    print(f"  Model Size: {best_acc['model_size_mb']:.2f} MB")
    
    # Best for speed
    best_speed = results_df.loc[results_df['inference_time_ms'].idxmin()]
    print("\nFastest Configuration:")
    print(f"  Inference Time: {best_speed['inference_time_ms']:.2f} ms")
    print(f"  Accuracy: {best_speed['accuracy']:.4f}")
    print(f"  Model Size: {best_speed['model_size_mb']:.2f} MB")
    
    # Best balanced (using a simple score)
    results_df['balanced_score'] = (
        results_df['accuracy'] / results_df['accuracy'].max() * 0.5 +
        (1 - results_df['inference_time_ms'] / results_df['inference_time_ms'].max()) * 0.3 +
        (1 - results_df['model_size_mb'] / results_df['model_size_mb'].max()) * 0.2
    )
    best_balanced = results_df.loc[results_df['balanced_score'].idxmax()]
    print("\nBest Balanced Configuration:")
    print(f"  Accuracy: {best_balanced['accuracy']:.4f}")
    print(f"  Inference Time: {best_balanced['inference_time_ms']:.2f} ms")
    print(f"  Model Size: {best_balanced['model_size_mb']:.2f} MB")
    print(f"  Balanced Score: {best_balanced['balanced_score']:.4f}")
    
    return results_df


if __name__ == "__main__":
    # Run tuning for baseline model
    print("Tuning Baseline Model...")
    baseline_results = run_hyperparameter_tuning(
        model_type='baseline',
        search_type='random',
        n_iter=20
    )
    
    # Run tuning for recurrent model
    print("\nTuning Recurrent Model...")
    recurrent_results = run_hyperparameter_tuning(
        model_type='recurrent',
        search_type='random',
        n_iter=20
    )
