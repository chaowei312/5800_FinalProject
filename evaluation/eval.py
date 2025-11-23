"""
Comprehensive evaluation module for model performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef
)
from collections import defaultdict
import time
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive evaluator for transformer models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        num_labels: int = 2
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_confusion_matrix: bool = True,
        compute_per_class_metrics: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model.
        
        Args:
            dataloader: Data loader for evaluation
            compute_confusion_matrix: Whether to compute confusion matrix
            compute_per_class_metrics: Whether to compute per-class metrics
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing all metrics
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        total_samples = 0
        inference_times = []
        
        # Disable gradient computation
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
            
            for batch in iterator:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].size(0)
                
                # Time inference
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids'),
                    labels=batch['labels']
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time / batch_size)
                
                # Get loss
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item() * batch_size
                    total_samples += batch_size
                
                # Get predictions
                logits = outputs['logits']
                probs = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = compute_classification_metrics(
            all_labels,
            all_predictions,
            all_probs,
            compute_confusion_matrix,
            compute_per_class_metrics
        )
        
        # Add loss and perplexity
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics['loss'] = avg_loss
            metrics['perplexity'] = np.exp(avg_loss)
        
        # Add inference statistics
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['std_inference_time'] = np.std(inference_times)
        metrics['samples_per_second'] = 1.0 / np.mean(inference_times) if inference_times else 0
        
        # Add model statistics
        metrics.update(compute_model_metrics(self.model))
        
        return metrics
    
    def evaluate_generation(
        self,
        dataloader: DataLoader,
        tokenizer,
        max_length: int = 128,
        num_beams: int = 4,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate generation capabilities (for models that support generation).
        
        Args:
            dataloader: Data loader
            tokenizer: Tokenizer for decoding
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            verbose: Whether to show progress
            
        Returns:
            Dictionary of generation metrics
        """
        # This would be implemented for generation tasks
        # Placeholder for now
        return {
            'bleu': 0.0,
            'rouge_1': 0.0,
            'rouge_2': 0.0,
            'rouge_l': 0.0,
            'meteor': 0.0
        }


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    compute_confusion_matrix: bool = True,
    compute_per_class_metrics: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        probabilities: Prediction probabilities
        compute_confusion_matrix: Whether to compute confusion matrix
        compute_per_class_metrics: Whether to compute per-class metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Macro averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(labels, predictions)
    
    # AUC-ROC if probabilities available
    if probabilities is not None and len(np.unique(labels)) == 2:
        # Binary classification
        metrics['auc_roc'] = roc_auc_score(labels, probabilities[:, 1])
    elif probabilities is not None:
        # Multi-class
        try:
            metrics['auc_roc'] = roc_auc_score(
                labels, probabilities, multi_class='ovr', average='weighted'
            )
        except:
            metrics['auc_roc'] = None
    
    # Confusion Matrix
    if compute_confusion_matrix:
        metrics['confusion_matrix'] = confusion_matrix(labels, predictions).tolist()
    
    # Per-class metrics
    if compute_per_class_metrics:
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, predictions, average=None)
        
        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        }
    
    # Classification report
    metrics['classification_report'] = classification_report(
        labels, predictions, output_dict=True
    )
    
    return metrics


def compute_generation_metrics(
    references: List[str],
    predictions: List[str]
) -> Dict[str, float]:
    """
    Compute text generation metrics.
    
    Args:
        references: Reference texts
        predictions: Generated texts
        
    Returns:
        Dictionary of generation metrics
    """
    # Note: For actual implementation, you would use libraries like:
    # - sacrebleu for BLEU
    # - rouge-score for ROUGE
    # - nltk for METEOR
    
    metrics = {}
    
    # Placeholder implementations
    # In practice, use proper libraries
    metrics['bleu'] = 0.0  # Would compute actual BLEU score
    metrics['rouge_1'] = 0.0  # ROUGE-1 F1
    metrics['rouge_2'] = 0.0  # ROUGE-2 F1
    metrics['rouge_l'] = 0.0  # ROUGE-L F1
    metrics['meteor'] = 0.0  # METEOR score
    
    return metrics


def compute_model_metrics(model: nn.Module) -> Dict[str, Any]:
    """
    Compute model-specific metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of model metrics
    """
    metrics = {}
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics['total_parameters'] = total_params
    metrics['trainable_parameters'] = trainable_params
    metrics['non_trainable_parameters'] = total_params - trainable_params
    
    # Model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    metrics['model_size_mb'] = (param_size + buffer_size) / 1024 / 1024
    
    # Memory footprint estimation (approximate)
    metrics['estimated_memory_mb'] = metrics['model_size_mb'] * 2  # Rough estimate
    
    return metrics


def evaluate_models_comparison(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    save_path: Optional[str] = None,
    plot_results: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of model_name -> model
        dataloader: Evaluation data loader
        save_path: Optional path to save results
        plot_results: Whether to plot comparison
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models.items():
        print(f"\\nEvaluating {model_name}...")
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(dataloader, verbose=True)
        
        # Add model name
        metrics['model'] = model_name
        results.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Select key metrics for display
    key_metrics = [
        'model', 'accuracy', 'f1', 'precision', 'recall',
        'loss', 'perplexity', 'avg_inference_time',
        'total_parameters', 'model_size_mb'
    ]
    
    display_df = df[key_metrics].round(4)
    
    # Save results
    if save_path:
        df.to_csv(f"{save_path}/model_comparison.csv", index=False)
        with open(f"{save_path}/model_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\nResults saved to {save_path}")
    
    # Plot comparison
    if plot_results:
        plot_model_comparison(df, save_path)
    
    return display_df


def plot_model_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot model comparison charts.
    
    Args:
        df: DataFrame with model metrics
        save_path: Optional path to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    axes[0, 0].bar(df['model'], df['accuracy'])
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(df['accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # F1 Score comparison
    axes[0, 1].bar(df['model'], df['f1'], color='orange')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(df['f1']):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Loss comparison
    axes[0, 2].bar(df['model'], df['loss'], color='red')
    axes[0, 2].set_title('Loss')
    axes[0, 2].set_ylabel('Loss')
    for i, v in enumerate(df['loss']):
        axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Inference time comparison
    axes[1, 0].bar(df['model'], df['avg_inference_time'] * 1000, color='green')
    axes[1, 0].set_title('Avg Inference Time')
    axes[1, 0].set_ylabel('Time (ms)')
    for i, v in enumerate(df['avg_inference_time'] * 1000):
        axes[1, 0].text(i, v + 0.1, f'{v:.1f}', ha='center')
    
    # Model size comparison
    axes[1, 1].bar(df['model'], df['model_size_mb'], color='purple')
    axes[1, 1].set_title('Model Size')
    axes[1, 1].set_ylabel('Size (MB)')
    for i, v in enumerate(df['model_size_mb']):
        axes[1, 1].text(i, v + 0.1, f'{v:.1f}', ha='center')
    
    # Parameters comparison
    axes[1, 2].bar(df['model'], df['total_parameters'] / 1e6, color='brown')
    axes[1, 2].set_title('Total Parameters')
    axes[1, 2].set_ylabel('Parameters (M)')
    for i, v in enumerate(df['total_parameters'] / 1e6):
        axes[1, 2].text(i, v + 0.1, f'{v:.1f}M', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}/model_comparison.png")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: Optional class names
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Example usage of evaluation module."""
    # This would be called from other scripts or notebooks
    pass


if __name__ == '__main__':
    main()
