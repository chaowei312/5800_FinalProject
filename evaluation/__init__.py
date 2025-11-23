"""
Evaluation module for model performance metrics.
"""

from .eval import (
    ModelEvaluator,
    compute_classification_metrics,
    compute_generation_metrics,
    compute_model_metrics,
    evaluate_models_comparison
)

__all__ = [
    'ModelEvaluator',
    'compute_classification_metrics',
    'compute_generation_metrics',
    'compute_model_metrics',
    'evaluate_models_comparison'
]
