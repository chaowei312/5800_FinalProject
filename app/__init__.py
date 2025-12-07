"""
Sentiment and Domain Classification App

This package provides inference capabilities for:
1. Binary sentiment classification (positive/negative)
2. Multi-domain classification (movie review, online shopping, local business review)

Models supported:
- Baseline Transformer (standard transformer with enhancements)
- Recurrent Transformer (iterative refinement architecture)
"""

__version__ = "1.0.0"

from .inference import SentimentClassifier, MultiDomainClassifier
from .label_mappings import SENTIMENT_LABELS, DOMAIN_LABELS

__all__ = [
    'SentimentClassifier',
    'MultiDomainClassifier',
    'SENTIMENT_LABELS',
    'DOMAIN_LABELS'
]

