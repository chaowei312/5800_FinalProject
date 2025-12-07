"""
Inference module for sentiment and multi-domain classification.
Provides easy-to-use classes for making predictions with trained models.
"""

import torch
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoTokenizer
import numpy as np

from .model_loader import load_baseline_model, load_recurrent_model
from .label_mappings import (
    get_sentiment_label, 
    get_domain_label,
    get_sentiment_description,
    get_domain_description
)


class BaseClassifier:
    """Base class for classifiers with common functionality."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cpu',
        max_length: int = 256
    ):
        """
        Initialize base classifier.
        
        Args:
            model: Trained PyTorch model
            tokenizer: HuggingFace tokenizer
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        self.model.to(device)
        self.model.eval()
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of tokenized inputs
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def predict_proba(self, text: str) -> np.ndarray:
        """
        Get class probabilities for input text.
        
        Args:
            text: Input text string
            
        Returns:
            Array of class probabilities
        """
        inputs = self.preprocess(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()[0]
    
    def predict(self, text: str) -> int:
        """
        Predict class for input text.
        
        Args:
            text: Input text string
            
        Returns:
            Predicted class ID
        """
        probs = self.predict_proba(text)
        return int(np.argmax(probs))
    
    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Predict classes for a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of predicted class IDs
        """
        predictions = []
        for text in texts:
            predictions.append(self.predict(text))
        return predictions


class SentimentClassifier(BaseClassifier):
    """
    Sentiment classifier for binary classification (positive/negative).
    
    Example:
        >>> classifier = SentimentClassifier.from_checkpoint(
        ...     'configs/Baseline_best_model.pt',
        ...     model_type='baseline'
        ... )
        >>> result = classifier.classify("This movie was amazing!")
        >>> print(result['label'])  # "Positive"
        >>> print(result['confidence'])  # 0.95
    """
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str = 'baseline',
        tokenizer_name: str = 'bert-base-uncased',
        device: Optional[str] = None,
        max_length: int = 256,
        # Model configuration parameters
        hidden_size: int = None,
        num_hidden_layers: int = None,
        num_attention_heads: int = None,
        intermediate_size: int = None
    ):
        """
        Create classifier from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            model_type: Type of model ('baseline' or 'recurrent')
            tokenizer_name: Name of tokenizer to use
            device: Device to run on (defaults to 'cuda' if available else 'cpu')
            max_length: Maximum sequence length
            hidden_size: Model hidden size (inferred if None)
            num_hidden_layers: Number of layers (inferred if None)
            num_attention_heads: Number of attention heads (inferred if None)
            intermediate_size: Intermediate size (inferred if None)
            
        Returns:
            SentimentClassifier instance
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model based on type
        if model_type.lower() == 'baseline':
            model = load_baseline_model(
                checkpoint_path, 
                num_labels=2, 
                device=device,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            )
        elif model_type.lower() == 'recurrent':
            model = load_recurrent_model(
                checkpoint_path, 
                num_labels=2, 
                device=device,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'recurrent'")
        
        return cls(model, tokenizer, device, max_length)
    
    def classify(self, text: str, return_probs: bool = False) -> Dict:
        """
        Classify sentiment of input text.
        
        Args:
            text: Input text to classify
            return_probs: Whether to return probabilities for all classes
            
        Returns:
            Dictionary containing:
                - label: Human-readable sentiment label
                - label_id: Numeric prediction (0 or 1)
                - confidence: Confidence score for prediction
                - description: Detailed description of prediction
                - probabilities: (optional) All class probabilities
        """
        probs = self.predict_proba(text)
        prediction_id = int(np.argmax(probs))
        confidence = float(probs[prediction_id])
        
        result = {
            'label': get_sentiment_label(prediction_id),
            'label_id': prediction_id,
            'confidence': confidence,
            'description': get_sentiment_description(prediction_id)
        }
        
        if return_probs:
            result['probabilities'] = {
                'Negative': float(probs[0]),
                'Positive': float(probs[1])
            }
        
        return result


class MultiDomainClassifier(BaseClassifier):
    """
    Multi-domain classifier for 3-class classification.
    Classifies text into: movie_review, online_shopping, or local_business_review.
    
    Example:
        >>> classifier = MultiDomainClassifier.from_checkpoint(
        ...     'configs/Baseline_best_model_multi.pt',
        ...     model_type='baseline'
        ... )
        >>> result = classifier.classify("The food at this restaurant was excellent!")
        >>> print(result['label'])  # "local_business_review"
        >>> print(result['confidence'])  # 0.89
    """
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str = 'baseline',
        tokenizer_name: str = 'bert-base-uncased',
        device: Optional[str] = None,
        max_length: int = 256,
        # Model configuration parameters
        hidden_size: int = None,
        num_hidden_layers: int = None,
        num_attention_heads: int = None,
        intermediate_size: int = None,
        recurrent_depth: int = None
    ):
        """
        Create classifier from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            model_type: Type of model ('baseline' or 'recurrent')
            tokenizer_name: Name of tokenizer to use
            device: Device to run on (defaults to 'cuda' if available else 'cpu')
            max_length: Maximum sequence length
            hidden_size: Model hidden size (inferred if None)
            num_hidden_layers: Number of layers (inferred if None)
            num_attention_heads: Number of attention heads (inferred if None)
            intermediate_size: Intermediate size (inferred if None)
            recurrent_depth: Recurrent depth for recurrent models (inferred if None)
            
        Returns:
            MultiDomainClassifier instance
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model based on type
        if model_type.lower() == 'baseline':
            model = load_baseline_model(
                checkpoint_path, 
                num_labels=3, 
                device=device,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            )
        elif model_type.lower() == 'recurrent':
            model = load_recurrent_model(
                checkpoint_path, 
                num_labels=3, 
                device=device,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                recurrent_depth=recurrent_depth
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'recurrent'")
        
        return cls(model, tokenizer, device, max_length)
    
    def classify(self, text: str, return_probs: bool = False) -> Dict:
        """
        Classify domain of input text.
        
        Args:
            text: Input text to classify
            return_probs: Whether to return probabilities for all classes
            
        Returns:
            Dictionary containing:
                - label: Human-readable domain label
                - label_id: Numeric prediction (0, 1, or 2)
                - confidence: Confidence score for prediction
                - description: Detailed description of prediction
                - probabilities: (optional) All class probabilities
        """
        probs = self.predict_proba(text)
        prediction_id = int(np.argmax(probs))
        confidence = float(probs[prediction_id])
        
        result = {
            'label': get_domain_label(prediction_id),
            'label_id': prediction_id,
            'confidence': confidence,
            'description': get_domain_description(prediction_id)
        }
        
        if return_probs:
            result['probabilities'] = {
                'movie_review': float(probs[0]),
                'online_shopping': float(probs[1]),
                'local_business_review': float(probs[2])
            }
        
        return result


class UnifiedClassifier:
    """
    Unified classifier that combines both sentiment and domain classification.
    Provides comprehensive analysis of input text.
    
    Example:
        >>> classifier = UnifiedClassifier(
        ...     sentiment_checkpoint='configs/Baseline_best_model.pt',
        ...     domain_checkpoint='configs/Baseline_best_model_multi.pt',
        ...     sentiment_model_type='baseline',
        ...     domain_model_type='recurrent'
        ... )
        >>> result = classifier.analyze("This movie was incredible!")
        >>> print(result['sentiment']['label'])  # "Positive"
        >>> print(result['domain']['label'])  # "movie_review"
    """
    
    def __init__(
        self,
        sentiment_checkpoint: str,
        domain_checkpoint: str,
        sentiment_model_type: str = 'baseline',
        domain_model_type: str = 'baseline',
        tokenizer_name: str = 'bert-base-uncased',
        device: Optional[str] = None
    ):
        """
        Initialize unified classifier with both sentiment and domain models.
        
        Args:
            sentiment_checkpoint: Path to sentiment model checkpoint
            domain_checkpoint: Path to domain model checkpoint
            sentiment_model_type: Type of sentiment model ('baseline' or 'recurrent')
            domain_model_type: Type of domain model ('baseline' or 'recurrent')
            tokenizer_name: Name of tokenizer to use
            device: Device to run on (defaults to 'cuda' if available else 'cpu')
        """
        self.sentiment_classifier = SentimentClassifier.from_checkpoint(
            sentiment_checkpoint,
            model_type=sentiment_model_type,
            tokenizer_name=tokenizer_name,
            device=device
        )
        
        self.domain_classifier = MultiDomainClassifier.from_checkpoint(
            domain_checkpoint,
            model_type=domain_model_type,
            tokenizer_name=tokenizer_name,
            device=device
        )
    
    def analyze(self, text: str, return_probs: bool = False) -> Dict:
        """
        Perform comprehensive analysis of text including sentiment and domain.
        
        Args:
            text: Input text to analyze
            return_probs: Whether to return probabilities for all classes
            
        Returns:
            Dictionary containing both sentiment and domain classifications
        """
        return {
            'text': text,
            'sentiment': self.sentiment_classifier.classify(text, return_probs),
            'domain': self.domain_classifier.classify(text, return_probs)
        }
    
    def analyze_batch(self, texts: List[str], return_probs: bool = False) -> List[Dict]:
        """
        Analyze a batch of texts.
        
        Args:
            texts: List of input texts
            return_probs: Whether to return probabilities for all classes
            
        Returns:
            List of analysis dictionaries
        """
        return [self.analyze(text, return_probs) for text in texts]

