"""
Label mappings for classification tasks.
Converts model predictions (0, 1, 2, etc.) to human-readable labels.
"""

# Binary sentiment classification (SST-2, Yelp)
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive"
}

# Multi-domain classification (movie reviews, shopping, business)
DOMAIN_LABELS = {
    0: "movie_review",
    1: "online_shopping",
    2: "local_business_review"
}

# Reverse mappings for convenience
SENTIMENT_TO_ID = {v: k for k, v in SENTIMENT_LABELS.items()}
DOMAIN_TO_ID = {v: k for k, v in DOMAIN_LABELS.items()}


def get_sentiment_label(prediction_id: int) -> str:
    """
    Convert sentiment prediction ID to human-readable label.
    
    Args:
        prediction_id: Model prediction (0 or 1)
        
    Returns:
        Human-readable sentiment label
    """
    return SENTIMENT_LABELS.get(prediction_id, f"Unknown (ID: {prediction_id})")


def get_domain_label(prediction_id: int) -> str:
    """
    Convert domain prediction ID to human-readable label.
    
    Args:
        prediction_id: Model prediction (0, 1, or 2)
        
    Returns:
        Human-readable domain label
    """
    return DOMAIN_LABELS.get(prediction_id, f"Unknown (ID: {prediction_id})")


def get_sentiment_description(prediction_id: int) -> str:
    """
    Get a detailed description of sentiment prediction.
    
    Args:
        prediction_id: Model prediction (0 or 1)
        
    Returns:
        Detailed description
    """
    descriptions = {
        0: "Negative - The text expresses a negative sentiment or opinion",
        1: "Positive - The text expresses a positive sentiment or opinion"
    }
    return descriptions.get(prediction_id, f"Unknown sentiment ID: {prediction_id}")


def get_domain_description(prediction_id: int) -> str:
    """
    Get a detailed description of domain prediction.
    
    Args:
        prediction_id: Model prediction (0, 1, or 2)
        
    Returns:
        Detailed description
    """
    descriptions = {
        0: "Movie Review - Review or commentary about films, TV shows, or entertainment content",
        1: "Online Shopping - Review or feedback about products purchased online",
        2: "Local Business Review - Review about local services, restaurants, or physical establishments"
    }
    return descriptions.get(prediction_id, f"Unknown domain ID: {prediction_id}")

