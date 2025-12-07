"""
Simple usage examples for the Text Classification App.
This file shows the most common use cases.
"""

import sys
import os

# Ensure the parent directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def example_1_basic_sentiment():
    """Example 1: Basic sentiment classification."""
    print("\n" + "="*60)
    print("Example 1: Basic Sentiment Classification")
    print("="*60)
    
    from app import SentimentClassifier
    
    # Load model
    classifier = SentimentClassifier.from_checkpoint(
        'configs/Baseline_best_model.pt',
        model_type='baseline'
    )
    
    # Classify some texts
    texts = [
        "I absolutely loved this movie!",
        "This product is terrible, waste of money.",
        "Pretty good, I'm satisfied with my purchase."
    ]
    
    for text in texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"→ {result['label']} (confidence: {result['confidence']:.1%})")


def example_2_domain_classification():
    """Example 2: Domain classification."""
    print("\n" + "="*60)
    print("Example 2: Domain Classification")
    print("="*60)
    
    from app import MultiDomainClassifier
    
    # Load model
    classifier = MultiDomainClassifier.from_checkpoint(
        'configs/Baseline_best_model_multi.pt',
        model_type='baseline'
    )
    
    # Classify texts from different domains
    texts = [
        "The movie had amazing special effects and great acting.",
        "Fast shipping, item arrived in perfect condition.",
        "This restaurant has the best pasta in town!"
    ]
    
    for text in texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"→ Domain: {result['label']} (confidence: {result['confidence']:.1%})")


def example_3_unified_analysis():
    """Example 3: Complete analysis with both sentiment and domain."""
    print("\n" + "="*60)
    print("Example 3: Unified Analysis (Sentiment + Domain)")
    print("="*60)
    
    from app import UnifiedClassifier
    
    # Load unified classifier
    classifier = UnifiedClassifier(
        sentiment_checkpoint='configs/Baseline_best_model.pt',
        domain_checkpoint='configs/Baseline_best_model_multi.pt'
    )
    
    # Analyze texts
    texts = [
        "This film is a masterpiece! Highly recommended!",
        "Poor quality product, very disappointed.",
        "Amazing food and excellent service at this place."
    ]
    
    for text in texts:
        result = classifier.analyze(text)
        print(f"\nText: {text}")
        print(f"→ Sentiment: {result['sentiment']['label']} ({result['sentiment']['confidence']:.1%})")
        print(f"→ Domain: {result['domain']['label']} ({result['domain']['confidence']:.1%})")


def example_4_with_probabilities():
    """Example 4: Get full probability distributions."""
    print("\n" + "="*60)
    print("Example 4: Get Probability Distributions")
    print("="*60)
    
    from app import SentimentClassifier
    
    classifier = SentimentClassifier.from_checkpoint(
        'configs/Baseline_best_model.pt',
        model_type='baseline'
    )
    
    text = "This movie is okay, not great but not terrible either."
    result = classifier.classify(text, return_probs=True)
    
    print(f"\nText: {text}")
    print(f"\nPrediction: {result['label']} (confidence: {result['confidence']:.1%})")
    print("\nProbability distribution:")
    for label, prob in result['probabilities'].items():
        bar = '█' * int(prob * 40)
        print(f"  {label:10} {bar} {prob:.1%}")


def example_5_compare_models():
    """Example 5: Compare baseline vs recurrent models."""
    print("\n" + "="*60)
    print("Example 5: Compare Baseline vs Recurrent Models")
    print("="*60)
    
    from app import SentimentClassifier
    
    # Load both models
    baseline = SentimentClassifier.from_checkpoint(
        'configs/Baseline_best_model.pt',
        model_type='baseline'
    )
    
    recurrent = SentimentClassifier.from_checkpoint(
        'configs/Recurrent_best_model.pt',
        model_type='recurrent'
    )
    
    text = "This is an incredible achievement in filmmaking!"
    
    print(f"\nText: {text}\n")
    
    baseline_result = baseline.classify(text, return_probs=True)
    print(f"Baseline Model:")
    print(f"  → {baseline_result['label']} (confidence: {baseline_result['confidence']:.1%})")
    print(f"  → Probabilities: {baseline_result['probabilities']}\n")
    
    recurrent_result = recurrent.classify(text, return_probs=True)
    print(f"Recurrent Model:")
    print(f"  → {recurrent_result['label']} (confidence: {recurrent_result['confidence']:.1%})")
    print(f"  → Probabilities: {recurrent_result['probabilities']}")


def example_6_batch_processing():
    """Example 6: Process multiple texts efficiently."""
    print("\n" + "="*60)
    print("Example 6: Batch Processing")
    print("="*60)
    
    from app import UnifiedClassifier
    
    classifier = UnifiedClassifier(
        sentiment_checkpoint='configs/Baseline_best_model.pt',
        domain_checkpoint='configs/Baseline_best_model_multi.pt'
    )
    
    texts = [
        "Great movie!",
        "Terrible service.",
        "Love this product!",
        "The food was amazing!",
        "Disappointing ending."
    ]
    
    print(f"\nProcessing {len(texts)} texts...\n")
    results = classifier.analyze_batch(texts)
    
    for text, result in zip(texts, results):
        print(f"'{text}'")
        print(f"  → {result['sentiment']['label']} | {result['domain']['label']}")


def example_7_custom_text():
    """Example 7: Interactive - classify your own text."""
    print("\n" + "="*60)
    print("Example 7: Try Your Own Text")
    print("="*60)
    
    from app import UnifiedClassifier
    
    classifier = UnifiedClassifier(
        sentiment_checkpoint='configs/Baseline_best_model.pt',
        domain_checkpoint='configs/Baseline_best_model_multi.pt'
    )
    
    print("\nEnter a text to classify (or press Enter to skip):")
    text = input("→ ")
    
    if text.strip():
        result = classifier.analyze(text, return_probs=True)
        
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        print(f"\n💭 Sentiment: {result['sentiment']['label']}")
        print(f"   Confidence: {result['sentiment']['confidence']:.1%}")
        print(f"   {result['sentiment']['description']}")
        
        print(f"\n🏷️  Domain: {result['domain']['label']}")
        print(f"   Confidence: {result['domain']['confidence']:.1%}")
        print(f"   {result['domain']['description']}")
        
        if 'probabilities' in result['sentiment']:
            print(f"\nSentiment Probabilities:")
            for label, prob in result['sentiment']['probabilities'].items():
                print(f"   {label:10} {prob:.1%}")
        
        if 'probabilities' in result['domain']:
            print(f"\nDomain Probabilities:")
            for label, prob in result['domain']['probabilities'].items():
                print(f"   {label:25} {prob:.1%}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("TEXT CLASSIFICATION APP - USAGE EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate how to use the classification app")
    print("in your Python code.")
    
    examples = [
        ("Basic Sentiment Classification", example_1_basic_sentiment),
        ("Domain Classification", example_2_domain_classification),
        ("Unified Analysis", example_3_unified_analysis),
        ("Probability Distributions", example_4_with_probabilities),
        ("Compare Models", example_5_compare_models),
        ("Batch Processing", example_6_batch_processing),
        ("Try Your Own Text", example_7_custom_text),
    ]
    
    try:
        for i, (name, example_func) in enumerate(examples, 1):
            print(f"\n\n{'#'*60}")
            print(f"Running Example {i}/{len(examples)}: {name}")
            print(f"{'#'*60}")
            
            try:
                example_func()
            except FileNotFoundError as e:
                print(f"\n⚠️  Skipping - Model file not found: {e}")
            except Exception as e:
                print(f"\n❌ Error in example: {e}")
            
            if i < len(examples):
                input("\n\nPress Enter to continue to next example...")
        
        print("\n\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        print("\nFor more advanced usage:")
        print("  - See app/README.md for full documentation")
        print("  - Run 'python app/demo.py' for comprehensive demos")
        print("  - Try 'python -m app.cli --interactive' for interactive mode")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")


if __name__ == '__main__':
    main()

