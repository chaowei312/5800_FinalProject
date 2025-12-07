"""
Demo script showcasing the classification app capabilities.

This script demonstrates:
1. Loading models
2. Making predictions
3. Comparing baseline vs recurrent models
4. Batch processing examples
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.inference import SentimentClassifier, MultiDomainClassifier, UnifiedClassifier


def demo_sentiment_classification():
    """Demonstrate sentiment classification."""
    print("\n" + "=" * 80)
    print("DEMO 1: Sentiment Classification")
    print("=" * 80)
    
    # Example texts
    texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible experience. Would not recommend to anyone.",
        "It was okay, nothing special but not bad either.",
        "Best purchase I've ever made! Highly recommended!",
        "Disappointing quality for the price."
    ]
    
    # Load both baseline and recurrent models
    print("\n📦 Loading Baseline Model...")
    baseline_classifier = SentimentClassifier.from_checkpoint(
        'configs/Baseline_best_model.pt',
        model_type='baseline'
    )
    
    print("\n📦 Loading Recurrent Model...")
    recurrent_classifier = SentimentClassifier.from_checkpoint(
        'configs/Recurrent_best_model.pt',
        model_type='recurrent'
    )
    
    print("\n" + "-" * 80)
    print("Comparing Baseline vs Recurrent Models")
    print("-" * 80)
    
    for i, text in enumerate(texts, 1):
        print(f"\n📝 Text {i}: {text}")
        
        # Baseline prediction
        baseline_result = baseline_classifier.classify(text, return_probs=True)
        print(f"\n   🔹 Baseline Model:")
        print(f"      Prediction: {baseline_result['label']}")
        print(f"      Confidence: {baseline_result['confidence']:.2%}")
        print(f"      Probs: Neg={baseline_result['probabilities']['Negative']:.2%}, "
              f"Pos={baseline_result['probabilities']['Positive']:.2%}")
        
        # Recurrent prediction
        recurrent_result = recurrent_classifier.classify(text, return_probs=True)
        print(f"\n   🔸 Recurrent Model:")
        print(f"      Prediction: {recurrent_result['label']}")
        print(f"      Confidence: {recurrent_result['confidence']:.2%}")
        print(f"      Probs: Neg={recurrent_result['probabilities']['Negative']:.2%}, "
              f"Pos={recurrent_result['probabilities']['Positive']:.2%}")


def demo_domain_classification():
    """Demonstrate multi-domain classification."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-Domain Classification")
    print("=" * 80)
    
    # Example texts from different domains
    texts = [
        "The cinematography was stunning and the plot kept me engaged throughout the entire film.",
        "Fast shipping and the product matches the description perfectly. Very satisfied!",
        "The service at this restaurant was excellent and the food was delicious.",
        "This TV series has amazing character development and unexpected plot twists.",
        "Great deals and easy checkout process. Will shop here again!",
        "Friendly staff and clean environment. Highly recommend this local business."
    ]
    
    # Load domain classifiers
    print("\n📦 Loading Baseline Model...")
    baseline_classifier = MultiDomainClassifier.from_checkpoint(
        'configs/Baseline_best_model_multi.pt',
        model_type='baseline'
    )
    
    print("\n📦 Loading Recurrent Model...")
    recurrent_classifier = MultiDomainClassifier.from_checkpoint(
        'configs/Recurrent_best_model_multi.pt',
        model_type='recurrent'
    )
    
    print("\n" + "-" * 80)
    print("Domain Classification Results")
    print("-" * 80)
    
    for i, text in enumerate(texts, 1):
        print(f"\n📝 Text {i}: {text}")
        
        # Baseline prediction
        baseline_result = baseline_classifier.classify(text, return_probs=True)
        print(f"\n   🔹 Baseline: {baseline_result['label']} ({baseline_result['confidence']:.2%})")
        
        # Recurrent prediction
        recurrent_result = recurrent_classifier.classify(text, return_probs=True)
        print(f"   🔸 Recurrent: {recurrent_result['label']} ({recurrent_result['confidence']:.2%})")
        
        # Show probabilities
        print(f"\n   Probability Distribution (Baseline):")
        for label, prob in baseline_result['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f"      {label:25} {bar} {prob:.2%}")


def demo_unified_analysis():
    """Demonstrate unified classification (sentiment + domain)."""
    print("\n" + "=" * 80)
    print("DEMO 3: Unified Analysis (Sentiment + Domain)")
    print("=" * 80)
    
    # Example texts
    texts = [
        "This movie was a masterpiece! Everything from acting to direction was perfect.",
        "Terrible online shopping experience. The product arrived damaged.",
        "Average restaurant with okay food and mediocre service.",
        "Best TV show I've watched in years! Absolutely loved it!",
        "Poor quality item, not worth the money. Very disappointed."
    ]
    
    print("\n📦 Loading Unified Classifier...")
    classifier = UnifiedClassifier(
        sentiment_checkpoint='configs/Baseline_best_model.pt',
        domain_checkpoint='configs/Baseline_best_model_multi.pt',
        sentiment_model_type='baseline',
        domain_model_type='baseline'
    )
    
    print("\n" + "-" * 80)
    print("Comprehensive Analysis Results")
    print("-" * 80)
    
    for i, text in enumerate(texts, 1):
        print(f"\n📝 Text {i}:")
        print(f"   {text}")
        
        result = classifier.analyze(text, return_probs=False)
        
        print(f"\n   💭 Sentiment: {result['sentiment']['label']} "
              f"({result['sentiment']['confidence']:.2%})")
        print(f"   🏷️  Domain: {result['domain']['label']} "
              f"({result['domain']['confidence']:.2%})")


def demo_batch_processing():
    """Demonstrate batch processing."""
    print("\n" + "=" * 80)
    print("DEMO 4: Batch Processing")
    print("=" * 80)
    
    texts = [
        "Great product, highly recommend!",
        "Worst movie ever made.",
        "The restaurant had excellent ambiance.",
        "Fast delivery and good packaging.",
        "The plot was confusing and poorly executed.",
        "Friendly customer service at this store.",
        "Amazing special effects and soundtrack!",
        "Overpriced for what you get.",
        "Cozy atmosphere and delicious food.",
        "The acting was phenomenal!"
    ]
    
    print(f"\n📦 Loading classifier...")
    classifier = UnifiedClassifier(
        sentiment_checkpoint='configs/Baseline_best_model.pt',
        domain_checkpoint='configs/Baseline_best_model_multi.pt'
    )
    
    print(f"\n🔄 Processing {len(texts)} texts...")
    results = classifier.analyze_batch(texts)
    
    # Summary statistics
    sentiment_counts = {'Positive': 0, 'Negative': 0}
    domain_counts = {'movie_review': 0, 'online_shopping': 0, 'local_business_review': 0}
    
    for result in results:
        sentiment_counts[result['sentiment']['label']] += 1
        domain_counts[result['domain']['label']] += 1
    
    print("\n" + "-" * 80)
    print("Batch Processing Summary")
    print("-" * 80)
    print(f"\n📊 Sentiment Distribution:")
    for label, count in sentiment_counts.items():
        percentage = (count / len(texts)) * 100
        bar = '█' * int(percentage / 2)
        print(f"   {label:12} {bar} {count}/{len(texts)} ({percentage:.1f}%)")
    
    print(f"\n📊 Domain Distribution:")
    for label, count in domain_counts.items():
        percentage = (count / len(texts)) * 100
        bar = '█' * int(percentage / 2)
        print(f"   {label:25} {bar} {count}/{len(texts)} ({percentage:.1f}%)")
    
    print("\n✅ Batch processing completed!")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "CLASSIFICATION APP DEMO")
    print("=" * 80)
    print("\nThis demo showcases the capabilities of the sentiment and domain")
    print("classification app using both baseline and recurrent transformer models.")
    
    try:
        # Check if model files exist
        required_files = [
            'configs/Baseline_best_model.pt',
            'configs/Recurrent_best_model.pt',
            'configs/Baseline_best_model_multi.pt',
            'configs/Recurrent_best_model_multi.pt'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("\n⚠️  Warning: Some model files are missing:")
            for f in missing_files:
                print(f"   - {f}")
            print("\nSome demos may not run. Please train the models first.")
            return
        
        # Run demos
        demos = [
            ("Sentiment Classification", demo_sentiment_classification),
            ("Domain Classification", demo_domain_classification),
            ("Unified Analysis", demo_unified_analysis),
            ("Batch Processing", demo_batch_processing)
        ]
        
        for name, demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ Error in {name} demo: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("🎉 Demo completed!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Try the interactive mode: python -m app.cli --interactive")
        print("  2. Classify your own text: python -m app.cli --text 'Your text here'")
        print("  3. Process a batch file: python -m app.cli --input-file texts.txt --output-file results.json")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

