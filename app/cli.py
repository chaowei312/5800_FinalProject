"""
Command-line interface for sentiment and domain classification.

Usage examples:

1. Sentiment classification only:
   python -m app.cli --text "This movie was amazing!" --task sentiment --model baseline

2. Domain classification only:
   python -m app.cli --text "Great restaurant!" --task domain --model recurrent

3. Full analysis (both sentiment and domain):
   python -m app.cli --text "This product is excellent!" --task both

4. Interactive mode:
   python -m app.cli --interactive

5. Batch processing from file:
   python -m app.cli --input-file texts.txt --output-file results.json
"""

import argparse
import json
import sys
import os
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.inference import SentimentClassifier, MultiDomainClassifier, UnifiedClassifier


def print_separator():
    """Print a visual separator."""
    print("=" * 80)


def print_result(result: Dict, task: str = 'sentiment'):
    """Pretty print classification results."""
    print_separator()
    if 'text' in result:
        print(f"📝 Input: {result['text']}")
        print()
    
    if task in ['sentiment', 'both'] and 'sentiment' in result:
        sentiment = result.get('sentiment', result)
        print(f"💭 Sentiment Classification:")
        print(f"   Label: {sentiment['label']}")
        print(f"   Confidence: {sentiment['confidence']:.2%}")
        print(f"   Description: {sentiment['description']}")
        
        if 'probabilities' in sentiment:
            print(f"   Probabilities:")
            for label, prob in sentiment['probabilities'].items():
                print(f"      {label}: {prob:.2%}")
        print()
    
    if task in ['domain', 'both'] and 'domain' in result:
        domain = result.get('domain', result)
        print(f"🏷️  Domain Classification:")
        print(f"   Label: {domain['label']}")
        print(f"   Confidence: {domain['confidence']:.2%}")
        print(f"   Description: {domain['description']}")
        
        if 'probabilities' in domain:
            print(f"   Probabilities:")
            for label, prob in domain['probabilities'].items():
                print(f"      {label}: {prob:.2%}")
        print()
    
    print_separator()


def interactive_mode(
    sentiment_classifier=None,
    domain_classifier=None,
    unified_classifier=None,
    task: str = 'both',
    show_probs: bool = False
):
    """Run in interactive mode for continuous predictions."""
    print("\n🚀 Interactive Classification Mode")
    print("Type 'quit' or 'exit' to stop, 'help' for commands\n")
    
    while True:
        try:
            text = input("📝 Enter text to classify: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if text.lower() == 'help':
                print("\nCommands:")
                print("  - Type any text to classify it")
                print("  - 'quit' or 'exit' to stop")
                print("  - 'help' to show this message\n")
                continue
            
            if not text:
                print("⚠️  Please enter some text\n")
                continue
            
            # Perform classification
            if task == 'both' and unified_classifier:
                result = unified_classifier.analyze(text, return_probs=show_probs)
            elif task == 'sentiment' and sentiment_classifier:
                result = sentiment_classifier.classify(text, return_probs=show_probs)
            elif task == 'domain' and domain_classifier:
                result = domain_classifier.classify(text, return_probs=show_probs)
            else:
                result = unified_classifier.analyze(text, return_probs=show_probs)
            
            print_result(result, task)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def batch_process(
    input_file: str,
    output_file: str,
    sentiment_classifier=None,
    domain_classifier=None,
    unified_classifier=None,
    task: str = 'both',
    show_probs: bool = False
):
    """Process multiple texts from a file."""
    # Read input texts
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"📂 Processing {len(texts)} texts from {input_file}...")
    
    results = []
    for i, text in enumerate(texts, 1):
        print(f"   Processing {i}/{len(texts)}...", end='\r')
        
        if task == 'both' and unified_classifier:
            result = unified_classifier.analyze(text, return_probs=show_probs)
        elif task == 'sentiment' and sentiment_classifier:
            result = {'text': text, 'sentiment': sentiment_classifier.classify(text, return_probs=show_probs)}
        elif task == 'domain' and domain_classifier:
            result = {'text': text, 'domain': domain_classifier.classify(text, return_probs=show_probs)}
        else:
            result = unified_classifier.analyze(text, return_probs=show_probs)
        
        results.append(result)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Text Classification CLI - Sentiment and Domain Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configuration
    parser.add_argument(
        '--sentiment-checkpoint',
        type=str,
        default='configs/Baseline_best_model.pt',
        help='Path to sentiment classification model checkpoint'
    )
    parser.add_argument(
        '--domain-checkpoint',
        type=str,
        default='configs/Baseline_best_model_multi.pt',
        help='Path to domain classification model checkpoint'
    )
    parser.add_argument(
        '--sentiment-model',
        type=str,
        choices=['baseline', 'recurrent'],
        default='baseline',
        help='Type of sentiment model to use'
    )
    parser.add_argument(
        '--domain-model',
        type=str,
        choices=['baseline', 'recurrent'],
        default='baseline',
        help='Type of domain model to use'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bert-base-uncased',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to run on (defaults to cuda if available)'
    )
    
    # Task configuration
    parser.add_argument(
        '--task',
        type=str,
        choices=['sentiment', 'domain', 'both'],
        default='both',
        help='Classification task to perform'
    )
    parser.add_argument(
        '--show-probs',
        action='store_true',
        help='Show probabilities for all classes'
    )
    
    # Input modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--text',
        type=str,
        help='Single text to classify'
    )
    group.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    group.add_argument(
        '--input-file',
        type=str,
        help='File containing texts to classify (one per line)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file for batch processing results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Initialize classifiers based on task
    print("🔧 Loading models...")
    
    sentiment_classifier = None
    domain_classifier = None
    unified_classifier = None
    
    try:
        if args.task in ['sentiment', 'both']:
            if args.task == 'sentiment':
                print(f"   Loading sentiment model ({args.sentiment_model})...")
                sentiment_classifier = SentimentClassifier.from_checkpoint(
                    args.sentiment_checkpoint,
                    model_type=args.sentiment_model,
                    tokenizer_name=args.tokenizer,
                    device=args.device
                )
        
        if args.task in ['domain', 'both']:
            if args.task == 'domain':
                print(f"   Loading domain model ({args.domain_model})...")
                domain_classifier = MultiDomainClassifier.from_checkpoint(
                    args.domain_checkpoint,
                    model_type=args.domain_model,
                    tokenizer_name=args.tokenizer,
                    device=args.device
                )
        
        if args.task == 'both':
            print(f"   Loading unified classifier...")
            print(f"      Sentiment: {args.sentiment_model}")
            print(f"      Domain: {args.domain_model}")
            unified_classifier = UnifiedClassifier(
                sentiment_checkpoint=args.sentiment_checkpoint,
                domain_checkpoint=args.domain_checkpoint,
                sentiment_model_type=args.sentiment_model,
                domain_model_type=args.domain_model,
                tokenizer_name=args.tokenizer,
                device=args.device
            )
        
        print("✅ Models loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return 1
    
    # Execute based on mode
    if args.text:
        # Single text classification
        if args.task == 'both' and unified_classifier:
            result = unified_classifier.analyze(args.text, return_probs=args.show_probs)
        elif args.task == 'sentiment' and sentiment_classifier:
            result = sentiment_classifier.classify(args.text, return_probs=args.show_probs)
        elif args.task == 'domain' and domain_classifier:
            result = domain_classifier.classify(args.text, return_probs=args.show_probs)
        
        print_result(result, args.task)
        
    elif args.interactive:
        # Interactive mode
        interactive_mode(
            sentiment_classifier=sentiment_classifier,
            domain_classifier=domain_classifier,
            unified_classifier=unified_classifier,
            task=args.task,
            show_probs=args.show_probs
        )
        
    elif args.input_file:
        # Batch processing
        if not args.output_file:
            print("❌ Error: --output-file is required for batch processing")
            return 1
        
        batch_process(
            input_file=args.input_file,
            output_file=args.output_file,
            sentiment_classifier=sentiment_classifier,
            domain_classifier=domain_classifier,
            unified_classifier=unified_classifier,
            task=args.task,
            show_probs=args.show_probs
        )
        
    else:
        # No input specified, show help
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

