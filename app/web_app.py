"""
Flask web application for text classification.

This provides a web interface for the classification app with:
- Interactive web form for text input
- Model selection (baseline vs recurrent)
- Real-time classification results
- Beautiful, modern UI
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import traceback

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

from app.inference import SentimentClassifier, MultiDomainClassifier

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'text-classification-app-secret-key'

# Global model storage
models = {
    'sentiment_baseline': None,
    'sentiment_recurrent': None,
    'domain_baseline': None,
    'domain_recurrent': None
}

# Model configurations based on your training
BASELINE_CONFIG = {
    'hidden_size': 384,
    'num_hidden_layers': 6,
    'num_attention_heads': 6,
    'intermediate_size': 1536,
}

RECURRENT_CONFIG = {
    'hidden_size': 256,
    'num_hidden_layers': 3,
    # 'recurrent_depth': 2,  # Effective depth: 3 × 2 = 6
    'num_attention_heads': 4,
    'intermediate_size': 1024,
}


def load_models():
    """Load all models on startup."""
    print("\n" + "="*70)
    print("Loading models...")
    print("="*70)
    
    try:
        # Check if model files exist
        model_files = {
            'sentiment_baseline': '../configs/Baseline_best_model.pt',
            'sentiment_recurrent': '../configs/Recurrent_best_model.pt',
            'domain_baseline': '../configs/Baseline_best_model_multi.pt',
            'domain_recurrent': '../configs/Recurrent_best_model_multi.pt'
        }
        
        print("\nChecking model files...")
        for key, path in model_files.items():
            exists = os.path.exists(path)
            status = "✓" if exists else "✗"
            print(f"  {status} {key}: {path}")
            if not exists:
                print(f"     ERROR: File not found!")
        
        # Load sentiment models
        print("\nLoading sentiment baseline model...")
        models['sentiment_baseline'] = SentimentClassifier.from_checkpoint(
            '../configs/Baseline_best_model.pt',
            model_type='baseline',
            **BASELINE_CONFIG
        )
        print("✓ Loaded sentiment baseline model")
        
        print("\nLoading sentiment recurrent model...")
        models['sentiment_recurrent'] = SentimentClassifier.from_checkpoint(
            '../configs/Recurrent_best_model.pt',
            model_type='recurrent',
            **RECURRENT_CONFIG
        )
        print("✓ Loaded sentiment recurrent model")
        
        # Load domain models
        print("\nLoading domain baseline model...")
        models['domain_baseline'] = MultiDomainClassifier.from_checkpoint(
            '../configs/Baseline_best_model_multi.pt',
            model_type='baseline',
            **BASELINE_CONFIG
        )
        print("✓ Loaded domain baseline model")
        
        print("\nLoading domain recurrent model...")
        models['domain_recurrent'] = MultiDomainClassifier.from_checkpoint(
            '../configs/Recurrent_best_model_multi.pt',
            model_type='recurrent',
            **RECURRENT_CONFIG
        )
        print("✓ Loaded domain recurrent model")
        
        print("\n" + "="*70)
        print("✅ All models loaded successfully!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ Error loading models: {e}")
        print("="*70)
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify input text using selected models.
    
    Expects JSON with:
    - text: Input text to classify
    - model_type: 'baseline' or 'recurrent'
    
    Returns JSON with:
    - sentiment: Sentiment classification result
    - domain: Domain classification result
    - model_type: Which model was used
    """
    print("\n" + "="*70)
    print("Received classification request")
    print("="*70)
    
    try:
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data or 'text' not in data:
            print("ERROR: No text provided")
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        print(f"Text to classify: '{text}'")
        
        if not text:
            print("ERROR: Text is empty")
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        model_type = data.get('model_type', 'baseline').lower()
        print(f"Model type: {model_type}")
        
        if model_type not in ['baseline', 'recurrent']:
            print(f"ERROR: Invalid model type: {model_type}")
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Get the appropriate models
        sentiment_model = models[f'sentiment_{model_type}']
        domain_model = models[f'domain_{model_type}']
        
        if sentiment_model is None or domain_model is None:
            print("ERROR: Models not loaded")
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Classify
        print("\nClassifying sentiment...")
        sentiment_result = sentiment_model.classify(text, return_probs=True)
        print(f"Sentiment: {sentiment_result['label']} ({sentiment_result['confidence']:.2%})")
        
        print("\nClassifying domain...")
        domain_result = domain_model.classify(text, return_probs=True)
        print(f"Domain: {domain_result['label']} ({domain_result['confidence']:.2%})")
        
        # Format response
        response = {
            'text': text,
            'model_type': model_type,
            'sentiment': {
                'label': sentiment_result['label'],
                'confidence': round(sentiment_result['confidence'] * 100, 2),
                'probabilities': {
                    k: round(v * 100, 2) 
                    for k, v in sentiment_result['probabilities'].items()
                }
            },
            'domain': {
                'label': domain_result['label'],
                'confidence': round(domain_result['confidence'] * 100, 2),
                'probabilities': {
                    k: round(v * 100, 2) 
                    for k, v in domain_result['probabilities'].items()
                }
            }
        }
        
        print("\n✅ Classification successful!")
        print("="*70 + "\n")
        return jsonify(response)
        
    except Exception as e:
        print(f"\n❌ ERROR in classify(): {e}")
        print("="*70)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    models_loaded = all(m is not None for m in models.values())
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'degraded',
        'models_loaded': models_loaded,
        'available_models': [k for k, v in models.items() if v is not None]
    })


@app.route('/api/info')
def info():
    """Get API information."""
    return jsonify({
        'app_name': 'Text Classification Web App',
        'version': '1.0.0',
        'models': {
            'sentiment': ['baseline', 'recurrent'],
            'domain': ['baseline', 'recurrent']
        },
        'labels': {
            'sentiment': ['Negative', 'Positive'],
            'domain': ['movie_review', 'online_shopping', 'local_business_review']
        }
    })


def main():
    """Run the Flask app."""
    print("="*70)
    print("TEXT CLASSIFICATION WEB APP")
    print("="*70)
    print()
    
    # Load models
    if not load_models():
        print("❌ Failed to load models. Please check model files.")
        return 1
    
    print()
    print("="*70)
    print("🚀 Starting web server...")
    print("="*70)
    print()
    print("📱 Open your browser and go to:")
    print("   http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("="*70)
    print()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Avoid loading models twice
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

