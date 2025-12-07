"""
Installation test script.

This script verifies that the classification app is properly installed
and can load models successfully.
"""

import sys
import os


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers not found: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy not found: {e}")
        return False
    
    try:
        # Add parent directory to path if needed
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import app
        print(f"  ✓ App package {app.__version__}")
    except ImportError as e:
        print(f"  ✗ App package not found: {e}")
        return False
    
    return True


def test_model_files():
    """Test if model checkpoint files exist."""
    print("\nChecking model files...")
    
    model_files = [
        'configs/Baseline_best_model.pt',
        'configs/Recurrent_best_model.pt',
        'configs/Baseline_best_model_multi.pt',
        'configs/Recurrent_best_model_multi.pt'
    ]
    
    all_found = True
    for file_path in model_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {file_path} (not found)")
            all_found = False
    
    if not all_found:
        print("\n  ⚠️  Some model files are missing. You need to train the models first.")
        print("     The app will still work, but you won't be able to make predictions.")
    
    return all_found


def test_app_functionality():
    """Test basic app functionality."""
    print("\nTesting app functionality...")
    
    try:
        from app.label_mappings import get_sentiment_label, get_domain_label
        
        # Test label mappings
        sentiment = get_sentiment_label(1)
        domain = get_domain_label(0)
        
        assert sentiment == "Positive", f"Expected 'Positive', got '{sentiment}'"
        assert domain == "movie_review", f"Expected 'movie_review', got '{domain}'"
        
        print("  ✓ Label mappings work correctly")
        
    except Exception as e:
        print(f"  ✗ Label mappings failed: {e}")
        return False
    
    return True


def test_model_loading():
    """Test if models can be loaded (if checkpoint files exist)."""
    print("\nTesting model loading...")
    
    try:
        from app.model_loader import get_model_info
        
        # Try to load baseline sentiment model
        baseline_path = 'configs/Baseline_best_model.pt'
        if os.path.exists(baseline_path):
            info = get_model_info(baseline_path)
            print(f"  ✓ Can read baseline model checkpoint")
            print(f"    - Epoch: {info.get('epoch', 'N/A')}")
            print(f"    - Best score: {info.get('best_score', 'N/A')}")
        else:
            print(f"  ⚠️  Baseline model not found (skipping)")
        
        # Try to load recurrent multi-domain model
        recurrent_path = 'configs/Recurrent_best_model_multi.pt'
        if os.path.exists(recurrent_path):
            info = get_model_info(recurrent_path)
            print(f"  ✓ Can read recurrent model checkpoint")
            print(f"    - Epoch: {info.get('epoch', 'N/A')}")
            print(f"    - Best score: {info.get('best_score', 'N/A')}")
        else:
            print(f"  ⚠️  Recurrent model not found (skipping)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test actual inference if models are available."""
    print("\nTesting inference...")
    
    try:
        from app import SentimentClassifier
        
        baseline_path = 'configs/Baseline_best_model.pt'
        if not os.path.exists(baseline_path):
            print("  ⚠️  Model not found, skipping inference test")
            return True
        
        print("  Loading model...")
        classifier = SentimentClassifier.from_checkpoint(
            baseline_path,
            model_type='baseline'
        )
        
        print("  Running test prediction...")
        test_text = "This is a test sentence."
        result = classifier.classify(test_text)
        
        print(f"  ✓ Inference successful!")
        print(f"    Text: '{test_text}'")
        print(f"    Prediction: {result['label']}")
        print(f"    Confidence: {result['confidence']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nChecking CUDA support...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"    - CUDA version: {torch.version.cuda}")
            print(f"    - Device count: {torch.cuda.device_count()}")
            print(f"    - Device name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA is not available (will use CPU)")
            print(f"    This is fine for testing, but inference will be slower")
        
        return True
        
    except Exception as e:
        print(f"  ✗ CUDA check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("TEXT CLASSIFICATION APP - INSTALLATION TEST")
    print("="*70)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Files", test_model_files),
        ("App Functionality", test_app_functionality),
        ("CUDA Support", test_cuda),
        ("Model Loading", test_model_loading),
        ("Inference", test_inference),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print()
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"🎉 All {total_count} tests passed!")
        print("\nYou're ready to use the app!")
        print("\nNext steps:")
        print("  1. Run the demo: python app/demo.py")
        print("  2. Try examples: python app/examples.py")
        print("  3. Interactive mode: python -m app.cli --interactive")
        return 0
    else:
        print(f"⚠️  {passed_count}/{total_count} tests passed")
        
        if not results.get("Package Imports", False):
            print("\n❌ Critical: Package imports failed")
            print("   Run: pip install -r requirements.txt")
        
        if not results.get("Model Files", False):
            print("\n⚠️  Warning: Model files not found")
            print("   You need to train the models first")
        
        if not results.get("Inference", False):
            print("\n⚠️  Warning: Inference test failed")
            print("   The app may not work correctly")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())

