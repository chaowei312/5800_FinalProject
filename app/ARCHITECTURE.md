# App Architecture Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT CLASSIFICATION APP                       │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │   CLI    │      │ Python   │      │  Demo/   │
    │ Interface│      │   API    │      │ Examples │
    └──────────┘      └──────────┘      └──────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │    UnifiedClassifier          │
              │  (Sentiment + Domain)         │
              └───────────────────────────────┘
                     │              │
        ┌────────────┘              └────────────┐
        │                                        │
        ▼                                        ▼
┌───────────────────┐                  ┌───────────────────┐
│ SentimentClassifier│                  │MultiDomainClassifier│
│  (Binary: 0/1)    │                  │  (3-class: 0/1/2) │
└───────────────────┘                  └───────────────────┘
        │                                        │
        ▼                                        ▼
┌───────────────────┐                  ┌───────────────────┐
│  Model Loader     │                  │  Model Loader     │
└───────────────────┘                  └───────────────────┘
        │                                        │
        ├────────────┬───────────                ├────────────┬───────────
        │            │                           │            │
        ▼            ▼                           ▼            ▼
    ┌────────┐  ┌────────┐                  ┌────────┐  ┌────────┐
    │Baseline│  │Recurr. │                  │Baseline│  │Recurr. │
    │ Model  │  │ Model  │                  │ Model  │  │ Model  │
    └────────┘  └────────┘                  └────────┘  └────────┘
        │            │                           │            │
        └────────────┴───────────────────────────┴────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Label Mappings  │
                    │  0,1,2 → Text    │
                    └──────────────────┘
```

## Data Flow

### 1. User Input Flow

```
User Input Text
    │
    ▼
┌─────────────────┐
│  Tokenization   │  (BERT tokenizer)
│  - Truncation   │
│  - Padding      │
│  - Max Length   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Model Forward  │  (Baseline or Recurrent)
│  - Embeddings   │
│  - Attention    │
│  - Layers       │
│  - Classifier   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Softmax        │
│  Get Logits     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Label Mapping  │
│  0 → "Negative" │
│  1 → "Positive" │
│  0 → "movie_review" │
│  etc.           │
└─────────────────┘
    │
    ▼
Return Results
{
  'label': 'Positive',
  'confidence': 0.95,
  'description': '...'
}
```

## Class Hierarchy

```
BaseClassifier (Abstract)
    │
    ├── SentimentClassifier
    │       │
    │       ├── from_checkpoint()
    │       ├── classify()
    │       ├── predict()
    │       └── predict_proba()
    │
    └── MultiDomainClassifier
            │
            ├── from_checkpoint()
            ├── classify()
            ├── predict()
            └── predict_proba()

UnifiedClassifier (Composition)
    │
    ├── sentiment_classifier: SentimentClassifier
    ├── domain_classifier: MultiDomainClassifier
    │
    ├── analyze()
    └── analyze_batch()
```

## Model Architecture Comparison

### Baseline Transformer
```
Input Text
    │
    ▼
Embeddings (Word + Position + Token Type)
    │
    ▼
┌───────────────────────────────┐
│  Transformer Block 1          │
│  ┌─────────────────────────┐  │
│  │ Self-Attention          │  │
│  │ (with Flash Attention)  │  │
│  └─────────────────────────┘  │
│           │                   │
│  ┌─────────────────────────┐  │
│  │ Feed-Forward Network    │  │
│  │ (with SwiGLU)          │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│  Transformer Block 2-N        │
│  (Same structure)             │
└───────────────────────────────┘
    │
    ▼
Classifier Head (Linear + Softmax)
    │
    ▼
Predictions
```

### Recurrent Transformer
```
Input Text
    │
    ▼
Embeddings
    │
    ▼
┌─────────────────────────────────────────┐
│  Iteration 1                            │
│  ┌───────────────────────────────────┐  │
│  │ Pass through all layers           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Iteration 2                            │
│  ┌───────────────────────────────────┐  │
│  │ Pass through SAME layers again    │  │
│  │ (with residual connection)        │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Iteration N (recurrent_depth times)    │
└─────────────────────────────────────────┘
    │
    ▼
Classifier Head
    │
    ▼
Predictions
```

## File Dependencies

```
app/
│
├── __init__.py
│   └── Exports: SentimentClassifier, MultiDomainClassifier,
│                SENTIMENT_LABELS, DOMAIN_LABELS
│
├── config.py
│   └── Defines: MODEL_PATHS, TOKENIZER_NAME, defaults
│
├── label_mappings.py
│   └── Defines: SENTIMENT_LABELS, DOMAIN_LABELS
│               get_sentiment_label(), get_domain_label()
│
├── model_loader.py
│   ├── Imports: models/baseline/baseline_model.py
│   │            models/recurrent/recurrent_model.py
│   └── Defines: load_baseline_model(), load_recurrent_model()
│
├── inference.py
│   ├── Imports: model_loader, label_mappings
│   ├── Defines: BaseClassifier (base class)
│   │            SentimentClassifier
│   │            MultiDomainClassifier
│   │            UnifiedClassifier
│   └── Uses: Transformers (AutoTokenizer), PyTorch
│
├── cli.py
│   ├── Imports: inference
│   └── Provides: Command-line interface with argparse
│
├── demo.py
│   ├── Imports: inference
│   └── Runs: Comprehensive demonstrations
│
└── examples.py
    ├── Imports: inference
    └── Shows: Common use cases
```

## Usage Patterns

### Pattern 1: Single Task Classification
```
User → CLI/API → SentimentClassifier → Model → Label Mapping → Result
```

### Pattern 2: Multi-Task Classification
```
User → CLI/API → UnifiedClassifier
                      ├→ SentimentClassifier → Result 1
                      └→ MultiDomainClassifier → Result 2
                Combined Results → User
```

### Pattern 3: Batch Processing
```
File with texts → CLI → UnifiedClassifier.analyze_batch()
                           ├→ Process text 1
                           ├→ Process text 2
                           └→ Process text N
                        Results → JSON file
```

## Label Mapping System

```
Model Output (Logits)
    │
    ▼
Softmax → Probabilities [0.1, 0.9]
    │
    ▼
Argmax → Numeric Label (1)
    │
    ▼
Label Mapping Dictionary
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive"
}
    │
    ▼
Human-Readable Label ("Positive")
    │
    ▼
Enhanced Description
"The text expresses a positive sentiment or opinion"
```

## Configuration Hierarchy

```
app/config.py (defaults)
    │
    ├─ MODEL_PATHS
    ├─ TOKENIZER_NAME
    ├─ MAX_LENGTH
    └─ DEVICE
         │
         ▼
CLI Arguments (override)
    │
    ├─ --sentiment-checkpoint
    ├─ --domain-checkpoint
    ├─ --tokenizer
    └─ --device
         │
         ▼
Runtime Detection (auto)
    │
    └─ CUDA availability
```

## Error Handling Flow

```
User Input
    │
    ▼
┌─────────────────────┐
│ Validate Input      │
│ - Text not empty?   │
│ - Files exist?      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Load Models         │
│ - Checkpoint exists?│
│ - Valid format?     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Tokenization        │
│ - Valid encoding?   │
│ - Truncate if long  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Model Inference     │
│ - Forward pass      │
│ - Handle CUDA OOM   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Post-processing     │
│ - Map labels        │
│ - Format output     │
└─────────────────────┘
    │
    ▼
Return Results or Error Message
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `inference.py` | High-level classification logic |
| `model_loader.py` | Load PyTorch checkpoints |
| `label_mappings.py` | Convert IDs to text labels |
| `config.py` | Configuration management |
| `cli.py` | Command-line interface |
| `demo.py` | Demonstrations |
| `examples.py` | Usage examples |
| `test_installation.py` | Verify installation |

This architecture ensures:
- ✅ Modularity (easy to extend)
- ✅ Reusability (classes can be used independently)
- ✅ Clarity (clear separation of concerns)
- ✅ Maintainability (easy to debug and update)

