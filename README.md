# BioMoQA Classifier

A machine learning pipeline for biodiversity research classification using transformer models. The system classifies scientific abstracts for biodiversity relevance with support for multiple model architectures, hyperparameter optimization, and ensemble methods.

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd BioMoQA-Classifier
```

2. **Install dependencies with uv**
```bash

Use curl to download the script and execute it with sh:

curl -LsSf https://astral.sh/uv/install.sh | sh

If your system doesn't have curl, you can use wget:

wget -qO- https://astral.sh/uv/install.sh | sh


# Install project dependencies
uv sync
```

3. **Verify installation**
```bash
uv run python -c "import torch; print('Installation successful!')"
```

## Training

### Data Preprocessing
First, prepare the data with cross-validation folds:
```bash
# Generate 5-fold CV splits with 500 optional negatives
uv run src/data_pipeline/biomoqa/preprocess_biomoqa.py \
  -nf 5 -nr 1 -on 500
```

### Hyperparameter Optimization (HPO)
Find optimal hyperparameters for each model:
```bash
# Run HPO for a specific model and fold
uv run src/models/biomoqa/hpo.py \
  --config configs/hpo.yaml \
  --fold 0 \
  --run 0 \
  --nb_opt_negs 500 \
  --n_trials 25 \
  --hpo_metric "eval_roc_auc" \
  -m "google-bert/bert-base-uncased" \
  --loss "BCE" \
  -t
```

### Final Training
Train models with optimized hyperparameters:
```bash
# Train with best HPO configuration
uv run src/models/biomoqa/train.py \
  --config configs/train.yaml \
  --hp_config configs/best_hpo.yaml \
  --fold 0 \
  --run 0 \
  -m "google-bert/bert-base-uncased" \
  --nb_opt_negs 500 \
  -bm "eval_roc_auc" \
  --loss "BCE" \
  -t
```

### Model Evaluation
Ensemble evaluation across all trained models:
```bash
# Generate ensemble predictions
uv run src/models/biomoqa/ensemble.py \
  --config configs/ensemble.yaml \
  --fold 0 \
  --run 0 \
  --nb_opt_negs 500 \
  --loss "BCE" \
  -t
```

### Automated Training Pipeline
Use the provided script to train multiple models automatically (with ensemble learning):
```bash
# Runs full pipeline: preprocessing → HPO → training → ensemble
./scripts/biomoqa/launch_final.sh
```

This script will:
- Train multiple model architectures (BERT, BioBERT, RoBERTa)
- Perform 5-fold cross-validation
- Run hyperparameter optimization
- Train final models with best parameters
- Generate ensemble predictions

## Testing

### Baseline Models
Run traditional ML baselines (Random, SVM, Random Forest):
```bash
# Run all baseline models
uv run src/models/biomoqa/baselines.py \
  -on 500 -nf 5 -nr 1 -t

# Or use the launch script
./scripts/launch_baselines.sh
```

### Ensemble Testing
Test the ensemble inference pipeline:
```bash
# Test ensemble scoring with sample texts
uv run web/test_ensemble.py

# This will test:
# - Single text ensemble scoring
# - Batch processing with GPU acceleration  
# - Model validation
# - Score interpretation
```

## Inference

The inference system provides multiple interfaces for classifying research abstracts:

### Command Line Interface (Recommended)
```bash
# Single text prediction
uv run experiments/inference.py \
  --model_name "BiomedBERT-abs" \
  --abstract "Your research abstract text here..." \
  --loss_type "BCE" \
  --with_title \
  --threshold 0.5

# Run demo with example texts
uv run experiments/inference.py \
  --model_name "BiomedBERT-abs" \
  --demo

# Process multiple texts (not yet implemented in CLI)
```

### Programmatic Usage
```python
# Using the ensemble predictor (recommended)
from src.models.biomoqa.ensemble import load_ensemble_predictor

predictor = load_ensemble_predictor(
    model_type="BiomedBERT-abs",
    loss_type="BCE",
    threshold=0.5
)

# Single text scoring with ensemble
result = predictor.score_text("Your abstract text here")
print(f"Ensemble Score: {result['ensemble_score']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")

# Batch processing with GPU acceleration
results = predictor.score_batch_optimized(
    ["Abstract 1", "Abstract 2", "Abstract 3"],
    batch_size=16
)
```

```python
# Using individual model predictor
from src.models.biomoqa.instantiation import load_predictor

predictor = load_predictor(
    model_name="BiomedBERT-abs",
    loss_type="BCE",
    with_title=False,
    threshold=0.5
)

result = predictor.evaluate_text(
    abstract="Your abstract text",
    return_binary=True
)
```

### Web Interface
Launch the interactive web application for ensemble scoring:
```bash
# From project root
uv run streamlit run web/app.py
```

The web interface provides:
- **Ensemble scoring** with 5-fold cross-validation models
- Single text classification with confidence metrics
- Batch file processing (JSON/CSV) with GPU acceleration  
- Interactive result visualization and statistics
- Score-based ranking and filtering
- Real-time performance metrics and fold-level analysis

**Supported Models in Web Interface:**
- BiomedBERT-abs, BiomedBERT-full, BioBERT, BERT, RoBERTa
- BCE and Focal loss variants
- Configurable thresholds and batch sizes

## Configuration

### Model Options
Supported model architectures:
- `google-bert/bert-base-uncased`
- `dmis-lab/biobert-v1.1`
- `FacebookAI/roberta-base`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract+full-text`

### Input Formats
**JSON format** (`examples/sample_texts.json`):
```json
[
  {
    "abstract": "Your research abstract text...",
    "title": "Optional title",
    "keywords": "Optional keywords"
  }
]
```

**CSV format**:
```csv
abstract,title,keywords
"Research abstract text...","Title","keywords"
```

### Key Parameters

**Training Parameters:**
- `--with_title` / `-t`: Include title in model input
- `--with_keywords` / `-k`: Include keywords in model input  
- `--nb_opt_negs` / `-on`: Number of optional negative samples
- `--loss`: Loss function (`BCE` or `focal`)
- `--n_trials`: Number of HPO trials
- `--fold`: Cross-validation fold (0-4)
- `--run`: Experiment run number

**Inference Parameters:**
- `--model_name`: Model architecture name (e.g., `BiomedBERT-abs`)
- `--loss_type`: Loss type used during training (`BCE`, `focal`)
- `--weights_parent_dir`: Directory containing model checkpoints
- `--threshold`: Classification threshold (default: 0.5)
- `--abstract`: Text to classify (for single prediction)
- `--demo`: Run with built-in example texts

## Project Structure

```
├── configs/           # Configuration files
├── src/
│   ├── data_pipeline/ # Data preprocessing
│   ├── models/
│   │   └── biomoqa/
│   │       ├── ensemble.py      # 🆕 Shared ensemble predictor
│   │       ├── instantiation.py # Individual model predictor  
│   │       ├── train.py         # Model training
│   │       ├── hpo.py          # Hyperparameter optimization
│   │       └── baselines.py    # Traditional ML baselines
│   └── utils/         # Utility functions
├── experiments/
│   └── inference.py   # 🔄 Updated CLI interface
├── web/              # 🔄 Streamlit ensemble interface
│   ├── app.py        # Main web application
│   ├── test_ensemble.py # Testing utilities
│   └── utils.py      # Web utilities
├── scripts/          # Launch scripts
└── results/          # Model outputs and metrics
    └── final_model/  # Cross-validation model checkpoints
```

**Key Changes in v2.0:**
- ✅ Unified inference pipeline with shared ensemble logic
- ✅ Eliminated code duplication (~400 lines removed)
- ✅ Consistent API across CLI, web, and programmatic interfaces
- ✅ Enhanced error handling and parameter validation
- ✅ GPU-accelerated batch processing

## Results

Trained models and evaluation metrics are saved in:
- `results/final_model/` - Final trained models
- `results/metrics/` - Performance metrics
- Model checkpoints follow the pattern: `best_model_cross_val_{loss}_{model}_{fold}/`

## Requirements

- Python ≥3.11
- CUDA-compatible GPU

## Troubleshooting

**Common Issues:**

1. **Out of memory errors**: 
   - Reduce batch size in configs or web interface
   - Use smaller batch sizes in `score_batch_optimized()`

2. **Missing models**: 
   - Check model paths: `results/final_model/best_model_cross_val_{loss}_{model}_fold-{N}/`
   - Ensure models are trained before inference
   - Verify `--model_name` matches trained model directory names

3. **Import errors**:
   - Run from project root directory
   - Ensure dependencies installed with `uv sync`

4. **CLI parameter errors**:
   - Use `--model_name` instead of `--model_path` (old interface)
   - Required parameters: `--model_name` and either `--abstract` or `--demo`

5. **Web interface issues**:
   - Launch with `uv run streamlit run web/app.py` from project root
   - Check model paths in sidebar configuration

