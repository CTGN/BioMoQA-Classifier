# BioMoQA Classifier

A machine learning pipeline for biodiversity research classification using transformer models. The system classifies scientific abstracts for biodiversity relevance with support for multiple model architectures, hyperparameter optimization, and ensemble methods.

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CTGN/BioMoQA-Classifier.git
cd BioMoQA-Classifier
```

2. **Install dependencies with uv**
```bash

Use curl to download the script and execute it with sh:

curl -LsSf https://astral.sh/uv/install.sh | sh

If your system does not have curl, you can use wget:

wget -qO- https://astral.sh/uv/install.sh | sh


# Install project dependencies
uv sync
```

3. **Verify installation**
```bash
uv run python -c "import torch; print('Installation successful!')"
```

## Download Dataset and Model Weights (Linux)

The dataset and pretrained model weights are publicly hosted in an S3 bucket: [biomoqa-classifier (public)](https://biomoqa-classifier.s3.text-analytics.ch/).

Use `wget` to download the contents into the correct project folders:
```bash
# From project root
mkdir -p data results/final_model

# Download model checkpoints → results/final_model/
wget -r -np -nH --cut-dirs=1 -R "index.html*" -e robots=off \
  -P results/final_model \
  https://biomoqa-classifier.s3.text-analytics.ch/checkpoints/

# Download dataset → data/
wget -r -np -nH --cut-dirs=1 -R "index.html*" -e robots=off \
  -P data \
  https://biomoqa-classifier.s3.text-analytics.ch/dataset/
```

After running the above commands:
- `results/final_model/` contains the cross-validation checkpoints
- `data/` contains the dataset files

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

### Ensemble Learning
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

## Inference

The inference system provides multiple interfaces for classifying research abstracts:

### Command Line Interface (Recommended)
```bash
# Single text prediction
uv run src/models/biomoqa/instantiation.py \
  --model_name "BiomedBERT-abs" \
  --input_file path/to/input_text \
  --loss_type "BCE" \
  --with_title \
  --threshold 0.5

# Process multiple texts (not yet implemented in CLI)
```

### Programmatic Usage
```python
# Using the ensemble predictor (recommended)
from src.models.biomoqa.folds_ensemble_predictor import load_ensemble_predictor

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
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

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
- `--model_name`: Model architecture name (can be one of the following : bert-base, roberta-base, BiomedBERT-abs, BiomedBERT-abs-ft,biobert-v1)
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
│   │       ├── ensemble.py      # Shared ensemble predictor
│   │       ├── instantiation.py # Individual model predictor  
│   │       ├── train.py         # Model training
│   │       ├── hpo.py          # Hyperparameter optimization
│   │       └── baselines.py    # Traditional ML baselines
│   └── utils/         # Utility functions
├── experiments/
│   └── inference.py   # Updated CLI interface
├── web/              # Streamlit ensemble interface
│   ├── app.py        # Main web application
│   ├── test_ensemble.py # Testing utilities
│   └── utils.py      # Web utilities
├── scripts/          # Launch scripts
└── results/          # Model outputs and metrics
    └── final_model/  # Cross-validation model checkpoints
```

## Results

Trained models and evaluation metrics are saved in:
- `results/final_model/` - Final trained models weights
- `results/metrics/` - Performance metrics
- Model checkpoints follow the pattern: `best_model_cross_val_{loss}_{model}_{fold}/`