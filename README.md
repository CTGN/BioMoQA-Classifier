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
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

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

### Automated Training Pipeline
Use the provided script to train multiple models automatically:
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

## Inference

### Single Text Prediction
```bash
# Predict on a single abstract
uv run src/models/biomoqa/instantiation.py \
  --model_path /path/to/trained/model \
  --abstract "Your research abstract text here..." \
  --with_title \
  --threshold 0.5
```

### Batch Prediction
```bash
# Process multiple texts from JSON file
uv run src/models/biomoqa/instantiation.py \
  --model_path /path/to/trained/model \
  --input_file examples/sample_texts.json \
  --with_title
```

### Using the Inference Script
```bash
# Alternative inference script
uv run experiments/inference.py \
  --model_path /path/to/model \
  --text "Your abstract here"
```

### Web Interface
Launch the interactive web application:
```bash
cd web
uv run streamlit run app.py
```

The web interface provides:
- Single text classification
- Batch file processing (JSON/CSV)
- Ensemble scoring with 5-fold cross-validation
- Interactive result visualization
- Score-based ranking and filtering

## Configuration

### Model Options
Supported model architectures:
- `google-bert/bert-base-uncased`
- `dmis-lab/biobert-v1.1`
- `FacebookAI/roberta-base`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`

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
- `--with_title` / `-t`: Include title in model input
- `--with_keywords` / `-k`: Include keywords in model input  
- `--nb_opt_negs` / `-on`: Number of optional negative samples
- `--loss`: Loss function (`BCE` or `focal`)
- `--n_trials`: Number of HPO trials
- `--fold`: Cross-validation fold (0-4)
- `--run`: Experiment run number

## Project Structure

```
├── configs/           # Configuration files
├── src/
│   ├── data_pipeline/ # Data preprocessing
│   ├── models/        # Training and model code
│   └── utils/         # Utility functions
├── experiments/       # Inference scripts
├── web/              # Web interface
├── scripts/          # Launch scripts
├── examples/         # Sample data
└── results/          # Model outputs and metrics
```

## Results

Trained models and evaluation metrics are saved in:
- `results/biomoqa/final_model/` - Final trained models
- `results/biomoqa/metrics/` - Performance metrics
- Model checkpoints follow the pattern: `best_model_cross_val_{loss}_{model}_{fold}/`

## Requirements

- Python ≥3.11
- CUDA-compatible GPU

## Troubleshooting

**Out of memory errors**: Reduce batch size in configs
**Missing models**: Check model paths in config files

