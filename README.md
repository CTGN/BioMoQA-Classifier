# BioMoQA Classifier

A machine learning system for classifying scientific abstracts for biodiversity relevance using ensemble transformer models. The system achieves 0.92+ ROC-AUC through 5-fold cross-validation with state-of-the-art biomedical language models.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Inference (Recommended)](#inference-recommended)
  - [Training Pipeline](#training-pipeline)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Results](#results)

---

## Overview

BioMoQA Classifier is designed to automatically assess whether scientific literature is relevant to biodiversity research. The system:

- **Supports 5 transformer architectures**: BERT, BioBERT variants, BiomedBERT variants, and RoBERTa
- **Uses ensemble learning**: Combines predictions from 5-fold cross-validation for robust performance
- **Achieves strong performance**: ~92% ROC-AUC, ~90% F1-score on test data
- **Provides multiple interfaces**: Python API, CLI, and web interface
- **Handles imbalanced data**: Supports optional negative sampling and focal loss

### Key Features

- **Ensemble Predictions**: 5-fold cross-validated models for reliable classification
- **Flexible Input**: Supports abstract-only or abstract+title classification
- **Loss Functions**: Binary Cross-Entropy (BCE) and Focal Loss
- **Hyperparameter Optimization**: Automated HPO with Ray Tune
- **Production Ready**: Optimized inference with FP16 support and batch processing

---

## Quick Start

### Installation

**Prerequisites**: Python 3.11+

1. **Clone the repository**
```bash
git clone https://github.com/CTGN/BioMoQA-Classifier.git
cd BioMoQA-Classifier
```

2. **Install dependencies with uv (recommended)**

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

3. **Download pretrained models and data**

```bash
./scripts/download_bucket.sh
```

This downloads from the public S3 bucket: [biomoqa-classifier.s3.text-analytics.ch](https://biomoqa-classifier.s3.text-analytics.ch/)

### Verify Installation

```bash
uv run python -c "import torch; from src.models.biomoqa.model_api import BioMoQAEnsemblePredictor; print('Installation successful!')"
```

---

## Project Structure

```
BioMoQA_Playground/
├── configs/                      # YAML configuration files
│   ├── paths.yaml               # Path and environment settings
│   ├── best_hpo.yaml           # Best hyperparameters from HPO
│   ├── ensemble.yaml           # Ensemble configuration
│   └── *.yaml                  # Training/HPO configs
│
├── src/
│   ├── config.py               # Configuration manager (YAML-based)
│   ├── data_pipeline/
│   │   └── biomoqa/
│   │       ├── preprocess_biomoqa.py    # Create cross-validation folds
│   │       └── create_raw.py            # Raw data preparation
│   ├── models/biomoqa/
│   │   ├── model_api.py        # Main inference API
│   │   ├── train.py            # Model training
│   │   ├── hpo.py              # Hyperparameter optimization
│   │   ├── ensemble.py         # Ensemble prediction
│   │   ├── baselines.py        # Baseline models
│   │   └── model_init.py       # Model initialization
│   └── utils/
│       ├── utils.py            # General utilities
│       └── plot_style.py       # Plotting configuration
│
├── web/
│   └── app.py                  # Streamlit web interface
│
├── scripts/
│   ├── biomoqa/
│   │   └── launch_final.sh     # Full training pipeline
│   ├── download_bucket.sh      # Download models/data
│   └── *.sh                    # Various utility scripts
│
├── data/                        # Dataset (downloaded)
│   ├── positives.csv           # Positive examples
│   ├── negatives.csv           # Negative examples
│   └── folds/                  # Cross-validation splits
│
└── results/
    ├── final_model/            # Trained model checkpoints
    ├── metrics/                # Performance metrics
    │   └── binary_metrics.csv
    └── test preds/             # Test predictions
```

---

## Usage

### Inference (Recommended)

The primary way to use BioMoQA Classifier is through the inference API, which loads pretrained ensemble models.

#### 1. Web Interface (Easiest)

```bash
uv run streamlit run web/app.py
```

Features:
- Single text classification
- Batch CSV/JSON upload
- Interactive results visualization
- Download predictions

#### 2. Python API

```python
from src.models.biomoqa.model_api import BioMoQAEnsemblePredictor

# Initialize predictor
predictor = BioMoQAEnsemblePredictor(
    model_type="BiomedBERT-abs",          # Model architecture
    loss_type="BCE",                       # Loss type used during training
    base_path="results/final_model",       # Path to trained models
    threshold=0.5,                         # Classification threshold
    nb_opt_negs=0                          # Optional negatives (0 for final models)
)

# Single prediction
result = predictor.score_text(
    abstract="This study examines the impact of climate change on species diversity...",
    title="Climate Change Effects on Biodiversity"  # Optional
)

print(f"Score: {result['score']:.3f}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
from src.models.biomoqa.model_api import load_data

data = load_data("data/sample_texts.csv")  # CSV or JSON
results = predictor.score_batch(data, batch_size=16)
```

#### 3. Command Line

**Single abstract:**
```bash
uv run python src/models/biomoqa/model_api.py \
  --model_type BiomedBERT-abs \
  --loss_type BCE \
  --abstract "Your abstract text here..." \
  --title "Paper Title"
```

**Batch processing:**
```bash
uv run python src/models/biomoqa/model_api.py \
  --model_type BiomedBERT-abs \
  --loss_type BCE \
  --input_file data/abstracts.csv \
  --output_file results/predictions.csv \
  --batch_size 16
```

#### Supported Model Types

Use these values for `--model_type`:
- `BiomedBERT-abs` - BiomedNLP-BiomedBERT-base-uncased-abstract
- `BiomedBERT-abs-ft` - BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
- `roberta-base` - FacebookAI/roberta-base
- `biobert-v1` - dmis-lab/biobert-v1.1
- `bert-base` - google-bert/bert-base-uncased

---

### Training Pipeline

To train models from scratch or reproduce results:

#### 1. Data Preprocessing

Create cross-validation folds:
```bash
uv run python src/data_pipeline/biomoqa/preprocess_biomoqa.py \
  -nf 5 \        # Number of folds
  -nr 1 \        # Number of runs
  -on 500        # Optional negatives per fold
```

**Data Quality**: See [DATA_QUALITY_REPORT.md](./DATA_QUALITY_REPORT.md) for fold creation details and coverage analysis.

#### 2. Hyperparameter Optimization (Optional)

```bash
uv run python src/models/biomoqa/hpo.py \
  --config configs/hpo.yaml \
  --fold 0 \
  --run 0 \
  --nb_opt_negs 500 \
  --n_trials 25 \
  --hpo_metric "eval_roc_auc" \
  -m "google-bert/bert-base-uncased" \
  --loss "BCE" \
  -t  # Include title
```

Best hyperparameters are automatically saved to `configs/best_hpo.yaml`.

#### 3. Model Training

Train a single fold:
```bash
uv run python src/models/biomoqa/train.py \
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

#### 4. Ensemble Prediction

Generate ensemble predictions across all folds:
```bash
uv run python src/models/biomoqa/ensemble.py \
  --config configs/ensemble.yaml \
  --fold 0 \
  --run 0 \
  --nb_opt_negs 500 \
  --loss "BCE" \
  -t
```

#### 5. Automated Full Pipeline

Run complete training for all models:
```bash
./scripts/biomoqa/launch_final.sh
```

This script trains all 5 model architectures with 5-fold cross-validation.

---

## Model Architecture

### Ensemble Approach

BioMoQA uses a 5-fold cross-validation ensemble:

1. **Data Splitting**: Dataset is split into 5 stratified folds maintaining class balance
2. **Training**: Each fold is held out once, and a model is trained on the remaining 4 folds
3. **Inference**: For new inputs, predictions from all 5 models are averaged
4. **Aggregation**: Mean of probabilities is used as the final ensemble score

### Transformer Models

All models are fine-tuned sequence classification models:
- **Input**: `[CLS] title [SEP] abstract [SEP]` (or abstract only)
- **Output**: Binary classification (relevant/not relevant to biodiversity)
- **Max Length**: 512 tokens
- **Optimization**: AdamW with linear warmup and decay

### Loss Functions

- **BCE (Binary Cross-Entropy)**: Standard for balanced datasets
- **Focal Loss**: Better for imbalanced data, reduces weight of easy examples

---

## Configuration

All configuration is managed via YAML files in `configs/`. **Never hardcode paths or settings in Python files.**

### Key Configuration Files

- **`configs/paths.yaml`**: Data paths, result directories, project structure
- **`configs/best_hpo.yaml`**: Optimized hyperparameters per model
- **`configs/ensemble.yaml`**: Ensemble settings
- **`configs/train.yaml`**: Training configuration

### Using Configuration in Code

```python
from src.config import get_config

config = get_config()
data_dir = config.get("data_dir")
fold_path = config.get_fold_path("train", fold=0, run=0)
```

### Important Settings

From `configs/paths.yaml`:
```yaml
data_dir: "data"
results_dir: "results"

data:
  folds_dir: "${data_dir}/folds"
  raw_data:
    positives: "${data_dir}/positives.csv"
    negatives: "${data_dir}/negatives.csv"

results:
  final_model_dir: "${results_dir}/final_model"
  metrics_dir: "${results_dir}/metrics"

training:
  num_folds: 5
  num_runs: 1
  default_optional_negatives: 500
```

---

## Results

### Reproducing Performance Metrics

The performance metrics below are obtained by running the complete training pipeline. To reproduce these results:

**Option 1: Run full automated pipeline**
```bash
./scripts/biomoqa/launch_final.sh
```

This script will:
1. Train all 5 model architectures with 5-fold cross-validation
2. Generate predictions on the test set for each fold
3. Calculate ensemble predictions by averaging fold predictions
4. Compute and save metrics to `results/metrics/binary_metrics.csv`

**Option 2: Train individual models**

Follow the [Training Pipeline](#training-pipeline) steps for each model:
1. Preprocess data (creates 5 folds)
2. (Optional) Run HPO for hyperparameter tuning
3. Train each fold (5 models per architecture)
4. Run ensemble prediction - **this generates the metrics**

The `ensemble.py` script automatically:
- Loads all 5 fold models
- Generates predictions on the test set
- Averages predictions across folds
- Calculates all metrics (F1, ROC-AUC, Precision, Recall, etc.)
- Saves results to `results/metrics/binary_metrics.csv`

### Model Performance

Performance metrics from 5-fold cross-validation on the test set (averaged across folds):

| Model | Loss | ROC-AUC | F1 | Precision | Recall | Accuracy |
|-------|------|---------|----|-----------| -------|----------|
| **Ensemble (BiomedBERT-abs)** | BCE | 0.920 | 0.905 | 0.894 | 0.916 | 0.869 |
| BiomedBERT-abs | BCE | 0.900 | 0.889 | 0.886 | 0.893 | 0.847 |
| BiomedBERT-abs-ft | BCE | 0.903 | 0.894 | 0.870 | 0.920 | 0.849 |
| RoBERTa-base | BCE | 0.898 | 0.894 | 0.888 | 0.901 | 0.852 |
| BioBERT-v1 | BCE | 0.893 | 0.884 | 0.850 | 0.922 | 0.831 |
| BERT-base | BCE | 0.900 | 0.892 | 0.879 | 0.906 | 0.848 |

**Key Findings**:
- Ensemble models consistently outperform individual folds
- Biomedical-pretrained models (BiomedBERT, BioBERT) show strong domain adaptation
- BCE loss generally performs better than Focal Loss for this balanced dataset
- Including title information improves performance by ~1-2% on most metrics

### Viewing Results

**Check all detailed metrics:**
```bash
# View all models and folds
cat results/metrics/binary_metrics.csv

# Or load in Python
import pandas as pd
metrics = pd.read_csv("results/metrics/binary_metrics.csv")
print(metrics[['model_name', 'loss_type', 'fold', 'f1', 'roc_auc', 'precision', 'recall']])
```

**Analyze specific model:**
```python
# Filter for ensemble results
ensemble_results = metrics[metrics['model_name'] == 'Ensemble']
print(f"Average ROC-AUC: {ensemble_results['roc_auc'].mean():.3f}")
```

### Output Files

- **`results/metrics/binary_metrics.csv`**: Complete metrics for all models/folds
- **`results/final_model/`**: Trained model checkpoints organized by fold
  - Format: `best_model_cross_val_{loss}_{model}_fold-{fold}_opt_negs-{nb_opt_negs}/`
- **`results/test preds/`**: Individual model predictions on test set
- **`plots/`**: Visualization of performance (ROC curves, confusion matrices)

### Key Metrics Explained

- **ROC-AUC**: Area under ROC curve, measures ranking quality (0.5-1.0, higher is better)
- **F1**: Harmonic mean of precision and recall (0-1, higher is better)
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many are detected
- **MCC**: Matthews Correlation Coefficient, balanced metric for imbalanced data
- **NDCG**: Normalized Discounted Cumulative Gain, ranking quality metric

---

## Best Practices

1. **Always use YAML configuration** - Never hardcode paths or settings in Python files
2. **Access config via manager** - Use `from src.config import get_config` in all scripts
3. **Use ensemble models for production** - They provide more robust predictions than single folds
4. **Monitor metrics** - Check `results/metrics/` for model performance
5. **Verify fold coverage** - Run `uv run python verify_folds.py` after preprocessing

---

## Additional Documentation

- **[DATA_QUALITY_REPORT.md](./DATA_QUALITY_REPORT.md)**: Data labeling verification and fold coverage
- **[FOLD_CREATION_DOCUMENTATION.md](./FOLD_CREATION_DOCUMENTATION.md)**: Detailed fold creation process
- **[PLOTTING_STYLE_GUIDE.md](./PLOTTING_STYLE_GUIDE.md)**: Visualization standards

---

## Citation

If you use this code or models, please cite:

```bibtex
@software{biomoqa_classifier,
  title={BioMoQA Classifier: Biodiversity Literature Classification},
  author={Your Name/Organization},
  year={2024},
  url={https://github.com/CTGN/BioMoQA-Classifier}
}
```

---

## License

[Add your license information here]

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- See documentation in `docs/`
- Check configuration examples in `configs/`
