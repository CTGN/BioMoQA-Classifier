# BioMoQA Classifier

A machine learning pipeline for biodiversity literature classification using robust ensemble transformer models. The system classifies scientific abstracts for biodiversity relevance with support for multiple model architectures, hyperparameter optimization, and cross-validation ensemble inference.

---

## **Table of Contents**
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training & Evaluation Pipeline](#training--evaluation-pipeline)
- [Inference & Interfaces](#inference--interfaces)
- [Project Structure](#project-structure)
- [Results](#results)

---

## **Quick Start**

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CTGN/BioMoQA-Classifier.git
cd BioMoQA-Classifier
```

2. **Install dependencies (recommended: [uv](https://github.com/astral-sh/uv))**
```bash
# Download and run install script for uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with wget
wget -qO- https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

3. **Verify installation**
```bash
uv run python -c "import torch; print('Installation successful!')"
```

---

## **Configuration**

**All configuration is managed via YAML files in `configs/`.**
- Main path and experiment config: `configs/paths.yaml`.
- *Never* edit src/config.py or any Python constants for configuration—**use YAML & the config manager only**.

Access configuration in any Python script via:
```python
from src.config import get_config
config = get_config()
data_dir = config.get("data_dir")
fold_path = config.get_fold_path("train", fold, run)
```

You can add/modify hyperparameters, data/model locations, and training settings in your YAML—these changes will be picked up automatically.

---

## **Download Dataset and Model Weights (Linux)**

The dataset and pretrained model weights are hosted here: [biomoqa-classifier (public)](https://biomoqa-classifier.s3.text-analytics.ch/).

```bash
# From project root
mkdir -p data results/final_model

# Download model checkpoints
wget -r -np -nH --cut-dirs=1 -R "index.html*" -e robots=off \
  -P results/final_model \
  https://biomoqa-classifier.s3.text-analytics.ch/checkpoints/

# Download dataset
wget -r -np -nH --cut-dirs=1 -R "index.html*" -e robots=off \
  -P data \
  https://biomoqa-classifier.s3.text-analytics.ch/dataset/
```
*Result: `results/final_model/` contains checkpoints, `data/` contains datasets.*

---

## **Training & Evaluation Pipeline**

### Data Preprocessing
Prepare the cross-validation folds using provided scripts/dynamic config:
```bash
uv run src/data_pipeline/biomoqa/preprocess_biomoqa.py \
  -nf 5 -nr 1 -on 500
```

### Hyperparameter Optimization (HPO)
Search for best hyperparameters per model with config-driven experiment control:
```bash
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

### Training
Train a model using ensemble best parameters and config-resolved data splits:
```bash
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
Aggregate and score with the 5-fold ensemble:
```bash
uv run src/models/biomoqa/ensemble.py \
  --config configs/ensemble.yaml \
  --fold 0 \
  --run 0 \
  --nb_opt_negs 500 \
  --loss "BCE" \
  -t
```

### Automated Pipeline
Run the full training and evaluation pipeline:
```bash
./scripts/biomoqa/launch_final.sh
```

---

## **Inference & Interfaces**

All model scoring is now handled by the unified API:

### Programmatic Usage (Python)
```python
from src.models.biomoqa.model_api import BioMoQAEnsemblePredictor, load_data

# Instantiate ensemble predictor (all config-driven)
predictor = BioMoQAEnsemblePredictor(
    model_type="BiomedBERT-abs",
    loss_type="BCE",
    base_path="results/final_model",
    threshold=0.5,
)

# Score a single abstract
result = predictor.score_text("Your abstract here")

# Score a batch of abstracts (can also use API's load_data)
texts_data = load_data("path/to/abstracts.csv")  # Supports JSON/CSV
batch_results = predictor.score_batch(texts_data, batch_size=8)
```

### Command Line Interface (Ensemble Inference)
You can use the model API directly from the command line for batch or single predictions:

**Single Abstract Prediction:**
```bash
uv run python src/models/biomoqa/model_api.py \
  --model_type BiomedBERT-abs \
  --loss_type BCE \
  --abstract "This study discusses ecosystem biodiversity..." \
  --title "Biodiversity Research"
```

**Batch File Prediction (CSV or JSON):**
```bash
uv run python src/models/biomoqa/model_api.py \
  --model_type BiomedBERT-abs \
  --loss_type focal \
  --input_file data/sample_texts.csv \
  --batch_size 16 \
  --output_file results/predictions.csv
```

- `--model_type` (required): see supported names in Model Options below.
- `--loss_type`, `--base_path`, `--threshold`, `--device` as per the API.
- `--batch_size` for file-level prediction.
- `--output_file` writes output to this file (format detected from file extension).
- Output is pretty-printed to stdout if no `--output_file` is given.

All logic matches programmatic usage—and all config remains YAML-driven as above.

### Web Interface (Recommended)
Run:
```bash
uv run streamlit run web/app.py
```
*Features:*
- Ensemble scoring, single/batch classification, CSV/JSON input
- Robust file validation
- Interactive results, download, and statistics

---

## **Project Structure**
```
├── configs/                # YAML-based configuration (centralized)
│   └── paths.yaml
├── src/
│   ├── data_pipeline/
│   ├── models/
│   │   └── biomoqa/
│   │        ├── model_api.py     # Unified ensemble inference API
│   │        ├── train.py, hpo.py, ensemble.py, baselines.py
│   └── utils/
├── web/
│   └── app.py               # Streamlit web app
├── scripts/
├── results/
│   └── final_model/
├── data/
```

---

## **Model Options**
Supported model architectures:
- `google-bert/bert-base-uncased`
- `dmis-lab/biobert-v1.1`
- `FacebookAI/roberta-base`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

---

## **Best Practices**
- **All config, paths, and experiment settings come from YAML** (see `configs/paths.yaml`); *never edit constants in Python files*.
- Any new script or module accessing settings/paths **must use** the config manager (`from src.config import get_config`).

---

## Results

See `results/metrics/` for evaluation reports and `results/final_model/` for model checkpoints. All result paths are guaranteed to be config-driven for reproducibility.

---

**For further guidance, see in-code docstrings and configs/paths.yaml as the canonical references for all parameters, file paths, and experiment options.**
