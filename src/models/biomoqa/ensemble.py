import logging
import os
import argparse
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from ray.tune.schedulers import ASHAScheduler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import transformers
import sys
import datasets
from time import perf_counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import yaml

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.biomoqa.HPO_callbacks import CleanupCallback
from src.utils import *
from src.models.biomoqa.model_init import *

from src.config import *
import pandas as pd

logger = logging.getLogger(__name__)

# Initialize config instance
config = get_config()

logger.info(f"Project root: {project_root}")

#TODO : Add some variables to the config file and lnk them to here from the config (ex: Early Stopping patience)
#TODO : Make the paths reproducible


def parse_args():
    parser = argparse.ArgumentParser(description="Run HPO for our BERT classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g. configs/hpo.yaml)"
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-m",
        "--model_names",
        nargs="*",
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-on",
        "--nb_opt_negs",
        type=int,
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        help="Type of loss to use during training"
    )
    parser.add_argument(
        "-t",
        "--with_title",
        nargs='?',
        const=True,
        default=None,
        type=lambda x: x.lower() == 'true' if isinstance(x, str) else bool(x),
        help="Include title in input (overrides config)"
    )
    parser.add_argument(
        "-k",
        "--with_keywords",
        nargs='?',
        const=True,
        default=None,
        type=lambda x: x.lower() == 'true' if isinstance(x, str) else bool(x),
        help="Include keywords in input (overrides config)"
    )
    
    return parser.parse_args()

def get_scores_by_model(cfg):
    scores_by_models=[]
    for model in cfg['model_names']:
        test_preds_path=config.get_path('results', 'test_preds_dir') / f"fold_{cfg['fold']}_{map_name(os.path.basename(model))}_{cfg['loss_type']}{'_with_title' if cfg['with_title'] else ''}{'_with_keywords' if cfg['with_keywords'] else ''}_run-{cfg['run']}_opt_neg-{cfg['nb_optional_negs']}.csv"
        logger.info(f"test_preds_path : {test_preds_path}")
        if os.path.isfile(test_preds_path):
            model_preds_df=pd.read_csv(test_preds_path)
            logger.info(model_preds_df.head())
            scores_by_models.append(model_preds_df['score'])
        else:
            logger.error(f"No model scores for config : {cfg} with model {model}. Cannot run ensemble learning.")
            logger.error(f"This may be due to a model not having been trained yet. The ensmble learning process requires all 5 individual models to have been trained and their predictions saved.")
            return None
    return scores_by_models

def ensemble_pred(cfg):
    scores_by_model=get_scores_by_model(cfg)
    if scores_by_model is None:
        raise ValueError("scores_by_model is None. Cannot calculate average scores.")
    avg_models_scores = np.mean(scores_by_model, axis=0)  # Average scores across models
    logger.info(f"\nfold number {cfg['fold'] + 1} | run no. {cfg['run']}")

    test_split = load_dataset("csv", data_files=str(config.get_fold_path("test", cfg['fold'], cfg['run'])),split="train")

    # Ensure avg_models_scores and test_split["labels"] have matching shapes
    if avg_models_scores.shape[0] != len(test_split["labels"]):
        raise ValueError("Mismatch between avg_models_scores and test_split labels.")

    preds = (avg_models_scores > 0.5).astype(int)
    result = detailed_metrics(preds, test_split["labels"],scores=avg_models_scores)

    result_metrics_path=config.get_path('results', 'metrics_dir') / 'binary_metrics.csv'

    if os.path.isfile(result_metrics_path):
        result_metrics=pd.read_csv(result_metrics_path)
        # Convert metric columns to numeric, coercing errors to NaN
        numeric_cols = ["f1", "recall", "precision", "accuracy", "roc_auc", "AP", "MCC", "NDCG", "kappa", "TN", "FP", "FN", "TP"]
        for col in numeric_cols:
            if col in result_metrics.columns:
                result_metrics[col] = pd.to_numeric(result_metrics[col], errors='coerce')
    else:
        result_metrics=pd.DataFrame(columns=["model_name", "loss_type","fold", "run", "with_title", "with_keywords","nb_added_negs"])

    #We update the results dataframe
    result_metrics = pd.concat([
        result_metrics,
        pd.DataFrame([{
            "model_name": "Ensemble",
            "loss_type": cfg['loss_type'],
            "fold": cfg['fold']+1,
            "run": cfg['run']+1, 
            "with_title": cfg['with_title'],
            "with_keywords":cfg['with_keywords'],
            "nb_added_negs": cfg['nb_optional_negs'],
            **result
        }])
    ],ignore_index=True)
    
    save_dataframe(result_metrics)

    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"prediction":preds,'score':avg_models_scores,"fold":[cfg['fold'] for _ in range(len(preds))],"title":test_split['title'] })
    test_preds_path=config.get_path('results', 'test_preds_dir') / f"fold-{cfg['fold']}_Ensemble_{cfg['loss_type']}{'_with_title' if cfg['with_title'] else ''}{'_with_keywords' if cfg['with_keywords'] else ''}_run-{cfg['run']}_opt_neg-{cfg['nb_optional_negs']}.csv"
    
    fold_preds_df.to_csv(test_preds_path)

    # Group by relevant columns and calculate mean metrics
    avg_metrics = result_metrics.groupby(
        ["loss_type", "model_name", "run"]
    )[["f1", "recall", "precision", "accuracy"]].mean().reset_index()

    # Filter metrics for the current data_type and loss_type
    filtered_metrics = avg_metrics[
        (avg_metrics["model_name"] == "Ensemble") &
        (avg_metrics["loss_type"] == cfg['loss_type'])
    ]

    return filtered_metrics

def main():
    args = parse_args()

    logger.info(args)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info(cfg)
    if args.nb_opt_negs is not None:
        cfg["nb_optional_negs"] = args.nb_opt_negs
    if args.model_names is not None:
        cfg["model_names"] = args.model_names
    if args.with_title is not None:
        cfg["with_title"] = args.with_title
    if args.with_keywords is not None:
        cfg["with_keywords"] = args.with_keywords
    if args.loss is not None:
        cfg["loss_type"] = args.loss
    if args.fold is not None:
        cfg['fold']=args.fold
    if args.run is not None:
        cfg['run']=args.run
    
    ensemble_pred(cfg)

if __name__ == "__main__":
    main()