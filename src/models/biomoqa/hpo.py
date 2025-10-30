import logging
import os
from typing import *
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
import pandas as pd

# Add project root to sys.path for imports
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent

from src.models.biomoqa.HPO_callbacks import CleanupCallback
from src.utils import *
from src.models.biomoqa.model_init import *
from src.config import get_config

logger = logging.getLogger(__name__)

def set_reproducibility(seed):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Randomness sources seeded with {seed} for reproducibility.")

    set_random_seeds(get_config().get('environment', {}).get('seed', 42))
    set_seed(get_config().get('environment', {}).get('seed', 42))

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
        default=0,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        default=0,
        help="Which run to execute (overrides config)"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        help="GPU that can handle batches of 100"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Model name to use for training (overrides config.hpo.model_name)"
    )
    parser.add_argument(
        "-on",
        "--nb_opt_negs",
        type=int,
        help="Number of optional negatives to use (overrides config.hpo.nb_optional_negs, if None it will use the default value from the config)"
    )
    parser.add_argument(
        "-nt",
        "--n_trials",
        type=int,
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-hpom",
        "--hpo_metric",
        type=str,
        help="Metric to optimize during HPO (overrides config.hpo.metric, e.g. 'accuracy', 'f1', 'roc_auc', etc.)"
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=str,
        help="Direction of optimization for the metric (overrides config.hpo.direction, e.g. 'max' or 'min')"
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
        action="store_true",
        help="Whether to include the title in the input text (overrides config.hpo.with_title)"
    )
    parser.add_argument(
        "-k",
        "--with_keywords",
        action="store_true",
        help="Whether to include the keywords in the input text (overrides config.hpo.with_keywords)"
    )
    
    return parser.parse_args()

@staticmethod
def trainable(config,model_name,loss_type,hpo_metric,tokenized_train,tokenized_dev,data_collator,tokenizer):
    
    clear_cuda_cache()
    
    #ray.tune.utils.wait_for_gpu(target_util=0.15)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=get_config().get('environment', {}).get('num_labels', 1))

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    batch_size = 100
    
    logger.info(f"Trial on GPU {gpu_id} using batch size {batch_size}")
    
    # Set up training arguments
    training_args = CustomTrainingArguments(
        output_dir=str(get_config().get_path("results", "models_dir")),
        seed=get_config().get('environment', {}).get('seed', 42),
        data_seed=get_config().get('environment', {}).get('seed', 42),
        **get_config().get('environment', {}).get('default_training_args', {}),
        loss_type=loss_type,
        pos_weight=config["pos_weight"] if loss_type=="BCE" else None,
        alpha=config["alpha"] if loss_type=="focal" else None,
        gamma=config["gamma"]if loss_type=="focal" else None,
        weight_decay=config["weight_decay"],
        disable_tqdm=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model=hpo_metric,
        load_best_model_at_end = True,
        save_strategy='no',
        eval_strategy="no",
    )
    training_args.learning_rate=config["learning_rate"]
    training_args.num_train_epochs=7

    # Initialize trainer for hyperparameter search
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        callbacks=[LearningRateCallback()],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )


    os.makedirs(training_args.output_dir, exist_ok=True)

    trainer.train()
    eval_result = trainer.evaluate()
    logger.info(f"eval_result: {eval_result}")

    clear_cuda_cache()

    return eval_result
    


def train_hpo(cfg,fold_idx,run_idx):
    """Fine-tune a pre-trained model with optimized loss parameters.

    Args:
        train_ds
        val_ds
        test_ds
        seed_set: Whether to set random seeds for reproducibility.
        loss_type: Type of loss function ("BCE" or "focal").
        model_name: Name of the pre-trained model to use.

    Returns:
        Dictionary with evaluation results of the best model.
    """
    
    ray.init(runtime_env={
        'excludes': os.listdir(os.path.join(project_root,'.git/objects'))
        ,"env_vars": {"VIRTUAL_ENV": os.path.join(project_root,".venv")}
    })

    #? When should we tokenize ? 
    #TODO : See how tokenizing is done to check if it is alrgiht like this -> ask julien if that's ok
    #It can be a problem since we are truncating
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True)

    test_metrics=[]
    scores_by_fold=[]
    preds_df_list=[]

    clear_cuda_cache()
    logger.info(f"\nfold number {fold_idx+1} | run no. {run_idx+1}")
    
    config_mgr = get_config()
    train_split = load_dataset("csv", data_files=str(config_mgr.get_fold_path("train", fold_idx, run_idx)), split="train")
    dev_split = load_dataset("csv", data_files=str(config_mgr.get_fold_path("dev", fold_idx, run_idx)), split="train")
    test_split = load_dataset("csv", data_files=str(config_mgr.get_fold_path("test", fold_idx, run_idx)), split="train")

    logger.info(f"train split size : {len(train_split)}")
    logger.info(f"dev split size : {len(dev_split)}")
    logger.info(f"test split size : {len(test_split)}")
    
    
    tokenized_train,tokenized_dev, tokenized_test = tokenize_datasets(train_split,dev_split,test_split, tokenizer=tokenizer,with_title=cfg['with_title'],with_keywords=cfg['with_keywords'])

    max_len = max(len(batch) for batch in tokenized_train["input_ids"])
    logger.info(f"Fold {fold_idx+1} max seq len = {max_len}")

    #We whould maybe perform cross val inside the model names loops ?
    #1. We run all models for each fold and we take the average of all of them at the end -> I think it is not good this way
    #2. (Inside loops) We go through all folds for each run and compare the means
    
    # Define hyperparameter search space based on loss_type
    if cfg['loss_type'] == "BCE":
        pos_weight_range=tune.uniform(0.0,1.0) if cfg['nb_optional_negs']==0 else tune.uniform(0.0,5.0)
        tune_config = {
            "pos_weight": pos_weight_range,
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            #"gradient_accumulation_steps": tune.choice([2,4,8]),
            "weight_decay":tune.loguniform(1e-6, 1e-1)
            #"num_train_epochs": tune.choice([2, 3, 4, 5, 6]),
            }  # Tune pos_weight for BCE
    elif cfg['loss_type'] == "focal":
        tune_config = {
            "alpha": tune.uniform(0.0, 1.0),  # Tune alpha for focal loss
            "gamma": tune.uniform(0.0, 10.0),   # Tune gamma for focal loss
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            #"gradient_accumulation_steps": tune.choice([2,4,8]),
            "weight_decay":tune.loguniform(1e-6, 1e-1)
            #"num_train_epochs": tune.choice([2, 3, 4, 5, 6]),
            }
    else:
        raise ValueError(f"Unsupported loss_type: {cfg['loss_type']}")

    # Set up scheduler for early stopping during hyperparameter search
    scheduler = ASHAScheduler(
        metric=cfg['hpo_metric'], #When set to objective, it takes the sum of the compute-metric output. if compute-metric isnt defined, it takes the loss.
        mode=cfg['direction']
    )
    
    # Perform hyperparameter search
    logger.info(f"Starting hyperparameter search for {cfg['loss_type']} loss")

    checkpoint_config = tune.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False)
    sync_config=tune.SyncConfig(sync_artifacts_on_checkpoint=False,sync_artifacts=False)
    
    wrapped_trainable=tune.with_parameters(trainable,model_name=cfg['model_name'],loss_type=cfg['loss_type'],hpo_metric=cfg['hpo_metric'],tokenized_train=tokenized_train,tokenized_dev=tokenized_dev,data_collator=data_collator,tokenizer=tokenizer)
    analysis = tune.run(
        wrapped_trainable,
        config=tune_config,
        sync_config=sync_config,
        scheduler=scheduler,
        search_alg=HyperOptSearch(metric=cfg['hpo_metric'], mode="max", random_state_seed=get_config().get('environment', {}).get('seed', 42)),
        checkpoint_config=checkpoint_config,
        num_samples=cfg['num_trials'],
        resources_per_trial={"cpu": 7, "gpu": 1},
        storage_path=str(os.path.join(project_root, "results/ray_results")),
        callbacks=[CleanupCallback(cfg['hpo_metric'])]
    )
    logger.info(f"Analysis results: {analysis}")

    # Handle case where no trials succeeded
    best_trial = analysis.get_best_trial(metric=cfg['hpo_metric'], mode="max")
    logger.info(f"Best trial : {best_trial}")
    if best_trial is None:
        logger.error("No successful trials found. Please check the training process and metric logging.")
        return None, None

    best_config = best_trial.config
    best_results = best_trial.last_result

    logger.info(f"Best config : {best_config}")
    logger.info(f"Best trial after optimization: {best_results}")
    best_config['loss_type'] = cfg['loss_type']
    best_config['model_name'] = cfg['model_name']
    best_config['with_title'] = cfg['with_title']
    best_config['with_keywords'] = cfg['with_keywords']
    best_config['fold'] = fold_idx
    best_config['run_idx'] = run_idx
    best_config['hpo_metric'] = cfg['hpo_metric']
    best_config['direction'] = cfg['direction']
    best_config['nb_optional_negs'] = cfg['nb_optional_negs']
    best_config['num_trials'] = cfg['num_trials']

    plot_trial_performance(analysis,logger=logger,plot_dir=get_config().get('plot_dir'),metric=cfg['hpo_metric'],file_name=f"metrics_evol_{map_name(cfg['model_name'])}_fold-{fold_idx}_title-{cfg['with_title']}_run_{run_idx}.png")

    return best_config

def main():
    args = parse_args()
    set_reproducibility(get_config().get('environment', {}).get('seed', 42))

    logger.info(args)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info(cfg)
    cfg["fold"] = args.fold
    if args.n_trials is not None:
        cfg["num_trials"] = args.n_trials
    if args.hpo_metric is not None:
        cfg["hpo_metric"] = args.hpo_metric
    if args.direction is not None:
        cfg["direction"] = args.direction
    if args.nb_opt_negs is not None:
        cfg["nb_optional_negs"] = args.nb_opt_negs
    if args.model_name is not None:
        cfg["model_name"] = args.model_name
    if args.with_title is not None:
        cfg["with_title"] = args.with_title
    if args.with_keywords is not None:
        cfg["with_keywords"] = args.with_keywords
    if args.loss is not None:
        cfg["loss_type"] = args.loss


    best_params=train_hpo(cfg,args.fold,args.run)
    
    logger.info(f"Best Hyperparameters after HPO : {best_params}")

    os.makedirs("configs", exist_ok=True)
    with open("configs/best_hpo.yaml", "w") as fout:
        yaml.safe_dump(best_params, fout)

if __name__ == "__main__":
    main()