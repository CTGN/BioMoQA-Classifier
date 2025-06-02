import logging
import os
from typing import Dict, List, Tuple, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from ray.tune.schedulers import ASHAScheduler
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from src.utils.utils import *
from model_init import *

#For 20 trials and 4 epochs, with BCE :
#- Best run : BestRun(run_id='e03ff073', objective=0.6607142857142857, hyperparameters={'pos_weight': 2.122305933450838, 'learning_rate': 3.8152580362510575e-06, 'weight_decay': 0.154749633579119}, run_summary=<ray.tune.analysis.experiment_analysis.ExperimentAnalysis object at 0x74f4bc15c6d0>)

# Ensure plot directory exists
os.makedirs(CONFIG["plot_dir"], exist_ok=True)

def optimize_model(
    seed_set: bool = True, loss_type: str = "BCE", model_name: str = CONFIG["model_name"],n_trials=10
) -> Dict:
    """Fine-tune a pre-trained model with optimized loss parameters.

    Args:
        seed_set: Whether to set random seeds for reproducibility.
        loss_type: Type of loss function ("BCE" or "focal").
        model_name: Name of the pre-trained model to use.

    Returns:
        Dictionary with evaluation results of the best model.
    """
    if seed_set:
        set_random_seeds(CONFIG["seed"])

    # Load and tokenize datasets
    train_ds, val_ds, test_ds = load_datasets()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train, tokenized_val,_ = tokenize_datasets(train_ds, val_ds,test_ds, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set up training arguments
    training_args = CustomTrainingArguments(
        output_dir=CONFIG["output_dir"],
        seed=CONFIG["seed"],
        data_seed=CONFIG["seed"],
        **CONFIG["default_training_args"],
        loss_type=loss_type,
        disable_tqdm=True,
    )

    # Define model initialization function
    def get_model():
        return BertForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])

    # Initialize trainer for hyperparameter search
    trainer = CustomTrainer(
        model_init=get_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        callbacks=[LearningRateCallback()],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
     # Define hyperparameter search space based on loss_type
    if loss_type == "BCE":
        tune_config = {
            "pos_weight": tune.uniform(1.0,10.0),
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            #"gradient_accumulation_steps": tune.choice([2,4,8]),
            "weight_decay": tune.uniform(0.0, 0.3),
            }  # Tune pos_weight for BCE
    elif loss_type == "focal":
        tune_config = {
            "alpha": tune.uniform(0.5, 1.0),  # Tune alpha for focal loss
            "gamma": tune.uniform(2.0, 10.0),   # Tune gamma for focal loss
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            #"gradient_accumulation_steps": tune.choice([2,4,8]),
            "weight_decay": tune.uniform(0.0, 0.3),
            }
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Set up scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="eval_f1", #When set to objective, it takes the sum of the compute-metric output. if compute-metric isnt defined, it takes the loss.
        mode="max"
    )
    """ 
    class MyCallback(tune.Callback):
        def on_trial_start(self, iteration, trials, trial, **info):
            logger.info(f"Trial successfully started with config : {trial.config}")
            return super().on_trial_start(iteration, trials, trial, **info)
        
        def on_trial_complete(self, iteration, trials, trial, **info):
            logger.info(f"Trial ended with config : {trial.config}")
            return super().on_trial_complete(iteration, trials, trial, **info)
        
        def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
            logger.info("Created checkpoint successfully")
            return super().on_checkpoint(iteration, trials, trial, checkpoint, **info)
    """
    
    # Perform hyperparameter search
    logger.info(f"Starting hyperparameter search for {loss_type} loss")

    checkpoint_config=tune.CheckpointConfig(num_to_keep=1,checkpoint_frequency=0,checkpoint_score_attribute="training_iteration",checkpoint_score_order="max")

    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        search_alg=HyperOptSearch(metric="eval_f1", mode="max",random_state_seed=CONFIG["seed"]),
        backend="ray",
        direction="maximize",
        n_trials=n_trials,
        resources_per_trial={"cpu": 16, "gpu": 1},
        #scheduler=scheduler,
        checkpoint_config=checkpoint_config,
        #callbacks=[MyCallback()],
        storage_path="/home/leandre/Projects/BioMoQA_Playground/model/ray_results/",
        name="tune_transformer_pbt",
    )
    logger.info(f"Best run : {best_run}")

    visualize_ray_tune_results(best_run, logger, plot_dir=CONFIG['plot_dir'])
    plot_trial_performance(best_run,logger=logger,plot_dir=CONFIG['plot_dir'])

    return best_run