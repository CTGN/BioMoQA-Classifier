import logging
import os
from typing import Dict, List, Tuple, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
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
import datasets
from utils import *
from model_init import *
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

#For 20 trials and 4 epochs, with BCE (with threshold 0.5):
#- Best run : BestRun(run_id='e03ff073', objective=0.6607142857142857, hyperparameters={'pos_weight': 2.122305933450838, 'learning_rate': 3.8152580362510575e-06, 'weight_decay': 0.154749633579119}, run_summary=<ray.tune.analysis.experiment_analysis.ExperimentAnalysis object at 0x74f4bc15c6d0>)

#For focal loss, 20 trials and 2-4 epochs and optimal threshold ( about 0.3 I think):
#Metrics: {'f1': {'f1': 0.5257731958762887}, 'recall': {'recall': 0.9107142857142857}, 'precision': {'precision': 0.3695652173913043}, 'accuracy': {'accuracy': 0.952626158599382}}
# Ensure plot directory exists
#TODO : To lower checkpointing storage usage -> don't checkpoint for each fold but for the final model after the cross val
#TODO : Add an argument whhich allows to to ensemble learning or not
#TODO : Error handling  and clean code

def optimize_model_cross_val(
    train_ds,test_ds,seed_set: bool = True, loss_type: str = "BCE", model_name: str = CONFIG["model_name"],n_trials=10, n_fold=5
) -> Dict:
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
    if seed_set:
        set_random_seeds(CONFIG["seed"])
    print("train size before HPO : ",len(train_ds))
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=CONFIG["seed"])
    folds = list(skf.split(train_ds['text'], train_ds['labels']))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train,tokenized_test = tokenize_datasets(train_ds,test_ds, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def train_bert(config):
        f1_scores = []
        optim_thresholds = []
        for fold_idx in range(len(folds)):
            logger.info(f"\nfold number {fold_idx+1} / {len(folds)}")
            train_indices, val_indices = folds[fold_idx]
            train_fold = tokenized_train.select(train_indices)
            val_fold = tokenized_train.select(val_indices)
            logger.info(f"train fold size : {len(train_fold)}")
            logger.info(f"val fold size : {len(val_fold)}")

            model=BertForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])

            # Set up training arguments
            training_args = CustomTrainingArguments(
                output_dir=f'.', #! Is that alright ?
                seed=CONFIG["seed"],
                data_seed=CONFIG["seed"],
                **CONFIG["default_training_args"],
                loss_type=loss_type,
                pos_weight=config["pos_weight"] if loss_type=="BCE" else None,
                alpha=config["alpha"] if loss_type=="focal" else None,
                gamma=config["gamma"]if loss_type=="focal" else None,
                weight_decay=config["weight_decay"],
                disable_tqdm=True,
                logging_dir=f'./logs_fold_{fold_idx}',
            )
            training_args.learning_rate=config["learning_rate"]
            training_args.num_train_epochs=config["num_train_epochs"]

            # Initialize trainer for hyperparameter search
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_fold,
                eval_dataset=val_fold,
                callbacks=[LearningRateCallback()],
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
            )

            trainer.train()
            eval_result = trainer.evaluate()
            print("eval_result",eval_result)
            optim_thresholds.append(eval_result["eval_optim_threshold"])
            f1_scores.append(eval_result["eval_f1"])

        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_optim_threshold = sum(optim_thresholds) / len(optim_thresholds)
        logger.info(f"Average F1 score for fold {fold_idx+1}: {avg_f1}")
        logger.info(f"Average optimal threshold for fold {fold_idx+1}: {avg_optim_threshold}")
        tune.report({"avg_f1":avg_f1,"optim_threshold":avg_optim_threshold})


    
    # Define hyperparameter search space based on loss_type
    if loss_type == "BCE":
        tune_config = {
            "pos_weight": tune.uniform(1.0,10.0),
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            #"gradient_accumulation_steps": tune.choice([2,4,8]),
            "weight_decay": tune.uniform(0.0, 0.3),
            "num_train_epochs": tune.choice([2, 3, 4]),
            }  # Tune pos_weight for BCE
    elif loss_type == "focal":
        tune_config = {
            "alpha": tune.uniform(0.5, 1.0),  # Tune alpha for focal loss
            "gamma": tune.uniform(2.0, 10.0),   # Tune gamma for focal loss
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            #"gradient_accumulation_steps": tune.choice([2,4,8]),
            "weight_decay": tune.uniform(0.0, 0.3),
            "num_train_epochs": tune.choice([2, 3, 4]),
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

    checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=0,checkpoint_at_end=False)

    analysis = tune.run(
        train_bert,
        config=tune_config,
        search_alg=HyperOptSearch(metric="avg_f1", mode="max",random_state_seed=CONFIG["seed"]),
        checkpoint_config=checkpoint_config,
        num_samples=n_trials,  # Number of trials
        metric="avg_f1",
        resources_per_trial={"cpu": 32, "gpu": 3},
        storage_path="/home/leandre/Projects/BioMoQA_Playground/model/ray_results/",
        mode="max",
    )
    print(analysis)
    # Step 10: Train Final Model with Best Hyperparameters
    best_config = analysis.get_best_config(metric="avg_f1", mode="max")
    best_results=analysis.best_result

    logger.info(f"Best config : {best_config}")
    logger.info(f"Best results ? : {best_results}")

    visualize_ray_tune_results(analysis, logger, plot_dir=CONFIG['plot_dir'])
    plot_trial_performance(analysis,logger=logger,plot_dir=CONFIG['plot_dir'])

    #TODO : Perform ensemble learning with 5/10 independent models by looping here, then take the majority vote for each test instance. 
    #TODO : Check how to do ensemble learning with transformers before
    #TODO : Check Julien's article about how to impèlement that (ask him about the threholding optimization)
    model=BertForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])
    
    # Set up training arguments
    training_args = CustomTrainingArguments(
            output_dir=CONFIG["output_dir"],
            seed=CONFIG["seed"],
            data_seed=CONFIG["seed"],
            **CONFIG["default_training_args"],
            loss_type=loss_type,
        )
    
    training_args.pos_weight = best_config["pos_weight"] if loss_type == "BCE" else None
    training_args.alpha = best_config["alpha"] if loss_type == "focal" else None
    training_args.gamma = best_config["gamma" ]if loss_type == "focal" else None
    training_args.learning_rate= best_config["learning_rate"]
    training_args.num_train_epochs=best_config["num_train_epochs"]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,#! Be careful with this
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    print("training size : ",len(tokenized_train))
    print("test size : ",len(tokenized_test))
    trainer.train()
    eval_results = trainer.evaluate(tokenized_test)
    logger.info(f"Evaluation results on test set: {eval_results}")

    if loss_type=="BCE":
        final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model_cross_val_BCE")
    elif loss_type=="focal":
        final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model_cross_val_Focal")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    
    trainer.save_model(final_model_path)
    logger.info(f"Best model saved to {final_model_path}")

    print("On test Set (with threshold 0.5) : ")
    # Compute detailed metrics
    predictions = trainer.predict(tokenized_test)
    
    scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
    preds = (scores > 0.5).astype(int)
    detailed_metrics(preds, test_ds["labels"])
    plot_roc_curve(test_ds["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
    plot_precision_recall_curve(test_ds["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

    threshold = best_results["optim_threshold"]
    print(f"\nOn test Set (optimal threshold of {threshold} according to cross validation on the training set): ")
    preds = (scores > threshold).astype(int)
    detailed_metrics(preds, test_ds["labels"])
    plot_precision_recall_curve(test_ds["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

    torch.cuda.empty_cache()
    
    return analysis,preds

if __name__ == "__main__":

    #results=fine_tune(loss_type='focal',optimize=True,n_trials=5)
    #print(results)
    ray.init()
    os.makedirs(CONFIG["plot_dir"], exist_ok=True)

    # Load and tokenize datasets
    train_ds, test_ds = load_datasets()
    results=optimize_model_cross_val(train_ds,test_ds,n_trials=20,loss_type="BCE",n_fold=5)
    print(results)
