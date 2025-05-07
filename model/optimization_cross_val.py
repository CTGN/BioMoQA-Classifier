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
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import datasets
from utils import *
from model_init import *
from time import perf_counter

#TODO : To lower checkpointing storage usage -> don't checkpoint for each fold but for the final model after the cross val
#TODO : Error handling  and clean code
#! Maybe use recall instead of f1 score for HPO
#! Maybe use the SHMOP or something instead of classic HPO
#? How do we implement ensemble learning with cross-validation ?
#->Since we have the same folds for each model, we can take the majority vote the test split of each fold
# TODO : use all GPUs when training final model while distribting for HPO
# TODO : add earlystoppingcallback

def optimize_model_cross_val(
    dataset,folds,seed_set: bool = True, loss_type: str = "BCE", model_name: str = CONFIG["model_name"],n_trials=10,with_title=False,data_type=None
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

    #? When should we tokenize ? 
    #TODO : See how tokenizing is done to check if it alrgiht like this -> ask julien if that's ok
    #It can be a problem since we are truncating
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_metrics=[]
    scores_by_fold=[]
    for fold_idx in range(len(folds)):
        logger.info(f"\nfold number {fold_idx+1} / {len(folds)}")
        
        train_dev_indices, test_indices = folds[fold_idx]
        train_dev_split = dataset.select(train_dev_indices)
        test_split = dataset.select(test_indices)

        logger.info(f"train+dev split size : {len(train_dev_split)}")
        logger.info(f"test split size : {len(test_split)}")

        # First split: 70% train, 30% temporary
        train_temp = train_dev_split.train_test_split(test_size=0.2, stratify_by_column="labels", seed=CONFIG["seed"])
        train_split = train_temp['train']
        dev_split = train_temp['test']
        logger.info(f"train split size : {len(train_split)}")
        logger.info(f"dev split size : {len(dev_split)}")
        
        tokenized_train,tokenized_dev, tokenized_test = tokenize_datasets(train_split,dev_split,test_split, tokenizer=tokenizer,with_title=with_title)

        #We whould maybe perform cross val inside the model names loops ?
        #1. We run all models for each fold and we take the average of all of them at the end -> I think it is not good this way
        #2. (Inside loops) We go through all folds for each run and compare the means

        def train_bert(config):

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])

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
                train_dataset=tokenized_train,
                eval_dataset=tokenized_dev,
                callbacks=[LearningRateCallback()],
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
            )

            trainer.train()
            eval_result = trainer.evaluate()
            logger.info(f"eval_result: {eval_result}")

            return eval_result
        
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

        #TODO : check and remove the following
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
            scheduler=scheduler,
            search_alg=HyperOptSearch(metric="eval_f1", mode="max",random_state_seed=CONFIG["seed"]),
            checkpoint_config=checkpoint_config,
            num_samples=n_trials,  # Number of trials
            resources_per_trial={"cpu": 10, "gpu": 1},
            storage_path="/home/leandre/Projects/BioMoQA_Playground/model/ray_results/",
        )
        logger.info(f"Analysis results: {analysis}")
        # Step 10: Train Final Model with Best Hyperparameters
        best_config = analysis.get_best_config(metric="eval_f1", mode="max")
        best_results=analysis.get_best_trial(metric="eval_f1",mode="max").last_result

        logger.info(f"Best config : {best_config}")
        logger.info(f"Best results after optimization: {best_results}")

        visualize_ray_tune_results(analysis, logger, plot_dir=CONFIG['plot_dir'])
        plot_trial_performance(analysis,logger=logger,plot_dir=CONFIG['plot_dir'])

        #TODO : Perform ensemble learning with 5/10 independent models by looping here, then take the majority vote for each test instance. 
        #TODO : Check how to do ensemble learning with transformers before
        #TODO : Check Julien's article about how to impèlement that (ask him about the threholding optimization)
        logger.info(f"Final training...")
        start_time=perf_counter()
        model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])
        
        if data_type is not None:
            output_dir=os.path.join(CONFIG["output_dir"], data_type + "_"  + loss_type + "_" + model_name)
        else:
            output_dir=os.path.join(CONFIG["output_dir"], loss_type + "_" + model_name)

        # Set up training arguments
        training_args = CustomTrainingArguments(
                output_dir=output_dir,
                seed=CONFIG["seed"],
                data_seed=CONFIG["seed"],
                **CONFIG["default_training_args"],
                loss_type=loss_type,
            )
        
        training_args.pos_weight = best_config["pos_weight"] if loss_type == "BCE" else None
        training_args.alpha = best_config["alpha"] if loss_type == "focal" else None
        training_args.gamma = best_config["gamma"] if loss_type == "focal" else None
        training_args.learning_rate = best_config["learning_rate"]
        training_args.num_train_epochs = best_config["num_train_epochs"]
        

        training_args.gradient_accumulation_steps = best_config.get("gradient_accumulation_steps", 1)

        #TODO : Impelement early stopping !!
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        logger.info(f"training size : {len(tokenized_train)}")
        logger.info(f"test size : {len(tokenized_test)}")

        metrics = trainer.train().metrics

        eval_results_dev=trainer.evaluate()

        end_time_train=perf_counter()
        logger.info(f"Training time : {end_time_train-start_time}")

        eval_results_test = trainer.evaluate(tokenized_test)

        end_time_val=perf_counter()
        logger.info(f"Evaluation time : {end_time_val-end_time_train}")
        logger.info(f"Evaluation results on test set: {eval_results_test}")
        # Number of optimizer updates performed:

        n_updates = trainer.state.global_step
        logger.info(f"Total updates (optimizer steps): {n_updates}")
        n_steps = metrics["train_steps_per_second"] * metrics["train_runtime"]
        avg_step_time = metrics["train_runtime"] / n_steps
        logger.info(f"Avg time / step: {avg_step_time:.3f}s")
        

        final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model_cross_val_"+str(loss_type)+str(model_name)+str(n_trials)+"trials_fold-"+str(fold_idx+1))
        
        trainer.save_model(final_model_path)
        logger.info(f"Best model saved to {final_model_path}")

        results=[]
        logger.info(f"On test Set (with threshold 0.5) : ")
        # Compute detailed metrics
        predictions = trainer.predict(tokenized_test)
        
        scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
        preds = (scores > 0.5).astype(int)
        res1=detailed_metrics(preds, test_split["labels"])
        results.append(res1)
        plot_roc_curve(test_split["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
        plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

        #! The following seems weird. we are talking about decision here. View it like a ranking problem. take a perspective for usage
        threshold = eval_results_dev["eval_optim_threshold"]
        logger.info(f"\nOn test Set (optimal threshold of {threshold} according to cross validation on the training set): ")
        preds = (scores > threshold).astype(int)
        res2=detailed_metrics(preds, test_split["labels"])
        results.append(res2)
        plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

        logger.info(f"Results for fold {fold_idx+1} : {results}")
        test_metrics.append(results)
        scores_by_fold.append(scores)

    #TODO : Get the average metrics from the cross validation
    torch.cuda.empty_cache()
    
    return test_metrics, scores_by_fold

if __name__ == "__main__":
    #! the following is obsolete

    #results=fine_tune(loss_type='focal',optimize=True,n_trials=5)
    #print(results)
    ray.init()
    os.makedirs(CONFIG["plot_dir"], exist_ok=True)

    # Load and tokenize datasets
    train_ds, test_ds = load_datasets()
    results=optimize_model_cross_val(train_ds,test_ds,n_trials=20,loss_type="BCE",n_fold=5)
    print(results)
