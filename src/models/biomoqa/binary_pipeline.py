import logging
import os
from typing import *

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
from time import perf_counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from .HPO_callbacks import CleanupCallback

from src.utils import *
from .model_init import *
from src.data_pipeline.biomoqa.preprocess_biomoqa import biomoqa_data_pipeline

from src.config import *
import pandas as pd

logger = logging.getLogger(__name__)

#? How do we implement ensemble learning with cross-validation
# -> like i did i guess
# look for other ways ? ask julien ?
# TODO : change ensemble implementation so that it takes scores and returns scores (average ? ) -> see julien's covid paper
# TODO : compare with and without title
# TODO : Put longer number of epoch and implement early stopping, ask julien to explain again why tha adam opt implies that early stopping is better for bigger epochs
# TODO : Better management of checkpoints -> see how it works, what we're doing, I think that what's taking most of the storage is coming from the /tmp/ray dir
# -> Here is what we're doing : 
# For each data type we train len(name_models)*n_folds
# -> You have to ask the question : why do we want checkpoints and the answer will show you how to implement it


#TODO : To lower checkpointing storage usage -> don't checkpoint for each fold but for the final model after the cross val -> no, just keep the best model's checkpoint
#TODO : Error handling  and clean code
#! Maybe use recall instead of f1 score for HPO
#! Maybe use the SHMOP or something instead of classic HPO
#? How do we implement ensemble learning with cross-validation ?
#->Since we have the same folds for each model, we can take the majority vote the test split of each fold
# TODO : use all GPUs when training final model while distribting for HPO
# TODO : add earlystoppingcallback
# TODO : ask julien about gradient checkpointing 
# Maybe make pandas df for results where you would have the following attributes : model_name, f1, accuracy,tp, fp, tn, fn 

class TrainPipeline:

    def __init__(self,loss_type="BCE",hpo_metric="eval_recall",ensemble : Optional[bool] =False,n_trials=10,num_runs=5,with_title : Optional[bool] =False,with_keywords : Optional[bool] =False,nb_optional_negs=5000,n_folds=5,model_names = ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext","FacebookAI/roberta-base", "dmis-lab/biobert-v1.1", "google-bert/bert-base-uncased"]):
        set_random_seeds(CONFIG["seed"])
        self.loss_type=loss_type
        self.ensemble=ensemble
        self.with_title=with_title
        self.with_keywords=with_keywords
        self.n_folds=n_folds
        self.n_trials=n_trials
        self.result_metrics=pd.DataFrame(columns=["model_name", "fold", "loss_type", "run","with_title","with_keywords"])
        self.svm_metrics=pd.DataFrame(columns=["model_name", "fold", "kernel", "run","with_title","with_keywords"])
        self.random_forest_metrics=pd.DataFrame(columns=["model_name", "fold", "criterion","num_trees","run","with_title","with_keywords"])
        self.hpo_metric=hpo_metric
        self.model_names=model_names
        self.run_idx=0
        self.num_runs=num_runs
        self.dataset,self.folds_per_run =biomoqa_data_pipeline(self.n_folds,self.num_runs,self.with_title,self.with_keywords,nb_optional_negs=nb_optional_negs)
        self.folds=[]

        naive_metrics=pd.DataFrame()
        folds=self.folds_per_run[0]
        for fold_idx,(_,_,test_indices) in enumerate(folds):
            test_split=self.dataset.select(test_indices)
            
            res=detailed_metrics(np.asarray([1 for _ in range(len(test_indices))]),test_split['labels'],scores=np.asarray([1.0 for _ in range(len(test_indices))]))
            naive_metrics = pd.concat([
                naive_metrics,
                pd.DataFrame([{
                    "approach": "always pos",
                    "fold": fold_idx+1,
                    **res
                }])
            ])

            res=detailed_metrics(np.asarray([0 for _ in range(len(test_indices))]),test_split['labels'],scores=np.asarray([0.0 for _ in range(len(test_indices))]))
            naive_metrics = pd.concat([
                naive_metrics,
                pd.DataFrame([{
                    "approach": "always neg",
                    "fold": fold_idx+1,
                    **res
                }])
            ])
        self.store_metrics(naive_metrics,file_name="naive_metrics.csv")
        logger.info(f"Pipeline for loss type {self.loss_type} and ensemble={self.ensemble}")

    @staticmethod
    def train_hpo(config,model_name,fold_idx,loss_type,hpo_metric,tokenized_train,tokenized_dev,data_collator,tokenizer):
                model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])

                if torch.cuda.current_device()==2:
                    batch_size=70
                else:
                    batch_size=19
                  
                # Set up training arguments
                training_args = CustomTrainingArguments(
                    output_dir="/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/models",
                    seed=CONFIG["seed"],
                    data_seed=CONFIG["seed"],
                    **CONFIG["default_training_args"],
                    loss_type=loss_type,
                    pos_weight=config["pos_weight"] if loss_type=="BCE" else None,
                    alpha=config["alpha"] if loss_type=="focal" else None,
                    gamma=config["gamma"]if loss_type=="focal" else None,
                    weight_decay=config["weight_decay"],
                    disable_tqdm=True,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    metric_for_best_model=hpo_metric,
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


                os.makedirs(training_args.output_dir, exist_ok=True)

                trainer.train()
                eval_result = trainer.evaluate()
                logger.info(f"eval_result: {eval_result}")

                torch.cuda.empty_cache()
                clear_cuda_cache()

                return eval_result
    
    def train(self,model_name):
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
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer,max_length=512)

        test_metrics=[]
        scores_by_fold=[]
        preds_df_list=[]
        for fold_idx in range(len(self.folds)):
            logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
            
            train_indices, dev_indices,test_indices = self.folds[fold_idx]
            train_split = self.dataset.select(train_indices)
            dev_split = self.dataset.select(dev_indices)
            test_split = self.dataset.select(test_indices)

            logger.info(f"train split size : {len(train_split)}")
            logger.info(f"dev split size : {len(dev_split)}")
            logger.info(f"test split size : {len(test_split)}")
            
            
            tokenized_train,tokenized_dev, tokenized_test = tokenize_datasets(train_split,dev_split,test_split, tokenizer=tokenizer,with_title=self.with_title,with_keywords=self.with_keywords)

            #We whould maybe perform cross val inside the model names loops ?
            #1. We run all models for each fold and we take the average of all of them at the end -> I think it is not good this way
            #2. (Inside loops) We go through all folds for each run and compare the means
            
            # Define hyperparameter search space based on loss_type
            if self.loss_type == "BCE":
                tune_config = {
                    "pos_weight": tune.uniform(1.0,10.0),
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    #"gradient_accumulation_steps": tune.choice([2,4,8]),
                    "weight_decay": tune.uniform(0.0, 0.3),
                    "num_train_epochs": tune.choice([2, 3, 4, 5, 6]),
                    }  # Tune pos_weight for BCE
            elif self.loss_type == "focal":
                tune_config = {
                    "alpha": tune.uniform(0.5, 1.0),  # Tune alpha for focal loss
                    "gamma": tune.uniform(2.0, 10.0),   # Tune gamma for focal loss
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    #"gradient_accumulation_steps": tune.choice([2,4,8]),
                    "weight_decay": tune.uniform(0.0, 0.3),
                    "num_train_epochs": tune.choice([2, 3, 4, 5, 6]),
                    }
            else:
                raise ValueError(f"Unsupported loss_type: {self.loss_type}")

            # Set up scheduler for early stopping
            scheduler = ASHAScheduler(
                metric=self.hpo_metric, #When set to objective, it takes the sum of the compute-metric output. if compute-metric isnt defined, it takes the loss.
                mode="max"
            )
            
            # Perform hyperparameter search
            logger.info(f"Starting hyperparameter search for {self.loss_type} loss")

            checkpoint_config = tune.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False)
            sync_config=tune.SyncConfig(sync_artifacts_on_checkpoint=False,sync_artifacts=False)
            
            wrapped_trainable=tune.with_parameters(self.train_hpo,model_name=model_name,fold_idx=fold_idx,loss_type=self.loss_type,hpo_metric=self.hpo_metric,tokenized_train=tokenized_train,tokenized_dev=tokenized_dev,data_collator=data_collator,tokenizer=tokenizer)
            analysis = tune.run(
                wrapped_trainable,
                config=tune_config,
                sync_config=sync_config,
                scheduler=scheduler,
                search_alg=HyperOptSearch(metric=self.hpo_metric, mode="max", random_state_seed=CONFIG["seed"]),
                checkpoint_config=checkpoint_config,
                num_samples=self.n_trials,
                resources_per_trial={"cpu": 10, "gpu": 1},
                storage_path="/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/ray_results/",
                callbacks=[CleanupCallback(self.hpo_metric)]
            )
            logger.info(f"Analysis results: {analysis}")

            # Handle case where no trials succeeded
            best_trial = analysis.get_best_trial(metric=self.hpo_metric, mode="max")
            logger.info(f"Best trial : {best_trial}")
            if best_trial is None:
                logger.error("No successful trials found. Please check the training process and metric logging.")
                return None, None

            best_config = best_trial.config
            best_results = best_trial.last_result

            logger.info(f"Best config : {best_config}")
            logger.info(f"Best trial after optimization: {best_results}")

            plot_trial_performance(analysis,logger=logger,plot_dir=CONFIG['plot_dir'])
            
            
            #TODO : Check Julien's article about how to implement that (ask him about the threholding optimization)
            logger.info(f"Final training...")
            start_time=perf_counter()
            model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=CONFIG["num_labels"])
            #model.gradient_checkpointing_enable()

            # Set up training arguments
            training_args = CustomTrainingArguments(
                    output_dir="/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/models",
                    seed=CONFIG["seed"],
                    data_seed=CONFIG["seed"],
                    **CONFIG["default_training_args"],
                    loss_type=self.loss_type,
                    metric_for_best_model=self.hpo_metric,
                )
            
            training_args.pos_weight = best_config["pos_weight"] if self.loss_type == "BCE" else None
            training_args.alpha = best_config["alpha"] if self.loss_type == "focal" else None
            training_args.gamma = best_config["gamma"] if self.loss_type == "focal" else None
            training_args.learning_rate = best_config["learning_rate"]
            training_args.num_train_epochs = best_config["num_train_epochs"]
            

            training_args.gradient_accumulation_steps = best_config.get("gradient_accumulation_steps", 1)

            class CustomEarlyStoppingCallback(EarlyStoppingCallback):
                def on_train_end(self, args, state, control, **kwargs):
                    if state.best_model_checkpoint:
                        logger.info(f"Early stopping triggered. Best model checkpoint saved at: {state.best_model_checkpoint}")
                    else:
                        logger.info("Early stopping triggered, but no best model checkpoint was saved.")

            early_stopping_callback = CustomEarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01,
            )

            #TODO : Impelement early stopping !!
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_dev,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                callbacks=[early_stopping_callback],
            )

            logger.info(f"training size : {len(tokenized_train)}")
            logger.info(f"dev size : {len(tokenized_dev)}")
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
            

            final_model_path = os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/final_model", "best_model_cross_val_"+str(self.loss_type)+str(model_name)+str(self.n_trials)+"trials_fold-"+str(fold_idx+1))
            
            trainer.save_model(final_model_path)
            logger.info(f"Best model saved to {final_model_path}")

            results=[]
            logger.info(f"On test Set (with threshold 0.5) : ")
            # Compute detailed metrics
            predictions = trainer.predict(tokenized_test)
            
            scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
            preds = (scores > 0.5).astype(int)
            logger.info(f"preds : {preds}")
            logger.info(f'test_split["labels"] : {test_split["labels"]}')
            res1=detailed_metrics(preds, test_split["labels"],scores=scores)

            #We update the results dataframe
            self.result_metrics = pd.concat([
                self.result_metrics,
                pd.DataFrame([{
                    "model_name": model_name,
                    "loss_type": self.loss_type,
                    "fold": fold_idx+1,
                    "run": self.run_idx+1, 
                    "with_title":self.with_title,
                    "with_keywords":self.with_keywords,
                    **res1
                }])
            ])
            fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
            preds_df_list.append(fold_preds_df)
            
            self.store_metrics(self.result_metrics)

            plot_roc_curve(test_split["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
            plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

            #! The following seems weird. we are talking about decision here. View it like a ranking problem. take a perspective for usage
            threshold = eval_results_dev["eval_optim_threshold"]
            logger.info(f"\nOn test Set (optimal threshold of {threshold} according to cross validation on the training set): ")
            preds = (scores > threshold).astype(int)
            res2=detailed_metrics(preds, test_split["labels"],scores=scores)
            results.append(res2)
            plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

            logger.info(f"Results for fold {fold_idx+1} : {results}")
            test_metrics.append(results)
            logger.info(f"scores : {scores}")
            scores_by_fold.append(scores)

            torch.cuda.empty_cache()
            clear_cuda_cache()
        torch.cuda.empty_cache()
        clear_cuda_cache()
        pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/bert",f"{os.path.basename(model_name)}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_run-{self.run_idx}.csv"))

        return scores_by_fold

    #def metrics_analysis(self):

    def run_pipeline(self):
        """
        This function runs the entire pipeline for training and evaluating a model using cross-validation.
        It includes data loading, preprocessing, model training, and evaluation.
        """
        #TODO : compare the loss functions inside the pipeline function to have the same test set for each run

        if self.ensemble==False:
            scores_by_fold =self.train()
            clear_cuda_cache() 
            logger.info(f"Results: {self.result_metrics}")
            return self.result_metrics

        elif self.ensemble==True:
            logger.info(f"Ensemble learning pipeline")
            
            scores_by_model = []
            for i,model_name in enumerate(self.model_names):
                logger.info(f"Training model {i+1}/{len(self.model_names)}: {model_name}")
                scores_by_fold = self.train(model_name=model_name)

                torch.cuda.empty_cache()
                clear_cuda_cache() 

                # Ensure scores_by_fold is a consistent array before appending
                scores_by_fold = np.array(scores_by_fold, dtype=object)
                scores_by_model.append(scores_by_fold)
                logger.info(f"Metrics for {model_name}: {self.result_metrics[ (self.result_metrics['loss_type'] == self.loss_type) & (self.result_metrics['model_name'] == model_name)]}")

            avg_ensemble_metrics=self.ensemble_pred(scores_by_model)

            return avg_ensemble_metrics
        
    def ensemble_pred(self, scores_by_model):
        # Ensure scores_by_model is a valid numpy array
        scores_by_model = np.array(scores_by_model)
        if scores_by_model.ndim != 3:
            raise ValueError("scores_by_model must be a 3D array with shape (n_models, n_folds, n_samples).")

        logger.info(f"score by model shape (before ensembling): {scores_by_model.shape}")
        scores_by_fold = scores_by_model.transpose(1, 0, 2)  # Transpose to (n_folds, n_models, n_samples)

        for fold_idx in range(len(self.folds)):
            avg_models_scores = np.mean(scores_by_fold[fold_idx], axis=0)  # Average scores across models
            logger.info(f"\nfold number {fold_idx + 1} / {len(self.folds)}")

            _,_, test_indices = self.folds[fold_idx]
            test_split = self.dataset.select(test_indices)

            # Ensure avg_models_scores and test_split["labels"] have matching shapes
            if avg_models_scores.shape[0] != len(test_split["labels"]):
                raise ValueError("Mismatch between avg_models_scores and test_split labels.")

            preds = (avg_models_scores > 0.5).astype(int)
            result = detailed_metrics(preds, test_split["labels"],scores=avg_models_scores)

            # Update the results DataFrame
            self.result_metrics = pd.concat([
                self.result_metrics,
                pd.DataFrame([{
                    "model_name": "Ensemble",
                    "loss_type": self.loss_type,
                    "fold": fold_idx+1,
                    "run": self.run_idx,
                    "with_title":self.with_title,
                    "with_keywords":self.with_keywords,
                    **result
                }])
            ], ignore_index=True)
        self.store_metrics(self.result_metrics)
        
        # Group by relevant columns and calculate mean metrics
        avg_metrics = self.result_metrics.groupby(
            ["loss_type", "model_name", "run"]
        )[["f1", "recall", "precision", "accuracy"]].mean().reset_index()

        # Filter metrics for the current data_type and loss_type
        filtered_metrics = avg_metrics[
            (avg_metrics["model_name"] == "Ensemble") &
            (avg_metrics["loss_type"] == self.loss_type)
        ]

        return filtered_metrics

    def whole_pipeline(self):

        loss_type_list=["BCE","focal"]
        for loss_type in loss_type_list:
            self.loss_type=loss_type
            for self.run_idx in range(self.num_runs):
                self.folds=self.folds_per_run[self.run_idx]
                logger.info(f"fold indexes for run no.{self.run_idx+1}: {self.folds[0]}")
                logger.info(f"Run no {self.run_idx+1}/{self.num_runs}")
                logger.info(f"Loss Type : {loss_type}")
                avg_ens_metrics=self.run_pipeline()
                self.store_metrics(self.result_metrics)
            self.folds=[]

        return self.result_metrics

    def store_metrics(self,metric_df,path="/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/metrics",file_name="binary_metrics.csv"):
        if metric_df is not None:
            metric_df.to_csv(os.path.join(path, file_name))
        else:
            raise ValueError("result_metrics is None. Consider running the model before storing metrics.")
    

    def random_forest(self):
        for num_trees in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            for criterion in "log_loss","gini","entropy":
                for self.run_idx in range(self.num_runs):
                    preds_df_list=[]
                    self.folds=self.folds_per_run[self.run_idx]
                    for fold_idx in range(len(self.folds)):
                        logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
                        
                        train_indices, dev_indices,test_indices = self.folds[fold_idx]
                        train_split = self.dataset.select(train_indices)
                        dev_split = self.dataset.select(dev_indices)
                        test_split = self.dataset.select(test_indices)

                        logger.info(f"train split size : {len(train_split)}")
                        logger.info(f"dev split size : {len(dev_split)}")
                        logger.info(f"test split size : {len(test_split)}")

                         # Prepare features
                        if self.with_title:
                            if self.with_keywords:
                                x_train_df = train_split.select_columns(["title","abstract","Keywords"]).to_pandas()
                                x_train = (x_train_df["title"].astype(str) + " " + x_train_df["abstract"].astype(str)+ " " + x_train_df["Keywords"].astype(str)).values
                                x_test_df = test_split.select_columns(["title","abstract","Keywords"]).to_pandas()
                                x_test = (x_test_df["title"].astype(str) + " " + x_test_df["abstract"].astype(str)+ " " + x_test_df["Keywords"].astype(str)).values
                            else:
                                x_train_df = train_split.select_columns(["title","abstract"]).to_pandas()
                                x_train = (x_train_df["title"].astype(str) + " " + x_train_df["abstract"].astype(str)).values
                                x_test_df = test_split.select_columns(["title","abstract"]).to_pandas()
                                x_test = (x_test_df["title"].astype(str) + " " + x_test_df["abstract"].astype(str)).values
                        elif self.with_keywords:
                            x_train_df = train_split.select_columns(["abstract","Keywords"]).to_pandas()
                            x_train = (x_train_df["abstract"].astype(str) + " " + x_train_df["Keywords"].astype(str)).values
                            x_test_df = test_split.select_columns(["abstract","Keywords"]).to_pandas()
                            x_test = (x_test_df["abstract"].astype(str) + " " + x_test_df["Keywords"].astype(str)).values
                        else :
                            x_train = np.asarray(train_split["abstract"])
                            x_test = np.asarray(test_split["abstract"])
                        y_train = np.asarray(train_split["labels"])

                        # Fit TF-IDF vectorizer
                        vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1,2), stop_words="english")
                        x_train_vec = vectorizer.fit_transform(x_train)
                        x_test_vec = vectorizer.transform(x_test)

                        # Fit MultiOutputClassifier with RF
                        clf = RandomForestClassifier(n_estimators=num_trees, max_depth=None, n_jobs=-1, criterion=criterion, random_state=CONFIG["seed"])
        
                        logger.info(f"X shape : {x_train_vec.shape}")
                        logger.info(f"y shape : {y_train.shape}")
                        fitted_model = clf.fit(x_train_vec, y_train)

                        preds = fitted_model.predict(x_test_vec)
                        results=detailed_metrics(preds, np.asarray(test_split["labels"]))

                        self.random_forest_metrics = pd.concat([
                            self.random_forest_metrics,
                            pd.DataFrame([{
                                "model_name": "Random Forest",
                                "fold": fold_idx+1,
                                "criterion": criterion,
                                "num_trees": num_trees,
                                "run": self.run_idx, 
                                "with_title": self.with_title,
                                "with_keywords":self.with_keywords,
                                **results
                            }])
                        ])

                        fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                        preds_df_list.append(fold_preds_df)

                    self.store_metrics(self.random_forest_metrics,file_name="random_forest_metrics.csv")
                    pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/rf",f"rf_{criterion}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_{num_trees}_run-{self.run_idx}.csv"))



    def svm(self):
        for kernel in "linear","rbf","poly","sigmoid":
            for self.run_idx in range(self.num_runs):
                self.folds=self.folds_per_run[self.run_idx]
                logger.info(f"fold indexes for run no.{self.run_idx+1}: {self.folds[0]}")
                logger.info(f"Run no {self.run_idx+1}/{self.num_runs}")
                logger.info(f"Kernel : {kernel}")
                preds_df_list=[]
                for fold_idx in range(len(self.folds)):
                    logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
                    
                    train_indices, dev_indices,test_indices = self.folds[fold_idx]
                    train_split = self.dataset.select(train_indices)
                    dev_split = self.dataset.select(dev_indices)
                    test_split = self.dataset.select(test_indices)

                    logger.info(f"train split size : {len(train_split)}")
                    logger.info(f"dev split size : {len(dev_split)}")
                    logger.info(f"test split size : {len(test_split)}")

                        # Prepare features
                    if self.with_title:
                        if self.with_keywords:
                            x_train_df = train_split.select_columns(["title","abstract","Keywords"]).to_pandas()
                            x_train = (x_train_df["title"].astype(str) + " " + x_train_df["abstract"].astype(str)+ " " + x_train_df["Keywords"].astype(str)).values
                            x_test_df = test_split.select_columns(["title","abstract","Keywords"]).to_pandas()
                            x_test = (x_test_df["title"].astype(str) + " " + x_test_df["abstract"].astype(str)+ " " + x_test_df["Keywords"].astype(str)).values
                        else:
                            x_train_df = train_split.select_columns(["title","abstract"]).to_pandas()
                            x_train = (x_train_df["title"].astype(str) + " " + x_train_df["abstract"].astype(str)).values
                            x_test_df = test_split.select_columns(["title","abstract"]).to_pandas()
                            x_test = (x_test_df["title"].astype(str) + " " + x_test_df["abstract"].astype(str)).values
                    elif self.with_keywords:
                        x_train_df = train_split.select_columns(["abstract","Keywords"]).to_pandas()
                        x_train = (x_train_df["abstract"].astype(str) + " " + x_train_df["Keywords"].astype(str)).values
                        x_test_df = test_split.select_columns(["abstract","Keywords"]).to_pandas()
                        x_test = (x_test_df["abstract"].astype(str) + " " + x_test_df["Keywords"].astype(str)).values
                    else :
                        x_train = np.asarray(train_split["abstract"])
                        x_test = np.asarray(test_split["abstract"])
                    y_train = np.asarray(train_split["labels"])

                    # Fit TF-IDF vectorizer
                    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1,2), stop_words="english")
                    x_train_vec = vectorizer.fit_transform(x_train)
                    x_test_vec = vectorizer.transform(x_test)

                    # Fit MultiOutputClassifier with SVC
                    clf = SVC(kernel=kernel, probability=True)
                    logger.info(f"X shape : {x_train_vec.shape}")
                    logger.info(f"y shape : {y_train.shape}")
                    fitted_model = clf.fit(x_train_vec, y_train)

                    preds = fitted_model.predict(x_test_vec)
                    results = detailed_metrics(preds, np.asarray(test_split["labels"]))

                    logger.info(f"Concatenating...")
                    self.svm_metrics = pd.concat([
                        self.svm_metrics,
                        pd.DataFrame([{
                            "model_name": "SVM",
                            "kernel": kernel,
                            "fold": fold_idx+1,
                            "run": self.run_idx,
                            "with_title": self.with_title,
                            "with_keywords":self.with_keywords,
                            **results
                        }])
                    ])
                    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                    preds_df_list.append(fold_preds_df)
                self.store_metrics(self.svm_metrics, file_name="svm_metrics.csv")
                pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/svm", f"svm_{kernel}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_run-{self.run_idx}.csv"))

    def svm_bert(self, model_name):
        """
        Train an SVM model using BERT embeddings as features.

        Args:
            model_name (str): Name of the pre-trained BERT model to use for embeddings.
        """
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="sequence-classification").to(device)
        model.eval()  # Set the model to evaluation mode

        for kernel in "linear", "rbf", "poly", "sigmoid":
            for self.run_idx in range(self.num_runs):
                logger.info(f"Run no {self.run_idx+1}/{self.num_runs}")
                logger.info(f"Kernel : {kernel}")
                preds_df_list = []
                self.folds=self.folds_per_run[self.run_idx]
                for fold_idx in range(len(self.folds)):
                    logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
                    
                    train_indices, dev_indices,test_indices = self.folds[fold_idx]
                    train_split = self.dataset.select(train_indices)
                    dev_split = self.dataset.select(dev_indices)
                    test_split = self.dataset.select(test_indices)

                    logger.info(f"train split size : {len(train_split)}")
                    logger.info(f"dev split size : {len(dev_split)}")
                    logger.info(f"test split size : {len(test_split)}")
                    logger.info(f"train split size : {len(train_split)}")
                    logger.info(f"dev split size : {len(dev_split)}")
                                

                    logger.info("Initializing BERT pipeline...")
                    bert_pipeline = transformers.pipeline(
                        task="feature-extraction",
                        model=model_name,
                        tokenizer=tokenizer,
                        truncation=True,
                        padding=True,
                        device=2,
                        batch_size=40,
                    )

                    logger.info("Generating BERT embeddings for train and test sets...")

                    x_train = bert_pipeline(train_split["abstract"])

                    logger.info(f"x_train: {x_train}")

                    x_train = bert_pipeline(train_split["abstract"], return_tensors=True)
                    logger.info(f"x_train : {x_train}")

                    x_test = bert_pipeline(test_split["abstract"],  return_tensors=True)
                    y_train = np.asarray(train_split["labels"])

                    logger.info(f"Training Classifier with kernel: {kernel}...")
                    # Train SVM
                    clf = SVC(kernel=kernel, probability=True)
                    logger.info(f"X shape : {x_train.shape}")
                    logger.info(f"y shape : {np.asarray(y_train).shape}")
                    fitted_model = clf.fit(x_train, y_train)

                    preds = fitted_model.predict(x_test)
                    results = detailed_metrics(preds, np.asarray(test_split["labels"]))

                    logger.info(f"Concatenating results...")
                    self.svm_metrics = pd.concat([
                        self.svm_metrics,
                        pd.DataFrame([{
                            "model_name": f"SVM with {model_name}",
                            "kernel": kernel,
                            "fold": fold_idx,
                            "run": self.run_idx,
                            "with_title":self.with_title,
                            "with_keywords":self.with_keywords,
                            **results
                        }])
                    ])
                    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                    preds_df_list.append(fold_preds_df)

                self.store_metrics(self.svm_metrics, file_name=f"svm_bert_{model_name.replace('/', '_')}_metrics.csv")
                pd.concat(preds_df_list).to_csv(os.path.join(
                    "/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/svm_bert",
                    f"svm_bert_{model_name.replace('/', '_')}_{kernel}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_run-{self.run_idx}.csv"
                ))

    def find_best(self,metrics_df,metric):
        metrics_df[metrics_df[metric]==metrics_df[metric].max()]

    def zero_shot(self):
        """
        Perform zero-shot classification using a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for zero-shot classification.
        """
        scores_by_model=[]
        for model_name in self.model_names:
            logger.info(f"Running zero-shot classification with model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="sequence-classification")

            # Initialize the zero-shot classification pipeline
            zero_shot_pipeline = transformers.pipeline(
                task="zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            for fold_idx in range(len(self.folds)):
                logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
                
                train_indices, dev_indices,test_indices = self.folds[fold_idx]
                train_split = self.dataset.select(train_indices)
                dev_split = self.dataset.select(dev_indices)
                test_split = self.dataset.select(test_indices)

                logger.info(f"train split size : {len(train_split)}")
                logger.info(f"dev split size : {len(dev_split)}")
                logger.info(f"test split size : {len(test_split)}")

                # Prepare abstract for zero-shot classification
                texts = test_split["abstract"]
                labels = ["relevant","not relevant"]

                # Perform zero-shot classification
                results = zero_shot_pipeline(texts, candidate_labels=labels, multi_label=True)

                # Process results and store metrics
                preds = np.array([result['scores'] for result in results])
                preds_df = pd.DataFrame(preds, columns=labels)
                preds_df["fold"] = fold_idx

                detailed_metrics = detailed_metrics(preds, np.asarray(test_split["labels"]))
                logger.info(f"Detailed metrics for fold {fold_idx+1}: {detailed_metrics}")

                # Save predictions
                preds_df.to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds", f"zero_shot_{model_name.replace('/', '_')}_fold-{fold_idx}.csv"), index=False)
                scores_by_model.append(preds_df)
        logger.info(f"Zero-shot ensemble classification : {self.ensemble_pred(scores_by_model=scores_by_model, zero_shot=True)}")