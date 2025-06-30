#Hyperparameters optimization script
#This include reading for agiven fold number, the train, dev and test set from the data directory

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
import datasets
from src.utils import *
import sys
import datasets
from time import perf_counter
from .model_init import *
from time import perf_counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from .HPO_callbacks import CleanupCallback
from src.data_pipeline.ipbes import data_pipeline
import pandas as pd

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
#TODO : Ask julien about multlabelstratifiedsplit
#TODO : Consider Merging this file into the original binary pipeline
#TODO : Check if it is relevant to use the weight attribute for BCEwithlogitloss in CustomTraier -> see model_init
#TODO : Check that we have same folds for different runs -> check if see avoid total randomness in folding

logger = logging.getLogger(__name__)

def __init__(self,loss_type="BCE",ensemble=False,balance_coeff=5,n_trials=10,num_runs=5,with_title=False,n_fold=5,model_names = ["FacebookAI/roberta-base","microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "dmis-lab/biobert-v1.1", "google-bert/bert-base-uncased"]):
    self.loss_type=loss_type
    self.ensemble=ensemble
    self.with_title=with_title
    self.n_fold=n_fold
    self.n_trials=n_trials
    self.result_metrics=pd.DataFrame(columns=["model_name", "fold", "data_type", "loss_type", "run","with_title","with_head"])
    self.svm_metrics=pd.DataFrame(columns=["model_name", "fold", "data_type", "kernel", "run","with_title"])
    self.random_forest_metrics=pd.DataFrame(columns=["model_name", "fold", "data_type", "criterion","num_trees","run","with_title"])
    self.folds=list()
    self.model_names=model_names
    self.run_idx=None
    self.num_runs=num_runs
    self.labels=["IAS","SUA","VA"]
    self.balance_coeff=balance_coeff
    self.dataset=self.data_loading()
    self.store_test_folds()

    logger.info(f"Multi Label pipeline for loss type {self.loss_type} and ensemble={self.ensemble}")

def _balance_dataset(self,dataset):
    """
    - Performs undersampling on the negatives
    - Renames the abstract column -> we should not hqve to do that 
    """
    labels=["IAS","SUA","VA"]

    
    def is_pos(batch):
        batch_bools=[False for _ in range(len(batch[labels[0]]))]
        for label in labels:
            for idx in range(len(batch[label])):
                if batch[label][idx] == 1:
                    batch_bools[idx]=True
        return batch_bools
    
    def is_neg(batch):
        batch_bools=[True for _ in range(len(batch[labels[0]]))]
        for label in labels:
            for idx in range(len(batch[label])):
                if batch[label][idx] == 1:
                    batch_bools[idx]=False
        return batch_bools

    pos = dataset.filter(lambda x : is_pos(x), batched=True, batch_size=1000, num_proc=os.cpu_count())
    neg = dataset.filter(lambda x : is_neg(x), batched=True, batch_size=1000, num_proc=os.cpu_count())
    logger.info(f"Number of positives: {len(pos)}")
    logger.info(f"Number of negatives: {len(neg)}")
    num_pos = len(pos)

    # Ensure there are more negatives than positives before subsampling
    if len(neg) > num_pos:
        neg_subset_train = neg.shuffle(seed=42).select(range(self.balance_coeff*num_pos))
    else:
        neg_subset_train = neg  # Fallback (unlikely in your case)

    balanced_ds = datasets.concatenate_datasets([pos, neg_subset_train])
    balanced_ds = balanced_ds.shuffle(seed=42)  # Final shuffle

    balanced_ds = balanced_ds.rename_column("abstract", "text")
    logger.info(f"Balanced columns: {balanced_ds.column_names}")
    logger.info(f"Balanced dataset size: {len(balanced_ds)}")

    return balanced_ds

def data_loading(self):
    set_random_seeds(CONFIG["seed"])

    unbalanced_dataset = data_pipeline(multi_label=True)

    # ? When and what should we balance ? 
    # the when depends on what. If only training is balances then it should be done inside the optimization_cros_val function
    #TODO : even if done on the whole dataset, we can move it into optimization_cros_val
    logger.info(f"Balancing dataset...")
    dataset = self._balance_dataset(unbalanced_dataset)

    logger.info(f"Balanced dataset : {dataset}")

    #First we do k-fold cross validation for testing
    #TODO : ensure that this does not change when running this pipeline different times for comparisons purposes
    mskf = MultilabelStratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=CONFIG["seed"])
    logger.info(f"dataset's labels : {dataset.select_columns(self.labels)}")
    if self.with_title:
        self.folds = list(mskf.split(dataset.select_columns(['text','title']).to_pandas(), dataset.select_columns(self.labels).to_pandas()))
        logging.info(f"fold 1 : {self.folds[0]}")

    else:
        self.folds = list(mskf.split(dataset['text'], dataset.select_columns(self.labels).to_pandas()))
        logging.info(f"fold 1 : {self.folds[0]}")
    return dataset


def fine_tune(self,model_name):
    """Fine-tune a pre-trained model with optimized loss parameters.

    Args:
        train_ds
        val_ds
        test_ds
        seed_set: Whether to set random seeds for reproducibility.
        loss_type: Type of loss function ("BCE" or "focal").
        model_name: Name of the pre-trained model to use.

    Returns:
        Classification score of the model by fold
    """

    #? When should we tokenize ? 
    #TODO : See how tokenizing is done to check if it alrgiht like this -> ask julien if that's ok
    #It can be a problem since we are truncating
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_metrics=[]
    scores_by_fold=[]
    scores_df_list=[]
    for fold_idx in range(len(self.folds)):
        logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
        
        train_dev_indices, test_indices = self.folds[fold_idx]
        train_dev_split = self.dataset.select(train_dev_indices)
        test_split = self.dataset.select(test_indices)

        logger.info(f"train+dev split size : {len(train_dev_split)}")
        logger.info(f"test split size : {len(test_split)}")

        # Train/dev multi-label stratified split :
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=0.3,    # 30% → dev set
            train_size=0.7,   # 70% → train set
            random_state=CONFIG["seed"]
        )

        # get train/dev indices, stratified on the labels
        train_idx, dev_idx = next(msss.split(np.arange(len(train_dev_split)),train_dev_split.select_columns(self.labels).to_pandas()))

        train_split = train_dev_split.select(train_idx)
        dev_split  = train_dev_split.select(dev_idx)

        logger.info(f"train split size : {len(train_split)}")
        logger.info(f"dev split size : {len(dev_split)}")

        def preprocess(batch):
            # join title & text, tokenize
            if self.with_title:
                enc = tokenizer(batch["title"], batch["text"], truncation=True, max_length=512)
            else:
                enc = tokenizer(batch["text"], truncation=True, max_length=512)
            # stack the 3 label columns into a single multi-hot vector
            enc["labels"] = [
                [i, s, v] for i, s, v in zip(batch["IAS"], batch["SUA"], batch["VA"]) #TODO: use self.labels
            ]
            return enc

        tokenized_train = train_split.map(preprocess, batched=True, remove_columns=train_split.column_names)
        tokenized_dev = dev_split.map(preprocess, batched=True, remove_columns=dev_split.column_names)
        tokenized_test = test_split.map(preprocess, batched=True, remove_columns=test_split.column_names)

        #We whould maybe perform cross val inside the model names loops ?
        #1. We run all models for each fold and we take the average of all of them at the end -> I think it is not good this way
        #2. (Inside loops) We go through all folds for each run and compare the means

        def train_hpo(config):
            
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.labels),problem_type="multi_label_classification")
            
            if torch.cuda.current_device()==2:
                batch_size=70
            else:
                batch_size=27


            # Set up training arguments
            training_args = CustomTrainingArguments(
                output_dir=f'.', #! Is that alright ?
                seed=CONFIG["seed"],
                data_seed=CONFIG["seed"],
                **CONFIG["default_training_args"],
                loss_type=self.loss_type,
                pos_weight=config["pos_weight"] if self.loss_type=="BCE" else None,
                alpha=config["alpha"] if self.loss_type=="focal" else None,
                gamma=config["gamma"]if self.loss_type=="focal" else None,
                weight_decay=config["weight_decay"],
                disable_tqdm=True,
                logging_dir=f'./logs_fold_{fold_idx}',
                multi_label=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                
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
                compute_metrics=multi_label_compute_metrics,
                tokenizer=tokenizer,
            )

            trainer.train()
            eval_result = trainer.evaluate()
            logger.info(f"eval_result: {eval_result}")

            torch.cuda.empty_cache()

            return eval_result
        
        # Define hyperparameter search space based on loss_type
        if self.loss_type == "BCE":
            tune_config = {
                "pos_weight": tune.uniform(1.0,10.0),
                "learning_rate": tune.loguniform(1e-6, 1e-4),
                #"gradient_accumulation_steps": tune.choice([2,4,8]),
                "weight_decay": tune.uniform(0.0, 0.3),
                "num_train_epochs": tune.choice([2, 3, 4,5]),
                }  # Tune pos_weight for BCE
        elif self.loss_type == "focal":
            tune_config = {
                "alpha": tune.uniform(0.5, 1.0),  # Tune alpha for focal loss
                "gamma": tune.uniform(2.0, 10.0),   # Tune gamma for focal loss
                "learning_rate": tune.loguniform(1e-6, 1e-4),
                #"gradient_accumulation_steps": tune.choice([2,4,8]),
                "weight_decay": tune.uniform(0.0, 0.3),
                "num_train_epochs": tune.choice([2, 3, 4,5]),
                }
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # Set up scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="eval_f1_weighted", #When set to objective, it takes the sum of the compute-metric output. if compute-metric isnt defined, it takes the loss.
            mode="max"
        )
        
        # Perform hyperparameter search
        logger.info(f"Starting hyperparameter search for {self.loss_type} loss")

        checkpoint_config = tune.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False)

        analysis = tune.run(
            train_hpo,
            config=tune_config,
            scheduler=scheduler,
            search_alg=HyperOptSearch(metric="eval_f1_weighted", mode="max", random_state_seed=CONFIG["seed"]),
            checkpoint_config=checkpoint_config,
            num_samples=self.n_trials,
            resources_per_trial={"cpu": 10, "gpu": 1},
            storage_path="/home/leandre/Projects/BioMoQA_Playground/model/ray_results/",
            callbacks=[CleanupCallback()],
        )
        logger.info(f"Analysis results: {analysis}")

        # Handle case where no trials succeeded
        best_trial = analysis.get_best_trial(metric="eval_f1_weighted", mode="max")
        logger.info(f"Best trial : {best_trial}")
        if best_trial is None:
            logger.error("No successful trials found. Please check the training process and metric logging.")
            return None, None

        best_config = best_trial.config
        best_results = best_trial.last_result

        logger.info(f"Best config : {best_config}")
        logger.info(f"Best trial after optimization: {best_results}")

        """
        try :
            visualize_ray_tune_results(analysis, logger, plot_dir=CONFIG['plot_dir'])
        except Exception as e:
            logger.error(f"Error visualizing Ray Tune results: {e}")
            logger.info("Skipping visualization due to error.")
        else:
            continue
        """

        plot_trial_performance(analysis,logger=logger,plot_dir=CONFIG['plot_dir'])

        #TODO : Perform ensemble learning with 5/10 independent models by looping here, then take the majority vote for each test instance. 
        #TODO : Check how to do ensemble learning with transformers before
        #TODO : Check Julien's article about how to implement that (ask him about the threholding optimization)
        logger.info(f"Final training...")
        start_time=perf_counter()
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.labels),problem_type="multi_label_classification")
        #model.gradient_checkpointing_enable()

        output_dir=os.path.join(CONFIG["output_dir"], self.loss_type  + "_" + str((fold_idx+1)))

        # Set up training arguments
        training_args = CustomTrainingArguments(
                output_dir=output_dir,
                seed=CONFIG["seed"],
                data_seed=CONFIG["seed"],
                **CONFIG["default_training_args"],
                loss_type=self.loss_type,
                multi_label=True,
            )
        if self.loss_type == "BCE":
            training_args.pos_weight = best_config["pos_weight"]
        if self.loss_type == "focal":
            training_args.alpha = best_config["alpha"]
            training_args.gamma = best_config["gamma"]
        training_args.learning_rate = best_config["learning_rate"]
        training_args.num_train_epochs = best_config["num_train_epochs"]
        training_args.weight_decay = best_config["weight_decay"]
        training_args.gradient_accumulation_steps = best_config.get("gradient_accumulation_steps", 1)

        early_stopping_callback = EarlyStoppingCallback(
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
            compute_metrics=multi_label_compute_metrics,
            tokenizer=tokenizer,
            callbacks=[early_stopping_callback],
        )

        logger.info(f"training size : {len(tokenized_train)}")
        logger.info(f"test size : {len(tokenized_test)}")

        trainer.train()

        #TODO : Early stopping here ?
        eval_results_dev=trainer.evaluate()

        end_time_train=perf_counter()
        logger.info(f"Training time : {end_time_train-start_time}")

        eval_results_test = trainer.evaluate(tokenized_test)

        end_time_val=perf_counter()
        logger.info(f"Evaluation time : {end_time_val-end_time_train}")
        logger.info(f"Evaluation results on test set: {eval_results_test}")
        # Number of optimizer updates performed:

        final_model_path = os.path.join(CONFIG["final_model_dir"], "best_model_cross_val_"+str(self.loss_type)+str(model_name)+str(self.n_trials)+"trials_fold-"+str(fold_idx+1))
        
        trainer.save_model(final_model_path)
        logger.info(f"Best model saved to {final_model_path}")

        results=[]
        logger.info(f"On test Set (with threshold 0.5) : ")
        # Compute detailed metrics
        pred_res = trainer.predict(tokenized_test)

        scores = 1 / (1 + np.exp(-pred_res.predictions.squeeze()))

        logger.info(f"Scores shape : {scores.shape}")
        fold_scores_df=pd.DataFrame(scores, columns=["IAS_scores", "SUA_scores", "VA_scores"])
        fold_scores_df["fold"]=[fold_idx for _ in range(len(fold_scores_df))]

        logger.info(f"Fold scores DataFrame shape: {fold_scores_df.shape}")
        scores_df_list.append(fold_scores_df)

        preds = (scores > 0.5).astype(int)
        res1=self.multi_label_detailed_metrics(preds, test_split.select_columns(self.labels).to_pandas().to_numpy())
        results.append(res1)
        #We update the results dataframe
        self.result_metrics = pd.concat([
            self.result_metrics,
            pd.DataFrame([{
                "model_name": model_name,
                "data_type": "MultiLabel",
                "loss_type": self.loss_type,
                "fold": fold_idx,
                "run": self.run_idx,
                "with_title": self.with_title,
                "with_head": True,
                **res1
            }])]
        )

        #TODO: Implement the follwing for multi-label (like the covid paper)
        #plot_roc_curve(test_split["labels"],scores,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")
        #plot_precision_recall_curve(test_split["labels"],preds,logger=logger,plot_dir=CONFIG["plot_dir"],data_type="test")

        logger.info(f"Results for fold {fold_idx+1} : {results}")
        test_metrics.append(results)
        scores_by_fold.append(scores)

        torch.cuda.empty_cache()
    if scores_df_list:
        logger.info(f"Scores DataFrame shape: {pd.concat(scores_df_list).shape}")
        pd.concat(scores_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/test_preds", f"{model_name}_loss-{self.loss_type}{'_with_title' if self.with_title else ''}_run-{self.run_idx}.csv"))
    else:
        logger.warning(f"No scores to save for model {model_name} with loss type {self.loss_type} and run {self.run_idx}.")

    torch.cuda.empty_cache()
    #TODO : Get the average metrics from the cross validation

    return scores_by_fold

def run_pipeline(self):
    """
    This function runs the entire pipeline for training and evaluating a model using cross-validation.
    It includes data loading, preprocessing, model training, and evaluation.
    """
    #TODO : compare the loss functions inside the pipeline function to have the same test set for each run

    if self.ensemble==False:
        scores_by_model = []
        for i,model_name in enumerate(self.model_names):
            scores_by_fold =self.fine_tune(model_name=model_name)
            torch.cuda.empty_cache()
            clear_cuda_cache() 

            scores_by_model.append(scores_by_fold)
            logger.info(f"Metrics for {model_name}: {self.result_metrics[ (self.result_metrics['loss_type'] == self.loss_type) & (self.result_metrics['model_name'] == model_name)]}")

        return self.result_metrics

    elif self.ensemble==True:
        logger.info(f"Ensemble learning pipeline")
        
        scores_by_model = []
        for i,model_name in enumerate(self.model_names):
            logger.info(f"Training model {i+1}/{len(self.model_names)}: {model_name}")
            scores_by_fold = self.fine_tune(model_name=model_name)


            torch.cuda.empty_cache()
            clear_cuda_cache() 

            scores_by_model.append(scores_by_fold)
            logger.info(f"Metrics for {model_name}: {self.result_metrics[ (self.result_metrics['loss_type'] == self.loss_type) & (self.result_metrics['model_name'] == model_name)]}")
            self.store_metrics(self.result_metrics)

        avg_ensemble_metrics=self.ensemble_pred(scores_by_model)
        self.store_metrics(self.result_metrics)

        return avg_ensemble_metrics

    else:
        #TODO : use the typing instead of this
        raise ValueError("Invalid value for ensemble. It must be True or False.")
    
def ensemble_pred(self,scores_by_model,zero_shot=False):
    """
    Ensemble the predictions from multiple models by averaging their scores.
    Args:
        scores_by_model (List[List[float]]): List of scores from each model for each fold.
    """
    #Then we can take the majority vote
    logger.info(f"score by model shape (before ensembling): {np.array(scores_by_model).shape}")
    scores_by_model=np.array(scores_by_model)
    scores_by_fold= scores_by_model.transpose(1, 0, 2, 3)
    scores_df_list=[]
    metrics_by_fold=[]
    for fold_idx in range(len(self.folds)):
        avg_models_scores=np.mean(scores_by_fold[fold_idx],axis=0)
        logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
        
        _, test_indices = self.folds[fold_idx]
        test_split = self.dataset.select(test_indices)

        preds = (avg_models_scores > 0.5).astype(int)
        result=self.multi_label_detailed_metrics(preds, test_split.select_columns(self.labels).to_pandas().to_numpy())
        metrics_by_fold.append(result)
        if not zero_shot:
            #We update the results dataframe
            self.result_metrics = pd.concat([
                self.result_metrics,
                pd.DataFrame([{
                    "model_name": "Ensemble",
                    "data_type": "MultiLabel",
                    "loss_type": self.loss_type,
                    "fold": fold_idx,
                    "run": self.run_idx,
                    "with_title": self.with_title,
                    "with_head": True,
                    **result
                }])
            ])
        fold_scores_df=pd.DataFrame(avg_models_scores, columns=["IAS_scores", "SUA_scores", "VA_scores"])
        fold_scores_df["fold"]=[fold_idx for _ in range(len(fold_scores_df))]
        scores_df_list.append(fold_scores_df)
    if not zero_shot:
        pd.concat(scores_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/test_preds", f"ensemble_loss-{self.loss_type}{'_with_title' if self.with_title else ''}.csv"))
    else:
        pd.concat(scores_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/test_preds", f"zero_shot_ensemble{'_with_title' if self.with_title else ''}.csv"))
        # Group by relevant columns and calculate mean metrics
    avg_metrics = self.result_metrics.groupby(
        ["data_type", "loss_type", "model_name", "run"]
    )[[key for key in metrics_by_fold[0]]].mean().reset_index()

    # Filter metrics for the current data_type and loss_type
    filtered_metrics = avg_metrics[
        (avg_metrics["model_name"]=="Ensemble") &
        (avg_metrics["data_type"] == None) &
        (avg_metrics["loss_type"] == self.loss_type)
    ]

    return filtered_metrics

def whole_pipeline(self):

    loss_type_list=["BCE","focal"]

    for loss_type in loss_type_list:
        self.loss_type=loss_type
        for i in range(self.num_runs):
            self.run_idx=i+1
            logger.info(f"Run no {i+1}/{self.num_runs}")
            logger.info(f"Loss Type : {loss_type}")
            avg_ens_metrics=self.run_pipeline()
            logger.info(f"Ensemble metrics for run no. {i+1}: {avg_ens_metrics}")
            logger.info(f"Metrics dataframe for run no. {i+1}: {self.dataset}")
    return self.result_metrics

def store_metrics(self,metric_df,path="/home/leandre/Projects/BioMoQA_Playground/results/metrics",file_name="multi_labels_metrics.csv"):
    if metric_df is not None:
        metric_df.to_csv(os.path.join(path, file_name))
    else:
        raise ValueError("result_metrics is None. Consider running the model before storing metrics.")

def multi_label_detailed_metrics(self,predictions: np.ndarray, labels_val: np.ndarray) -> Dict[str, float]:
    """Compute and display detailed metrics including confusion matrix."""
    logger.info(f"label vals shape : {labels_val.shape}")
    logger.info(f"predictions vals shape : {predictions.shape}")
    cm = confusion_matrix(np.asarray(labels_val).argmax(axis=1), np.asarray(predictions).argmax(axis=1))

    disp = ConfusionMatrixDisplay(cm, display_labels=["IAS","SUA","VA"])
    disp.plot()
    plt.savefig(os.path.join(CONFIG["plot_dir"], "confusion_matrix.png"))
    plt.close()

    f1_per_label=list(f1_score(labels_val,predictions,average=None))
    recall_per_label=list(recall_score(labels_val,predictions,average=None))
    precision_per_label=list(precision_score(labels_val,predictions,average=None))

    metrics = {"f1_IAS":f1_per_label[0],"f1_SUA":f1_per_label[1],"f1_VA":f1_per_label[2],"f1_weighted":f1_score(labels_val,predictions,average="weighted"),"f1_macro":f1_score(labels_val,predictions,average="macro"),"f1_micro":f1_score(labels_val,predictions,average="micro"),
        "recall_IAS":recall_per_label[0],"recall_SUA":recall_per_label[1],"recall_VA":recall_per_label[2],"recall_weighted":recall_score(labels_val,predictions,average="weighted"),"recall_macro":recall_score(labels_val,predictions,average="macro"),"recall_micro":recall_score(labels_val,predictions,average="micro"),
        "precision_IAS":precision_per_label[0],"precision_SUA":precision_per_label[1],"precision_VA":precision_per_label[2],"precision_weighted":precision_score(labels_val,predictions,average="weighted"),"precision_macro":precision_score(labels_val,predictions,average="macro"),"precision_micro":precision_score(labels_val,predictions,average="micro"),
        "accuracy":accuracy_score(labels_val,predictions)
    }
    metrics={key : float(val) for key,val in metrics.items()}
    
    logger.info(f"Metrics: {metrics}")
    return metrics

def store_test_folds(self,path="/home/leandre/Projects/BioMoQA_Playground/results/test preds",file_name="test_folds.csv"):
    test_split_list=[]
    for fold_idx in range(len(self.folds)):
        logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
        
        _, test_indices = self.folds[fold_idx]
        test_split = self.dataset.select(test_indices)
        test_split.add_column("fold",[fold_idx for _ in range(len(test_split))])
        test_split_list.append(test_split)
    datasets.concatenate_datasets(test_split_list).to_csv(os.path.join(path, file_name))

def random_forest(self):
    for num_trees in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        for criterion in "log_loss","gini","entropy":
            for self.run_idx in range(self.num_runs):
                preds_df_list=[]
                for fold_idx in range(len(self.folds)):
                    logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
                    
                    train_dev_indices, test_indices = self.folds[fold_idx]
                    train_dev_split = self.dataset.select(train_dev_indices)
                    test_split = self.dataset.select(test_indices)

                    logger.info(f"train+dev split size : {len(train_dev_split)}")
                    logger.info(f"test split size : {len(test_split)}")

                    # Train/dev multi-label stratified split :
                    msss = MultilabelStratifiedShuffleSplit(
                        n_splits=1,
                        test_size=0.3,    # 30% → dev set
                        train_size=0.7,   # 70% → train set
                        random_state=CONFIG["seed"]
                    )

                    # get train/dev indices, stratified on the labels
                    train_idx, dev_idx = next(msss.split(np.arange(len(train_dev_split)),train_dev_split.select_columns(self.labels).to_pandas()))

                    train_split = train_dev_split.select(train_idx)
                    dev_split  = train_dev_split.select(dev_idx)

                        # Prepare features
                    if self.with_title:
                        x_train_df = train_split.select_columns(["title","text"]).to_pandas()
                        x_train = (x_train_df["title"].astype(str) + " " + x_train_df["text"].astype(str)).values
                        x_test_df = test_split.select_columns(["title","text"]).to_pandas()
                        x_test = (x_test_df["title"].astype(str) + " " + x_test_df["text"].astype(str)).values
                    else:
                        x_train = np.asarray(train_split["text"])
                        x_test = np.asarray(test_split["text"])
                    y_train = train_split.select_columns(self.labels).to_pandas().to_numpy()

                    # Fit TF-IDF vectorizer
                    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1,2), stop_words="english")
                    x_train_vec = vectorizer.fit_transform(x_train)
                    x_test_vec = vectorizer.transform(x_test)

                    # Fit MultiOutputClassifier with RF
                    clf =  MultiOutputClassifier(
                                    RandomForestClassifier(n_estimators=num_trees,
                                                        max_depth=None,
                                                        n_jobs=-1,
                                                        criterion=criterion,
                                                        random_state=CONFIG["seed"]),
                            )
                    logger.info(f"X shape : {x_train_vec.shape}")
                    logger.info(f"y shape : {y_train.shape}")
                    fitted_model = clf.fit(x_train_vec, y_train)

                    preds = fitted_model.predict(x_test_vec)
                    results=self.multi_label_detailed_metrics(preds, test_split.select_columns(self.labels).to_pandas().to_numpy())

                    self.random_forest_metrics = pd.concat([
                        self.random_forest_metrics,
                        pd.DataFrame([{
                            "model_name": "Random Forest",
                            "data_type": "MultiLabel",
                            "fold": fold_idx,
                            "criterion": criterion,
                            "num_trees": num_trees,
                            "run": self.run_idx, 
                            "with_title": self.with_title,
                            **results
                        }])
                    ])
                    
                    fold_preds_df=pd.DataFrame(preds, columns=["IAS_scores", "SUA_scores", "VA_scores"])
                    fold_preds_df["fold"]=[fold_idx for _ in range(len(fold_preds_df))]
                    preds_df_list.append(fold_preds_df)

                self.store_metrics(self.random_forest_metrics,file_name="random_forest_metrics.csv")
                pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/test preds",f"rf_{criterion}{'_with_title' if self.with_title else ''}_{num_trees}_run-{self.run_idx}.csv"))



def svm(self):
    for kernel in "linear","rbf","poly","sigmoid":
        for self.run_idx in range(self.num_runs):
            logger.info(f"Run no {self.run_idx+1}/{self.num_runs}")
            logger.info(f"Kernel : {kernel}")
            preds_df_list=[]
            for fold_idx in range(len(self.folds)):
                logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
                
                train_dev_indices, test_indices = self.folds[fold_idx]
                train_dev_split = self.dataset.select(train_dev_indices)
                test_split = self.dataset.select(test_indices)

                logger.info(f"train+dev split size : {len(train_dev_split)}")
                logger.info(f"test split size : {len(test_split)}")

                # Train/dev multi-label stratified split :
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.3,    # 30% → dev set
                    train_size=0.7,   # 70% → train set
                    random_state=CONFIG["seed"]
                )

                # get train/dev indices, stratified on the labels
                train_idx, dev_idx = next(msss.split(np.arange(len(train_dev_split)),train_dev_split.select_columns(self.labels).to_pandas()))

                train_split = train_dev_split.select(train_idx)
                dev_split  = train_dev_split.select(dev_idx)

                # Prepare features
                if self.with_title:
                    x_train_df = train_split.select_columns(["title","text"]).to_pandas()
                    x_train = (x_train_df["title"].astype(str) + " " + x_train_df["text"].astype(str)).values
                    x_test_df = test_split.select_columns(["title","text"]).to_pandas()
                    x_test = (x_test_df["title"].astype(str) + " " + x_test_df["text"].astype(str)).values
                else:
                    x_train = np.asarray(train_split["text"])
                    x_test = np.asarray(test_split["text"])
                y_train = train_split.select_columns(self.labels).to_pandas().to_numpy()

                # Fit TF-IDF vectorizer
                vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1,2), stop_words="english")
                x_train_vec = vectorizer.fit_transform(x_train)
                x_test_vec = vectorizer.transform(x_test)

                # Fit MultiOutputClassifier with SVC
                clf = MultiOutputClassifier(SVC(kernel=kernel, probability=True))
                logger.info(f"X shape : {x_train_vec.shape}")
                logger.info(f"y shape : {y_train.shape}")
                fitted_model = clf.fit(x_train_vec, y_train)

                preds = fitted_model.predict(x_test_vec)
                results = self.multi_label_detailed_metrics(preds, test_split.select_columns(self.labels).to_pandas().to_numpy())

                logger.info(f"Concatenating...")
                self.svm_metrics = pd.concat([
                    self.svm_metrics,
                    pd.DataFrame([{
                        "model_name": "SVM",
                        "data_type": "MultiLabel",
                        "kernel": kernel,
                        "fold": fold_idx,
                        "run": self.run_idx,
                        "with_title": self.with_title,
                        **results
                    }])
                ])
                fold_preds_df = pd.DataFrame(preds, columns=["IAS_scores", "SUA_scores", "VA_scores"])
                fold_preds_df["fold"] = [fold_idx for _ in range(len(fold_preds_df))]
                preds_df_list.append(fold_preds_df)
            self.store_metrics(self.svm_metrics, file_name="svm_metrics.csv")
            pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/test preds", f"svm_{kernel}{'_with_title' if self.with_title else ''}_run-{self.run_idx}.csv"))

def svm_bert(self, model_name):
    """
    Train an SVM model using BERT embeddings as features.

    Args:
        model_name (str): Name of the pre-trained BERT model to use for embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.labels), problem_type="multi_label_classification")
    model.eval()  # Set the model to evaluation mode

    for kernel in "linear", "rbf", "poly", "sigmoid":
        for self.run_idx in range(self.num_runs):
            logger.info(f"Run no {self.run_idx+1}/{self.num_runs}")
            logger.info(f"Kernel : {kernel}")
            preds_df_list = []

            for fold_idx in range(len(self.folds)):
                logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")

                train_dev_indices, test_indices = self.folds[fold_idx]
                train_dev_split = self.dataset.select(train_dev_indices)
                test_split = self.dataset.select(test_indices)

                logger.info(f"train+dev split size : {len(train_dev_split)}")
                logger.info(f"test split size : {len(test_split)}")

                # Train/dev multi-label stratified split
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.3,  # 30% → dev set
                    train_size=0.7,  # 70% → train set
                    random_state=CONFIG["seed"]
                )

                train_idx, dev_idx = next(msss.split(
                    np.arange(len(train_dev_split)),
                    train_dev_split.select_columns(self.labels).to_pandas()
                ))

                train_split = train_dev_split.select(train_idx)
                dev_split = train_dev_split.select(dev_idx)
                    

                logger.info("Initializing BERT pipeline...")
                bert_pipeline = transformers.pipeline(
                    task="feature-extraction",
                    model=model_name,
                    tokenizer=tokenizer,
                    truncation=True,
                    padding=True,
                )

                logger.info("Generating BERT embeddings for train and test sets...")

                x_train = bert_pipeline(train_split["text"])

                logger.info(f"x_train: {x_train}")

                x_train = bert_pipeline(train_split["text"], return_tensors=True)
                logger.info(f"x_train : {x_train}")

                x_test = bert_pipeline(test_split["text"],  return_tensors=True)
                y_train = train_split.select_columns(self.labels).to_pandas().to_numpy()

                logger.info(f"Training Classifier with kernel: {kernel}...")
                # Train SVM
                clf = MultiOutputClassifier(SVC(kernel=kernel, probability=True))
                logger.info(f"X shape : {x_train.shape}")
                logger.info(f"y shape : {np.asarray(y_train).shape}")
                fitted_model = clf.fit(x_train, y_train)

                preds = fitted_model.predict(x_test)
                results = self.multi_label_detailed_metrics(preds, test_split.select_columns(self.labels).to_pandas().to_numpy())

                logger.info(f"Concatenating results...")
                self.svm_metrics = pd.concat([
                    self.svm_metrics,
                    pd.DataFrame([{
                        "model_name": f"SVM with {model_name}",
                        "data_type": "MultiLabel",
                        "kernel": kernel,
                        "fold": fold_idx,
                        "run": self.run_idx,
                        "with_title": self.with_title,
                        **results
                    }])
                ])
                fold_preds_df = pd.DataFrame(preds, columns=["IAS_scores", "SUA_scores", "VA_scores"])
                fold_preds_df["fold"] = [fold_idx for _ in range(len(fold_preds_df))]
                preds_df_list.append(fold_preds_df)

            self.store_metrics(self.svm_metrics, file_name=f"svm_bert_{model_name.replace('/', '_')}_metrics.csv")
            pd.concat(preds_df_list).to_csv(os.path.join(
                "/home/leandre/Projects/BioMoQA_Playground/results/test preds",
                f"svm_bert_{model_name.replace('/', '_')}_{kernel}{'_with_title' if self.with_title else ''}_run-{self.run_idx}.csv"
            ))

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
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.labels), problem_type="multi_label_classification")

        # Initialize the zero-shot classification pipeline
        zero_shot_pipeline = transformers.pipeline(
            task="zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        for fold_idx in range(len(self.folds)):
            logger.info(f"\nfold number {fold_idx+1} / {len(self.folds)}")
            
            train_dev_indices, test_indices = self.folds[fold_idx]
            train_dev_split = self.dataset.select(train_dev_indices)
            test_split = self.dataset.select(test_indices)

            logger.info(f"train+dev split size : {len(train_dev_split)}")
            logger.info(f"test split size : {len(test_split)}")

            # Prepare text for zero-shot classification
            texts = test_split["text"]
            labels = self.labels

            # Perform zero-shot classification
            results = zero_shot_pipeline(texts, candidate_labels=labels, multi_label=True)

            # Process results and store metrics
            preds = np.array([result['scores'] for result in results])
            preds_df = pd.DataFrame(preds, columns=labels)
            preds_df["fold"] = fold_idx

            detailed_metrics = self.multi_label_detailed_metrics(preds, test_split.select_columns(self.labels).to_pandas().to_numpy())
            logger.info(f"Detailed metrics for fold {fold_idx+1}: {detailed_metrics}")

            # Save predictions
            preds_df.to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/test_preds", f"zero_shot_{model_name.replace('/', '_')}_fold-{fold_idx}.csv"), index=False)
            scores_by_model.append(preds_df)
    logger.info(f"Zero-shot ensemble classification : {self.ensemble_pred(scores_by_model=scores_by_model, zero_shot=True)}")

if __name__ == "__main__":
begin_pipeline=perf_counter()
ray.init()
ablation=False
pipeline=TrainMultiLabelPipeline(ensemble=True,n_fold=5,n_trials=6,balance_coeff=5,num_runs=1)

if ablation:
    ablations = [True,False]
else:
    ablations = [False]
for with_title in ablations:
    pipeline.with_title=with_title
    logger.info(f"Running baseline SVM model...")
    """
    for model_name in pipeline.model_names:
        pipeline.svm_bert(model_name=model_name)
    """
    #pipeline.zero_shot()
    #pipeline.whole_pipeline()
    logger.info(f"Running baseline Random Forest model...")
    #pipeline.svm()
    pipeline.random_forest()
    logger.info(f"Running training pipeline for BERT based multi-label classification model...")
    torch.cuda.empty_cache()  # Clear CUDA cache after pipeline
    clear_cuda_cache()  # Log memory usage
end_pipeline=perf_counter()
pipeline_runtime=end_pipeline-begin_pipeline
logger.info(f"Total running time : {pipeline_runtime}")
#TODO : Add a statistical test function