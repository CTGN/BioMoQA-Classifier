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
    # Class-level constants (shared defaults)
    DEFAULT_MODEL_NAMES: List[str] = [
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "FacebookAI/roberta-base",
        "dmis-lab/biobert-v1.1",
        "google-bert/bert-base-uncased",
    ]
    RESULT_COLUMNS = ["model_name", "fold", "loss_type", "run", "with_title", "with_keywords"]
    SVM_COLUMNS = ["model_name", "fold", "kernel", "run", "with_title", "with_keywords"]
    RF_COLUMNS = ["model_name", "fold", "criterion", "num_trees", "run", "with_title", "with_keywords"]

    def __init__(
        self,
        loss_type: str = "BCE",
        hpo_metric: str = "eval_recall",
        ensemble: bool = False,
        n_trials: int = 10,
        num_runs: int = 5,
        with_title: bool = False,
        with_keywords: bool = False,
        nb_optional_negs: int = 5000,
        n_folds: int = 5,
        model_names: Optional[List[str]] = None,
    ):
        from transformers import set_seed
        from src.config import CONFIG
        from src.data_pipeline.biomoqa.preprocess_biomoqa import biomoqa_data_pipeline
        from src.utils import set_random_seeds, detailed_metrics

        # Initialize random seeds
        set_random_seeds(CONFIG["seed"])
        set_seed(CONFIG["seed"])

        # Instance attributes
        self.loss_type = loss_type
        self.hpo_metric = hpo_metric
        self.ensemble = ensemble
        self.n_trials = n_trials
        self.num_runs = num_runs
        self.with_title = with_title
        self.with_keywords = with_keywords
        self.nb_optional_negs = nb_optional_negs
        self.n_folds = n_folds
        # allow override of default models
        self.model_names = model_names or self.DEFAULT_MODEL_NAMES

        self.run_idx = 0
        self.dataset = None
        self.folds_per_run = None
        self.folds = None

        # Metrics storage
        self.result_metrics = pd.DataFrame(columns=self.RESULT_COLUMNS)
        self.svm_metrics = pd.DataFrame(columns=self.SVM_COLUMNS)
        self.random_forest_metrics = pd.DataFrame(columns=self.RF_COLUMNS)

    def load_dataset(self, pipeline_fn=biomoqa_data_pipeline):
        self.dataset, self.folds_per_run = pipeline_fn(
            self.n_folds,
            self.num_runs,
            self.with_title,
            self.with_keywords,
            nb_optional_negs=self.nb_optional_negs,
        )
        self.folds = self.folds_per_run[0]

    def _compute_naive_metrics(self, metrics_fn=detailed_metrics):
        naive_metrics = pd.DataFrame()
        for fold_idx, (_, _, test_indices) in enumerate(self.folds):
            test_split = self.dataset.select(test_indices)

            # always positive
            res_pos = metrics_fn(
                np.asarray([1 for _ in range(len(test_indices))]),
                test_split["labels"],
                scores=np.asarray([1.0 for _ in range(len(test_indices))]),
            )
            naive_metrics = pd.concat(
                [
                    naive_metrics,
                    pd.DataFrame(
                        [{
                            "approach": "always pos",
                            "fold": fold_idx + 1,
                            "nb_added_negs": self.nb_optional_negs,
                            **res_pos,
                        }]
                    ),
                ],
                ignore_index=True,
            )

            # always negative
            res_neg = metrics_fn(
                np.asarray([0 for _ in range(len(test_indices))]),
                test_split["labels"],
                scores=np.asarray([0.0 for _ in range(len(test_indices))]),
            )
            naive_metrics = pd.concat(
                [
                    naive_metrics,
                    pd.DataFrame(
                        [{
                            "approach": "always neg",
                            "fold": fold_idx + 1,
                            "nb_added_negs": self.nb_optional_negs,
                            **res_neg,
                        }]
                    ),
                ],
                ignore_index=True,
            )
        # Persist results
        from src.utils import save_dataframe

        save_dataframe(naive_metrics, file_name="naive_metrics.csv")
        
    #def metrics_analysis(self):

    #translate this into bash
    def run_pipeline(self):
        """
        This function runs the entire pipeline for training and evaluating a model using cross-validation.
        It includes data loading, preprocessing, model training, and evaluation.
        """
        #TODO : compare the loss functions inside the pipeline function to have the same test set for each run

        if self.ensemble==False:
            scores_by_fold_by_model = [ [] for _ in range(self.n_folds)]
            for i,model_name in enumerate(self.model_names):
                logger.info(f"Training model {i+1}/{len(self.model_names)}: {model_name}")
                scores_by_fold = self.train(model_name=model_name)

                clear_cuda_cache() 

                # Ensure scores_by_fold is a consistent array before appending
                for k in range(self.n_folds):
                    scores_by_fold_by_model[k].append(scores_by_fold[k]) 

                logger.info(f"Metrics for {model_name}: {self.result_metrics[ (self.result_metrics['loss_type'] == self.loss_type) & (self.result_metrics['model_name'] == self.map_name(model_name))]}")

            clear_cuda_cache() 
            logger.info(f"Results: {self.result_metrics}")
            return self.result_metrics

        elif self.ensemble==True:
            logger.info(f"Ensemble learning pipeline")
            
            scores_by_fold_by_model = [ [] for _ in range(self.n_folds)]
            for i,model_name in enumerate(self.model_names):
                logger.info(f"Training model {i+1}/{len(self.model_names)}: {model_name}")
                scores_by_fold = self.train(model_name=model_name)

                clear_cuda_cache() 

                # Ensure scores_by_fold is a consistent array before appending
                for k in range(self.n_folds):
                    scores_by_fold_by_model[k].append(scores_by_fold[k]) 

                logger.info(f"Metrics for {model_name}: {self.result_metrics[ (self.result_metrics['loss_type'] == self.loss_type) & (self.result_metrics['model_name'] == model_name)]}")

            avg_ensemble_metrics=self.ensemble_pred(scores_by_fold_by_model)

            return avg_ensemble_metrics
    
    #TODO: translate this into bash
    def whole_pipeline(self):
        logger.info(f"with_title : {self.with_title}")
        logger.info(f"with_keywords : {self.with_keywords}")

        loss_type_list=["BCE","focal"]
        for loss_type in loss_type_list:
            self.loss_type=loss_type
            for self.run_idx in range(self.num_runs):
                self.folds=self.folds_per_run[self.run_idx]
                logger.info(f"fold indexes for run no.{self.run_idx+1}: {self.folds[0]}")
                logger.info(f"Run no {self.run_idx+1}/{self.num_runs}")
                logger.info(f"Loss Type : {loss_type}")
                avg_ens_metrics=self.run_pipeline()
                save_dataframe(self.result_metrics)
            self.folds=[]

        return self.result_metrics
    

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
                                "nb_added_negs": self.nb_optional_negs,
                                **results
                            }])
                        ])

                        fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                        preds_df_list.append(fold_preds_df)

                    save_dataframe(self.random_forest_metrics,file_name="random_forest_metrics.csv")
                    pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/rf",f"rf_{criterion}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_{num_trees}_run-{self.run_idx}_opt_neg-{self.nb_optional_negs}.csv"))



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
                            "nb_added_negs": self.nb_optional_negs,
                            **results
                        }])
                    ])
                    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                    preds_df_list.append(fold_preds_df)
                save_dataframe(self.svm_metrics, file_name="svm_metrics.csv")
                pd.concat(preds_df_list).to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/svm", f"svm_{kernel}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_run-{self.run_idx}_opt_neg-{self.nb_optional_negs}.csv"))

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
                            "nb_added_negs": self.nb_optional_negs,
                            **results
                        }])
                    ])
                    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                    preds_df_list.append(fold_preds_df)

                save_dataframe(self.svm_metrics, file_name=f"svm_bert_{model_name.replace('/', '_')}_metrics.csv")
                pd.concat(preds_df_list).to_csv(os.path.join(
                    "/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/svm_bert",
                    f"svm_bert_{model_name.replace('/', '_')}_{kernel}{'_with_title' if self.with_title else ''}{'_with_keywords' if self.with_keywords else ''}_run-{self.run_idx}_opt_neg-{self.nb_optional_negs}.csv"
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
                preds_df.to_csv(os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds", f"zero_shot_{model_name.replace('/', '_')}_fold-{fold_idx}_opt_neg-{self.nb_optional_negs}.csv"), index=False)
                scores_by_model.append(preds_df)
        logger.info(f"Zero-shot ensemble classification : {self.ensemble_pred(scores_by_model=scores_by_model, zero_shot=True)}")