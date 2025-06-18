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



def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline models (random, SVM, Random Forest)")
    parser.add_argument(
        "-f",
        "--n_folds",
        type=int,
        required=True,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-r",
        "--n_runs",
        type=int,
        required=True,
        help="Which CV fold to run (overrides config)"
    )
    parser.add_argument(
        "-on",
        "--nb_opt_negs",
        type=int,
        required=True,
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-t",
        "--with_title",
        action="store_true",
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    parser.add_argument(
        "-k",
        "--with_keywords",
        action="store_true",
        help="Number of HPO trials (overrides config.hpo.num_trials)"
    )
    
    return parser.parse_args()

def compute_naive_metrics(metrics_fn=detailed_metrics):
    naive_metrics = pd.DataFrame()
    for fold_idx, (_, _, test_indices) in enumerate(self.folds):
        
        test_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/test{cfg['fold']}_run-{cfg['run']}.csv",split="train")

        logger.info(f"test split size : {len(test_split)}")
        

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

def random_forest(num_folds,num_runs,with_title,with_keywords,nb_optional_negs):
    for num_trees in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        for criterion in "log_loss","gini","entropy":
            for run_idx in range(num_runs):
                for fold_idx in range(num_folds):
                    logger.info(f"\nfold number {fold_idx+1} / {num_folds}")


                    train_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/train{fold_idx}_run-{run_idx}.csv",split="train")
                    dev_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/dev{fold_idx}_run-{run_idx}.csv",split="train")
                    test_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/test{fold_idx}_run-{run_idx}.csv",split="train")

                    logger.info(f"train split size : {len(train_split)}")
                    logger.info(f"dev split size : {len(dev_split)}")
                    logger.info(f"test split size : {len(test_split)}")
    
                        # Prepare features
                    if with_title:
                        if with_keywords:
                            x_train_df = train_split.select_columns(["title","abstract","Keywords"]).to_pandas()
                            x_train = (x_train_df["title"].astype(str) + " " + x_train_df["abstract"].astype(str)+ " " + x_train_df["Keywords"].astype(str)).values
                            x_test_df = test_split.select_columns(["title","abstract","Keywords"]).to_pandas()
                            x_test = (x_test_df["title"].astype(str) + " " + x_test_df["abstract"].astype(str)+ " " + x_test_df["Keywords"].astype(str)).values
                        else:
                            x_train_df = train_split.select_columns(["title","abstract"]).to_pandas()
                            x_train = (x_train_df["title"].astype(str) + " " + x_train_df["abstract"].astype(str)).values
                            x_test_df = test_split.select_columns(["title","abstract"]).to_pandas()
                            x_test = (x_test_df["title"].astype(str) + " " + x_test_df["abstract"].astype(str)).values
                    elif with_keywords:
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


                    result_metrics_path="/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/metrics/random_forest_metrics.csv"

                    if os.path.isfile(result_metrics_path):
                        result_metrics=pd.read_csv(result_metrics_path)
                    else:
                        result_metrics=pd.DataFrame(columns=["model_name", "fold", "run","criterion","num_trees", "with_title", "with_keywords","nb_added_negs"])


                    random_forest_metrics = pd.concat([
                        random_forest_metrics,
                        pd.DataFrame([{
                            "model_name": "Random Forest",
                            "fold": fold_idx+1,
                            "run": run_idx+1, 
                            "criterion": criterion,
                            "num_trees": num_trees,
                            "with_title": with_title,
                            "with_keywords":with_keywords,
                            "nb_added_negs": nb_optional_negs,
                            **results
                        }])
                    ])
                    
                    save_dataframe(random_forest_metrics,file_name="random_forest_metrics.csv")
                    fold_preds_df=pd.DataFrame(data={"label":test_split["labels"],"score":preds,"fold":[fold_idx for _ in range(len(preds))]})
                    test_preds_path=os.path.join("/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/test preds/rf",f"fold-{fold_idx}_rf_{criterion}{'_with_title' if with_title else ''}{'_with_keywords' if with_keywords else ''}_{num_trees}_run-{run_idx}_opt_neg-{nb_optional_negs}.csv")
                    
                    fold_preds_df.to_csv(test_preds_path)



def svm(num_folds,num_runs):
    for kernel in "linear","rbf","poly","sigmoid":
        for run_idx in range(num_runs):
            folds=self.folds_per_run[run_idx]
            logger.info(f"fold indexes for run no.{run_idx+1}: {folds[0]}")
            logger.info(f"Run no {run_idx+1}/{num_runs}")
            logger.info(f"Kernel : {kernel}")
            preds_df_list=[]
            for fold_idx in range(len(folds)):
                logger.info(f"\nfold number {fold_idx+1} / {len(folds)}")
                
                
                train_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/train{cfg['fold']}_run-{cfg['run']}.csv",split="train")
                dev_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/dev{cfg['fold']}_run-{cfg['run']}.csv",split="train")
                test_split = load_dataset("csv", data_files=f"/home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/test{cfg['fold']}_run-{cfg['run']}.csv",split="train")

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


def main():
    args = parse_args()

    svm(args.n_folds,args.n_runs)
    random_forest(args.n_folds,args.n_runs,args.with_title,args.with_keywords,args.nb_opt_negs)

    

if __name__ == "__main__":
    main()