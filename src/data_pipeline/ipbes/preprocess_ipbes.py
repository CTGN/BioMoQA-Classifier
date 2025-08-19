import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset,concatenate_datasets,ClassLabel, Features, Value, Sequence,IterableDataset
import datasets
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/leandre/Projects/BioMoQA_Playground"))  # Adjust ".." based on your structure

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data_pipeline.ipbes.create_ipbes_raw import loading_pipeline_from_raw
from src.data_pipeline.ipbes.fetch import fill_missing_metadata

import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from src.config import *
from src.utils import *
from collections import defaultdict
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import argparse

import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#Since the goal of this project is to find super-positives, we should consider only the multi-label dataset
#TODO : Delete the multi_label argument
#? How do we construct negatives ? We cannot take all of them so which of them should we consider ? Random ? Or do we look for variety across MESH terms ?
#TODO : See if there is a simple way to take a stratified random sample of all the negatives on the MESH terms
#TODO : Include MESH terms when building and cleaning the set
#TODO : Look for what features we should keep for the model
#TODO : Get the DOI of instances based on their abstract (I think) -> use Fetch APi

"""
We need to rewrite everything in order to :
- First assign labels to each instance, since we already have the seperated data
- Then unify all of them
- Then clean the whole dataset ie. ->
    - Check for conflicts ie. instances that are in both unified positives and negatives
    - Check for duplicates across the same labels combinations
    - Check for None values
- Split the dataset and create the folds so that we store each fold -> Is it memory efficient ? Look for a better way to do this
In conclusion the pipeline should take as input the raw Datasets object built from the corpus, clean them, unify them, create and store the folds.

How can I use less memory ? 
-> Store only the indices of the instances you use in each fold so that when the model use the fold it needs, it recreates the fold data from the indices.
This allows us to clear the cache at each fold and thus always having one fold data stored in the cache instead of 5
Conclusion : 5 times more memory efficient then the classical approach of storing the folds

This pipeline should also recreate a brand new dataset from the Fetch API (or not) with all the relevant informations we need
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess IPBES dataset")
    parser.add_argument("-b","--balanced", action="store_true", help="Whether to balance the dataset")
    parser.add_argument("-bc","--balance_coeff", type=int, default=5, help="Coefficient for balancing the dataset")
    parser.add_argument("-nf","--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("-nr","--n_runs", type=int, default=2, help="Number of runs for cross-validation")
    parser.add_argument("-s","--seed", type=int, default=42, help="Seed for reproducibility")
    return parser.parse_args()

#Use this function in the preprocess pipeline
def set_reproducibility(seed):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Randomness sources seeded with {seed} for reproducibility.")

def merge_pos_neg(pos_ds, neg_ds, store=False):
    """
    Merge positive and negative datasets.
    """
    # ! Not Sure it is useful because of the cast
    #TODO : Change that
    label_feature = ClassLabel(names=["irrelevant", "relevant"])
    
    # Mappign function to add labels to positives and negatives data
    def add_labels(examples, label_value):
        examples["labels"] = [label_value] * len(examples["title"])
        return examples
    
    # Create the labels column
    new_features = pos_ds.features.copy()
    new_features["labels"] = label_feature
    
    #Add postives labels by batch
    pos_ds = pos_ds.map(
        lambda x: add_labels(x, 1),
        batched=True,
        batch_size=100,
        num_proc=min(4, os.cpu_count() or 1)
    )
    pos_ds = pos_ds.cast(new_features)

    print("pos_ds size:", len(pos_ds))
    
    #Add negatives labels by batch
    neg_ds = neg_ds.map(
        lambda x: add_labels(x, 0),
        batched=True,
        batch_size=100,
        num_proc=min(4, os.cpu_count() or 1)
    )
    neg_ds = neg_ds.cast(new_features)

    print("neg_ds size:", len(neg_ds))
    
    #Merge positives and negatives 
    merged_ds = datasets.concatenate_datasets([pos_ds, neg_ds])
    print(merged_ds)

    if store:
        # Save in chunks to reduce memory pressure
        #? What are num_shards ? -> see doc
        merged_ds.save_to_disk("/home/leandre/Projects/BioMoQA_Playground/data/corpus", 
                              num_shards=4)  # Split into multiple files)
    print("Number of Positives before cleaning :",len(pos_ds))
    return merged_ds

def clean_ipbes(dataset,label_cols=["labels"]):
    """
    Clean dataset :
    - 
    - 
    """
    #TODO : I think it would be more memory efficient to first clean the positives set and negatives set while they seperated, and then merging them knowing that they are not overlapping
    print("Filtering out rows with no abstracts or DOI...")


    # Process conflicts and duplicates using map
    seen_texts = set()
    seen_dois = set()
    # Initialize empty sets
    pos_abstracts = set()
    neg_abstracts = set()
    # Process in batches using map
    
    def collect_abstracts(examples):
        for i in range(len(examples['abstract'])):
                if any([bool(examples[label_name][i]) for label_name in label_cols]):
                    pos_abstracts.add(examples['abstract'][i])
                else:
                    neg_abstracts.add(examples['abstract'][i])
        return examples
    
    # Process in parallel with batching
    dataset=dataset.map(collect_abstracts, 
                batched=True, 
                batch_size=1000, 
                num_proc=min(4, os.cpu_count() or 1))
    
    conflicting_texts = set()
    print("Size of the dataset before cleaning:", len(dataset))

    #Check if, in a given batch, there is an instance for which the title or the abstract is None + check for conflicts and duplicates
    def clean_filter(examples):
        keep = [True] * len(examples['abstract'])
        
        for i in range(len(examples['abstract'])):
            text = examples['abstract'][i]
            title=examples['title'][i]
            
            # Check for None values
            if text is None or title is None:
                keep[i] = False
                continue
            
            # Check for conflicts
            if text in pos_abstracts and text in neg_abstracts:
                conflicting_texts.add(text)
                keep[i] = False
                continue
            
            # Check for duplicates
            if text in seen_texts:
                keep[i] = False
                continue

            if examples['doi'][i] is None or examples['doi'][i] in seen_dois:
                keep[i] = False
                continue
            
            seen_texts.add(text)
            seen_dois.add(examples['doi'][i])
        
        return keep

    #Apply the clean function acrross the whole dataset
    print("Applying clean_filter...")
    dataset = dataset.filter(clean_filter, batched=True, batch_size=1000, num_proc=os.cpu_count())
    logger.info(f"Dataset size after cleaing : {len(dataset)}")
    return dataset

def create_folds(dataset,n_folds,n_runs):
    """
    Give the indices from k-fold stratified cross validation, on each different run (with a different seed for each one)

    Returns the the folds for each run (List of list of folds)
    """
    labels=['IAS','SUA','VA']
    rng = np.random.RandomState(CONFIG["seed"])
    derived_seeds = rng.randint(0, 1000000, size=n_runs)
    folds_per_run = []
    df=dataset.to_pandas()

    for seed in derived_seeds:
        # Stratified K‐Fold on only the original (non‐optional) data
        mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CONFIG["seed"])

        run_folds = []
        for train_dev_indices, test_indices in mskf.split(dataset['abstract'], dataset.select_columns(labels).to_pandas()):
            # Convert positional indices back to the DataFrame's original index

            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=0.3,
                train_size=0.7,
                random_state=seed
            )

            # get train/dev indices, stratified on the labels
            train_indices, dev_indices = next(msss.split(np.arange(len(train_dev_indices)),dataset.select(train_dev_indices).select_columns(labels).to_pandas()))

            train_indices = train_indices.tolist()
            dev_indices = dev_indices.tolist()

            run_folds.append([train_indices, dev_indices, test_indices])

            # Log distributions
            train_labels = df.loc[train_indices, labels]
            test_labels = df.loc[test_indices, labels]
            train_label_dist = train_labels.value_counts(normalize=True)
            test_label_dist = test_labels.value_counts(normalize=True)
            logger.info(f"Fold {len(run_folds)}:")
            logger.info(f"  Train label distribution: {train_label_dist.to_dict()}")
            logger.info(f"  Test label distribution: {test_label_dist.to_dict()}")

            
        #TODO : Store the indices and return them
        #TODO : Use the indices in the pipeline to get the corresponding data, and clear the cache after using them.
        
        folds_per_run.append(run_folds)

        return folds_per_run

def fill_missing_metadata_for_positives(pos_ds_list, output_dir="data/IPBES/modified_instances", max_workers=5):
    """
    Fill missing metadata (title/abstract) for positive datasets using CrossRef API.
    
    Args:
        pos_ds_list (List[Dataset]): List of positive datasets
        output_dir (str): Directory to save modified instances
        max_workers (int): Maximum concurrent workers for API requests
        
    Returns:
        List[Dataset]: List of updated positive datasets with filled metadata
    """
    logger.info("Starting metadata filling for positive datasets...")
    
    updated_pos_ds_list = []
    all_modified_instances = []
    
    data_type_names = ["IAS", "SUA", "VA"]
    
    for i, pos_ds in enumerate(pos_ds_list):
        data_type = data_type_names[i]
        logger.info(f"Processing {data_type} dataset with {len(pos_ds)} instances...")
        
        # Fill missing metadata
        updated_ds, modified_instances = fill_missing_metadata(
            pos_ds,
            output_file=output_dir+ f"/{data_type}_modified_instances.csv",
            max_workers=max_workers
        )
        
        updated_pos_ds_list.append(updated_ds)
        
        # Track modifications for this dataset
        for instance in modified_instances:
            instance['dataset_type'] = data_type
        all_modified_instances.extend(modified_instances)
        
        logger.info(f"Completed {data_type}: Updated {len(modified_instances)} instances")
    
    logger.info(f"Metadata filling completed for all datasets. Total instances modified: {len(all_modified_instances)}")
    return updated_pos_ds_list


def unify_multi_label(pos_ds_list,neg_ds,label_cols,balance_coeff=None):
    """
    Unify all positives with the negative data and add a label for each positive type (3 in our case)
    """
    
    #Merge the positives between each other
    pos_combined = concatenate_datasets(pos_ds_list)

    pos_combined=pos_combined.filter(lambda batch : [batch['Item Type'][i]=='journalArticle' for i in range(len(batch['Item Type']))],batched=True,batch_size=1000,num_proc=30)

    pos_combined=pos_combined.filter(lambda batch : [batch['doi'][i] is not None for i in range(len(batch['doi']))],batched=True,batch_size=1000,num_proc=(os.cpu_count()-5))

    doi_set = [set(ds["doi"]) for ds in pos_ds_list]

    pos_combined_df=pos_combined.to_pandas()
    pos_combined_df=pos_combined_df.drop_duplicates(ignore_index=True)
    pos_combined=Dataset.from_pandas(pos_combined_df)


    print("pos_combined",pos_combined)


    gcombined=concatenate_datasets([pos_combined,neg_ds])
    
    #For each batch, check if dois belongs to each labels dois set assigns 1 if so, 0 if not
    def assign_membership(batch):
        dois = batch['doi']
        for i, s in enumerate(doi_set):
            batch[label_cols[i]]=[int(a in s) for a in dois]
        return batch

    unified_dataset = gcombined.map(
        assign_membership,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count()
    )
    
    return unified_dataset


def prereprocess_ipbes(pos_ds,neg_ds):
    """
    Function to preprocess the IPBES dataset
    """

    all_ds=merge_pos_neg(pos_ds,neg_ds)

    #We consider only the one with an abstract, and we remove the duplicates
    clean_ds=clean_ipbes(all_ds)

    def is_label(batch,label):
        batch_bools=[]
        for ex_label in batch['labels']:
            if ex_label == label:
                batch_bools.append(True)
            else:
                batch_bools.append(False)
        return batch_bools

    pos_dataset = clean_ds.filter(lambda x : is_label(x,1), batched=True, batch_size=1000, num_proc=os.cpu_count())
    print("Number of positives after cleaning:", len(pos_dataset))
    print(clean_ds)
    
    return clean_ds


def data_pipeline(n_folds,n_runs,balance_coeff=None,multi_label=True,fill_metadata=True,max_workers=5):
    """
    Load the data and preprocess it
    
    Args:
        n_folds (int): Number of folds for cross-validation
        n_runs (int): Number of runs for cross-validation
        balance_coeff (int): Coefficient for balancing the dataset
        multi_label (bool): Whether to use multi-label approach
        fill_metadata (bool): Whether to fill missing metadata using CrossRef API
        max_workers (int): Maximum concurrent workers for API requests
    """
    data_type_list=["IAS","SUA","VA"]
    pos_ds_list, neg_ds = loading_pipeline_from_raw(multi_label=multi_label)

    # Fill missing metadata for positive datasets if requested
    if fill_metadata:
        logger.info("Filling missing metadata for positive datasets...")
        pos_ds_list = fill_missing_metadata_for_positives(pos_ds_list, max_workers=max_workers)
    
    unified_ds = unify_multi_label(pos_ds_list,neg_ds,data_type_list,balance_coeff=balance_coeff)
    clean_ds=clean_ipbes(unified_ds,label_cols=data_type_list)
    folds_per_run=create_folds(clean_ds,n_folds,n_runs)

    return clean_ds,folds_per_run



#TODO : Double/triple check that we indeed delete cases where you have a positive into negatives -> I think we did it with conflicts


def main():

    parser = argparse.ArgumentParser(description="Preprocess IPBES dataset")
    parser.add_argument("-bc","--balance_coeff", type=int, default=None, help="Coefficient for balancing the dataset")
    parser.add_argument("-nf","--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("-nr","--n_runs", type=int, default=2, help="Number of runs for cross-validation")
    parser.add_argument("-fm","--fill_metadata", action="store_true", help="Fill missing metadata using CrossRef API")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum concurrent workers for API requests")


    args = parser.parse_args()
    set_reproducibility(CONFIG["seed"])

    logger.info(args)

    clean_ds, folds_per_run=data_pipeline(
        args.n_folds, 
        n_runs=args.n_runs, 
        balance_coeff=args.balance_coeff,
        multi_label=True,
        fill_metadata=args.fill_metadata,
        max_workers=args.max_workers
    )
    clean_ds.to_pandas().to_csv("/home/leandre/Projects/BioMoQA_Playground/data/IPBES/cleaned_dataset.csv", index=False)

    for run_idx in range(len(folds_per_run)):
        folds=folds_per_run[run_idx]
        for fold_idx in range(args.n_folds):

            train_indices, dev_indices,test_indices = folds[fold_idx]

            logger.info(f"\nfold number {fold_idx+1} / {len(folds)}")
            
            logger.info(f"train split size : {len(train_indices)}")
            logger.info(f"dev split size : {len(dev_indices)}")
            logger.info(f"test split size : {len(test_indices)}")
            
            np.savetxt(f"/home/leandre/Projects/BioMoQA_Playground/data/IPBES/folds/train{fold_idx}_run-{run_idx}.csv", train_indices, delimiter=',', fmt='%g')
            np.savetxt(f"/home/leandre/Projects/BioMoQA_Playground/data/IPBES/folds/dev{fold_idx}_run-{run_idx}.csv", dev_indices, delimiter=',', fmt='%g')
            np.savetxt(f"/home/leandre/Projects/BioMoQA_Playground/data/IPBES/folds/test{fold_idx}_run-{run_idx}.csv", test_indices, delimiter=',', fmt='%g')

if __name__ == "__main__":
    main()