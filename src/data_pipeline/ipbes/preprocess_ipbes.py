import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset,concatenate_datasets,ClassLabel, Features, Value, Sequence,IterableDataset
import datasets
from .create_ipbes_raw import loading_pipeline_from_raw
import os
import gc
from collections import defaultdict
import random
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
We need to rewrite everything on order to :
- First assign labels to each instance, since we already have the seperated data
- Then unify all of them
- Then clean the whole dataset ie. ->
    - Check for conflicts ie. instances that are in both positives and negatives
    - Check for duplicates across labels combination
    - Check for None values
- Split the dataset and create the folds so that we store each fold -> Is it memory efficient ? Look for a better way to do this
In conclusion the pipeline should take as input the raw Datasets object built from the corpus, clean them, unify them, create and store the folds.

How can I use less memory ? 
-> Store only the indices of the instances you use in each fold so that when the model use the fold it needs, it recreates the fold data from the indices.
This allows us to clear the cache at each fold and thus always having one fold data stored in the cache instead of 5
Conclusion : 5 times more memory efficient then the classical approach of storing the folds

This pipeline should also recreate a brand new dataset from the Fetch API with all the relevant informations we need
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

def add_labels(dataset,label):
    # Mappign function to add labels to positives and negatives data
    def add_labels(examples, label_value):
        examples["labels"] = [label_value] * len(examples["title"])
        return examples

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


    for label_name in label_cols:
        # Process conflicts and duplicates using map
        seen_texts = set()
        # Initialize empty sets
        pos_abstracts = set()
        neg_abstracts = set()
        # Process in batches using map
        def collect_abstracts(examples):
            for i, label_val in enumerate(examples[label_name]):
                if label_val == 1:
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
        # Initialize result array
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
            
            seen_texts.add(text)
        
        return keep

    #Apply the clean function acrross the whole dataset
    print("Applying clean_filter...")
    dataset = dataset.filter(clean_filter, batched=True, batch_size=1000, num_proc=os.cpu_count())
    return dataset

def unify_multi_label(pos_ds_list,neg_ds,label_cols,balance_coeff=None):
    """
    Unify all positives with the negative data and add a label for each positive type (3 in our case)
    """
    for pos_ds in pos_ds_list:


def unify_multi_label(pos_ds_list,neg_ds,label_cols,balance_coeff=None):
    """
    Unify all positives with the negative data and add a label for each positive type (3 in our case)
    """
    
    # 1. Create sets of abstracts for each positive dataset
    abstract_sets = [set(ds["abstract"]) for ds in pos_ds_list]

    # 2. Unify all positives into one Dataset
    pos_combined = concatenate_datasets(pos_ds_list)

    pos_combined_df=pos_combined.to_pandas()
    pos_combined_df=pos_combined_df.drop_duplicates(ignore_index=True)
    pos_combined=Dataset.from_pandas(pos_combined_df)

    print("pos_combined",pos_combined)

    # 3. Concatenate positives with the negative dataset
    gcombined = concatenate_datasets([pos_combined, neg_ds])

    #We assign membership of each instance based on the abstract which is not a good idea
    def assign_membership(batch):
        abstracts = batch['abstract']
        # for each example in the batch, check membership in each set
        for i, s in enumerate(abstract_sets):
            batch[label_cols[i]]=[int(a in s) for a in abstracts]
        # return new columns; existing columns are kept by default
        return batch

    # Use a reasonable batch size for efficiency
    unified_dataset = gcombined.map(
        assign_membership,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count()
    )

    clean_unified_dataset=clean_ipbes(unified_dataset,label_cols=label_cols)
    
    return clean_unified_dataset



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


def data_pipeline(multi_label=True):
    """
    Load the data and preprocess it
    """
    if multi_label:
        data_type_list=["IAS","SUA","VA"]
        pos_ds_list, neg_ds, _ = loading_pipeline_from_raw(multi_label=multi_label)

        clean_ds = unify_multi_label(pos_ds_list,neg_ds,data_type_list)
        return clean_ds
    else:
        
        pos_ds_list, neg_ds_list, _ = loading_pipeline_from_raw()
        dataset_dict = {}
        data_type_list=["IAS","SUA","VA"]
        for i in range(len(pos_ds_list)):
            print("Processing dataset for type:", data_type_list[i])
            print("Positive dataset size:", len(pos_ds_list[i]))
            print("Negative dataset size:", len(neg_ds_list[i]))
            dataset_dict[data_type_list[i]]= prereprocess_ipbes(pos_ds_list[i],neg_ds_list[i])
        
        print("Completed processing all datasets.")

        return dataset_dict


#TODO : Double check that we indeed delete cases where you have a positive into negatives -> I think we did it with conflicts

def main():
    args = parse_args()
    set_reproducibility(args.seed)

    dataset_dict=data_label()

    logger.info(dataset_dict)

    logger.info(args)
    


if __name__ == "__main__":

    main()