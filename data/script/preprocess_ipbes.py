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

def merge_pos_neg(pos_ds, neg_ds, store=False):
    """
    Merge positive and negative datasets with streaming support.
    Memory-optimized version.
    """
    # Define label feature
    label_feature = ClassLabel(names=["irrelevant", "relevant"])
    
    # Process datasets with more efficient mapping
    def add_labels(examples, label_value):
        examples["labels"] = [label_value] * len(examples["title"])
        return examples
    
    # Create new features once
    new_features = pos_ds.features.copy()
    new_features["labels"] = label_feature
    
    # Apply transformations one at a time to reduce memory usage
    pos_ds = pos_ds.map(
        lambda x: add_labels(x, 1),
        batched=True,
        batch_size=100,  # Smaller batch size to reduce memory usage
        num_proc=min(4, os.cpu_count() or 1)  # Limit CPU usage
    )
    pos_ds = pos_ds.cast(new_features)

    print("pos_ds size:", len(pos_ds))
    
    neg_ds = neg_ds.map(
        lambda x: add_labels(x, 0),
        batched=True,
        batch_size=100,
        num_proc=min(4, os.cpu_count() or 1)
    )
    neg_ds = neg_ds.cast(new_features)

    print("neg_ds size:", len(neg_ds))
    
    # Interleave datasets instead of concatenating all at once
    merged_ds = datasets.concatenate_datasets([pos_ds, neg_ds])
    print(merged_ds)
    if store:
        # Save in chunks to reduce memory pressure
        merged_ds.save_to_disk("/home/leandre/Projects/BioMoQA_Playground/data/corpus", 
                              num_shards=4)  # Split into multiple files)
    print("Number of Positives before cleaning :",len(pos_ds))
    return merged_ds

def clean_ipbes(dataset,label_cols=["labels"]):
    """
    Clean dataset using streaming operations
    """
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

    print("Applying clean_filter...")
    dataset = dataset.filter(clean_filter, batched=True, batch_size=1000, num_proc=os.cpu_count())
    return dataset

def unify_multi_label(pos_ds_list,neg_ds,label_cols):
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


def data_pipeline(multi_label=False):
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

if __name__ == "__main__":

    print("IPBES data preprocessing pipeline")

    #Final clean, training and test sets, with journal abstracts and labels
    train_ds, test_ds, clean_ds=data_pipeline()