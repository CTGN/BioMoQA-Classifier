import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset,concatenate_datasets,Dataset,Features, Value,Sequence
import datasets
import pyarrow.parquet as pq
from pyalex import Works
import pyalex

pyalex.config.email="leandre.catogni@hesge.ch"
from pyalex import config

config.max_retries = 1
config.retry_backoff_factor = 0.1

#TODO : combine the 3 files into one with a class indicator -> not sure it is a great idea
#TODO : change functions and variable names for readability
#TODO : our data_types depends entirely on the reading order of the data directories, we should solve this by creating a dictionarry with the name of the data directory type
#! solve the last comment

def get_ipbes_corpus(directory='/home/leandre/Projects/BioMoQA_Playground/data/corpus/Raw/Corpus'):
    # Create the corpus dataset
    print("creating corpus dataset")
    dataset = load_dataset(
        'parquet', 
        data_files=[
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith('.parquet')
    ],
    split='train'
    )
    
    print("dataset loaded")
    print(dataset.column_names)

    #To add language and doc_type columns, see below
    """
    print("adding new columns")
    def add_new_column(batch):
        languages = []
        doc_types = []
        works = Works()

        # Filter out None DOIs
        valid_dois = [doi for doi in batch["doi"] if doi is not None]
        
        # Batch request all DOIs at once
        if valid_dois:
            try:
                results = works.filter(doi=valid_dois).get()
                # Create a lookup dictionary
                doi_info = {work['doi']: work for work in results}
            except:
                results = []
                doi_info = {}
        else:
            doi_info = {}

        # Process each DOI
        for doi in batch["doi"]:
            if doi is None or doi not in doi_info:
                languages.append('unknown')
                doc_types.append('unknown')
            else:
                work = doi_info[doi]
                languages.append(work.get('language', 'unknown'))
                doc_types.append(work.get('type', 'unknown'))

        batch["language"] = languages
        batch["doc_type"] = doc_types
        
        return batch
    
    dataset = dataset.map(
        add_new_column,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count()
    )
    """
    return dataset


def get_ipbes_positives(directory = '/home/leandre/Projects/BioMoQA_Playground/data/corpus/Raw/Positives'):
    """
    Creates 3 positives datasets from a directory containing all the positives.
    The returned datasets are raw and are just the combination of the several original files.
    """
    pos_datasets = []
    
    # Get all IPBES subdirectories
    directories = [
        os.path.join(directory, dirname)
        for dirname in os.listdir(directory)
        if dirname.startswith('IPBES') and os.path.isdir(os.path.join(directory, dirname))
    ]
    
    for dir_path in directories:
        print(dir_path)
        csv_files = [
            os.path.join(dir_path, filename)
            for filename in os.listdir(dir_path)
            if filename.endswith('.csv') and os.path.isfile(os.path.join(dir_path, filename))
        ]
        
        if not csv_files:
            print(f"No CSV files found in the directory: {dir_path}")
            continue
        
        try:
            print(len(csv_files))
            # Load dataset with explicit features
            combined_dataset = load_dataset(
                "csv",
                data_files=csv_files, 
                split='train',
                features=Features({
                    #"Key":Value(dtype="string"),
                    "DOI": Value(dtype="string"),
                    "Title": Value(dtype="string"),
                    "Abstract Note": Value(dtype="string"),
                    "Language": Value(dtype="string"),
                    #"ISBN":Value(dtype="string"),
                    #"ISSN":Value(dtype="string"),
                    #"Url":Value(dtype="string")
                }),
            )
            
            pos_datasets.append(combined_dataset)
            print(f"Successfully loaded dataset from: {dir_path}")
            
        except Exception as e:
            print(f"Error loading files from {dir_path}: {str(e)}")
    return pos_datasets

def create_ipbes_negatives(pos_raw, corpus_ds):
    "Deletes all instances from the corpus dataset that are in the positives dataset with respect to the the abstract,title and doi or that are None"

    # Create a set of positive DOIs for faster lookup
    neg_ds=corpus_ds.remove_columns(['author','topics', 'author_abbr',"id"])

    abs_set=set(e.strip() for e in pos_raw['Abstract Note'] if e is not None)
    titles_set=set(e.strip() for e in pos_raw['Title'] if e is not None)
    dois_set=set(e.strip() for e in pos_raw['DOI'] if e is not None)
    if None in dois_set : dois_set.remove(None) 
    if None in titles_set : titles_set.remove(None) 
    if None in abs_set : abs_set.remove(None) 

    def find(batch):
        batch_bools=[]
        for j in range(len(batch['display_name'])):
            title=batch['display_name'][j]
            title=title.strip() if title is not None else None
            abstract=batch['ab'][j]
            abstract=abstract.strip() if abstract is not None else None
            doi=batch['doi'][j]
            doi=doi.strip() if doi is not None else None
            
            if abstract is None or (abstract in abs_set):
                batch_bools.append(False)
            elif title is None or (title in titles_set):
                batch_bools.append(False)
            elif (doi is not None) and (any(doi.endswith(p_doi) for p_doi in dois_set)):
                batch_bools.append(False)
            else:
                batch_bools.append(True)
        return batch_bools
    neg_ds=neg_ds.filter(find, batched=True, batch_size=1000,num_proc=32)
    neg_ds=neg_ds.rename_column("display_name", "title")
    neg_ds=neg_ds.rename_column("ab", "abstract")

    return neg_ds

def create_ipbes_positives(pos_raw):
    """
    This function creates the positives dataset from the IPBES data.
    It loads the data from the specified directory, processes it, and returns the dataset.
    The function filters instances where the DOI ends with any DOI in the positive_dois set.
    """
    
    pos_ds = pos_raw.rename_column("DOI", "doi")
    pos_ds = pos_ds.rename_column("Title", "title")
    pos_ds = pos_ds.rename_column("Abstract Note", "abstract")
    pos_ds=pos_ds.remove_columns(["Language"])
            
    return pos_ds

def loading_pipeline_from_raw(multi_label=False):
    """
    This function runs the entire pipeline for creating the IPBES dataset.
    It includes data loading, preprocessing, and saving the final dataset.
    """
    #TODO : Optimize this code !

    if multi_label:
        #Here we return the list of the 3 positves, the unified negataives dataset (which deducts instances of the three positives from the corpus)

        # Get the 3 positives from the raw directory
        pos_ds_list = get_ipbes_positives()
        print("pos_ds features for IAS : ",pos_ds_list[0].features)

        # Get the corpus from the raw directory
        corpus_ds = get_ipbes_corpus()

        #Merge positives to create the unified negative dataset
        print("Concatenating positive datasets...")
        unify_pos_ds=concatenate_datasets([ds for ds in pos_ds_list])
        unify_pos_dataframe=unify_pos_ds.to_pandas()
        unify_pos_dataframe=unify_pos_dataframe.drop_duplicates()
        unify_pos_ds=Dataset.from_pandas(unify_pos_dataframe)

        print("creating raw negative dataset")
        # Create a unified negative dataset that deducts instances from all positives type from the corpus
        neg_ds = create_ipbes_negatives(unify_pos_ds, corpus_ds)

        print("creating raw positive dataset")
        # Create 3 positives dataset for each data type
        final_pos_ds_list = [create_ipbes_positives(ds) for ds in pos_ds_list]
        print("Finished positives and negatives creation pipeline")

        return final_pos_ds_list, neg_ds, corpus_ds
    else:
        # Get the 3 positives from the raw directory
        pos_ds_list = get_ipbes_positives()
        print("pos_ds features for IAS : ",pos_ds_list[0].features)

        # Get the corpus from the raw directory
        corpus_ds = get_ipbes_corpus()

        print("1")

        # Create 3 negatives
        neg_ds_list = [create_ipbes_negatives(pos_ds_list[i], corpus_ds) for i in range(len(pos_ds_list))]

        print("2")
        # Create 3 positives
        final_pos_ds_list = [create_ipbes_positives(pos_ds_list[i]) for i in range(len(pos_ds_list))]
        print("Finished positives and negatives creation pipeline")

        return final_pos_ds_list, neg_ds_list, corpus_ds

if __name__ == "__main__":
    # Create the IPBES dataset
    gathered_datasets = loading_pipeline_from_raw()