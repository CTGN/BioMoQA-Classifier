import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset,concatenate_datasets,Dataset,Features, Value,Sequence
import datasets
import pyarrow.parquet as pq
from pyalex import Works
import pyalex
import logging
import re


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pyalex.config.email="leandre.catogni@hesge.ch"
pyalex.config.max_retries = 1
pyalex.config.retry_backoff_factor = 0.1


def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text, particularly common formatting tags like <i>, <b>, <em>, etc.
    
    Args:
        text (str): Text that may contain HTML tags
        
    Returns:
        str: Cleaned text with HTML tags removed
    """
    if not text or not isinstance(text, str):
        return text
    
    # Remove HTML tags using regex
    # This pattern matches any tag <tag> or </tag>
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up any extra whitespace that might result from tag removal
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text



#TODO : Use the fetch API to get the dois and get more features -> including MESH terms
#TODO : combine the 3 files into one with a class indicator -> not sure it is a great idea
#TODO : change functions and variable names for readability
#TODO : our data_types depends entirely on the reading order of the data directories, we should solve this by creating a dictionarry with the name of the data directory type
#! solve the last comment

def get_ipbes_negatives(directory='/home/leandre/Projects/BioMoQA_Playground/data/IPBES/Raw/Corpus'):
    # Create the corpus dataset
    logger.info(f"creating corpus dataset")
    dataset = load_dataset(
        'parquet', 
        data_files=[
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith('.parquet')
    ],
    split='train'
    )
    
    logger.info(f"dataset loaded")
    logger.info(f"{dataset.column_names}")

    #To add language and doc_type columns, see below
    """
    logger.info(f"adding new columns")
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

def get_ipbes_positives(directory = '/home/leandre/Projects/BioMoQA_Playground/data/IPBES/Raw/Positives'):
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
        logger.info(dir_path)
        csv_files = [
            os.path.join(dir_path, filename)
            for filename in os.listdir(dir_path)
            if filename.endswith('.csv') and os.path.isfile(os.path.join(dir_path, filename))
        ]
        
        if not csv_files:
            logger.info(f"No CSV files found in the directory: {dir_path}")
            continue
        
        try:
            logger.info(len(csv_files))
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
                    "Item Type": Value(dtype="string"),
                    "Publication Year": Value(dtype="string")
                    #"ISBN":Value(dtype="string"),
                    #"ISSN":Value(dtype="string"),
                    #"Url":Value(dtype="string")
                }),
            )
            combined_dataset=combined_dataset.cast_column("Publication Year", Value(dtype="int32"))
            
            pos_datasets.append(combined_dataset)
            logger.info(f"Successfully loaded dataset from: {dir_path}")
            
        except Exception as e:
            logger.info(f"Error loading files from {dir_path}: {str(e)}")
    return pos_datasets

def delete_conflicts(unified_pos_raw, neg_ds_raw):
    "Deletes all instances from the corpus dataset that are in the positives dataset with respect to the the abstract,title and doi or that are None"

    neg_ds=neg_ds_raw.remove_columns(['author','topics', 'author_abbr',"id"])

    abs_set=set(e.strip() for e in unified_pos_raw['Abstract Note'] if e is not None)
    titles_set=set(e.strip() for e in unified_pos_raw['Title'] if e is not None)
    dois_set=set(e.strip() for e in unified_pos_raw['DOI'] if e is not None)
    if None in dois_set : dois_set.remove(None) 
    if None in titles_set : titles_set.remove(None) 
    if None in abs_set : abs_set.remove(None) 

    def is_not_in_pos(batch):
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
            elif doi is None or ( (doi is not None) and (any(doi.endswith(p_doi) for p_doi in dois_set))):
                batch_bools.append(False)
            else:
                batch_bools.append(True)
        return batch_bools
    
    neg_ds=neg_ds.filter(is_not_in_pos, batched=True, batch_size=1000,num_proc=32)
    neg_ds=neg_ds.rename_column("display_name", "title")
    neg_ds=neg_ds.rename_column("ab", "abstract")

    # Clean HTML tags from titles and abstracts in negative data
    def clean_text_fields(batch):
        # Clean titles
        batch['title'] = [clean_html_tags(title) if title else title for title in batch['title']]
        # Clean abstracts
        batch['abstract'] = [clean_html_tags(abstract) if abstract else abstract for abstract in batch['abstract']]
        return batch
    
    neg_ds = neg_ds.map(
        clean_text_fields,
        batched=True,
        batch_size=1000,
        num_proc=min(4, os.cpu_count() or 1)
    )

    return neg_ds

def rename_positives(pos_raw):
    """
    This function creates the positives dataset from the IPBES data.
    It loads the data from the specified directory, processes it, and returns the dataset.
    The function filters instances where the DOI ends with any DOI in the positive_dois set.
    """
    
    pos_ds = pos_raw.rename_column("DOI", "doi")
    pos_ds = pos_ds.rename_column("Title", "title")
    pos_ds = pos_ds.rename_column("Abstract Note", "abstract")
    pos_ds = pos_ds.remove_columns(["Language"])
    
    # Clean HTML tags from titles and abstracts
    def clean_text_fields(batch):
        # Clean titles
        batch['title'] = [clean_html_tags(title) if title else title for title in batch['title']]
        # Clean abstracts
        batch['abstract'] = [clean_html_tags(abstract) if abstract else abstract for abstract in batch['abstract']]
        return batch
    
    pos_ds = pos_ds.map(
        clean_text_fields,
        batched=True,
        batch_size=1000,
        num_proc=min(4, os.cpu_count() or 1)
    )
            
    return pos_ds

def rename_negatives(neg_raw):
    pos_ds = pos_raw.rename_column("DOI", "doi")
    pos_ds = pos_ds.rename_column("display_name", "title")
    pos_ds = pos_ds.rename_column("ab", "abstract")
    pos_ds=pos_ds.remove_columns(["Language"])
            
    return pos_ds

def loading_pipeline_from_raw(multi_label=True):
    """
    This function runs the entire loading pipeline of the IPBES dataset.
    """
    #TODO : Optimize this code !

    if multi_label:
        #Here we return the list of the 3 positves, the unified negataives dataset (which deducts instances of the three positives from the corpus)

        # Get the 3 positives from the raw directory
        logger.info(f"load raw positive datasets")
        pos_ds_list = get_ipbes_positives()
        logger.info(f"pos_ds features for IAS : {pos_ds_list[0].features}")

        logger.info(f"load raw negative dataset")
        # Create a unified negative dataset that deducts instances from all positives types from the corpus
        neg_ds = get_ipbes_negatives()

        logger.info(f"renaming raw positive datasets")
        # Create 3 positives dataset for each data type
        final_pos_ds_list = [rename_positives(ds) for ds in pos_ds_list]
        logger.info(f"Finished positives and negatives creation pipeline")

        return final_pos_ds_list, neg_ds
    else:
        # Get the 3 positives from the raw directory
        pos_ds_list = get_ipbes_positives()
        logger.info(f"pos_ds features for IAS : {pos_ds_list[0].features}")

        # Get the corpus from the raw directory
        corpus_ds = get_ipbes_corpus()

        logger.info("1")

        # Create 3 negatives
        neg_ds_list = [create_ipbes_negatives(pos_ds_list[i], corpus_ds) for i in range(len(pos_ds_list))]

        logger.info("2")
        # Create 3 positives
        final_pos_ds_list = [rename_positives(pos_ds_list[i]) for i in range(len(pos_ds_list))]
        logger.info("Finished positives and negatives creation pipeline")

        return final_pos_ds_list, neg_ds_list, corpus_ds

def main():
    loading_pipeline_from_raw(multi_label=True)

if __name__ == "__main__":
    # Create the IPBES dataset
    main()