import os
import requests
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from datasets import Dataset
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

# CrossRef REST API base URL
CROSSREF_BASE_URL = "https://api.crossref.org/works/"

# Rate limiting configuration
RATE_LIMIT = 50  # requests per second
RATE_LIMIT_WINDOW = 1.0  # seconds

# Thread-safe rate limiter
class RateLimiter:
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                # Need to wait
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            self.calls.append(now)

# Global rate limiter instance
rate_limiter = RateLimiter(RATE_LIMIT, RATE_LIMIT_WINDOW)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_crossref_metadata(doi: str, params: dict = None, filters: dict = None) -> dict | None:
    """
    Fetch metadata for a given DOI from the CrossRef REST API.
    
    Args:
        doi (str): The DOI of the work to fetch metadata for
        params (dict): Query parameters for the API request
        filters (dict): Filters to apply to the API request
        
    Returns:
        dict | None: The metadata record if successful, None if failed
    """
    if params is None:
        params = {}
    if filters is None:
        filters = {}
    
    # Apply rate limiting
    rate_limiter.wait_if_needed()
    
    try:
        # Make request to CrossRef API
        url = f"{CROSSREF_BASE_URL}{doi}"
        headers = {
            'User-Agent': 'BioMoQA-Classifier/1.0 (mailto:leandre.catogni@hesge.ch)' 
        }
        
        # Add default mailto parameter
        if 'mailto' not in params:
            params['mailto'] = 'leandre.catogni@hesge.ch'
        
        if len(list(filters.keys())) > 0:
            fval = ''
            # add each filter key and value to the string
            for f in filters:
                fval += str(f) + ':' + str(filters[f]) + ','
            fval = fval[:-1] # removing trailing comma
            params['filter'] = fval

        # make the query
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        return data
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for DOI {doi}: {e}")
        return None
    except KeyError as e:
        logger.warning(f"Error parsing response for DOI {doi}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error occurred for DOI {doi}: {e}")
        return None

def clean_jats_tags(text: str) -> str:
    """
    Remove JATS XML tags from text, particularly <jats:p> and </jats:p> tags.
    
    Args:
        text (str): Text that may contain JATS tags
        
    Returns:
        str: Cleaned text with JATS tags removed
    """
    if not text:
        return text
    
    # Remove <jats:p> and </jats:p> tags from beginning and end
    text = text.strip()
    if text.startswith('<jats:p>'):
        text = text[8:]  # Remove '<jats:p>'
    if text.endswith('</jats:p>'):
        text = text[:-9]  # Remove '</jats:p>'
    
    return text.strip()

def extract_metadata(record: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract formatted metadata information from a CrossRef record.
    
    Args:
        record (dict): The metadata record from CrossRef API
        
    Returns:
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]: 
        (abstract, journal_title, language, article_title)
    """
    abstract, journal_title, language, article_title = None, None, None, None
    
    record = record.get('message', {})
    
    if 'abstract' in record and len(record['abstract']) > 0 and record['abstract'] is not None:
        abstract = clean_jats_tags(record['abstract'])
    
    if 'container-title' in record and len(record['container-title']) > 0:
        journal_title = record['container-title'][0]
    
    if 'language' in record and len(record['language']) > 0:
        language = record['language']
        
    # Extract article title
    if 'title' in record and len(record['title']) > 0:
        article_title = record['title'][0]
    
    return abstract, journal_title, language, article_title


def get_metadata(record: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Print and return formatted metadata information from a CrossRef record.
    
    Args:
        record (dict): The metadata record from CrossRef API
        
    Returns:
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]: 
        (abstract, journal_title, language, article_title)
    """
    print("Record fetched successfully:")
    abstract, journal_title, language, article_title = extract_metadata(record)
    
    print(f"Abstract: {abstract if abstract else 'Not available'}")
    print(f"Journal title: {journal_title if journal_title else 'Not available'}")
    print(f"Language: {language if language else 'Not available'}")
    print(f"Article title: {article_title if article_title else 'Not available'}")
    
    # Print authors
    record_data = record.get('message', {})
    if 'author' in record_data:
        authors = [f"{author.get('given', '')} {author.get('family', '')}" for author in record_data['author']]
        print(f"Authors: {', '.join(authors)}")
    
    return abstract, journal_title, language, article_title


def identify_missing_metadata(dataset: Dataset) -> List[Tuple[int, str]]:
    """
    Identify instances in the dataset that have missing title or abstract but have valid DOIs.
    
    Args:
        dataset (Dataset): The dataset to check
        
    Returns:
        List[Tuple[int, str]]: List of (index, doi) pairs for instances with missing metadata
    """
    # Add original indices to the dataset for tracking
    dataset_with_indices = dataset.add_column("original_index", list(range(len(dataset))))
    
    def filter_condition_batch(batch):
        """Filter condition to identify instances with missing metadata (batched version)."""
        results = []
        
        for i in range(len(batch['doi'])):
            doi = batch['doi'][i] if batch['doi'][i] is not None else None
            title = batch['title'][i] if batch['title'][i] is not None else None
            abstract = batch['abstract'][i] if batch['abstract'][i] is not None else None
            
            # Check if DOI exists and either title or abstract is missing
            has_valid_doi = doi and doi.strip()
            missing_title_or_abstract = not title or not abstract
            
            results.append(has_valid_doi and missing_title_or_abstract)
        
        return results
    
    # Filter the dataset using the Dataset's filter method with batched processing
    filtered_dataset = dataset_with_indices.filter(filter_condition_batch, batched=True,batch_size=1000,num_proc=os.cpu_count()-5)
    
    # Extract the results as (index, doi) tuples
    missing_instances = [
        (row["original_index"], row["doi"].strip()) 
        for row in filtered_dataset
    ]
    
    logger.info(f"Found {len(missing_instances)} instances with missing title/abstract")
    return missing_instances


def fetch_metadata_batch(doi_list: List[str], max_workers: int = 5) -> Dict[str, Dict]:
    """
    Fetch metadata for a batch of DOIs using ThreadPoolExecutor with rate limiting.
    
    Args:
        doi_list (List[str]): List of DOIs to fetch
        max_workers (int): Maximum number of concurrent workers
        
    Returns:
        Dict[str, Dict]: Dictionary mapping DOI to metadata record
    """
    results = {}
    
    def fetch_single_doi(doi: str) -> Tuple[str, Optional[Dict]]:
        metadata = fetch_crossref_metadata(doi)
        return doi, metadata
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_doi = {executor.submit(fetch_single_doi, doi): doi for doi in doi_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_doi):
            doi, metadata = future.result()
            if metadata:
                results[doi] = metadata
                logger.info(f"Successfully fetched metadata for DOI: {doi}")
            else:
                logger.warning(f"Failed to fetch metadata for DOI: {doi}")
    
    logger.info(f"Successfully fetched metadata for {len(results)}/{len(doi_list)} DOIs")
    return results


def update_dataset_with_metadata(dataset: Dataset, metadata_results: Dict[str, Dict], 
                                missing_instances: List[Tuple[int, str]], 
                                output_file: str = None) -> Tuple[Dataset, List[Dict]]:
    """
    Update dataset instances with fetched metadata and save modified instances to file.
    
    Args:
        dataset (Dataset): The original dataset
        metadata_results (Dict[str, Dict]): Fetched metadata keyed by DOI
        missing_instances (List[Tuple[int, str]]): List of (index, doi) pairs to update
        output_file (str): Path to save modified instances
        
    Returns:
        Tuple[Dataset, List[Dict]]: Updated dataset and list of modified instances
    """
    modified_instances = []
    dataset_dict = dataset.to_dict()
    
    for idx, doi in missing_instances:
        if doi in metadata_results:
            metadata = metadata_results[doi]
            abstract, journal_title, language, article_title = extract_metadata(metadata)
            
            # Store original values for tracking
            original_title = dataset_dict['title'][idx]
            original_abstract = dataset_dict['abstract'][idx]
            
            # Update missing values
            if not original_title and article_title:
                dataset_dict['title'][idx] = article_title
                logger.info(f"Updated title for DOI {doi}")
            
            if not original_abstract and abstract:
                dataset_dict['abstract'][idx] = abstract
                logger.info(f"Updated abstract for DOI {doi}")
            
            # Track modifications
            modified_instance = {
                'index': idx,
                'doi': doi,
                'title': dataset_dict['title'][idx],
                'abstract': dataset_dict['abstract'][idx],
                'journal_title': journal_title,
                'language': language,
                'updated_fields': []
            }
            
            if not original_title and article_title:
                modified_instance['updated_fields'].append('title')
            if not original_abstract and abstract:
                modified_instance['updated_fields'].append('abstract')
                
            modified_instances.append(modified_instance)
    
    # Create updated dataset
    updated_dataset = Dataset.from_dict(dataset_dict)
    
    # Save modified instances to file if specified
    if output_file and modified_instances:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = pd.DataFrame(modified_instances)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(modified_instances)} modified instances to {output_file}")
    
    return updated_dataset, modified_instances


def fill_missing_metadata(dataset: Dataset, output_file: str = None, max_workers: int = 5) -> Tuple[Dataset, List[Dict]]:
    """
    Complete pipeline to identify and fill missing metadata in a dataset.
    
    Args:
        dataset (Dataset): The dataset to process
        output_file (str): Path to save modified instances
        max_workers (int): Maximum concurrent workers for API requests
        
    Returns:
        Tuple[Dataset, List[Dict]]: Updated dataset and list of modified instances
    """
    logger.info("Starting metadata filling pipeline...")
    
    # Step 1: Identify instances with missing metadata
    missing_instances = identify_missing_metadata(dataset)
    
    if not missing_instances:
        logger.info("No instances with missing metadata found")
        return dataset, []
    
    # Step 2: Extract unique DOIs
    dois_to_fetch = list(set([doi for _, doi in missing_instances]))
    logger.info(f"Fetching metadata for {len(dois_to_fetch)} unique DOIs...")
    
    # Step 3: Fetch metadata in batches
    metadata_results = fetch_metadata_batch(dois_to_fetch, max_workers=max_workers)
    
    # Step 4: Update dataset with fetched metadata
    updated_dataset, modified_instances = update_dataset_with_metadata(
        dataset, metadata_results, missing_instances, output_file
    )
    
    logger.info(f"Metadata filling pipeline completed. Updated {len(modified_instances)} instances.")
    return updated_dataset, modified_instances


if __name__ == "__main__":
    # DOI to fetch (same as before)
    doi = "10.1890/02-5002"

    # enter query parameters and filters
    params = {
        'mailto': 'leandre.catogni@hesge.ch'
    }
    filters = {
    }
    
    # Fetch metadata
    metadata = fetch_crossref_metadata(doi, params, filters)
    
    if metadata:
        get_metadata(metadata)
    else:
        print("Failed to fetch metadata")
