import os
import requests
import json

# CrossRef REST API base URL
CROSSREF_BASE_URL = "https://api.crossref.org/works/"


def fetch_crossref_metadata(doi: str, params:dict,filters:dict) -> dict | None:
    """
    Fetch metadata for a given DOI from the CrossRef REST API.
    
    Args:
        doi (str): The DOI of the work to fetch metadata for
        
    Returns:
        dict | None: The metadata record if successful, None if failed
    """
    try:
        # Make request to CrossRef API
        url = f"{CROSSREF_BASE_URL}{doi}"
        
        if len(list(filters.keys())) > 0:
            fval = ''
            # add each filter key and value to the string
            for f in filters:
                fval += str(f) + ':' + str(filters[f]) + ','
            fval = fval[:-1] # removing trailing comma
            params['filter'] = fval

        # make the query
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None
    except KeyError as e:
        print(f"Error parsing response: {e}")
        return None
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def print_metadata(record: dict) -> None:
    """
    Print formatted metadata information from a CrossRef record.
    
    Args:
        record (dict): The metadata record from CrossRef API
    """
    print("Record fetched successfully:")
    print(json.dumps(record, indent=2))
    
    # Extract and print journal title
    if 'container-title' in record and len(record['container-title']) > 0:
        journal_title = record['container-title'][0]
        print(f"Journal title: {journal_title}")
    else:
        print("Journal title: Not available")
        
    # Print article title
    if 'title' in record and len(record['title']) > 0:
        print(f"Article title: {record['title'][0]}")
    
    # Print authors
    if 'author' in record:
        authors = [f"{author.get('given', '')} {author.get('family', '')}" for author in record['author']]
        print(f"Authors: {', '.join(authors)}")


if __name__ == "__main__":
    # DOI to fetch (same as before)
    doi = "10.1890/02-5002"

    # enter query parameters and filters
    params = {
        'mailto': 'demo@crossref.org'
    }
    filters = {
    }
    
    # Fetch metadata
    metadata = fetch_crossref_metadata(doi)
    
    if metadata:
        print_metadata(metadata)
    else:
        print("Failed to fetch metadata")
