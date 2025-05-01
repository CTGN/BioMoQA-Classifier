from Bio import Entrez
from bs4 import BeautifulSoup
import arxiv
import pandas as pd
import time
import random


Entrez.email = "leandre.catogni@hesge.ch" 
big_slow_client = arxiv.Client(
    page_size=5000,
    delay_seconds=10.0,
    num_retries=5
)

def get_pubmed_negatives(query, max_results=5000):
    handle = Entrez.esearch(
        db="pubmed", 
        term=query, 
        retmax=max_results, 
        sort="relevance"
    )
    pmids = Entrez.read(handle)["IdList"]
    handle.close()

    batch_size = 100
    all_articles = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml")
        xml_data = handle.read()
        handle.close()

        soup = BeautifulSoup(xml_data, "xml")
        for article in soup.find_all("PubmedArticle"):
            # Extract PMID
            pmid_tag = article.find('PMID')
            pmid = pmid_tag.get_text() if pmid_tag else None
            
            # Extract Abstract
            abstract_tag = article.find("AbstractText")
            if abstract_tag:
                all_articles.append({
                    'pmid': pmid,
                    'abstract': abstract_tag.get_text()
                })
        time.sleep(0.5)

    return all_articles

def get_arxiv_abstracts(categories, max_results):
    all_abstracts = []
    try:
        search = arxiv.Search(
            query=f"cat:{categories}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(big_slow_client.results(search))
        for paper in results:
            abstract = paper.summary.replace('\n', ' ').strip()
            all_abstracts.append(abstract)
    except Exception as e:
        print(f"Error fetching arXiv {categories}: {e}")
    return all_abstracts[:max_results]




def get_pubmed_positives(max_results=200):
    handle = Entrez.esearch(
        db="pubmed",
        term='(English[Language]) AND ("Microbial Consortia"[Mesh])',
        retmax=max_results,
        sort="relevance"
    )
    pmids = Entrez.read(handle)["IdList"]
    handle.close()

    positives = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml")
        xml_data = handle.read()
        soup = BeautifulSoup(xml_data, "xml")
        
        for article in soup.find_all("PubmedArticle"):
            pmid_tag = article.find('PMID')
            abstract_tag = article.find("AbstractText")
            
            # Validate both PMID and abstract exist and are non-empty
            if all([pmid_tag, abstract_tag]):
                pmid = pmid_tag.get_text().strip()
                abstract = abstract_tag.get_text().strip()
                if pmid and abstract:  # Ensure non-empty strings
                    positives.append({
                        'pmid': pmid,
                        'abstract': abstract
                    })
        time.sleep(0.5)

    return positives

# Fetch data
pubmed_negatives = get_pubmed_negatives('(English[Language]) AND ("Microbiota"[Mesh]) NOT ("Microbial Consortia"[Mesh])')
cs_negatives = get_arxiv_abstracts("cs.*", max_results=2500)
physics_negatives = get_arxiv_abstracts("physics.*", max_results=2500)
arxiv_negatives = cs_negatives + physics_negatives
provisional_positives = get_pubmed_positives(max_results=300)

# Create DataFrames
pubmed_neg_df = pd.DataFrame({
    'text': [art['abstract'] for art in pubmed_negatives],
    'PMID': [art['pmid'] for art in pubmed_negatives],
    'label': 0,
    'domain': 'in-neg'
})

arxiv_neg_df = pd.DataFrame({
    'text': arxiv_negatives,
    'PMID': [None] * len(arxiv_negatives),
    'label': 0,
    'domain': 'out-neg'
})

positive_df = pd.DataFrame({
    'text': [art['abstract'] for art in provisional_positives],
    'PMID': [art['pmid'] for art in provisional_positives],
    'label': 1,
    'domain': 'pos'
})

assert positive_df['PMID'].notnull().all(), "Some positive instances are missing PMIDs!"


# Combine datasets
dataset = pd.concat([positive_df, pubmed_neg_df, arxiv_neg_df], axis=0)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Save and report
dataset.to_csv("data/corpus/all.csv", index=False)
print(f"Dataset saved! Samples: {len(dataset)}")
print(f"Class balance:\n{dataset['label'].value_counts()}")
print(dataset.head())