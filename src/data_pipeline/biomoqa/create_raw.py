from Bio import Entrez
from bs4 import BeautifulSoup
import arxiv
import pandas as pd
import time
import random
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import argparse


Entrez.email = "leandre.catogni@hesge.ch" 
big_slow_client = arxiv.Client(
    page_size=5000,
    delay_seconds=3.0,
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
            abstract = abstract_tag.get_text() if abstract_tag else None
            
            # Extract Title
            title_tag = article.find("ArticleTitle")
            title = title_tag.get_text() if title_tag else None
            
            # Extract DOI
            doi_tag = article.find("ELocationID", {"EIdType": "doi"})
            doi = doi_tag.get_text() if doi_tag else None
            
            # Extract MeSH terms
            mesh_terms = []
            for mesh_heading in article.find_all("MeshHeading"):
                descriptor = mesh_heading.find("DescriptorName")
                if descriptor:
                    mesh_terms.append(descriptor.get_text())

            all_articles.append({
                'pmid': pmid,
                'abstract': abstract,
                'title': title,
                'doi': doi,
                'mesh_terms': mesh_terms
            })
        time.sleep(0.5)

    return all_articles

def get_arxiv_abstracts(categories, max_results):
    all_papers = []
    try:
        search = arxiv.Search(
            query=f"cat:{categories}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(big_slow_client.results(search))
        for paper in results:
            abstract = paper.summary.replace('\n', ' ').strip()
            title = paper.title.replace('\n', ' ').strip()
            doi = paper.doi if paper.doi else None
            all_papers.append({
                'title': title,
                'abstract': abstract,
                'doi': doi
            })
    except Exception as e:
        logger.info(f"Error fetching arXiv {categories}: {e}")
    return all_papers[:max_results]

def fetch_negatives(include_arxiv=False):
    # Fetch data
    logger.info(f"Fetching PubMed negatives...")
    pubmed_negatives = get_pubmed_negatives('(English[Language]) AND Environment[MeSH Terms] AND ("2021/01/01"[Date - Publication] : "2025/12/31"[Date - Publication]) NOT (Islands[MeSH Terms]) NOT Islands[MeSH:noexp] NOT (island*[Title/Abstract] OR archipelago*[Title/Abstract] OR atoll[Title/Abstract] OR insular[Title/Abstract] OR "Hawaii"[Title/Abstract] OR "Galapagos"[Title/Abstract]) AND (fha[Filter])')
    logger.info(f"Fetched {len(pubmed_negatives)} PubMed negative articles.")
    if include_arxiv:
        logger.info(f"Fetching arXiv negatives...")
        cs_negatives = get_arxiv_abstracts("cs.*", max_results=2500)
        logger.info(f"Fetched {len(cs_negatives)} arXiv CS negative articles.")
        physics_negatives = get_arxiv_abstracts("physics.*", max_results=2500)
        logger.info(f"Fetched {len(physics_negatives)} arXiv Physics negative articles.")
        arxiv_negatives = cs_negatives + physics_negatives

    # Create DataFrames
    pubmed_neg_df = pd.DataFrame({
        'title': [art['title'] for art in pubmed_negatives],
        'text': [art['abstract'] for art in pubmed_negatives],
        'PMID': [art['pmid'] for art in pubmed_negatives],
        'MESH_terms': [', '.join(art['mesh_terms']) for art in pubmed_negatives],
        'doi': [art['doi'] for art in pubmed_negatives],
        'source': 'pubmed',
        'labels': -1,
    })
    if include_arxiv:
        logger.info("Arxiv negatives included in the dataset.")
        arxiv_neg_df = pd.DataFrame({
            'title': [art['title'] for art in arxiv_negatives],
            'text': arxiv_negatives,
            'PMID': [None] * len(arxiv_negatives),
            'MESH_terms': [None] * len(arxiv_negatives),
            'doi': [art['doi'] for art in arxiv_negatives],
            'source': 'arxiv',
            'labels': -1,
        })
        optional_negatives = pd.concat([pubmed_neg_df, arxiv_neg_df])
    else:
        optional_negatives = pubmed_neg_df

    # Save to project data directory
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    optional_negatives.to_csv(data_dir / "optional_negatives.csv", index=False)
    logger.info(f"Optional negatives saved to CSV with {len(optional_negatives)} entries.")

    return optional_negatives

def loading_pipeline(fetch=False):
    if fetch:
        optional_negatives=fetch_negatives()
    else:
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = project_root / "data"
        optional_negatives = pd.read_csv(data_dir / "optional_negatives.csv")
        logger.info(f"Optional negatives loaded from CSV with {len(optional_negatives)} entries.")

    logger.info(f"Loading original negatives...")
    negatives_df = pd.read_csv(data_dir / "negatives.csv")
    negatives_df["labels"] = 0
    negatives_df["Keywords"]=""
    logger.info(f"Negatives column names: {negatives_df.columns.tolist()}")

    logger.info(f"Loading original positives...")
    positive_df = pd.read_csv(data_dir / "positives.csv")
    positive_df["labels"] = 1
    logger.info(f"Positives column names: {positive_df.columns.tolist()}")


    # Combine datasets
    og_dataset = pd.concat([positive_df, negatives_df], axis=0)
    og_dataset = og_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Class balance:\n{og_dataset['labels'].value_counts()}")
    logger.info(og_dataset.head())
    logger.info(f"Original dataset size: {len(og_dataset)}")
    logger.info(f"Optional negatives size: {len(optional_negatives)}")
    logger.info(f"Optional negatives: {optional_negatives.head()}")

    return og_dataset, optional_negatives

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BioMoQA dataset")
    parser.add_argument("-f","--fetch", action="store_true", help="Whether to fetch optional negatives from PubMed")


    args = parser.parse_args()

    logger.info("Starting BioMoQA data loading pipeline...")
    og_dataset, optional_negatives = loading_pipeline(fetch=args.fetch)
    