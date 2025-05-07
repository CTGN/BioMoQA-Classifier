# src/fetch.py
import asyncio
import aiohttp
import logging
import pandas as pd
import os
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import utils.pub_lib as pub_lib
import hydra
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def fetch_publications_from_sibils(session, url, ids, collection):
    """
    Fetch publications from the SIBILS API using the given IDs and collection.
    """
    try:
        params = {"ids": ",".join(map(str, ids)), "col": collection}
        async with session.post(url + "/fetch", params=params) as response:
            if response.status != 200:
                log.error(f"Error fetching publications from {collection}: HTTP {response.status}")
                return {}
            res = await response.json()
        if "sibils_article_set" in res:
            log.info(f"Fetched {len(res['sibils_article_set'])} publications from {collection}")
            return {item.pop("_id"): item for item in res["sibils_article_set"]}
    except aiohttp.ClientError as e:
        log.error(f"ClientError while fetching publications from {collection}: {e}")
    except asyncio.TimeoutError as e:
        log.error(f"TimeoutError while fetching publications from {collection}: {e}")
    return {}

# async def extract_fulltext(document):
#     """
#     Extract full text from a document by concatenating all 'p' tag contents from body_sections.
#     Ensures each paragraph ends with a period if it doesn't already.
#     """
#     paragraphs = []
#     for section in document.get("body_sections", []):
#         for content in section.get("contents", []):
#             if content.get("tag") == "p":
#                 text = content.get("text", "").strip()
#                 if text and not text.endswith("."):
#                     text += "."
#                 if text:
#                     paragraphs.append(text)
#     return " ".join(paragraphs)


async def extract_fulltext(document):
    """
    Extracts full text with cleaner logic:
    - Skips generic titles like "Title", "Abstract"
    - Keeps box/section captions and labels if sentence-like
    - Adds 'Box:' for labeled sections
    - Ensures punctuation on all text
    - Bullet-formats list items
    - Moves footnotes to end
    """
    sentence_end_re = re.compile(r'[.?!…]["\')\]]?$')
    main_text_segments = []
    footnotes = []

    def ensure_punctuated(text):
        text = text.strip()
        if text and not sentence_end_re.search(text):
            return text + "."
        return text

    def is_valid_header(text):
        # Avoid including generic headers like "Title", "Abstract", or empty ones
        return text and text.lower().strip() not in {"title", "abstract"}

    for section in document.get("body_sections", []):
        section_parts = []

        # Mark boxed content
        if section.get("label", "").strip().lower() == "box":
            section_parts.append("Box:")

        # Include caption and label if valid
        for key in ["label", "caption"]:
            val = section.get(key, "").strip()
            if is_valid_header(val) and val.lower() != "box":
                section_parts.append(ensure_punctuated(val))

        # Include title only if it's not a generic header
        title = section.get("title", "").strip()
        if is_valid_header(title):
            section_parts.append(ensure_punctuated(title))

        # Process section content
        for content in section.get("contents", []):
            text = content.get("text", "").strip()
            tag = content.get("tag", "")

            if not text:
                continue

            formatted = ensure_punctuated(text)

            if tag == "list-item":
                section_parts.append(f"- {formatted}")
            elif tag == "fn":
                footnotes.append(formatted)
            else:
                section_parts.append(formatted)

        if section_parts:
            main_text_segments.append(" ".join(section_parts))

    full_text = " ".join(main_text_segments)
    if footnotes:
        full_text += "\n\nFootnotes:\n" + "\n".join(f"- {fn}" for fn in footnotes)

    return full_text

async def fetch_and_store_publications(sem, session, url, pmids, publication_dir, log_file, overwrite):
    """
    Fetch publications using PMIDs, store those with or without full text, and log failures or missing full text.
    """
    try:
        async with sem:
            # Skip PMIDs that already exist unless overwrite is True
            if not overwrite:
                filtered_pmids = [pmid for pmid in pmids if not pub_lib.pub_exists(publication_dir, pmid)]
            else:
                filtered_pmids = pmids

            if not filtered_pmids:
                log.info(f"All PMIDs already exist, skipping batch.")
                return

            # Fetch Medline data
            medline_publications = await fetch_publications_from_sibils(session, url, filtered_pmids, "medline")
            pmcids, publications = [], []
            for pmid, publication_data in medline_publications.items():
                pmcid = publication_data["document"].get("pmcid")
                publication = {
                    "PMID": pmid,
                    "TITLE": publication_data["document"]["title"],
                    "ABSTRACT": publication_data["document"]["abstract"],
                    "FULLTEXT": None,
                    "PMCID": pmcid,
                }
                if pmcid:
                    pmcids.append(pmcid)
                publications.append(publication)

            # Fetch full-text data from PMC if available
            if pmcids:
                pmc_publications = await fetch_publications_from_sibils(session, url, pmcids, "pmc")
                for publication in publications:
                    if publication["PMCID"] in pmc_publications:
                        publication_data = pmc_publications[publication["PMCID"]]["document"]
                        publication["FULLTEXT"] = await extract_fulltext(publication_data)

            # Store publications (with or without full-text) and log any that are missing full-text
            stored_count = 0
            for publication in publications:
                try:
                    pub_lib.save_pub(publication, publication_dir, overwrite=overwrite)
                    stored_count += 1
                    if not publication["FULLTEXT"]:
                        with open(log_file, 'a') as lf:
                            lf.write(f"PMID: {publication['PMID']} - Only abstract available, no full-text.\n")
                except FileExistsError:
                    log.info(f"Publication for PMID {publication['PMID']} already exists, skipping save.")

            log.info(f"Stored {stored_count} publications in {publication_dir} (abstract and/or full-text).")
    except Exception as e:
        log.error(f"Error processing publications: {e}")
        with open(log_file, 'a') as lf:
            lf.write(f"Error processing batch: {e}\n")

async def run(cfg: DictConfig):
    """
    Run the main publication fetching process based on the datasets listed in the configuration.
    """
    try:
        dataset_info = cfg.dataset
        log.info(f"Processing dataset: {dataset_info.id}")

        # Load PMIDs from the processed CSV (standardized dataset)
        processed_csv = to_absolute_path(cfg.dataset.fetch.input.std)
        df = pd.read_csv(processed_csv)
        pmid_set = df['PMID'].drop_duplicates().astype(str).tolist()

        log.info(f"{len(pmid_set)} PMIDs to process for {dataset_info.id}")
        batches = [pmid_set[i:i + cfg.sibils.batch_size] for i in range(0, len(pmid_set), cfg.sibils.batch_size)]

        # Use dynamically generated absolute paths for shared publication directory and dataset-specific logs
        publication_dir = to_absolute_path(cfg.dataset.fetch.output.publication_dir)
        log_file = to_absolute_path(cfg.dataset.fetch.output.log)
        summary_file = to_absolute_path(cfg.dataset.fetch.output.summary)

        # Ensure the log file directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Write initial message to log file to ensure it's being created
        with open(log_file, 'w') as lf:
            lf.write(f"Fetch process started for dataset {dataset_info.id}\n")
        log.info(f"Log file created at {log_file}")

        connector = aiohttp.TCPConnector(limit_per_host=cfg.sibils.limit_per_host)
        sem = asyncio.Semaphore(cfg.sibils.semaphore_size)

        # Pass the overwrite option to the fetch function
        overwrite = cfg.overwrite

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                fetch_and_store_publications(sem, session, cfg.sibils.url, batch, publication_dir, log_file, overwrite)
                for batch in batches
            ]
            await asyncio.gather(*tasks)

        # Write summary information to summary file
        with open(summary_file, 'w') as sf:
            sf.write(f"Fetch process completed for dataset {dataset_info.id}\n")
        log.info(f"Fetch process completed for dataset {dataset_info.id}. Summary saved to {summary_file}")
        
    except Exception as e:
        log.error(f"Unexpected error in run: {e}")
        with open(log_file, 'a') as lf:
            lf.write(f"Error during fetch process: {e}\n")

@hydra.main(config_path="../conf", config_name="fetch")
def main(cfg: DictConfig):
    asyncio.run(run(cfg))

if __name__ == "__main__":
    main()
