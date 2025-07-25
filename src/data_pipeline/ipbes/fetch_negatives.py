import os
import sys

parent_dir = os.path.abspath("/home/leandre/Projects/BioMoQA_Playground")

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data_pipeline.ipbes.create_ipbes_raw import loading_pipeline_from_raw

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from tqdm import tqdm
from time import sleep

# ------------- CONFIGURATION -------------
EMAIL = "leandre.catogni@hesge.ch"
USER_AGENT = f"mailto:{EMAIL}"
BATCH_SIZE = 100
TIMEOUT = 60  # seconds
EXPORT_CSV = True
CSV_PATH = "/home/leandre/Projects/BioMoQA_Playground/data/IPBES/modified_instances/openalex_sync_results.csv"

# Retry on 429, 5xx, connection errors, timeouts
RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],       # <- updated parameter name
    raise_on_status=False
)
# -----------------------------------------

# 1) Prepare session
session = requests.Session()
adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
session.mount("https://", adapter)
session.headers.update({"User-Agent": USER_AGENT})

datasets = loading_pipeline_from_raw(multi_label=True)
neg_ds=datasets[1]

# 2) Clean IDs
ids = [i.replace("https://openalex.org/", "") for i in neg_ds['id'] if isinstance(i, str) and i]
total = len(ids)

# 3) Batch generator
def chunker(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

results = []
failures = 0

# 4) Loop with progress bar
with tqdm(total=total, desc="OpenAlex works fetched") as pbar:
    for batch in chunker(ids, BATCH_SIZE):
        params = {
            "filter": f"openalex:{'|'.join(batch)}",
            "per-page": len(batch),
            "select": "id,doi,title,publication_year"
        }
        try:
            resp = session.get("https://api.openalex.org/works", params=params, timeout=TIMEOUT)
            data = resp.json()
            if resp.status_code == 200 and "results" in data:
                results.extend(data["results"])
                pbar.update(len(batch))
            else:
                # log partial failure and retry individually
                failures += len(batch)
                pbar.update(len(batch))
                for single in batch:
                    try:
                        single_resp = session.get(
                            f"https://api.openalex.org/works/{single}",
                            timeout=TIMEOUT,
                            params={"select": "id,doi,title,publication_year"}
                        )
                        single_data = single_resp.json()
                        if single_resp.status_code == 200:
                            results.append(single_data)
                        else:
                            failures += 1
                    except Exception:
                        failures += 1
                    finally:
                        pbar.update(1)
        except Exception as e:
            # whole-batch failure: treat as individual failures
            failures += len(batch)
            pbar.update(len(batch))

        # polite pause
        sleep(0.2)

# 5) Summary & Export
print(f"\n✅ Done! Fetched {len(results)} works; {failures} failures.")
if EXPORT_CSV and results:
    pd.DataFrame(results).to_csv(CSV_PATH, index=False)
    print(f"📁 Saved results to {CSV_PATH}")
