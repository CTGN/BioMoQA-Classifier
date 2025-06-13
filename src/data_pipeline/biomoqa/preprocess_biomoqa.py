import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import sys
import datasets
import numpy as np
from .create_raw import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # Adjust ".." based on your structure

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import CONFIG

def clean_data(df):
    logger.info("Before cleaning, head:\n%s", df.head())
    logger.info("Null counts before cleaning:\n%s", df.isnull().sum())

    # Drop any row missing at least one of these four required columns
    df = df.dropna(subset=['title', 'abstract', 'Keywords', 'labels'])

    # Iteratively drop duplicates on title → abstract → doi
    for col in ('title', 'abstract', 'doi'):
        # Drop any row where this col is missing
        df = df.dropna(subset=[col])

        # Log how many true duplicates (i.e., identical values in this column)
        duplicates = df[df.duplicated(subset=[col], keep=False)]
        logger.info("Total duplicate %s: %d", col, len(duplicates))

        # If there are any conflicts (same col value but different labels)
        conflict_counts = (
            df.groupby(col)['labels']
              .nunique()
              .reset_index(name='n_labels')
              .query("n_labels > 1")
        )
        logger.info("Conflicts on %s (same value, multiple labels):\n%s", col, conflict_counts)

        # Now actually drop duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=[col], keep='first')

    return df



def pmids_to_text(train_df, test_df):
    """Write PMIDs to files in chunks to reduce memory usage."""
    chunk_size = 10000
    
    for df, prefix in [(train_df, 'train'), (test_df, 'test')]:
        for label, label_text in [(1, 'pos'), (0, 'neg')]:
            pmids = (df[df["labels"]==label]['PMID']).dropna().astype(int).astype(str)
            
            with open(f"{prefix}_{label_text}_pmids.txt", "w") as f:
                for i in range(0, len(pmids), chunk_size):
                    chunk = pmids[i:i + chunk_size].tolist()
                    f.write(" ".join(chunk) + " ")
            
            logger.info(f"\nSaved {len(pmids)} PMIDs to {prefix}_{label_text}_pmids.txt")
            del pmids
            gc.collect()

def balance_dataset(df, balance_coeff):
    """
    - Performs undersampling on the negatives
    - Renames the abstract column -> we should not hqve to do that 
    """

    def is_label(batch,label):
        batch_bools=[]
        for ex_label in batch['labels']:
            if ex_label == label:
                batch_bools.append(True)
            else:
                batch_bools.append(False)
        return batch_bools

    # Assuming your dataset has a 'label' column (adjust if needed)
    pos = df[df['labels'] == 1]
    neg = df[df['labels'] == 0]
    logger.info(f"Number of positives: {len(pos)}")
    logger.info(f"Number of negatives: {len(neg)}")
    num_pos = len(pos)

    # Ensure there are more negatives than positives before subsampling
    if len(neg) > num_pos:
        #TODO : Change the proportion value here for les or more imbalance -> compare different values, plot ? try less
        neg_subset_train = neg.sample(n=balance_coeff*num_pos, random_state=CONFIG["seed"])
    else:
        neg_subset_train = neg  # Fallback (unlikely in your case)

    balanced_df = pd.concat([pos, neg_subset_train], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)  # Final shuffle

    logger.info(f"Balanced columns: {balanced_df.columns}")
    logger.info(f"Balanced dataset size: {len(balanced_df)}")

    return balanced_df

def biomoqa_data_pipeline(
    n_folds,
    n_runs,
    with_title,
    with_keywords,
    balanced=False,
    balance_coeff=5,
    nb_optional_negs=5000
):
    og_df, optional_negatives_df = loading_pipeline()
    og_df = og_df[['Title', 'Abstract', 'Keywords', 'DOI', 'labels']]
    og_df.rename(columns={'Title': 'title', 'Abstract': 'abstract', 'DOI': 'doi'}, inplace=True)

    optional_negatives_df = optional_negatives_df[['title', 'text', 'MESH_terms', 'doi', 'labels']]
    optional_negatives_df.rename(columns={'MESH_terms': 'Keywords', 'text': 'abstract'}, inplace=True)

    optional_negatives_df = optional_negatives_df.sample(n=nb_optional_negs, random_state=CONFIG['seed'])
    all_df = pd.concat([og_df, optional_negatives_df], ignore_index=True)
    logger.info(f"Combined dataset size: {len(all_df)}")

    clean_df = clean_data(all_df)
    clean_df = clean_df.reset_index(drop=True)
    logger.info(f"Cleaned dataset size: {len(clean_df)}")
    logger.info(f"Number of positives : {len(clean_df[clean_df['labels'] == 1])}")
    logger.info(f"Number of negatives : {len(clean_df[clean_df['labels'] != 1])}")

    clean_og_df = clean_df[clean_df['labels'] != -1]
    opt_neg_df = clean_df[clean_df['labels'] == -1]
    logger.info(f"clean_og_df size : {len(clean_og_df)}")
    logger.info(f"opt_neg_df size : {len(opt_neg_df)}")

    rng = np.random.RandomState(CONFIG["seed"])
    derived_seeds = rng.randint(0, 1000000, size=n_runs)
    folds_per_run = []

    for seed in derived_seeds:
        # Stratified K‐Fold on only the original (non‐optional) data
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        run_folds = []
        for train_dev_pos_idx, test_pos_idx in skf.split(clean_og_df['abstract'], clean_og_df['labels']):
            # Convert positional indices back to the DataFrame’s original index
            train_dev_indices = clean_og_df.index[train_dev_pos_idx]
            test_indices = clean_og_df.index[test_pos_idx].to_list()

            # Split train_dev into train vs. dev
            train_indices, dev_indices = train_test_split(
                train_dev_indices,
                stratify=clean_og_df.loc[train_dev_indices, "labels"],
                shuffle=True,
                random_state=seed
            )

            train_indices = train_indices.to_list()
            dev_indices = dev_indices.to_list()

            # Add all optional negatives to the train set
            train_indices.extend(opt_neg_df.index.to_list())

            run_folds.append([train_indices, dev_indices, test_indices])

            # Log distributions
            train_labels = clean_df.loc[train_indices, "labels"]
            test_labels = clean_df.loc[test_indices, "labels"]
            train_label_dist = train_labels.value_counts(normalize=True)
            test_label_dist = test_labels.value_counts(normalize=True)
            logger.info(f"Fold {len(run_folds)}:")
            logger.info(f"  Train label distribution: {train_label_dist.to_dict()}")
            logger.info(f"  Test label distribution: {test_label_dist.to_dict()}")

        folds_per_run.append(run_folds)

    # Rename optional‐negative labels from -1 → 0
    clean_df.loc[clean_df["labels"] == -1, "labels"] = 0

    # Check if reset_index altered the indices
    original_indices = clean_df.index
    reset_indices = clean_df.reset_index().index
    if not original_indices.equals(reset_indices):
        logger.warning("Indices of clean_df have changed after resetting.")
    else:
        logger.info("Indices of clean_df remain the same after resetting.")

    clean_ds = datasets.Dataset.from_pandas(clean_df)
    logger.info(f"clean_df index : {clean_df.index}")

    clean_ds = clean_ds.class_encode_column("labels")
    logger.info(f"Number of positives : {len(clean_df[clean_df['labels'] == 1])}")
    logger.info(f"Number of negatives : {len(clean_df[clean_df['labels'] == 0])}")

    # Ensure test indices do not include any optional negatives (just in case)
    for run_folds in folds_per_run:
        for i in range(len(run_folds)):
            train_indices, dev_indices, test_indices = run_folds[i]
            opt_neg_indices = set(opt_neg_df.index)
            test_indices = [idx for idx in test_indices if idx not in opt_neg_indices]
            run_folds[i] = [train_indices, dev_indices, test_indices]

    # === New block: Check independence of test‐folds within each run ===
    for run_idx, run_folds in enumerate(folds_per_run):
        # Collect all test‐sets in this run
        test_sets = [set(fold[2]) for fold in run_folds]
        n_f = len(test_sets)
        for i in range(n_f):
            for j in range(i + 1, n_f):
                overlap = test_sets[i].intersection(test_sets[j])
                if len(overlap) > 0:
                    raise ValueError(
                        f"Overlap detected between fold {i+1} and fold {j+1} in run {run_idx+1}: "
                        f"{len(overlap)} shared indices."
                    )
        logger.info(f"All {n_f} test‐folds in run {run_idx+1} are pairwise disjoint.")

    return clean_ds, folds_per_run


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess BioMoQA dataset")
    parser.add_argument("--balanced", action="store_true", help="Whether to balance the dataset")
    parser.add_argument("--balance_coeff", type=int, default=5, help="Coefficient for balancing the dataset")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--with_title", action="store_true", help="Whether to include title in the dataset")
    parser.add_argument("--with_keywords", action="store_true", help="Whether to include keywords in the dataset")
    args = parser.parse_args()

    biomoqa_data_pipeline(args.n_folds, args.with_title, args.with_keywords, balanced=args.balanced, balance_coeff=args.balance_coeff)

    # Example usage of pmids_to_text
    # pmids_to_text(train_df, test_df)