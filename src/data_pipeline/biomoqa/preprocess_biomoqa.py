import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import sys
import datasets
import numpy as np
import argparse
import logging
import torch
from transformers import set_seed
import matplotlib.pyplot as plt

# Add project root to sys.path for imports
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data_pipeline.biomoqa.create_raw import *
from src.config import get_config
from src.utils.plot_style import (
    FIGURE_SIZES, PRIMARY_COLORS,
    create_figure, save_figure, setup_plotting_style
)

# Apply unified style
setup_plotting_style()

logger = logging.getLogger(__name__)

def set_reproducibility(seed):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Randomness sources seeded with {seed} for reproducibility.")

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

    config = get_config()
    seed = config.get('environment', {}).get('seed', 42)

    # Assuming your dataset has a 'label' column (adjust if needed)
    pos = df[df['labels'] == 1]
    neg = df[df['labels'] == 0]
    logger.info(f"Number of positives: {len(pos)}")
    logger.info(f"Number of negatives: {len(neg)}")
    num_pos = len(pos)

    # Ensure there are more negatives than positives before subsampling
    if len(neg) > num_pos:
        #TODO : Change the proportion value here for les or more imbalance -> compare different values, plot ? try less
        neg_subset_train = neg.sample(n=balance_coeff*num_pos, random_state=seed)
    else:
        neg_subset_train = neg  # Fallback (unlikely in your case)

    balanced_df = pd.concat([pos, neg_subset_train], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Final shuffle

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
    nb_optional_negs=5000,
    store=True
):
    og_df, optional_negatives_df = loading_pipeline()
    og_df = og_df[['Title', 'Abstract', 'Keywords', 'DOI', 'labels']]
    og_df.rename(columns={'Title': 'title', 'Abstract': 'abstract', 'DOI': 'doi'}, inplace=True)

    optional_negatives_df = optional_negatives_df[['title', 'text', 'MESH_terms', 'doi', 'labels']]
    optional_negatives_df.rename(columns={'MESH_terms': 'Keywords', 'text': 'abstract'}, inplace=True)

    config = get_config()
    seed = config.get('environment', {}).get('seed', 42)

    optional_negatives_df = optional_negatives_df.sample(n=nb_optional_negs, random_state=seed)
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


    # Count the number of positives, original negatives, and optional negatives
    n_positives = len(clean_og_df[clean_og_df['labels'] == 1])
    n_original_negatives = len(clean_og_df[clean_og_df['labels'] == 0])
    n_optional_negatives = len(opt_neg_df)

    labels = ['Original Positives', 'Original Negatives', 'Added Negatives']
    sizes = [n_positives, n_original_negatives, n_optional_negatives]
    colors = [PRIMARY_COLORS['blue'], PRIMARY_COLORS['orange'], PRIMARY_COLORS['green']]

    fig, ax = create_figure(figsize='square')
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')
    ax.set_title('Dataset Distribution', fontsize=14, fontweight='bold', pad=20)

    config = get_config()
    plots_dir = Path(config.get("plots_dir"))
    data_plots_dir = plots_dir / "data"
    data_plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, data_plots_dir / "dataset_distribution.png")
    plt.show()

    rng = np.random.RandomState(seed)
    derived_seeds = rng.randint(0, 1000000, size=n_runs)
    folds_per_run = []

    for seed in derived_seeds:
        # Stratified K‐Fold on only the original (non‐optional) data
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        run_folds = []
        for train_dev_pos_idx, test_pos_idx in skf.split(clean_og_df['abstract'], clean_og_df['labels']):
            # Convert positional indices back to the DataFrame's original index
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

    for run_folds in folds_per_run:
        for i in range(len(run_folds)):
            train_indices, dev_indices, test_indices = run_folds[i]
            opt_neg_indices = set(opt_neg_df.index)
            test_indices = [idx for idx in test_indices if idx not in opt_neg_indices]
            run_folds[i] = [train_indices, dev_indices, test_indices]

    for run_idx, run_folds in enumerate(folds_per_run):
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

def main():

    parser = argparse.ArgumentParser(description="Preprocess BioMoQA dataset")
    parser.add_argument("-b","--balanced", action="store_true", help="Whether to balance the dataset")
    parser.add_argument("-bc","--balance_coeff", type=int, default=5, help="Coefficient for balancing the dataset")
    parser.add_argument("-nf","--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("-nr","--n_runs", type=int, default=2, help="Number of runs for cross-validation")
    parser.add_argument("-t","--with_title", action="store_true", help="Whether to include title in the dataset")
    parser.add_argument("-k","--with_keywords", action="store_true", help="Whether to include keywords in the dataset")
    parser.add_argument("-on","--nb_opt_negs", type=int, default=500, help="Number of optional negatives to add in the train splits")


    args = parser.parse_args()

    config = get_config()
    seed = config.get('environment', {}).get('seed', 42)
    set_reproducibility(seed)

    logger.info(args)

    dataset,folds_per_run=biomoqa_data_pipeline(args.n_folds, n_runs=args.n_runs, with_title=args.with_title, with_keywords=args.with_keywords, balanced=args.balanced, balance_coeff=args.balance_coeff,nb_optional_negs=args.nb_opt_negs)

    for run_idx in range(len(folds_per_run)):
        folds=folds_per_run[run_idx]
        for fold_idx in range(args.n_folds):

            train_indices, dev_indices,test_indices = folds[fold_idx]

            logger.info(f"\nfold number {fold_idx+1} / {len(folds)}")
            
            train_split = dataset.select(train_indices)
            dev_split = dataset.select(dev_indices)
            test_split = dataset.select(test_indices)

            logger.info(f"train split size : {len(train_split)}")
            logger.info(f"dev split size : {len(dev_split)}")
            logger.info(f"test split size : {len(test_split)}")

            config = get_config()
            test_split.to_pandas().to_csv(config.get_fold_path("test", fold_idx, run_idx))
            train_split.to_pandas().to_csv(config.get_fold_path("train", fold_idx, run_idx))
            dev_split.to_pandas().to_csv(config.get_fold_path("dev", fold_idx, run_idx))


if __name__ == "__main__":
    main()