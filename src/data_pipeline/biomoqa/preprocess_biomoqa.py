import gc
from sklearn.model_selection import StratifiedKFold
import os
import sys
import datasets
from .create_raw import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # Adjust ".." based on your structure

# Add it to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import CONFIG

def clean_data(df):
    """
    Deletes duplicates and conflicts in the dataset.
    """
    logger.info(f"{df.head()}")

    logger.info(f"{df.isnull().sum()}")

    df = df.dropna(subset=['title', 'abstract', 'Keywords', 'labels'])

    duplicates = df[df.duplicated(subset=['abstract','title','Keywords'], keep=False)]
    logger.info(f"Total duplicate abstracts: {len(duplicates)}")
    logger.info(f"{duplicates[['abstract', 'labels','Keywords']].sort_values('abstract')}")

    df_clean = df.drop_duplicates(subset=['abstract','title','Keywords'], keep='first')

    conflicts = df.groupby(['doi'])['labels'].nunique().reset_index()
    conflicts = conflicts[conflicts['labels'] > 1]

    logger.info(f"Abstracts appearing in both classes: {len(conflicts)}")
    logger.info(f"{conflicts}")

    # Free memory
    del duplicates
    del conflicts
    gc.collect()
    return df_clean


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
        neg_subset_train = neg.sample(n=balance_coeff*num_pos, random_state=42)
    else:
        neg_subset_train = neg  # Fallback (unlikely in your case)

    balanced_df = pd.concat([pos, neg_subset_train], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Final shuffle

    logger.info(f"Balanced columns: {balanced_df.columns}")
    logger.info(f"Balanced dataset size: {len(balanced_df)}")

    return balanced_df

def biomoqa_data_pipeline(n_folds,with_title, with_keywords, balanced=False, balance_coeff=5):
    og_df, optional_negatives_df = loading_pipeline()
    og_df=og_df[['Title', 'Abstract', 'Keywords', 'DOI','labels']]
    og_df.rename(columns={'Title': 'title', 'Abstract': 'abstract', 'DOI': 'doi'}, inplace=True)

    optional_negatives_df = optional_negatives_df[['title', 'text', 'MESH_terms', 'doi','labels']]
    optional_negatives_df.rename(columns={'MESH_terms': 'Keywords', 'text': 'abstract'}, inplace=True)

    all_df = pd.concat([og_df, optional_negatives_df], axis=0)
    logger.info(f"Combined dataset size: {len(all_df)}")
    clean_df = clean_data(all_df)
    logger.info(f"Cleaned dataset size: {len(clean_df)}")

     #First we do k-fold cross validation for testing
    #TODO : ensure that this does not change when running this pipeline different times for comparisons purposes
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CONFIG["seed"])
    folds = list()
    if with_title:
        if with_keywords:
            folds = list(skf.split(clean_df[['abstract','title','Keywords']], clean_df['labels']))
        else:
            folds = list(skf.split(clean_df[['abstract','title']], clean_df['labels']))
    elif not with_title:
        if with_keywords:
            folds = list(skf.split(clean_df[['abstract','Keywords']], clean_df['labels']))
        else:
            folds = list(skf.split(clean_df['abstract'].to_list(), clean_df['labels']))
        logging.info(f"fold 1 : {folds[0]}")
    else:
        raise ValueError("Invalid value for with_title. It must be True or False.")
    
    # Check distribution of labels in each fold
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_labels = clean_df.iloc[train_idx]["labels"]
        test_labels = clean_df.iloc[test_idx]["labels"]

        train_label_dist = train_labels.value_counts(normalize=True)
        test_label_dist = test_labels.value_counts(normalize=True)

        logger.info(f"Fold {fold_idx + 1}:")
        logger.info(f"  Train label distribution: {train_label_dist.to_dict()}")
        logger.info(f"  Test label distribution: {test_label_dist.to_dict()}")
    
    clean_df.loc[clean_df["labels"] == -1, "labels"] = 0

    clean_ds = datasets.Dataset.from_pandas(clean_df)

    clean_ds = clean_ds.class_encode_column("labels")

    return clean_ds, folds


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