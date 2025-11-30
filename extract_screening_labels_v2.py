#!/usr/bin/env python3
"""
Extract positives and negatives from screening outcome - improved version.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_labels(file_path: str, output_dir: str):
    """
    Extract positives (included) and negatives (excluded) from screening outcome.

    Positives = instances with ANY rating from ANY rater
    Negatives = instances with NO ratings from any rater
    """
    logger.info(f"Loading: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows")

    # Skip the header row with rater names (row 0)
    df = df.iloc[1:].reset_index(drop=True)

    logger.info(f"Processing {len(df)} instances after skipping header")

    # Define rating columns for both raters
    # Rater 1 (Giorgia): Comment columns
    # Rater 2 (Jordyn): Unnamed columns
    rating_columns = [
        'Comment: Island (Population); ',
        'Unnamed: 39',
        'Comment: Terrestrial Biodiversity (Population); ',
        'Unnamed: 41',
        'Comment: Monitoring&nbsp; (Outcome); ',
        'Unnamed: 43'
    ]

    # Check which columns exist
    available_rating_cols = [col for col in rating_columns if col in df.columns]
    logger.info(f"Using rating columns: {available_rating_cols}")

    # Classify instances
    positives_indices = []
    negatives_indices = []

    for idx, row in df.iterrows():
        # Check if ANY rating column has a non-empty value
        has_any_rating = False

        for col in available_rating_cols:
            val = str(row.get(col, '')).strip()
            # Check for meaningful values (not NaN, not empty)
            if val and val.lower() not in ['nan', 'none', '']:
                has_any_rating = True
                break

        if has_any_rating:
            positives_indices.append(idx)
        else:
            negatives_indices.append(idx)

    # Create DataFrames
    positives_df = df.loc[positives_indices].copy()
    negatives_df = df.loc[negatives_indices].copy()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save
    pos_path = output_dir / 'screening_positives.csv'
    neg_path = output_dir / 'screening_negatives.csv'

    positives_df.to_csv(pos_path, index=False)
    negatives_df.to_csv(neg_path, index=False)

    logger.info(f"\n{'='*70}")
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nTotal instances: {len(df)}")
    logger.info(f"  - POSITIVES (included): {len(positives_df)} ({100*len(positives_df)/len(df):.1f}%)")
    logger.info(f"  - NEGATIVES (excluded): {len(negatives_df)} ({100*len(negatives_df)/len(df):.1f}%)")

    logger.info(f"\nSaved to:")
    logger.info(f"  - {pos_path}")
    logger.info(f"  - {neg_path}")

    # Show sample ratings
    if len(positives_df) > 0:
        logger.info(f"\nSample POSITIVE:")
        sample = positives_df.iloc[0]
        title = str(sample.get('Title', 'N/A'))
        if len(title) > 70:
            title = title[:70] + "..."
        logger.info(f"  Title: {title}")
        for col in available_rating_cols:
            val = sample.get(col)
            if pd.notna(val) and str(val).strip():
                logger.info(f"  {col}: {val}")

    if len(negatives_df) > 0:
        logger.info(f"\nSample NEGATIVE:")
        sample = negatives_df.iloc[0]
        title = str(sample.get('Title', 'N/A'))
        if len(title) > 70:
            title = title[:70] + "..."
        logger.info(f"  Title: {title}")
        logger.info(f"  (No ratings from any rater)")

    return pos_path, neg_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract screening labels v2')
    parser.add_argument('--input', '-i', required=True, help='Input screening outcome CSV')
    parser.add_argument('--output', '-o', default='results/screening_labels', help='Output directory')

    args = parser.parse_args()

    extract_labels(args.input, args.output)
