#!/usr/bin/env python3
"""
Score OpenAlex data directly using the BioMoQA predictor (without API).
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Setup
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.models.biomoqa.model_api import BioMoQAEnsemblePredictor


def score_openalex(input_file: str, output_file: str):
    """Score OpenAlex data file"""

    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")

    # Check columns
    if 'abstract' not in df.columns:
        logger.error("Error: 'abstract' column is required")
        logger.info(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Prepare batch
    batch = []
    for idx, row in df.iterrows():
        abstract = str(row.get('abstract', '')).strip()
        title = str(row.get('title', row.get('display_name', ''))).strip()

        if abstract and abstract != 'nan':
            batch.append({
                'abstract': abstract,
                'title': title if title != 'nan' else None
            })

    logger.info(f"Prepared {len(batch)} valid records for scoring")

    # Load predictor
    logger.info("Loading BioMoQA ensemble predictor...")
    predictor = BioMoQAEnsemblePredictor(
        model_type="roberta-base",  # or use default from config
        loss_type="BCE",
        base_path=None,  # uses default
        threshold=0.5,
        nb_opt_negs=0,  # update this if models were trained with optional negatives
        device="auto",
    )

    # Score
    logger.info("Scoring batch...")
    results = predictor.score_batch_ultra_optimized(
        batch,
        base_batch_size=16,
        use_dynamic_batching=True
    )

    logger.info(f"✓ Scored {len(results)} records")

    # Merge results back with original data
    scored_df = df.copy()

    # Add score columns
    scored_df['ensemble_score'] = None
    scored_df['ensemble_prediction'] = None
    scored_df['std_score'] = None
    scored_df['min_score'] = None
    scored_df['max_score'] = None
    scored_df['consensus_strength'] = None
    scored_df['positive_folds'] = None

    result_idx = 0
    for df_idx, row in df.iterrows():
        abstract = str(row.get('abstract', '')).strip()
        if abstract and abstract != 'nan' and result_idx < len(results):
            result = results[result_idx]
            scored_df.at[df_idx, 'ensemble_score'] = result['ensemble_score']
            scored_df.at[df_idx, 'ensemble_prediction'] = result['ensemble_prediction']
            scored_df.at[df_idx, 'std_score'] = result['statistics']['std_score']
            scored_df.at[df_idx, 'min_score'] = result['statistics']['min_score']
            scored_df.at[df_idx, 'max_score'] = result['statistics']['max_score']
            scored_df.at[df_idx, 'consensus_strength'] = result['statistics']['consensus_strength']
            scored_df.at[df_idx, 'positive_folds'] = result['statistics']['positive_folds']
            result_idx += 1

    # Add rank
    scored_df['Score_Rank'] = scored_df['ensemble_score'].rank(ascending=False, method='min')

    # Save
    scored_df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved scored results to: {output_file}")

    # Statistics
    scored_count = scored_df['ensemble_score'].notna().sum()
    logger.info(f"\nStatistics:")
    logger.info(f"  - Total records: {len(scored_df)}")
    logger.info(f"  - Scored records: {scored_count}")
    logger.info(f"  - Score range: {scored_df['ensemble_score'].min():.4f} - {scored_df['ensemble_score'].max():.4f}")
    logger.info(f"  - Mean score: {scored_df['ensemble_score'].mean():.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Score OpenAlex data directly')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file')

    args = parser.parse_args()

    score_openalex(args.input, args.output)
