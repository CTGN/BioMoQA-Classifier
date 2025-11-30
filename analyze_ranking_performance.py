#!/usr/bin/env python3
"""
Analyze ranking performance by matching labeled instances with predictions.

This script:
1. Takes labeled instances (positives/negatives) from a CSV file
2. Matches them with a predictions file containing scores/ranks
3. Reports detailed statistics on ranking performance
"""

import pandas as pd
import argparse
import sys
import re
from pathlib import Path


def normalize_text(text):
    """Normalize text for matching."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace various dash/hyphen types
    text = text.replace('–', '-').replace('—', '-').replace('‐', '-')
    # Remove special characters except spaces and dashes
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def load_labeled_data(filepath, label_column=None):
    """
    Load labeled instances from CSV.

    Tries to auto-detect label column if not specified.
    Expected labels: 'positive', 'negative', or binary 0/1
    """
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} instances from {filepath}")
    print(f"Columns: {df.columns.tolist()}")

    # Try to detect label column
    if label_column is None:
        possible_label_cols = ['label', 'class', 'rating', 'category', 'type']
        for col in possible_label_cols:
            if col in df.columns.str.lower():
                label_column = df.columns[df.columns.str.lower() == col][0]
                print(f"Auto-detected label column: '{label_column}'")
                break

    if label_column and label_column in df.columns:
        df['label'] = df[label_column]
    else:
        # No label column - assume separate positive/negative files
        print("No label column found - assuming all instances have same label")
        df['label'] = None

    return df


def load_predictions(filepath):
    """Load predictions file with scores and ranks."""
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} predictions from {filepath}")
    print(f"Columns: {df.columns.tolist()}")

    # Auto-detect score and rank columns
    score_col = None
    rank_col = None

    # First pass: look for rank column
    for col in df.columns:
        col_lower = col.lower()
        if 'rank' in col_lower:
            rank_col = col
            break

    # Second pass: look for score column (not containing 'rank')
    for col in df.columns:
        col_lower = col.lower()
        if 'score' in col_lower and 'rank' not in col_lower:
            score_col = col
            break

    # If no pure score column found, look for any score
    if score_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'score' in col_lower:
                score_col = col
                break

    if score_col:
        print(f"Auto-detected score column: '{score_col}'")
    if rank_col:
        print(f"Auto-detected rank column: '{rank_col}'")

    return df, score_col, rank_col


def match_by_field(labeled_df, predictions_df, field, rank_col, score_col):
    """Match labeled instances with predictions by a specific field."""

    # Normalize the field for matching
    if field.lower() in ['title', 'name']:
        labeled_df['match_key'] = labeled_df[field].apply(normalize_text)
        predictions_df['match_key'] = predictions_df[field].apply(normalize_text)
    elif field.lower() in ['doi', 'id', 'key', 'match_field']:
        labeled_df['match_key'] = labeled_df[field].astype(str).str.strip().str.lower()
        predictions_df['match_key'] = predictions_df[field].astype(str).str.strip().str.lower()
    else:
        labeled_df['match_key'] = labeled_df[field].astype(str)
        predictions_df['match_key'] = predictions_df[field].astype(str)

    # For predictions with duplicates, keep best rank
    if rank_col and rank_col in predictions_df.columns:
        predictions_dedup = predictions_df.sort_values(
            rank_col,
            ascending=True
        ).groupby('match_key').first().reset_index()
    elif score_col and score_col in predictions_df.columns:
        predictions_dedup = predictions_df.sort_values(
            score_col,
            ascending=False
        ).groupby('match_key').first().reset_index()
    else:
        predictions_dedup = predictions_df.groupby('match_key').first().reset_index()

    # Merge
    merged = labeled_df.merge(
        predictions_dedup,
        on='match_key',
        how='inner',
        suffixes=('_labeled', '_pred')
    )

    return merged


def print_ranking_statistics(merged_df, score_col, rank_col, label_col='label'):
    """Print comprehensive ranking statistics."""

    print("\n" + "="*80)
    print("RANKING PERFORMANCE ANALYSIS")
    print("="*80)

    # Overall matching statistics
    print(f"\nTotal instances matched: {len(merged_df)}")

    if label_col in merged_df.columns and merged_df[label_col].notna().any():
        # Separate by label
        labels = merged_df[label_col].unique()

        for label in labels:
            if pd.isna(label):
                continue

            subset = merged_df[merged_df[label_col] == label]

            print(f"\n{'='*80}")
            print(f"CLASS: {label.upper()}")
            print(f"{'='*80}")
            print(f"Count: {len(subset)}")

            if score_col and score_col in merged_df.columns:
                scores = subset[score_col]
                print(f"\nScore statistics:")
                print(f"  - Range: {scores.min():.4f} - {scores.max():.4f}")
                print(f"  - Mean: {scores.mean():.4f}")
                print(f"  - Median: {scores.median():.4f}")
                print(f"  - Std: {scores.std():.4f}")

                # Score distribution
                high = (scores >= 0.9).sum()
                mid = ((scores >= 0.5) & (scores < 0.9)).sum()
                low = (scores < 0.5).sum()

                print(f"\nScore distribution:")
                print(f"  - High (≥ 0.9): {high}/{len(subset)} ({100*high/len(subset):.1f}%)")
                print(f"  - Medium (0.5-0.9): {mid}/{len(subset)} ({100*mid/len(subset):.1f}%)")
                print(f"  - Low (< 0.5): {low}/{len(subset)} ({100*low/len(subset):.1f}%)")

            if rank_col and rank_col in merged_df.columns:
                ranks = subset[rank_col]
                print(f"\nRank statistics:")
                print(f"  - Range: {ranks.min():.0f} - {ranks.max():.0f}")
                print(f"  - Mean: {ranks.mean():.1f}")
                print(f"  - Median: {ranks.median():.0f}")

                # Percentages in top N
                total_predictions = merged_df[rank_col].max()
                print(f"\nPercentage in top ranks (out of {total_predictions:.0f} total):")

                for threshold in [10, 50, 100, 500, 1000, 5000]:
                    if threshold <= total_predictions:
                        in_top = (ranks <= threshold).sum()
                        print(f"  - Top {threshold:5d}: {in_top:4d}/{len(subset):4d} ({100*in_top/len(subset):5.1f}%)")

                # Show best and worst ranked
                print(f"\nBest ranked (top 5):")
                best = subset.nsmallest(5, rank_col)
                for idx, row in best.iterrows():
                    title = row.get('Title', row.get('title', row.get('Title_labeled', row.get('title_labeled', 'N/A'))))
                    if isinstance(title, str) and len(title) > 60:
                        title = title[:60] + "..."
                    print(f"  Rank {row[rank_col]:5.0f}, Score {row[score_col]:.4f}: {title}")

                print(f"\nWorst ranked (bottom 5):")
                worst = subset.nlargest(5, rank_col)
                for idx, row in worst.iterrows():
                    title = row.get('Title', row.get('title', row.get('Title_labeled', row.get('title_labeled', 'N/A'))))
                    if isinstance(title, str) and len(title) > 60:
                        title = title[:60] + "..."
                    print(f"  Rank {row[rank_col]:5.0f}, Score {row[score_col]:.4f}: {title}")

    else:
        # No labels - just overall statistics
        print("\nNo labels found - showing overall statistics")

        if score_col and score_col in merged_df.columns:
            scores = merged_df[score_col]
            print(f"\nScore statistics:")
            print(f"  - Range: {scores.min():.4f} - {scores.max():.4f}")
            print(f"  - Mean: {scores.mean():.4f}")
            print(f"  - Median: {scores.median():.4f}")

        if rank_col and rank_col in merged_df.columns:
            ranks = merged_df[rank_col]
            print(f"\nRank statistics:")
            print(f"  - Range: {ranks.min():.0f} - {ranks.max():.0f}")
            print(f"  - Mean: {ranks.mean():.1f}")
            print(f"  - Median: {ranks.median():.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ranking performance of labeled instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match positives and negatives with predictions by DOI
  python analyze_ranking_performance.py \\
    --labeled data/positives.csv \\
    --predictions data/export.csv \\
    --match-by DOI \\
    --label positive

  # Analyze with both positives and negatives
  python analyze_ranking_performance.py \\
    --labeled data/positives.csv \\
    --predictions data/export.csv \\
    --match-by DOI \\
    --label positive \\
    --output results/positives_ranking.csv
        """
    )

    parser.add_argument('--labeled', '-l', required=True,
                        help='CSV file with labeled instances')
    parser.add_argument('--predictions', '-p', required=True,
                        help='CSV file with predictions (scores/ranks)')
    parser.add_argument('--match-by', '-m', required=True,
                        help='Column name to match on (e.g., DOI, Title, ID)')
    parser.add_argument('--label', default=None,
                        help='Label to assign to instances (e.g., positive, negative)')
    parser.add_argument('--label-column', default=None,
                        help='Column name containing labels in labeled file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file for matched instances')

    args = parser.parse_args()

    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    labeled_df = load_labeled_data(args.labeled, args.label_column)
    predictions_df, score_col, rank_col = load_predictions(args.predictions)

    # If label specified via command line, use it
    if args.label:
        labeled_df['label'] = args.label
        print(f"Assigned label '{args.label}' to all instances")

    # Check if matching field exists (case-insensitive)
    labeled_match_col = None
    for col in labeled_df.columns:
        if col.lower() == args.match_by.lower():
            labeled_match_col = col
            break

    if labeled_match_col is None:
        print(f"\nError: Column '{args.match_by}' not found in labeled file")
        print(f"Available columns: {labeled_df.columns.tolist()}")
        sys.exit(1)

    predictions_match_col = None
    for col in predictions_df.columns:
        if col.lower() == args.match_by.lower():
            predictions_match_col = col
            break

    if predictions_match_col is None:
        print(f"\nError: Column '{args.match_by}' not found in predictions file")
        print(f"Available columns: {predictions_df.columns.tolist()}")
        sys.exit(1)

    print(f"\nMatching on:")
    print(f"  Labeled file: '{labeled_match_col}'")
    print(f"  Predictions file: '{predictions_match_col}'")

    # Match
    print("\n" + "="*80)
    print("MATCHING INSTANCES")
    print("="*80)

    # Rename columns to have consistent names for matching
    labeled_df_renamed = labeled_df.rename(columns={labeled_match_col: 'match_field'})
    predictions_df_renamed = predictions_df.rename(columns={predictions_match_col: 'match_field'})

    merged = match_by_field(labeled_df_renamed, predictions_df_renamed, 'match_field', rank_col, score_col)

    print(f"\nMatched {len(merged)}/{len(labeled_df)} instances ({100*len(merged)/len(labeled_df):.1f}%)")

    if len(merged) == 0:
        print("\nNo matches found! Check that the match field contains compatible values.")
        sys.exit(1)

    # Analyze and print statistics
    print_ranking_statistics(merged, score_col, rank_col)

    # Save output if requested
    if args.output:
        merged.to_csv(args.output, index=False)
        print(f"\n✓ Saved {len(merged)} matched instances to {args.output}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
