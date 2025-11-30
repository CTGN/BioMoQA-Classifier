#!/usr/bin/env python3
"""
Enhanced version: Extract FP and FN with multiple data sources for complete metadata.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
import re


def normalize_title(title):
    """Normalize title for matching."""
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r'<[^>]+>', '', title)
    title = title.replace('–', '-').replace('—', '-').replace('‐', '-')
    title = re.sub(r'[^\w\s-]', ' ', title)
    title = ' '.join(title.split())
    return title.strip()


def normalize_doi(doi):
    """Normalize DOI for matching."""
    if not isinstance(doi, str):
        return ""
    # Remove common prefixes
    doi = doi.lower().strip()
    doi = doi.replace('https://doi.org/', '')
    doi = doi.replace('http://doi.org/', '')
    doi = doi.replace('doi:', '')
    return doi.strip()


def load_all_fold_predictions(predictions_dir, pattern="fold_*.csv"):
    """Load all fold prediction files and combine them."""
    predictions_path = Path(predictions_dir)

    if not predictions_path.exists():
        print(f"Error: Directory {predictions_dir} does not exist")
        sys.exit(1)

    fold_files = sorted(predictions_path.glob(pattern))

    if not fold_files:
        print(f"Error: No files matching pattern '{pattern}' found in {predictions_dir}")
        sys.exit(1)

    print(f"Found {len(fold_files)} fold prediction files")

    all_predictions = []
    for file_path in fold_files:
        df = pd.read_csv(file_path)
        filename = file_path.stem
        df['source_file'] = filename
        all_predictions.append(df)

    combined_df = pd.concat(all_predictions, ignore_index=True)
    print(f"Loaded {len(combined_df)} total predictions")

    return combined_df


def load_optional_negatives(optional_negatives_file):
    """Load optional negatives from PubMed/arXiv."""
    if not Path(optional_negatives_file).exists():
        print(f"Warning: Optional negatives file not found: {optional_negatives_file}")
        return None

    opt_neg_df = pd.read_csv(optional_negatives_file)
    print(f"Loaded {len(opt_neg_df)} optional negatives")

    # Normalize titles and DOIs for matching
    opt_neg_df['title_normalized'] = opt_neg_df['title'].apply(normalize_title)
    if 'doi' in opt_neg_df.columns:
        opt_neg_df['doi_normalized'] = opt_neg_df['doi'].apply(normalize_doi)

    return opt_neg_df


def filter_optional_negatives(error_df, optional_negatives_df):
    """Filter out optional negatives from error dataframe."""
    if optional_negatives_df is None:
        return error_df

    # Normalize error titles for matching
    error_df['title_normalized'] = error_df['title'].apply(normalize_title)

    # Create a set of optional negative titles
    opt_neg_titles = set(optional_negatives_df['title_normalized'].dropna())

    # Filter out matches
    before_count = len(error_df)
    error_df = error_df[~error_df['title_normalized'].isin(opt_neg_titles)].copy()
    after_count = len(error_df)

    filtered_count = before_count - after_count
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} optional negatives from PubMed/arXiv")

    # Clean up temporary column
    error_df = error_df.drop(columns=['title_normalized'])

    return error_df


def identify_fp_fn(predictions_df, optional_negatives_df=None):
    """Identify False Positives and False Negatives."""
    fp_mask = (predictions_df['label'] == 0) & (predictions_df['prediction'] == 1)
    fp_df = predictions_df[fp_mask].copy()
    fp_df['error_type'] = 'False Positive'

    fn_mask = (predictions_df['label'] == 1) & (predictions_df['prediction'] == 0)
    fn_df = predictions_df[fn_mask].copy()
    fn_df['error_type'] = 'False Negative'

    print(f"\nFound {len(fp_df)} False Positives")
    print(f"Found {len(fn_df)} False Negatives")
    print(f"Total errors: {len(fp_df) + len(fn_df)}")

    # Filter out optional negatives if provided
    if optional_negatives_df is not None:
        print("\nFiltering out optional negatives from errors...")
        fp_df = filter_optional_negatives(fp_df, optional_negatives_df)
        fn_df = filter_optional_negatives(fn_df, optional_negatives_df)
        print(f"\nAfter filtering:")
        print(f"  False Positives: {len(fp_df)}")
        print(f"  False Negatives: {len(fn_df)}")
        print(f"  Total errors: {len(fp_df) + len(fn_df)}")

    return fp_df, fn_df


def load_metadata_sources(openalex_file, positives_file, negatives_file):
    """Load all metadata sources."""
    metadata_sources = {}

    # Load OpenAlex data
    if openalex_file and Path(openalex_file).exists():
        openalex_df = pd.read_csv(openalex_file)
        openalex_df['title_normalized'] = openalex_df['title'].apply(normalize_title)
        if 'doi' in openalex_df.columns:
            openalex_df['doi_normalized'] = openalex_df['doi'].apply(normalize_doi)
        # Ensure we have the required columns
        cols_to_keep = ['title_normalized', 'doi', 'abstract']
        available_cols = [c for c in cols_to_keep if c in openalex_df.columns or c == 'title_normalized']
        metadata_sources['openalex'] = openalex_df
        print(f"Loaded {len(openalex_df)} records from OpenAlex")

    # Load positives
    if positives_file and Path(positives_file).exists():
        positives_df = pd.read_csv(positives_file)
        # Create normalized names, keep originals
        if 'Title' in positives_df.columns:
            positives_df['title'] = positives_df['Title']
            positives_df['title_normalized'] = positives_df['Title'].apply(normalize_title)
        if 'DOI' in positives_df.columns:
            positives_df['doi'] = positives_df['DOI']
        if 'Abstract' in positives_df.columns:
            positives_df['abstract'] = positives_df['Abstract']
        metadata_sources['positives'] = positives_df
        print(f"Loaded {len(positives_df)} records from positives")

    # Load negatives
    if negatives_file and Path(negatives_file).exists():
        negatives_df = pd.read_csv(negatives_file)
        # Create normalized names, keep originals
        if 'Title' in negatives_df.columns:
            negatives_df['title'] = negatives_df['Title']
            negatives_df['title_normalized'] = negatives_df['Title'].apply(normalize_title)
        if 'DOI' in negatives_df.columns:
            negatives_df['doi'] = negatives_df['DOI']
        if 'Abstract' in negatives_df.columns:
            negatives_df['abstract'] = negatives_df['Abstract']
        metadata_sources['negatives'] = negatives_df
        print(f"Loaded {len(negatives_df)} records from negatives")

    return metadata_sources


def match_with_metadata_enhanced(error_df, metadata_sources):
    """Match error instances with metadata from multiple sources."""

    error_df = error_df.reset_index(drop=True)
    error_df['title_normalized'] = error_df['title'].apply(normalize_title)

    # Initialize result columns
    error_df['doi'] = pd.NA
    error_df['abstract'] = pd.NA
    error_df['source'] = pd.NA

    # Try matching with each source in order of preference
    unmatched_mask = pd.Series([True] * len(error_df), index=error_df.index)

    for source_name, metadata_df in metadata_sources.items():
        if unmatched_mask.sum() == 0:
            break

        print(f"\nMatching with {source_name}...")

        # Get unmatched subset
        unmatched_df = error_df[unmatched_mask].copy()

        if len(unmatched_df) == 0:
            print(f"  No unmatched instances remaining")
            continue

        # Prepare metadata for merge
        metadata_cols = ['title_normalized']
        if 'doi' in metadata_df.columns:
            metadata_cols.append('doi')
        if 'abstract' in metadata_df.columns:
            metadata_cols.append('abstract')

        metadata_subset = metadata_df[metadata_cols].drop_duplicates('title_normalized')

        # Match by title
        matches = unmatched_df.merge(
            metadata_subset,
            on='title_normalized',
            how='left',
            suffixes=('_orig', '_new')
        )

        # Count and update matches
        matched_count = 0
        for i in range(len(matches)):
            # Check if we got a match (doi_new exists and is not NA)
            has_match = False
            doi_val = None
            abstract_val = None

            if 'doi_new' in matches.columns and pd.notna(matches.iloc[i]['doi_new']):
                has_match = True
                doi_val = matches.iloc[i]['doi_new']
                if 'abstract_new' in matches.columns:
                    abstract_val = matches.iloc[i]['abstract_new']
            elif 'doi' in matches.columns and pd.notna(matches.iloc[i]['doi']) and 'doi_orig' not in matches.columns:
                # If there's no suffix (only one doi column), use it directly
                has_match = True
                doi_val = matches.iloc[i]['doi']
                if 'abstract' in matches.columns:
                    abstract_val = matches.iloc[i]['abstract']

            if has_match:
                orig_idx = unmatched_df.index[i]
                error_df.at[orig_idx, 'doi'] = doi_val
                error_df.at[orig_idx, 'abstract'] = abstract_val
                error_df.at[orig_idx, 'source'] = source_name
                unmatched_mask.at[orig_idx] = False
                matched_count += 1

        print(f"  Matched {matched_count} new instances")

    total_matched = error_df['doi'].notna().sum()
    print(f"\nTotal matched: {total_matched}/{len(error_df)} ({100*total_matched/len(error_df):.1f}%)")

    # Second pass: Fill in missing abstracts from other sources
    missing_abstract_mask = error_df['doi'].notna() & error_df['abstract'].isna()
    missing_abstract_count = missing_abstract_mask.sum()

    if missing_abstract_count > 0:
        print(f"\n[Second pass] Filling in {missing_abstract_count} missing abstracts...")

        for source_name, metadata_df in metadata_sources.items():
            # Get entries with DOI but no abstract
            need_abstract_df = error_df[missing_abstract_mask].copy()

            if len(need_abstract_df) == 0:
                break

            # Skip if this source doesn't have abstracts
            if 'abstract' not in metadata_df.columns:
                continue

            print(f"  Checking {source_name}...")

            # Prepare metadata for merge
            metadata_cols = ['title_normalized', 'abstract']
            metadata_subset = metadata_df[metadata_cols].drop_duplicates('title_normalized')
            metadata_subset = metadata_subset[metadata_subset['abstract'].notna()]

            if len(metadata_subset) == 0:
                continue

            # Match by title
            matches = need_abstract_df.merge(
                metadata_subset,
                on='title_normalized',
                how='left',
                suffixes=('', '_new')
            )

            # Update abstracts
            filled_count = 0
            for i in range(len(matches)):
                if 'abstract_new' in matches.columns and pd.notna(matches.iloc[i]['abstract_new']):
                    orig_idx = need_abstract_df.index[i]
                    error_df.at[orig_idx, 'abstract'] = matches.iloc[i]['abstract_new']
                    missing_abstract_mask.at[orig_idx] = False
                    filled_count += 1

            if filled_count > 0:
                print(f"    Filled {filled_count} abstracts from {source_name}")

        final_missing = missing_abstract_mask.sum()
        print(f"  Total abstracts filled: {missing_abstract_count - final_missing}")
        if final_missing > 0:
            print(f"  Still missing: {final_missing}")

    return error_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract False Positives and False Negatives with enhanced metadata matching',
    )

    parser.add_argument('--predictions_dir', '-p', required=True,
                        help='Directory containing fold prediction CSV files')
    parser.add_argument('--openalex', default='data/openalex_prepared.csv',
                        help='OpenAlex metadata CSV file')
    parser.add_argument('--positives', default='data/positives.csv',
                        help='Positives CSV file')
    parser.add_argument('--negatives', default='data/negatives.csv',
                        help='Negatives CSV file')
    parser.add_argument('--optional_negatives', default='data/optional_negatives.csv',
                        help='Optional negatives CSV file (PubMed/arXiv) to exclude')
    parser.add_argument('--pattern', default='fold_*.csv',
                        help='Glob pattern for prediction files')
    parser.add_argument('--output', '-o', default='results/fp_fn_complete.csv',
                        help='Output CSV file for all errors')
    parser.add_argument('--output_fp', default='results/false_positives_complete.csv',
                        help='Output CSV file for FP only')
    parser.add_argument('--output_fn', default='results/false_negatives_complete.csv',
                        help='Output CSV file for FN only')

    args = parser.parse_args()

    print("="*80)
    print("EXTRACTING FALSE POSITIVES AND FALSE NEGATIVES (Enhanced)")
    print("="*80)

    # Load predictions
    print("\n[Step 1] Loading predictions...")
    predictions_df = load_all_fold_predictions(args.predictions_dir, args.pattern)

    # Load optional negatives to exclude
    print("\n[Step 2] Loading optional negatives...")
    optional_negatives_df = load_optional_negatives(args.optional_negatives)

    # Identify FP and FN
    print("\n[Step 3] Identifying errors...")
    fp_df, fn_df = identify_fp_fn(predictions_df, optional_negatives_df)

    # Load metadata sources
    print("\n[Step 4] Loading metadata sources...")
    metadata_sources = load_metadata_sources(args.openalex, args.positives, args.negatives)

    # Match with metadata
    print("\n[Step 5] Matching False Positives with metadata...")
    fp_with_metadata = match_with_metadata_enhanced(fp_df, metadata_sources)

    print("\n[Step 6] Matching False Negatives with metadata...")
    fn_with_metadata = match_with_metadata_enhanced(fn_df, metadata_sources)

    # Save results
    print("\n[Step 7] Saving results...")

    output_columns = [
        'error_type', 'fold', 'source_file',
        'doi', 'title', 'abstract',
        'label', 'prediction', 'score',
        'source'
    ]

    # Combined output
    combined_df = pd.concat([fp_with_metadata, fn_with_metadata], ignore_index=True)
    combined_df = combined_df.sort_values(['error_type', 'fold'])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df[output_columns].to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(combined_df)} total errors to {output_path}")

    # FP output
    fp_path = Path(args.output_fp)
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_with_metadata[output_columns].to_csv(fp_path, index=False)
    print(f"✓ Saved {len(fp_with_metadata)} False Positives to {fp_path}")

    # FN output
    fn_path = Path(args.output_fn)
    fn_path.parent.mkdir(parents=True, exist_ok=True)
    fn_with_metadata[output_columns].to_csv(fn_path, index=False)
    print(f"✓ Saved {len(fn_with_metadata)} False Negatives to {fn_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total predictions analyzed: {len(predictions_df)}")
    print(f"False Positives: {len(fp_with_metadata)}")
    print(f"False Negatives: {len(fn_with_metadata)}")
    print(f"Total errors: {len(combined_df)}")
    print(f"Error rate: {100*len(combined_df)/len(predictions_df):.2f}%")

    print("\nData completeness:")
    print(f"FP with DOI: {fp_with_metadata['doi'].notna().sum()}/{len(fp_with_metadata)} ({100*fp_with_metadata['doi'].notna().sum()/len(fp_with_metadata):.1f}%)")
    print(f"FP with abstract: {fp_with_metadata['abstract'].notna().sum()}/{len(fp_with_metadata)} ({100*fp_with_metadata['abstract'].notna().sum()/len(fp_with_metadata):.1f}%)")
    print(f"FN with DOI: {fn_with_metadata['doi'].notna().sum()}/{len(fn_with_metadata)} ({100*fn_with_metadata['doi'].notna().sum()/len(fn_with_metadata):.1f}%)")
    print(f"FN with abstract: {fn_with_metadata['abstract'].notna().sum()}/{len(fn_with_metadata)} ({100*fn_with_metadata['abstract'].notna().sum()/len(fn_with_metadata):.1f}%)")

    # Per-fold statistics
    print("\nErrors per fold:")
    fold_stats = combined_df.groupby(['fold', 'error_type']).size().unstack(fill_value=0)
    print(fold_stats)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
