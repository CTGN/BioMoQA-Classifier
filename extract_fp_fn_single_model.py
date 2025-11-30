#!/usr/bin/env python3
"""
Extract FP and FN for a single specific model.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# Import the extraction functions from the enhanced script
from extract_fp_fn_enhanced import (
    normalize_title,
    load_all_fold_predictions,
    identify_fp_fn,
    load_metadata_sources,
    match_with_metadata_enhanced,
    load_optional_negatives
)


def list_available_models(predictions_dir):
    """List all available model patterns."""
    test_preds_path = Path(predictions_dir)
    all_files = list(test_preds_path.glob('fold_*.csv'))

    patterns = set()
    for f in all_files:
        parts = f.stem.split('_', 2)
        if len(parts) >= 3:
            pattern = parts[2].replace('_run-0_opt_neg-500', '')
            patterns.add(pattern)

    return sorted(patterns)


def evaluate_model(predictions_dir, model_pattern):
    """Evaluate a specific model and return metrics."""
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
    import numpy as np

    test_preds_path = Path(predictions_dir)
    all_labels = []
    all_predictions = []

    for fold in range(5):
        pred_file = test_preds_path / f'fold_{fold}_{model_pattern}_run-0_opt_neg-500.csv'
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            all_labels.extend(df['label'].values)
            all_predictions.extend(df['prediction'].values)

    if not all_labels:
        return None

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0),
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'total': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract FP and FN for a specific model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python extract_fp_fn_single_model.py --list

  # Extract for a specific model
  python extract_fp_fn_single_model.py --model roberta-base_BCE_with_title

  # Extract with custom output directory
  python extract_fp_fn_single_model.py \\
    --model roberta-base_BCE_with_title \\
    --output-dir results/roberta_errors
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List all available model patterns')
    parser.add_argument('--model', '-m', type=str,
                        help='Model pattern to extract (e.g., roberta-base_BCE_with_title)')
    parser.add_argument('--predictions-dir', default='results/test preds/bert',
                        help='Directory containing prediction files')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for results')
    parser.add_argument('--openalex', default='data/openalex_prepared.csv',
                        help='OpenAlex metadata CSV file')
    parser.add_argument('--positives', default='data/positives.csv',
                        help='Positives CSV file')
    parser.add_argument('--negatives', default='data/negatives.csv',
                        help='Negatives CSV file')
    parser.add_argument('--optional_negatives', default='data/optional_negatives.csv',
                        help='Optional negatives CSV file (PubMed/arXiv) to exclude')

    args = parser.parse_args()

    # List models
    if args.list:
        print("Available models:")
        print("="*80)
        models = list_available_models(args.predictions_dir)
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model}")
        print(f"\nTotal: {len(models)} models")
        sys.exit(0)

    # Check if model is specified
    if not args.model:
        print("Error: --model is required (or use --list to see available models)")
        sys.exit(1)

    # Verify model exists
    available_models = list_available_models(args.predictions_dir)
    if args.model not in available_models:
        print(f"Error: Model '{args.model}' not found")
        print(f"\nAvailable models:")
        for model in available_models:
            print(f"  - {model}")
        sys.exit(1)

    print("="*80)
    print(f"EXTRACTING FP/FN FOR MODEL: {args.model}")
    print("="*80)

    # Evaluate model first
    print("\n[Step 1] Evaluating model performance...")
    metrics = evaluate_model(args.predictions_dir, args.model)
    if metrics:
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}, Total: {metrics['total']}")

    # Load predictions
    print("\n[Step 2] Loading predictions...")
    pattern = f"fold_*_{args.model}_run-0_opt_neg-500.csv"
    predictions_df = load_all_fold_predictions(args.predictions_dir, pattern)

    # Load optional negatives to exclude
    print("\n[Step 3] Loading optional negatives...")
    optional_negatives_df = load_optional_negatives(args.optional_negatives)

    # Identify FP and FN
    print("\n[Step 4] Identifying errors...")
    fp_df, fn_df = identify_fp_fn(predictions_df, optional_negatives_df)

    # Load metadata sources
    print("\n[Step 5] Loading metadata sources...")
    metadata_sources = load_metadata_sources(args.openalex, args.positives, args.negatives)

    # Match with metadata
    print("\n[Step 6] Matching False Positives with metadata...")
    fp_with_metadata = match_with_metadata_enhanced(fp_df, metadata_sources)

    print("\n[Step 7] Matching False Negatives with metadata...")
    fn_with_metadata = match_with_metadata_enhanced(fn_df, metadata_sources)

    # Save results
    print("\n[Step 8] Saving results...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_safe = args.model.replace('/', '_').replace(' ', '_')

    output_columns = [
        'error_type', 'fold', 'source_file',
        'doi', 'title', 'abstract',
        'label', 'prediction', 'score',
        'source'
    ]

    # Combined output
    combined_df = pd.concat([fp_with_metadata, fn_with_metadata], ignore_index=True)
    combined_df = combined_df.sort_values(['error_type', 'fold'])

    combined_file = output_dir / f"fp_fn_{model_safe}.csv"
    combined_df[output_columns].to_csv(combined_file, index=False)
    print(f"\n✓ Saved {len(combined_df)} total errors to {combined_file}")

    # FP output
    fp_file = output_dir / f"false_positives_{model_safe}.csv"
    fp_with_metadata[output_columns].to_csv(fp_file, index=False)
    print(f"✓ Saved {len(fp_with_metadata)} False Positives to {fp_file}")

    # FN output
    fn_file = output_dir / f"false_negatives_{model_safe}.csv"
    fn_with_metadata[output_columns].to_csv(fn_file, index=False)
    print(f"✓ Saved {len(fn_with_metadata)} False Negatives to {fn_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"False Positives: {len(fp_with_metadata)}")
    print(f"False Negatives: {len(fn_with_metadata)}")
    print(f"Total errors: {len(combined_df)}")
    print(f"Error rate: {100*len(combined_df)/len(predictions_df):.2f}%")

    print("\nData completeness:")
    print(f"  FP with DOI: {fp_with_metadata['doi'].notna().sum()}/{len(fp_with_metadata)} ({100*fp_with_metadata['doi'].notna().sum()/len(fp_with_metadata):.1f}%)")
    print(f"  FP with abstract: {fp_with_metadata['abstract'].notna().sum()}/{len(fp_with_metadata)} ({100*fp_with_metadata['abstract'].notna().sum()/len(fp_with_metadata):.1f}%)")
    print(f"  FN with DOI: {fn_with_metadata['doi'].notna().sum()}/{len(fn_with_metadata)} ({100*fn_with_metadata['doi'].notna().sum()/len(fn_with_metadata):.1f}%)")
    print(f"  FN with abstract: {fn_with_metadata['abstract'].notna().sum()}/{len(fn_with_metadata)} ({100*fn_with_metadata['abstract'].notna().sum()/len(fn_with_metadata):.1f}%)")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
