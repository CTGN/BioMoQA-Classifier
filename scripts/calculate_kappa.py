#!/usr/bin/env python3
"""
Calculate Cohen's Kappa scores for inter-rater reliability
from abstract and fulltext screening data.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse
import sys
from pathlib import Path


def calculate_kappa_for_file(filepath, criteria_config):
    """
    Calculate Cohen's kappa for all criteria in a screening file.

    Args:
        filepath: Path to the CSV file
        criteria_config: Dict with 'criteria' (list of criterion names)
                        and 'start_col' (starting column index)

    Returns:
        List of dicts with kappa scores and statistics
    """
    df = pd.read_csv(filepath)

    # Get rater names from first data row
    rater_names = df.iloc[0, criteria_config['start_col']:criteria_config['start_col']+2].values

    # Extract data rows (skip header rows)
    data = df.iloc[2:].copy()

    results = []

    for i, criterion in enumerate(criteria_config['criteria']):
        col_idx = criteria_config['start_col'] + i * 2

        # Get ratings from both raters
        rater1 = data.iloc[:, col_idx].dropna()
        rater2 = data.iloc[:, col_idx + 1].dropna()

        # Ensure same length
        min_len = min(len(rater1), len(rater2))
        rater1 = rater1.iloc[:min_len]
        rater2 = rater2.iloc[:min_len]

        # Calculate agreement statistics
        agreements = (rater1 == rater2).sum()
        total = len(rater1)
        agreement_pct = (agreements / total * 100) if total > 0 else 0

        # Calculate Cohen's kappa
        try:
            kappa = cohen_kappa_score(rater1, rater2)
            # Check if kappa is NaN (perfect agreement with single class)
            if pd.isna(kappa):
                kappa_str = "Perfect (100%)"
                kappa_val = 1.0
            else:
                kappa_str = f"{kappa:.4f}"
                kappa_val = kappa
        except Exception as e:
            kappa_str = "Error"
            kappa_val = None

        results.append({
            'criterion': criterion,
            'kappa': kappa_val,
            'kappa_str': kappa_str,
            'agreements': agreements,
            'total': total,
            'agreement_pct': agreement_pct
        })

    return results, rater_names


def interpret_kappa(kappa):
    """Return interpretation of kappa score."""
    if kappa is None:
        return "N/A"
    if kappa >= 0.81:
        return "Almost Perfect"
    elif kappa >= 0.61:
        return "Substantial"
    elif kappa >= 0.41:
        return "Moderate"
    elif kappa >= 0.21:
        return "Fair"
    else:
        return "Slight/Poor"


def print_results(title, results, rater_names):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print(f"{title}")
    print("=" * 90)
    print(f"\nRaters: {rater_names[0]} and {rater_names[1]}")
    print(f"\nNumber of papers evaluated: {results[0]['total']}")
    print("\n" + "-" * 90)
    print(f"{'Criterion':<40} {'Kappa':<15} {'Interpretation':<18} {'Agreement':<15}")
    print("-" * 90)

    for result in results:
        criterion = result['criterion']
        kappa_str = result['kappa_str']
        interpretation = interpret_kappa(result['kappa'])
        agreement = f"{result['agreements']}/{result['total']} ({result['agreement_pct']:.1f}%)"

        print(f"{criterion:<40} {kappa_str:<15} {interpretation:<18} {agreement:<15}")

    # Calculate average kappa (excluding None and perfect scores)
    valid_kappas = [r['kappa'] for r in results if r['kappa'] is not None and r['kappa'] < 1.0]
    if valid_kappas:
        avg_kappa = sum(valid_kappas) / len(valid_kappas)
        print("-" * 90)
        print(f"{'Average Kappa (excluding perfect)':<40} {avg_kappa:.4f}")


def save_results_to_csv(abstract_results, fulltext_results, output_path):
    """Save kappa results to a CSV file."""
    all_results = []

    for result in abstract_results:
        all_results.append({
            'Screening_Phase': 'Abstract',
            'Criterion': result['criterion'],
            'Kappa': result['kappa'],
            'Kappa_Interpretation': interpret_kappa(result['kappa']),
            'Agreements': result['agreements'],
            'Total_Papers': result['total'],
            'Agreement_Percentage': result['agreement_pct']
        })

    for result in fulltext_results:
        all_results.append({
            'Screening_Phase': 'Fulltext',
            'Criterion': result['criterion'],
            'Kappa': result['kappa'],
            'Kappa_Interpretation': interpret_kappa(result['kappa']),
            'Agreements': result['agreements'],
            'Total_Papers': result['total'],
            'Agreement_Percentage': result['agreement_pct']
        })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Cohen\'s Kappa scores for systematic review screening'
    )
    parser.add_argument(
        '--abstract',
        default='data/For Leandre/kappa_outcome_abstract.xlsx - kappa evaluation.csv',
        help='Path to abstract screening CSV file'
    )
    parser.add_argument(
        '--fulltext',
        default='data/For Leandre/kappa_outcome_fulltext.xlsx - kappa evaluation.csv',
        help='Path to fulltext screening CSV file'
    )
    parser.add_argument(
        '--output',
        default='results/kappa_scores.csv',
        help='Path to save results CSV file'
    )

    args = parser.parse_args()

    # Check if files exist
    abstract_path = Path(args.abstract)
    fulltext_path = Path(args.fulltext)

    if not abstract_path.exists():
        print(f"Error: Abstract file not found: {abstract_path}")
        sys.exit(1)

    if not fulltext_path.exists():
        print(f"Error: Fulltext file not found: {fulltext_path}")
        sys.exit(1)

    # Configuration for abstract screening
    abstract_config = {
        'criteria': [
            'Island (Population)',
            'Terrestrial Biodiversity (Population)',
            'Monitoring (Outcome)'
        ],
        'start_col': 4
    }

    # Configuration for fulltext screening
    fulltext_config = {
        'criteria': [
            'Island (Population)',
            'Terrestrial Biodiversity (Population)',
            'Monitoring (Outcome)',
            'Specified location',
            'Not a review',
            'Not a book'
        ],
        'start_col': 4
    }

    # Calculate kappa for abstract screening
    abstract_results, abstract_raters = calculate_kappa_for_file(
        args.abstract,
        abstract_config
    )

    # Calculate kappa for fulltext screening
    fulltext_results, fulltext_raters = calculate_kappa_for_file(
        args.fulltext,
        fulltext_config
    )

    # Print results
    print_results("ABSTRACT SCREENING - Cohen's Kappa Scores",
                  abstract_results,
                  abstract_raters)

    print_results("FULLTEXT SCREENING - Cohen's Kappa Scores",
                  fulltext_results,
                  fulltext_raters)

    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(abstract_results, fulltext_results, output_path)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("\nKappa Interpretation Guidelines (Landis & Koch, 1977):")
    print("  0.81 - 1.00: Almost Perfect Agreement")
    print("  0.61 - 0.80: Substantial Agreement")
    print("  0.41 - 0.60: Moderate Agreement")
    print("  0.21 - 0.40: Fair Agreement")
    print("  0.00 - 0.20: Slight Agreement")
    print("  < 0.00: Poor Agreement")
    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
