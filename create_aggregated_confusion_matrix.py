#!/usr/bin/env python3
"""
Create an aggregated confusion matrix from all folds with percentages.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import argparse
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.plot_style import (
    FIGURE_SIZES, CONFUSION_CMAP, PLOT_PARAMS,
    create_figure, save_figure, setup_plotting_style
)

# Apply unified style
setup_plotting_style()


def create_aggregated_confusion_matrix(
    model_pattern: str,
    test_preds_dir: str = "results/test preds/bert",
    output_dir: str = "plots",
    num_folds: int = 5
):
    """
    Aggregate predictions from all folds and create a confusion matrix with percentages.

    Args:
        model_pattern: Pattern to match model files (e.g., 'roberta-base_BCE_with_title')
        test_preds_dir: Directory containing test prediction files
        output_dir: Directory to save the confusion matrix plot
        num_folds: Number of folds to aggregate
    """
    test_preds_path = Path(test_preds_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect predictions from all folds
    all_labels = []
    all_predictions = []

    print(f"Loading predictions for pattern: {model_pattern}")

    for fold in range(num_folds):
        # Find the prediction file for this fold
        pattern = f"fold_{fold}_{model_pattern}_run-0_opt_neg-500.csv"
        pred_file = test_preds_path / pattern

        if not pred_file.exists():
            print(f"Warning: File not found: {pred_file}")
            continue

        # Load predictions
        df = pd.read_csv(pred_file)
        all_labels.extend(df['label'].values)
        all_predictions.extend(df['prediction'].values)
        print(f"  Fold {fold}: {len(df)} samples")

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    total_samples = len(all_labels)
    print(f"\nTotal samples across all folds: {total_samples}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"  TN={tn} ({100*tn/total_samples:.2f}%)")
    print(f"  FP={fp} ({100*fp/total_samples:.2f}%)")
    print(f"  FN={fn} ({100*fn/total_samples:.2f}%)")
    print(f"  TP={tp} ({100*tp/total_samples:.2f}%)")

    # Create confusion matrix with percentages
    fig, ax = create_figure(figsize='medium')

    # Create the heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=CONFUSION_CMAP)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Negative (0)', 'Positive (1)'],
           yticklabels=['Negative (0)', 'Positive (1)'],
           title=f'Confusion Matrix - All Folds Aggregated\n{model_pattern}',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations with counts and percentages
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = 100 * count / total_samples
            text = f'{count:,}\n({percentage:.2f}%)'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')

    fig.tight_layout()

    # Save the plot
    output_file = output_path / f"confusion_matrix_all_folds_{model_pattern}.png"
    save_figure(fig, output_file)
    print(f"\nConfusion matrix saved to: {output_file}")

    # Also save a copy as the default confusion_matrix.png
    default_output = output_path / "confusion_matrix.png"
    save_figure(fig, default_output)
    print(f"Also saved as: {default_output}")

    plt.close(fig)

    # Return metrics
    return {
        'total_samples': total_samples,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'accuracy': (tp + tn) / total_samples,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create aggregated confusion matrix from all folds')
    parser.add_argument(
        '--model',
        type=str,
        default='roberta-base_BCE_with_title',
        help='Model pattern to match (default: roberta-base_BCE_with_title)'
    )
    parser.add_argument(
        '--test-preds-dir',
        type=str,
        default='results/test preds/bert',
        help='Directory containing test prediction files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--num-folds',
        type=int,
        default=5,
        help='Number of folds to aggregate'
    )

    args = parser.parse_args()

    metrics = create_aggregated_confusion_matrix(
        model_pattern=args.model,
        test_preds_dir=args.test_preds_dir,
        output_dir=args.output_dir,
        num_folds=args.num_folds
    )

    print("\n=== Overall Metrics ===")
    for key, value in metrics.items():
        if key != 'total_samples':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
