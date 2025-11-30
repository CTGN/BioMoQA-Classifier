"""
Script to verify fold creation and check if all instances appear in training at some point.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import get_config

def verify_fold_coverage(n_folds=5, n_runs=1, n_simulations=100):
    """
    Verify that all instances appear in training set at least once during CV.
    Run multiple simulations to check the probability.
    """
    config = get_config()
    env_config = config.get('environment') or {}
    seed = env_config.get('seed', 42)

    # Create a mock dataset similar to the actual one
    n_samples = 100
    labels = np.array([0] * 50 + [1] * 50)  # Balanced for simplicity

    instances_never_in_train = []

    for sim in range(n_simulations):
        # Track which instances appear in training
        in_train_count = np.zeros(n_samples, dtype=int)

        rng = np.random.RandomState(seed + sim)
        derived_seeds = rng.randint(0, 1000000, size=n_runs)

        for run_seed in derived_seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run_seed)

            for train_dev_idx, _ in skf.split(np.arange(n_samples), labels):
                # Split train_dev into train and dev
                train_idx, _ = train_test_split(
                    train_dev_idx,
                    stratify=labels[train_dev_idx],
                    shuffle=True,
                    random_state=run_seed
                )

                # Mark instances that are in training
                in_train_count[train_idx] += 1

        # Check if any instance never appeared in training
        never_in_train = np.sum(in_train_count == 0)
        instances_never_in_train.append(never_in_train)

    print(f"\n{'='*80}")
    print(f"FOLD COVERAGE VERIFICATION")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Number of folds: {n_folds}")
    print(f"  - Number of runs: {n_runs}")
    print(f"  - Number of simulations: {n_simulations}")
    print(f"  - Total samples per simulation: {n_samples}")
    print(f"\nResults:")
    print(f"  - Simulations where ALL instances appeared in train: {np.sum(np.array(instances_never_in_train) == 0)}/{n_simulations}")
    print(f"  - Average instances never in train: {np.mean(instances_never_in_train):.2f}")
    print(f"  - Max instances never in train: {np.max(instances_never_in_train)}")
    print(f"  - Min instances never in train: {np.min(instances_never_in_train)}")

    if np.max(instances_never_in_train) > 0:
        print(f"\n⚠️  WARNING: Some instances may NEVER appear in the training set!")
        print(f"   This happens because train_test_split randomly assigns instances")
        print(f"   to train vs dev, and there's a small probability that an instance")
        print(f"   is always assigned to dev when it's not in the test set.")
        print(f"\n   Probability an instance is NEVER in train:")
        # For k-fold CV, instance is in test once, in train_dev (k-1) times
        # Each time in train_dev, probability of being in dev ≈ 0.25 (default test_size)
        prob_never_train = (0.25 ** (n_folds - 1)) * 100
        print(f"   Theoretical: ~{prob_never_train:.3f}%")
        print(f"   Empirical: {(np.mean(instances_never_in_train) / n_samples) * 100:.3f}%")
    else:
        print(f"\n✓ All instances appeared in training at least once in all simulations!")

    return instances_never_in_train


def check_actual_folds():
    """
    Check the actual folds that were created during preprocessing.
    """
    config = get_config()
    print(f"\n{'='*80}")
    print(f"CHECKING ACTUAL FOLDS")
    print(f"{'='*80}")

    # Check how many runs and folds exist
    data_dir = Path(config.get("data_dir"))
    train_files = sorted(data_dir.glob("train_fold_*.csv"))

    if not train_files:
        print("No fold files found. Please run preprocessing first.")
        return

    # Extract run and fold information
    run_fold_pairs = set()
    for train_file in train_files:
        # Format: train_fold_X_run_Y.csv or trainX_run-Y.csv
        parts = train_file.stem.split('_')
        if len(parts) >= 4:
            fold_idx = int(parts[2])
            run_idx = int(parts[4])
            run_fold_pairs.add((run_idx, fold_idx))

    n_runs = len(set(run_idx for run_idx, _ in run_fold_pairs))
    n_folds = len(set(fold_idx for _, fold_idx in run_fold_pairs))

    print(f"Found {n_runs} run(s) with {n_folds} fold(s) each")

    # For each run, check coverage
    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx} ---")

        # Load all folds for this run
        all_train_indices = set()
        all_dev_indices = set()
        all_test_indices = set()
        all_indices = set()

        fold_train_sets = []
        fold_test_sets = []

        for fold_idx in range(n_folds):
            train_file_path = config.get_fold_path("train", fold_idx, run_idx)
            dev_file_path = config.get_fold_path("dev", fold_idx, run_idx)
            test_file_path = config.get_fold_path("test", fold_idx, run_idx)

            if train_file_path is None or not Path(train_file_path).exists():
                print(f"  Fold {fold_idx}: Files not found")
                continue

            train_df = pd.read_csv(train_file_path)
            dev_df = pd.read_csv(dev_file_path)
            test_df = pd.read_csv(test_file_path)

            # Get the original indices (they should be in the dataframe)
            train_idx = set(train_df.index) if 'index' not in train_df.columns else set(train_df['index'])
            dev_idx = set(dev_df.index) if 'index' not in dev_df.columns else set(dev_df['index'])
            test_idx = set(test_df.index) if 'index' not in test_df.columns else set(test_df['index'])

            fold_train_sets.append(train_idx)
            fold_test_sets.append(test_idx)

            all_train_indices.update(train_idx)
            all_dev_indices.update(dev_idx)
            all_test_indices.update(test_idx)
            all_indices.update(train_idx.union(dev_idx).union(test_idx))

            print(f"  Fold {fold_idx}: train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}")

        print(f"\n  Summary:")
        print(f"    - Total unique indices across all folds: {len(all_indices)}")
        print(f"    - Indices that appeared in train at least once: {len(all_train_indices)}")
        print(f"    - Indices that appeared in dev at least once: {len(all_dev_indices)}")
        print(f"    - Indices that appeared in test at least once: {len(all_test_indices)}")

        # Check if any index never appeared in training
        never_in_train = all_indices - all_train_indices
        if never_in_train:
            print(f"\n  ⚠️  WARNING: {len(never_in_train)} indices NEVER appeared in training!")
            print(f"    These indices: {sorted(list(never_in_train))[:10]}{'...' if len(never_in_train) > 10 else ''}")
        else:
            print(f"\n  ✓ All indices appeared in training at least once!")

        # Check for overlaps between test sets (should be none)
        print(f"\n  Checking test set overlaps:")
        has_overlap = False
        for i in range(len(fold_test_sets)):
            for j in range(i + 1, len(fold_test_sets)):
                overlap = fold_test_sets[i].intersection(fold_test_sets[j])
                if overlap:
                    print(f"    ⚠️  Fold {i} and Fold {j} have {len(overlap)} overlapping test instances!")
                    has_overlap = True
        if not has_overlap:
            print(f"    ✓ No overlaps between test sets!")


if __name__ == "__main__":
    print("Verifying fold creation and coverage...\n")

    # Run simulations to understand the theoretical probability
    verify_fold_coverage(n_folds=5, n_runs=1, n_simulations=1000)

    # Check actual folds if they exist
    check_actual_folds()
