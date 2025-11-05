#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Models to iterate over
MODELS=(
  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
  "FacebookAI/roberta-base"
  "dmis-lab/biobert-v1.1"
  "google-bert/bert-base-uncased"
)

# Optional negatives to try
NUM_OPT_NEGS=(500)

# Number of HPO/train runs and folds
NUM_RUNS=1
NUM_FOLDS=5

for opt_negs in "${NUM_OPT_NEGS[@]}"; do
  echo "==> Adding ${opt_negs} optional negatives"
  uv run src/data_pipeline/biomoqa/preprocess_biomoqa.py \
    -nf "${NUM_FOLDS}" \
    -nr "${NUM_RUNS}" \
    -on "${opt_negs}" 

  # repeat for each run
  for (( run=0; run<NUM_RUNS; run++ )); do
    echo "--> Run #${run}"
    # try both losses
    # each fold
    for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    echo "------> Fold: ${fold}"
      for loss in focal; do
        echo "----> Loss function: ${loss}"
        # each model
        for model in "${MODELS[@]}"; do
          echo "--------> Model: ${model}"
          # HPO
          uv run src/models/biomoqa/hpo.py \
            --config configs/hpo.yaml \
            --fold "${fold}" \
            --run "${run}" \
            --nb_opt_negs "${opt_negs}" \
            --n_trials 25 \
            --hpo_metric "eval_roc_auc" \
            -m "${model}" \
            --loss "${loss}" \
            -t

          # Training with best HPO config
          uv run src/models/biomoqa/train.py \
            --config configs/train.yaml \
            --hp_config configs/best_hpo.yaml \
            --fold "${fold}" \
            --run "${run}" \
            -m "${model}" \
            --nb_opt_negs "${opt_negs}" \
            -bm "eval_roc_auc" \
            --gpu 2 \
            --loss "${loss}" \
            -t
        done

        # Ensemble step for this fold/run
        echo "--------> Ensemble for fold ${fold}, run ${run}"
        uv run src/models/biomoqa/ensemble.py \
          --config configs/ensemble.yaml \
          --fold "${fold}" \
          --run "${run}" \
          --nb_opt_negs "${opt_negs}" \
          --loss "${loss}" \
          -t
      done
    done
  done

  # clean up folds
  echo "==> Cleaning up folds directory"
  rm -r "data/folds/"*
done
