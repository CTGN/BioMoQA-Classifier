#!/usr/bin/env bash

# Correct syntax for declaring an array in bash
MODELS=(
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  FacebookAI/roberta-base
  dmis-lab/biobert-v1.1
  google-bert/bert-base-uncased
)

NUM_OPT_NEGS=(500 100)

NUM_RUNS=2
NUM_FOLDS=5

for opt_negs in "${NUM_OPT_NEGS[@]}"; do
    echo "adding $opt_negs optional negatives"
    uv run src/data_pipeline/biomoqa/preprocess_biomoqa.py -t -nf "$NUM_FOLDS" -nr "$NUM_RUNS" -on "$opt_negs"
    # Run loop: run 1 and 2
    for run in {0..1}; do
        for loss in BCE focal; do
            echo "Loss function: $loss"

            # Fold loop: 0 to 4 (not 1 to 5)
            for fold in {0..4}; do
                echo "Fold number: $fold"
                echo "Run number: $run"

                for model in "${MODELS[@]}"; do
                    echo "Model: $model"
                    uv run src/models/biomoqa/hpo.py \
                        --config configs/hpo.yaml \
                        --fold "$fold" \
                        --run "$run" \
                        --nb_opt_negs "$opt_negs" \
                        --n_trials 3 \
                        --hpo_metric "eval_f1" \
                        -m "$model" \
                        --with_title \
                        --loss "$loss"
                    uv run src/models/biomoqa/train.py \
                    --config configs/train.yaml \
                    --hp_config configs/best_hpo.yaml \
                    --fold "$fold" \
                    --run "$run" \
                    -m "$model" \
                    --nb_opt_negs "$opt_negs" \
                    -bm "eval_f1" \
                    --gpu 2\
                    --with_title \
                    --loss "$loss"
                done
                echo "Ensemble :"
                #Ensemble Learning after running every model on one fold 
                uv run src/models/biomoqa/ensemble.py \
                --config configs/ensemble.yaml \
                --fold "$fold" \
                --run "$run" \
                --nb_opt_negs "$opt_negs" \
                --with_title \
                --loss "$loss"
            done
        done
    done
    rm -r /home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/*
done