#!/usr/bin/env bash

# Correct syntax for declaring an array in bash
MODELS=(
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  FacebookAI/roberta-base
  dmis-lab/biobert-v1.1
  google-bert/bert-base-uncased
)

NUM_OPT_NEGS=(100)

NUM_RUNS=1
NUM_FOLDS=4

for opt_negs in "${NUM_OPT_NEGS[@]}"; do
  echo "adding $opt_negs optional negatives"
  uv run src/data_pipeline/biomoqa/preprocess_biomoqa.py -t -nf "$NUM_FOLDS" -nr "$NUM_RUNS" -on "$opt_negs"
  uv run src/models/biomoqa/baselines.py -on "$opt_negs" -nf "$NUM_FOLDS" -nr "$NUM_RUNS" -t
  rm -r /home/leandre/Projects/BioMoQA_Playground/data/biomoqa/folds/*
done