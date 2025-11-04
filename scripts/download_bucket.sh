#!/usr/bin/env bash

# Detect grep command
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Running on macOS — installing via Homebrew..."
    brew install grep
    GREP_CMD="ggrep"
else
    GREP_CMD="grep"
fi

# Create the target directories
mkdir -p results/final_model
mkdir -p data

# Download and extract keys
curl -s https://biomoqa-classifier.s3.text-analytics.ch/ \
| $GREP_CMD -oP '(?<=<Key>)[^<]+' \
| while read -r file; do
    if [[ $file == checkpoints/* ]]; then
        target_file="results/final_model/${file#checkpoints/}"
        mkdir -p "$(dirname "$target_file")"
        echo "Downloading checkpoint: $file -> $target_file"
        wget -O "$target_file" "https://biomoqa-classifier.s3.text-analytics.ch/$file"
    elif [[ $file == dataset/* ]]; then
        target_file="data/${file#dataset/}"
        mkdir -p "$(dirname "$target_file")"
        echo "Downloading dataset: $file -> $target_file"
        wget -O "$target_file" "https://biomoqa-classifier.s3.text-analytics.ch/$file"
    else
        echo "Skipping: $file (not in checkpoints or dataset directory)"
    fi
done

echo "Download complete!"
echo "Checkpoints: results/final_model/"
echo "Dataset: data/"