# BioMoQA Model Instantiation

This module provides functionality to load trained BioMoQA models and make predictions on new texts.

## Features

- Load trained model checkpoints
- Support for different input configurations (abstract only, with title, with keywords)
- Single text prediction
- Batch prediction from JSON files
- Configurable prediction threshold
- GPU/CPU device selection
- JSON output for results

## Command Line Usage

### Basic Usage

```bash
# Show help
python src/models/biomoqa/instantiation.py --help

# Run built-in example
python src/models/biomoqa/instantiation.py --model_path /path/to/model --example
```

### Single Text Prediction

```bash
# Abstract only
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --abstract "Your abstract text here"

# With title (if model was trained with titles)
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --with_title \
    --abstract "Your abstract text" \
    --title "Your title"

# With keywords (if model was trained with keywords) 
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --with_keywords \
    --abstract "Your abstract text" \
    --keywords "keyword1, keyword2, keyword3"

# With both title and keywords
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --with_title --with_keywords \
    --abstract "Your abstract text" \
    --title "Your title" \
    --keywords "keyword1, keyword2"
```

### Batch Prediction

```bash
# From JSON file
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --input_file examples/sample_texts.json

# With title and keywords enabled
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --with_title --with_keywords \
    --input_file examples/sample_texts.json \
    --output_file results.json
```

### Advanced Options

```bash
# Custom threshold
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --abstract "Your text" \
    --threshold 0.7

# Force CPU usage
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --abstract "Your text" \
    --device cpu

# Verbose output
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --abstract "Your text" \
    --verbose

# Save results to file
python src/models/biomoqa/instantiation.py \
    --model_path /path/to/model \
    --input_file texts.json \
    --output_file predictions.json
```

## Arguments Reference

### Required Arguments
- `--model_path`: Path to the trained model checkpoint

### Input Arguments (choose one)
- `--abstract`: Abstract text for single prediction
- `--input_file`: JSON file with texts for batch prediction  
- `--example`: Run built-in example

### Optional Text Arguments
- `--title`: Title text (required if `--with_title` is set)
- `--keywords`: Keywords (required if `--with_keywords` is set)

### Model Configuration
- `--with_title`: Model was trained with titles
- `--with_keywords`: Model was trained with keywords
- `--threshold`: Classification threshold (default: 0.5)
- `--device`: Device to use (`cuda`, `cpu`, `auto` - default: auto)

### Output Options
- `--output_file`: Save results to JSON file
- `--verbose`: Enable verbose logging

## Input File Format

For batch prediction, use JSON format with an array of objects:

```json
[
  {
    "abstract": "Your abstract text here...",
    "title": "Optional title",
    "keywords": "optional, keywords, here"
  },
  {
    "abstract": "Another abstract...",
    "title": "Another title",
    "keywords": "more, keywords"
  }
]
```

## Output Format

Results are returned in JSON format:

```json
[
  {
    "abstract": "Your abstract text...",
    "title": "Your title",
    "keywords": "your, keywords",
    "score": 0.8456,
    "prediction": 1
  }
]
```

## Programmatic Usage

```python
from src.models.biomoqa.instantiation import load_predictor

# Load model
predictor = load_predictor(
    model_path="/path/to/model",
    with_title=True,
    with_keywords=False,
    threshold=0.6
)

# Single prediction
score = predictor.predict_score(
    abstract="Your abstract",
    title="Your title"
)

# Batch prediction
scores = predictor.predict_batch(
    abstracts=["abstract1", "abstract2"],
    titles=["title1", "title2"]
)

# Full evaluation
result = predictor.evaluate_text(
    abstract="Your abstract",
    title="Your title",
    return_binary=True
)
```

## Model Path Examples

Typical model paths in your project:
```bash
# Single model
/home/leandre/Projects/BioMoQA_Playground/results/biomoqa/final_model/best_model_cross_val_BCE_BiomedBERT-abs_fold-1

# Different models with different configurations
/path/to/model_with_title_and_keywords
/path/to/model_abstract_only
/path/to/model_with_title_only
```

## Tips

1. **Model Configuration**: Make sure to set `--with_title` and `--with_keywords` flags to match how your model was trained
2. **Threshold Tuning**: Adjust `--threshold` based on your use case (lower for higher recall, higher for higher precision)
3. **Batch Processing**: Use `--input_file` for processing multiple texts efficiently
4. **Error Handling**: Use `--verbose` for detailed error messages during debugging 