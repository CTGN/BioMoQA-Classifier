# BioMoQA Classifier Web Application

A simple web interface for BioMoQA (Biodiversity Modeling with Question Answering) text classification using Streamlit and Hugging Face transformers.

## Features

- **Single Text Classification**: Classify individual research abstracts
- **Batch Processing**: Upload JSON or CSV files for bulk classification
- **Example Texts**: Test with pre-loaded example texts
- **Simple Interface**: Clean, intuitive web interface
- **Multiple Formats**: Support for both JSON and CSV file uploads
- **Results Download**: Download results in JSON or CSV format

## Requirements

- Python 3.8+
- streamlit
- torch
- transformers
- pandas
- plotly

## Installation

1. Install dependencies:
```bash
pip install streamlit torch transformers pandas plotly
```

2. Navigate to the web directory:
```bash
cd web
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Model Setup

1. **Load a Model**: In the sidebar, enter the path to your trained BioMoQA model checkpoint
2. **Configure Settings**: Adjust the classification threshold if needed
3. **Select Device**: Choose CPU or CUDA for inference
4. **Click "Load Model"**: Load the model for inference

### Single Text Classification

1. Select "Single Text" mode
2. Enter a research abstract in the text area
3. Click "Classify Text"
4. View the prediction results and confidence score

### Batch Processing

#### JSON Format
Upload a JSON file with the following structure:
```json
[
  {
    "abstract": "Your research abstract text here..."
  },
  {
    "abstract": "Another research abstract..."
  }
]
```

#### CSV Format
Upload a CSV file with an "abstract" column:
```csv
abstract,title,category
"Research abstract 1...","Title 1","category1"
"Research abstract 2...","Title 2","category2"
```

### Example Files

Sample files are provided in the `examples/` directory:
- `sample_texts.json`: JSON format examples
- `sample_texts.csv`: CSV format examples

## Model Requirements

The application expects Hugging Face format models with:
- `config.json`: Model configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `tokenizer.json` and related tokenizer files (optional, falls back to BERT tokenizer)

## Model Training

This web app is designed to work with models trained **without titles**, using only abstracts for classification. The model should be a binary classifier for biodiversity-related research questions.

## Output

The application provides:
- **Binary prediction**: 0 (not biodiversity-related) or 1 (biodiversity-related)
- **Confidence score**: Probability score between 0 and 1
- **Visual feedback**: Gauge chart showing confidence level
- **Batch results**: Summary statistics and downloadable results

## Troubleshooting

- **Model loading errors**: Ensure the model path is correct and contains valid Hugging Face model files
- **Memory issues**: Use CPU device for large models if GPU memory is limited
- **File upload errors**: Check that JSON files contain valid arrays and CSV files have an "abstract" column

## Example Model Paths

```
results/biomoqa/best_model_fold_1/
path/to/your/huggingface/model/
models/biomoqa_bert_base/
```

## Development

The application uses a simplified inference pipeline built on top of Hugging Face transformers, focusing on ease of use and deployment rather than complex model configurations. 