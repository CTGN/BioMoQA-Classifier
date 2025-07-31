# BioMoQA Cross-Validation Classifier Web Application

A web interface for BioMoQA (Biodiversity Modeling with Question Answering) ensemble classification using **5-fold cross-validation** for enhanced reliability and robust predictions.

## Features

- **Ensemble Classification**: Uses all 5 fold models trained via cross-validation
- **Consensus Validation**: Implements multi-model consensus for reliable predictions
- **Single Text Classification**: Classify individual research abstracts with ensemble scoring
- **Batch Processing**: Upload JSON or CSV files for bulk ensemble classification
- **Example Texts**: Test with pre-loaded example texts
- **Detailed Analytics**: View individual fold results, consensus strength, and statistical analysis
- **Multiple Formats**: Support for both JSON and CSV file uploads
- **Enhanced Results**: Download ensemble results with consensus metrics

## Cross-Validation Approach

This application implements a **consensus validation framework** inspired by blockchain-style decentralized validation. Instead of relying on a single model, it:

1. **Loads 5 fold models** trained during cross-validation
2. **Queries each fold independently** for the same input
3. **Calculates ensemble statistics** (mean, std, min, max scores)
4. **Provides consensus metrics** showing agreement between folds
5. **Delivers robust predictions** with confidence intervals

This approach significantly improves reliability compared to single-model inference, similar to the [consensus validation patterns](https://discuss.huggingface.co/t/consensus-validation-for-llm-outputs-applying-blockchain-inspired-models-to-ai-reliability/158143) used in production AI systems.

## Requirements

- Python 3.8+
- streamlit
- torch
- transformers
- pandas
- plotly
- numpy

## Installation

1. Install dependencies:
```bash
pip install streamlit torch transformers pandas plotly numpy
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

1. **Select Model Type**: Choose from available architectures:
   - `bert-base`
   - `biobert-v1`
   - `BiomedBERT-abs`
   - `BiomedBERT-abs-ft`
   - `roberta-base`

2. **Select Loss Type**: Choose the loss function used during training:
   - `BCE` (Binary Cross Entropy)
   - `focal` (Focal Loss)

3. **Configure Base Path**: Set the directory containing fold models (default: `results/biomoqa/final_model`)

4. **Adjust Threshold**: Set classification threshold for ensemble mean score

5. **Click "Load Ensemble Models"**: Load all 5 fold models for the selected configuration

### Single Text Classification

1. Select "Single Text" mode
2. Enter a research abstract in the text area
3. Click "Classify with Ensemble"
4. View comprehensive results including:
   - **Ensemble prediction** and confidence score
   - **Consensus strength** (agreement between folds)
   - **Statistical metrics** (mean, std, min, max)
   - **Individual fold results** with visualization
   - **Fold agreement breakdown**

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

## Model Directory Structure

The application expects models organized as follows:
```
results/biomoqa/final_model/
├── best_model_cross_val_BCE_bert-base_fold-1/
├── best_model_cross_val_BCE_bert-base_fold-2/
├── best_model_cross_val_BCE_bert-base_fold-3/
├── best_model_cross_val_BCE_bert-base_fold-4/
├── best_model_cross_val_BCE_bert-base_fold-5/
├── best_model_cross_val_focal_bert-base_fold-1/
└── ... (other model types and loss functions)
```

Each fold directory should contain:
- `config.json`: Model configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- Tokenizer files (optional - falls back to BERT tokenizer)

## Ensemble Output

The application provides comprehensive ensemble results:

### Single Text Results
- **Ensemble Score**: Mean score across all 5 folds
- **Ensemble Prediction**: Binary classification based on mean score
- **Consensus Strength**: Percentage of folds agreeing with ensemble decision
- **Statistical Analysis**: Mean, standard deviation, min/max scores
- **Individual Fold Results**: Score and prediction for each fold
- **Visualization**: Bar chart showing fold scores vs. threshold

### Batch Results
- **Summary Statistics**: Total processed, positive/negative counts, average consensus
- **Detailed Table**: Per-item ensemble scores, predictions, and consensus metrics
- **Download Options**: JSON and CSV formats with full ensemble data

## Consensus Validation Benefits

1. **Error Tolerance**: Individual fold errors are mitigated by ensemble averaging
2. **Reliability Metrics**: Consensus strength indicates prediction confidence
3. **Variance Analysis**: Standard deviation reveals prediction stability
4. **Auditable Results**: Complete fold-by-fold breakdown for transparency
5. **Robust Predictions**: Less susceptible to single-model biases or failures

## Example Model Configurations

```
Model Type: BiomedBERT-abs
Loss Type: BCE
Result: 5 fold models for biomedical BERT with binary cross-entropy loss

Model Type: roberta-base  
Loss Type: focal
Result: 5 fold models for RoBERTa with focal loss
```

## Troubleshooting

- **Ensemble loading errors**: Ensure all 5 fold directories exist for the selected model/loss combination
- **Memory issues**: Use CPU device for large model ensembles
- **Missing folds**: Check that fold directories follow the exact naming convention
- **File upload errors**: Verify JSON arrays and CSV columns as specified

## Performance Considerations

- **Model Loading**: Initial loading takes longer as 5 models are loaded into memory
- **Inference Time**: ~5x slower than single model inference (running 5 models)
- **Memory Usage**: ~5x memory requirements (5 models in memory simultaneously)
- **GPU Efficiency**: Benefits from GPU acceleration for faster ensemble processing

## Development

This application implements the [multi-model inference pattern](https://www.philschmid.de/multi-model-inference-endpoints) adapted for cross-validation ensembles, providing a practical "repair kit" for AI reliability through consensus validation. 