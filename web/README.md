# BioMoQA Scoring & Ranking System

A web interface for BioMoQA (Biodiversity Modeling with Question Answering) **ensemble scoring and ranking** using **5-fold cross-validation** for robust relevance assessment and research prioritization.

## Features

- **Ensemble Scoring**: Uses all 5 fold models trained via cross-validation for robust scoring
- **Intelligent Ranking**: Sort and rank research abstracts by relevance scores
- **Score-Based Analysis**: Focus on raw scores rather than binary classification
- **Batch Scoring & Ranking**: Upload JSON or CSV files for bulk scoring with ranking capabilities
- **Interactive Filtering**: Filter results by score thresholds and sort in any order
- **Detailed Score Analytics**: View individual fold scores, statistical analysis, and score distributions
- **Score Interpretation**: Automatic relevance level assessment (High/Medium/Low)
- **Enhanced Downloads**: Export ranked results with comprehensive score metadata

## Scoring Approach

This application implements a **consensus scoring framework** for research relevance assessment. Instead of simple binary classification, it:

1. **Loads 5 fold models** trained during cross-validation
2. **Scores each text independently** with all fold models
3. **Calculates ensemble statistics** (mean, median, std, min, max scores)
4. **Provides ranking capabilities** to identify most relevant research
5. **Delivers interpretable scores** with confidence intervals and stability metrics

This approach enables **research prioritization** and **relevance ranking**, similar to scoring systems used in [medical AI applications](https://github.com/wisdomml2020/brain-tumour-webapp) and follows [model interpretation best practices](https://walkwithfastai.com/interp) for transparency.

## Score Interpretation

| Score Range | Relevance Level | Description |
|-------------|----------------|-------------|
| 0.8 - 1.0   | 🟢 High Relevance | Strong biodiversity research content |
| 0.6 - 0.8   | 🟡 Medium-High | Likely biodiversity-related research |
| 0.4 - 0.6   | 🟠 Medium | Mixed or unclear biodiversity content |
| 0.2 - 0.4   | 🔴 Low | Unlikely to be biodiversity-focused |
| 0.0 - 0.2   | ⚫ Very Low | Not biodiversity research |

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

4. **Set Reference Threshold**: Optional threshold for binary reference (scoring focuses on raw scores)

5. **Click "Load Ensemble Models"**: Load all 5 fold models for the selected configuration

### Single Text Scoring

1. Select "Single Text Scoring" mode
2. Enter a research abstract in the text area
3. Click "Score Text"
4. View comprehensive scoring results including:
   - **Ensemble Score**: Mean score across all 5 folds (0.0-1.0)
   - **Score Range**: Min-max range showing prediction stability
   - **Score Stability**: Standard deviation indicating consistency
   - **Relevance Interpretation**: Automatic categorization (High/Medium/Low)
   - **Statistical Analysis**: Mean, median, std deviation, consensus metrics
   - **Individual Fold Scores**: Ranked visualization of all fold predictions

### Batch Scoring & Ranking

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

#### Ranking Features
- **Automatic Ranking**: Results sorted by ensemble score (highest first)
- **Score Distribution**: Histogram showing score distribution across all texts
- **Interactive Filtering**: Slider to filter by minimum score threshold
- **Sort Controls**: Toggle between highest-to-lowest and lowest-to-highest ranking
- **Summary Statistics**: Total scored, high relevance count, average score, highest score

### Example Files

Sample files are provided in the `examples/` directory:
- `sample_texts.json`: JSON format examples with varying relevance levels
- `sample_texts.csv`: CSV format examples with expected score ranges

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

## Scoring Output

The application provides comprehensive scoring results:

### Single Text Results
- **Ensemble Score**: Mean score across all 5 folds (primary ranking metric)
- **Score Range**: Min-max scores showing prediction variability
- **Score Stability**: Standard deviation indicating model consensus
- **Relevance Level**: Automatic interpretation (High/Medium/Low/Very Low)
- **Statistical Breakdown**: Mean, median, std deviation, quartiles
- **Individual Fold Analysis**: Score from each fold with ranking visualization
- **Reference Classification**: Optional binary prediction for comparison

### Batch Results
- **Ranked Results Table**: All texts sorted by ensemble score with rank numbers
- **Score Distribution**: Interactive histogram showing score patterns
- **Summary Statistics**: Total count, relevance breakdowns, averages
- **Interactive Controls**: Filtering, sorting, and display options
- **Export Options**: JSON and CSV downloads with full ranking data

## Scoring Benefits

1. **Research Prioritization**: Rank papers by biodiversity relevance for efficient review
2. **Quality Assessment**: Score stability indicates prediction confidence
3. **Comparative Analysis**: Direct score comparison across multiple texts
4. **Threshold Flexibility**: No fixed cutoffs - use scores for custom ranking
5. **Transparency**: Complete fold-by-fold breakdown for interpretability
6. **Batch Processing**: Efficient scoring and ranking of large document collections

## Use Cases

- **Literature Review**: Rank papers by biodiversity relevance for systematic reviews
- **Grant Evaluation**: Score research proposals for biodiversity content assessment
- **Database Curation**: Identify and prioritize biodiversity-related research
- **Content Discovery**: Find most relevant papers in large document collections
- **Research Validation**: Assess consistency of biodiversity research classification

## Example Scoring Results

```
🎯 Ensemble Score: 0.8234
📈 Score Range: 0.801 - 0.891
📊 Score Stability: σ = 0.042
🟢 High Relevance - Strong biodiversity research content

Individual Fold Scores (Ranked):
   Fold 3: 0.891
   Fold 5: 0.887
   Fold 1: 0.853
   Fold 4: 0.834
   Fold 2: 0.801
```

## Performance Considerations

- **Model Loading**: Initial loading takes longer as 5 models are loaded into memory
- **Scoring Time**: ~5x slower than single model inference (running 5 models)
- **Memory Usage**: ~5x memory requirements (5 models in memory simultaneously)
- **Batch Efficiency**: Optimized for processing large collections with ranking
- **GPU Acceleration**: Benefits significantly from GPU for faster ensemble scoring

## Troubleshooting

- **Ensemble loading errors**: Ensure all 5 fold directories exist for the selected model/loss combination
- **Memory issues**: Use CPU device for large model ensembles
- **Missing folds**: Check that fold directories follow the exact naming convention
- **File upload errors**: Verify JSON arrays and CSV columns as specified
- **Slow scoring**: Consider GPU acceleration for faster batch processing

## Development

This application implements ensemble scoring and ranking patterns adapted for research relevance assessment, providing a robust framework for biodiversity research prioritization and content discovery. 