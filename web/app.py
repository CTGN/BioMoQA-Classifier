import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src to path to import your modules
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from web.utils import get_example_texts, validate_model_path, format_confidence_score

# Page config
st.set_page_config(
    page_title="BioMoQA Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CrossValidationPredictor:
    """Cross-validation predictor using ensemble of 5 fold models"""
    
    def __init__(self, model_type: str, loss_type: str, base_path: str = "results/biomoqa/final_model", 
                 threshold: float = 0.5, device: str = None):
        self.model_type = model_type
        self.loss_type = loss_type
        self.base_path = base_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store fold models and tokenizers
        self.fold_models = {}
        self.fold_tokenizers = {}
        self.num_folds = 5
        
        self._load_fold_models()
    
    def _load_fold_models(self):
        """Load all 5 fold models for the specified model type and loss type"""
        print(f"Loading {self.num_folds} fold models for {self.model_type} with {self.loss_type} loss...")
        
        for fold in range(1, self.num_folds + 1):
            fold_path = os.path.join(
                self.base_path, 
                f"best_model_cross_val_{self.loss_type}_{self.model_type}_fold-{fold}"
            )
            
            if not os.path.exists(fold_path):
                raise FileNotFoundError(f"Fold model not found: {fold_path}")
            
            try:
                # Load tokenizer (try from model path, fallback to default)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(fold_path)
                except:
                    print(f"Using default BERT tokenizer for fold {fold}")
                    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
                
                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(fold_path)
                model.to(self.device)
                model.eval()
                
                self.fold_tokenizers[fold] = tokenizer
                self.fold_models[fold] = model
                
                print(f"✅ Loaded fold {fold} successfully")
                
            except Exception as e:
                raise Exception(f"Failed to load fold {fold}: {str(e)}")
        
        print(f"Successfully loaded all {self.num_folds} fold models!")
    
    def predict_single_fold(self, abstract: str, fold: int) -> dict:
        """Predict using a single fold model"""
        tokenizer = self.fold_tokenizers[fold]
        model = self.fold_models[fold]
        
        # Tokenize input
        inputs = tokenizer(
            abstract,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).squeeze().cpu().item()
        
        prediction = int(score > self.threshold)
        return {
            "fold": fold,
            "score": score,
            "prediction": prediction
        }
    
    def predict(self, abstract: str) -> dict:
        """Predict using ensemble of all fold models"""
        fold_results = []
        scores = []
        
        # Get predictions from all folds
        for fold in range(1, self.num_folds + 1):
            fold_result = self.predict_single_fold(abstract, fold)
            fold_results.append(fold_result)
            scores.append(fold_result["score"])
        
        # Calculate ensemble statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Ensemble prediction based on mean score
        ensemble_prediction = int(mean_score > self.threshold)
        
        # Count individual fold predictions
        positive_folds = sum(1 for result in fold_results if result["prediction"] == 1)
        negative_folds = self.num_folds - positive_folds
        
        # Consensus strength (percentage of folds agreeing with ensemble)
        consensus_strength = max(positive_folds, negative_folds) / self.num_folds
        
        return {
            "abstract": abstract,
            "ensemble_score": mean_score,
            "ensemble_prediction": ensemble_prediction,
            "fold_results": fold_results,
            "statistics": {
                "mean_score": mean_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score,
                "positive_folds": positive_folds,
                "negative_folds": negative_folds,
                "consensus_strength": consensus_strength
            }
        }
    
    def predict_batch(self, abstracts: list) -> list:
        """Predict on a batch of abstracts using ensemble"""
        results = []
        for abstract in abstracts:
            result = self.predict(abstract)
            results.append(result)
        return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 BioMoQA Cross-Validation Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Sidebar for model configuration
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Input mode selection
        input_mode = st.radio(
            "Select input mode:",
            ["Single Text", "Batch Upload (JSON/CSV)", "Example Texts"],
            horizontal=True
        )
        
        if input_mode == "Single Text":
            render_single_text_input()
        elif input_mode == "Batch Upload (JSON/CSV)":
            render_batch_upload()
        else:
            render_example_texts()
    
    with col2:
        st.header("⚙️ Model Status")
        render_model_status()

def render_sidebar():
    """Render the sidebar for model configuration"""
    st.sidebar.header("🔧 Cross-Validation Model Configuration")
    
    # Available model types (based on your directory listing)
    model_types = [
        "bert-base",
        "biobert-v1", 
        "BiomedBERT-abs",
        "BiomedBERT-abs-ft",
        "roberta-base"
    ]
    
    # Available loss types
    loss_types = ["BCE", "focal"]
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        model_types,
        help="Select the base model architecture"
    )
    
    loss_type = st.sidebar.selectbox(
        "Loss Type", 
        loss_types,
        help="Select the loss function used during training"
    )
    
    # Base path for models
    base_path = st.sidebar.text_input(
        "Models Base Path",
        value="results/biomoqa/final_model",
        help="Base directory containing the fold model checkpoints"
    )
    
    # Threshold
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Threshold for binary classification (applied to ensemble mean)"
    )
    
    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        help="Select computation device"
    )
    device = None if device_option == "auto" else device_option
    
    # Show model info
    st.sidebar.subheader("📋 Model Configuration")
    st.sidebar.info(f"**Model:** {model_type}")
    st.sidebar.info(f"**Loss:** {loss_type}")
    st.sidebar.info(f"**Folds:** 5 models")
    
    # Load model button
    if st.sidebar.button("🚀 Load Ensemble Models", type="primary", use_container_width=True):
        load_ensemble_models(model_type, loss_type, base_path, threshold, device)
    
    # Example paths
    st.sidebar.subheader("💡 Example Model Configurations")
    st.sidebar.code("Model: BiomedBERT-abs\nLoss: BCE")
    st.sidebar.code("Model: bert-base\nLoss: focal")

def load_ensemble_models(model_type: str, loss_type: str, base_path: str, threshold: float, device: str):
    """Load the ensemble of fold models"""
    with st.spinner("Loading ensemble models..."):
        try:
            # Validate base path
            if not os.path.exists(base_path):
                st.sidebar.error("Base path does not exist. Please check the path.")
                return
            
            # Load predictor
            predictor = CrossValidationPredictor(
                model_type=model_type,
                loss_type=loss_type,
                base_path=base_path,
                threshold=threshold,
                device=device
            )
            
            # Store in session state
            st.session_state.predictor = predictor
            st.session_state.model_loaded = True
            
            st.sidebar.success(f"✅ Loaded {predictor.num_folds} fold models successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Failed to load ensemble models: {str(e)}")
            st.session_state.model_loaded = False

def render_single_text_input():
    """Render the single text input interface"""
    # Abstract input
    abstract = st.text_area(
        "Abstract*",
        height=200,
        placeholder="Enter the research abstract here...",
        help="The abstract text to classify using cross-validation ensemble"
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔍 Classify with Ensemble",
            type="primary",
            use_container_width=True
        )
    
    # Validation and prediction
    if predict_button:
        if not abstract.strip():
            st.error("Please enter an abstract.")
            return
        
        if not st.session_state.model_loaded:
            st.error("Please load ensemble models first in the sidebar.")
            return
        
        # Make prediction
        with st.spinner("Running ensemble classification..."):
            try:
                result = st.session_state.predictor.predict(abstract)
                render_ensemble_results(result)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def render_batch_upload():
    """Render the batch upload interface"""
    st.info("Upload a JSON or CSV file with multiple texts for batch ensemble classification.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['json', 'csv'],
        help="Upload a JSON file (array of objects) or CSV file with 'abstract' column"
    )
    
    if uploaded_file:
        try:
            # Parse file based on type
            if uploaded_file.name.endswith('.json'):
                texts_data = json.load(uploaded_file)
                if not isinstance(texts_data, list):
                    st.error("JSON file must contain an array of objects.")
                    return
                abstracts = [item.get('abstract', '') for item in texts_data]
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'abstract' not in df.columns:
                    st.error("CSV file must contain an 'abstract' column.")
                    return
                abstracts = df['abstract'].fillna('').tolist()
                texts_data = df.to_dict('records')
            
            st.success(f"Loaded {len(abstracts)} texts for ensemble classification.")
            
            # Show preview
            with st.expander("Preview uploaded data"):
                if uploaded_file.name.endswith('.json'):
                    st.json(texts_data[:3])
                else:
                    st.dataframe(df.head(3))
            
            # Batch processing
            if st.button("🚀 Process Batch with Ensemble", type="primary"):
                if not st.session_state.model_loaded:
                    st.error("Please load ensemble models first.")
                    return
                
                process_batch_ensemble(abstracts)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_example_texts():
    """Render the example texts interface"""
    st.info("Try the ensemble classifier with pre-loaded example texts.")
    
    examples = get_example_texts()
    
    selected_example = st.selectbox(
        "Choose an example:",
        range(len(examples)),
        format_func=lambda x: f"Example {x+1}: {examples[x]['title'][:50]}..."
    )
    
    example = examples[selected_example]
    
    # Display example
    st.markdown("**Title:** " + example['title'])
    st.markdown("**Keywords:** " + example['keywords'])
    st.markdown("**Abstract:** " + example['abstract'][:200] + "...")
    
    # Classify example
    if st.button("🔍 Classify with Ensemble", type="primary"):
        if not st.session_state.model_loaded:
            st.error("Please load ensemble models first.")
            return
        
        with st.spinner("Running ensemble classification..."):
            try:
                result = st.session_state.predictor.predict(example['abstract'])
                render_ensemble_results(result)
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")

def render_model_status():
    """Render model loading status"""
    if st.session_state.model_loaded:
        st.success("✅ Ensemble models loaded!")
        st.info(f"**Model Type:** {st.session_state.predictor.model_type}")
        st.info(f"**Loss Type:** {st.session_state.predictor.loss_type}")
        st.info(f"**Folds:** {st.session_state.predictor.num_folds}")
        st.info(f"**Device:** {st.session_state.predictor.device}")
        st.info(f"**Threshold:** {st.session_state.predictor.threshold}")
    else:
        st.warning("⚠️ No ensemble models loaded")
        st.info("Please configure and load ensemble models in the sidebar.")

def render_ensemble_results(result):
    """Render ensemble prediction results with detailed analysis"""
    st.markdown("---")
    st.header("🎯 Ensemble Classification Results")
    
    ensemble_score = result['ensemble_score']
    ensemble_prediction = result['ensemble_prediction']
    stats = result['statistics']
    
    # Main ensemble result
    col1, col2 = st.columns(2)
    
    with col1:
        if ensemble_prediction == 1:
            st.success("🎯 **POSITIVE** - Biomedical Research Question")
        else:
            st.info("❌ **NEGATIVE** - Not a Biomedical Research Question")
    
    with col2:
        st.metric("Ensemble Confidence", f"{ensemble_score:.1%}")
    
    # Consensus strength indicator
    consensus_strength = stats['consensus_strength']
    if consensus_strength >= 0.8:
        consensus_color = "🟢 Strong"
    elif consensus_strength >= 0.6:
        consensus_color = "🟡 Moderate"
    else:
        consensus_color = "🔴 Weak"
    
    st.metric("Consensus Strength", f"{consensus_strength:.1%}", help="Percentage of folds agreeing with ensemble decision")
    st.markdown(f"**Consensus Level:** {consensus_color}")
    
    # Ensemble statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Score", f"{stats['mean_score']:.3f}")
    with col2:
        st.metric("Std Deviation", f"{stats['std_score']:.3f}")
    with col3:
        st.metric("Min Score", f"{stats['min_score']:.3f}")
    with col4:
        st.metric("Max Score", f"{stats['max_score']:.3f}")
    
    # Fold agreement breakdown
    st.subheader("📊 Fold Agreement")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Positive Folds", f"{stats['positive_folds']}/5")
    with col2:
        st.metric("Negative Folds", f"{stats['negative_folds']}/5")
    
    # Individual fold results
    with st.expander("🔍 Individual Fold Results"):
        fold_df = pd.DataFrame(result['fold_results'])
        fold_df['prediction_label'] = fold_df['prediction'].map({0: 'Negative', 1: 'Positive'})
        fold_df['score'] = fold_df['score'].round(4)
        
        st.dataframe(fold_df[['fold', 'score', 'prediction_label']], use_container_width=True)
        
        # Fold scores visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i}" for i in range(1, 6)],
            y=[result['fold_results'][i-1]['score'] for i in range(1, 6)],
            name="Fold Scores",
            marker_color=['red' if score < st.session_state.predictor.threshold else 'green' 
                         for score in [result['fold_results'][i-1]['score'] for i in range(1, 6)]]
        ))
        
        # Add threshold line
        fig.add_hline(y=st.session_state.predictor.threshold, 
                     line_dash="dash", line_color="blue",
                     annotation_text="Threshold")
        
        # Add ensemble mean line
        fig.add_hline(y=ensemble_score, 
                     line_dash="dot", line_color="purple",
                     annotation_text="Ensemble Mean")
        
        fig.update_layout(
            title="Individual Fold Scores",
            xaxis_title="Fold",
            yaxis_title="Score",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def process_batch_ensemble(abstracts):
    """Process batch of abstracts using ensemble"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, abstract in enumerate(abstracts):
        status_text.text(f"Processing {i+1}/{len(abstracts)} with ensemble...")
        
        try:
            result = st.session_state.predictor.predict(abstract)
            # Simplify result for batch display
            simplified_result = {
                "abstract": abstract,
                "ensemble_score": result["ensemble_score"],
                "ensemble_prediction": result["ensemble_prediction"],
                "consensus_strength": result["statistics"]["consensus_strength"],
                "positive_folds": result["statistics"]["positive_folds"]
            }
            results.append(simplified_result)
        except Exception as e:
            st.warning(f"Failed to process item {i+1}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(abstracts))
    
    status_text.text("Ensemble processing complete!")
    
    # Display batch results
    st.header("📊 Batch Ensemble Results")
    
    # Summary stats
    positive_count = sum(1 for r in results if r.get('ensemble_prediction') == 1)
    avg_consensus = np.mean([r['consensus_strength'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", len(results))
    with col2:
        st.metric("Positive Classifications", positive_count)
    with col3:
        st.metric("Negative Classifications", len(results) - positive_count)
    with col4:
        st.metric("Avg Consensus", f"{avg_consensus:.1%}")
    
    # Results table
    df_results = pd.DataFrame([
        {
            "Index": i,
            "Abstract": r['abstract'][:100] + "..." if len(r['abstract']) > 100 else r['abstract'],
            "Ensemble Score": f"{r['ensemble_score']:.3f}",
            "Prediction": "Positive" if r['ensemble_prediction'] == 1 else "Negative",
            "Consensus": f"{r['consensus_strength']:.1%}",
            "Positive Folds": f"{r['positive_folds']}/5"
        }
        for i, r in enumerate(results)
    ])
    
    st.dataframe(df_results, use_container_width=True)
    
    # Download results
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download Ensemble Results (JSON)",
        data=results_json,
        file_name="biomoqa_ensemble_results.json",
        mime="application/json"
    )
    
    # Download as CSV
    df_download = pd.DataFrame(results)
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="📥 Download Ensemble Results (CSV)",
        data=csv,
        file_name="biomoqa_ensemble_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()