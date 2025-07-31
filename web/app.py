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
    page_title="BioMoQA Scorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CrossValidationPredictor:
    """Cross-validation predictor using ensemble of 5 fold models for scoring"""
    
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
        """Score using a single fold model"""
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
        
        # Optional binary prediction for reference
        prediction = int(score > self.threshold)
        return {
            "fold": fold,
            "score": score,
            "prediction": prediction
        }
    
    def score_text(self, abstract: str) -> dict:
        """Score text using ensemble of all fold models"""
        fold_results = []
        scores = []
        
        # Get scores from all folds
        for fold in range(1, self.num_folds + 1):
            fold_result = self.predict_single_fold(abstract, fold)
            fold_results.append(fold_result)
            scores.append(fold_result["score"])
        
        # Calculate ensemble statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        median_score = np.median(scores)
        
        # Optional ensemble prediction for reference
        ensemble_prediction = int(mean_score > self.threshold)
        
        # Count individual fold predictions for consensus
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
                "median_score": median_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score,
                "positive_folds": positive_folds,
                "negative_folds": negative_folds,
                "consensus_strength": consensus_strength
            }
        }
    
    def score_batch(self, abstracts: list) -> list:
        """Score a batch of abstracts using ensemble"""
        results = []
        for abstract in abstracts:
            result = self.score_text(abstract)
            results.append(result)
        return results

    def predict_batch_single_fold(self, abstracts: list, fold: int, batch_size: int = 16) -> list:
        """Score a batch of abstracts using a single fold model with GPU optimization"""
        tokenizer = self.fold_tokenizers[fold]
        model = self.fold_models[fold]
        
        all_results = []
        
        # Process in batches to manage GPU memory
        for i in range(0, len(abstracts), batch_size):
            batch_abstracts = abstracts[i:i + batch_size]
            
            # Batch tokenization - much faster than individual tokenization
            inputs = tokenizer(
                batch_abstracts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Batch inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                scores = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # Handle single item case
                if len(batch_abstracts) == 1:
                    scores = [scores.item()]
                else:
                    scores = scores.tolist()
                
                # Create results for this batch
                for j, score in enumerate(scores):
                    prediction = int(score > self.threshold)
                    all_results.append({
                        "fold": fold,
                        "score": score,
                        "prediction": prediction
                    })
        
        return all_results

    def score_batch_optimized(self, abstracts: list, batch_size: int = 16) -> list:
        """Optimized batch scoring using ensemble of all fold models with GPU acceleration"""
        num_texts = len(abstracts)
        
        # Initialize results structure
        all_fold_results = {fold: [] for fold in range(1, self.num_folds + 1)}
        
        # Process each fold with batch inference
        for fold in range(1, self.num_folds + 1):
            fold_results = self.predict_batch_single_fold(abstracts, fold, batch_size)
            all_fold_results[fold] = fold_results
            
            # Clear GPU cache between folds to prevent OOM
            if self.device == "cuda" or (self.device is None and torch.cuda.is_available()):
                torch.cuda.empty_cache()
        
        # Combine results for each text
        final_results = []
        for text_idx in range(num_texts):
            # Collect scores from all folds for this text
            fold_results = []
            scores = []
            
            for fold in range(1, self.num_folds + 1):
                fold_result = all_fold_results[fold][text_idx]
                fold_results.append(fold_result)
                scores.append(fold_result["score"])
            
            # Calculate ensemble statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            median_score = np.median(scores)
            
            # Ensemble prediction
            ensemble_prediction = int(mean_score > self.threshold)
            
            # Count individual fold predictions for consensus
            positive_folds = sum(1 for result in fold_results if result["prediction"] == 1)
            negative_folds = self.num_folds - positive_folds
            
            # Consensus strength
            consensus_strength = max(positive_folds, negative_folds) / self.num_folds
            
            final_results.append({
                "abstract": abstracts[text_idx],
                "ensemble_score": mean_score,
                "ensemble_prediction": ensemble_prediction,
                "fold_results": fold_results,
                "statistics": {
                    "mean_score": mean_score,
                    "median_score": median_score,
                    "std_score": std_score,
                    "min_score": min_score,
                    "max_score": max_score,
                    "positive_folds": positive_folds,
                    "negative_folds": negative_folds,
                    "consensus_strength": consensus_strength
                }
            })
        
        # Final GPU memory cleanup
        if self.device == "cuda" or (self.device is None and torch.cuda.is_available()):
            torch.cuda.empty_cache()
        
        return final_results

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 BioMoQA Scoring & Ranking System</h1>', unsafe_allow_html=True)
    st.markdown("**Score and rank research abstracts using ensemble cross-validation models**")
    st.markdown("---")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 16
    
    # Sidebar for model configuration
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Input mode selection
        input_mode = st.radio(
            "Select input mode:",
            ["Single Text Scoring", "Batch Scoring & Ranking", "Example Texts"],
            horizontal=True
        )
        
        if input_mode == "Single Text Scoring":
            render_single_text_input()
        elif input_mode == "Batch Scoring & Ranking":
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
    
    # Threshold (less important for scoring, but kept for reference)
    threshold = st.sidebar.slider(
        "Reference Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Reference threshold for binary classification (scoring focuses on raw scores)"
    )
    
    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        help="Select computation device"
    )
    device = None if device_option == "auto" else device_option
    
    # GPU Optimization Settings
    st.sidebar.subheader("🚀 GPU Optimization")
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=64,
        value=16,
        step=1,
        help="Number of texts to process together (increase for A100 GPU, decrease if OOM)"
    )
    
    # Store batch size in session state
    st.session_state.batch_size = batch_size
    
    # GPU memory info
    if device_option in ["auto", "cuda"] and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.sidebar.info(f"🔥 GPU Memory: {gpu_memory:.1f} GB")
        if gpu_memory >= 15:  # A100 or similar
            st.sidebar.success("🚀 High-end GPU detected! Consider batch_size=32-64")
        elif gpu_memory >= 8:
            st.sidebar.info("💡 Mid-range GPU: batch_size=16-32 recommended")
        else:
            st.sidebar.warning("⚠️ Low GPU memory: use batch_size=8 or less")
    
    # Show model info
    st.sidebar.subheader("📋 Model Configuration")
    st.sidebar.info(f"**Model:** {model_type}")
    st.sidebar.info(f"**Loss:** {loss_type}")
    st.sidebar.info(f"**Folds:** 5 models")
    st.sidebar.info(f"**Batch Size:** {batch_size}")
    
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
        help="The abstract text to score using cross-validation ensemble"
    )
    
    # Scoring button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score_button = st.button(
            "📊 Score Text",
            type="primary",
            use_container_width=True
        )
    
    # Validation and scoring
    if score_button:
        if not abstract.strip():
            st.error("Please enter an abstract.")
            return
        
        if not st.session_state.model_loaded:
            st.error("Please load ensemble models first in the sidebar.")
            return
        
        # Make prediction
        with st.spinner("Scoring with ensemble..."):
            try:
                result = st.session_state.predictor.score_text(abstract)
                render_scoring_results(result)
                
            except Exception as e:
                st.error(f"Scoring failed: {str(e)}")

def render_batch_upload():
    """Render the batch upload interface"""
    st.info("🚀 Upload a JSON or CSV file with multiple texts for GPU-accelerated batch scoring and ranking.")
    st.success("⚡ Optimized for A100 GPU: Process hundreds of texts in seconds with batch inference!")
    
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
            
            st.success(f"Loaded {len(abstracts)} texts for ensemble scoring.")
            
            # Show preview
            with st.expander("Preview uploaded data"):
                if uploaded_file.name.endswith('.json'):
                    st.json(texts_data[:3])
                else:
                    st.dataframe(df.head(3))
            
            # Batch processing
            if st.button("🚀 Score & Rank Batch", type="primary"):
                if not st.session_state.model_loaded:
                    st.error("Please load ensemble models first.")
                    return
                
                process_batch_scoring(abstracts)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_example_texts():
    """Render the example texts interface"""
    st.info("Try the ensemble scorer with pre-loaded example texts.")
    
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
    
    # Score example
    if st.button("📊 Score This Example", type="primary"):
        if not st.session_state.model_loaded:
            st.error("Please load ensemble models first.")
            return
        
        with st.spinner("Scoring example..."):
            try:
                result = st.session_state.predictor.score_text(example['abstract'])
                render_scoring_results(result)
                
            except Exception as e:
                st.error(f"Scoring failed: {str(e)}")

def render_model_status():
    """Render model loading status"""
    if st.session_state.model_loaded:
        st.success("✅ Ensemble models loaded!")
        st.info(f"**Model Type:** {st.session_state.predictor.model_type}")
        st.info(f"**Loss Type:** {st.session_state.predictor.loss_type}")
        st.info(f"**Folds:** {st.session_state.predictor.num_folds}")
        st.info(f"**Device:** {st.session_state.predictor.device}")
        st.info(f"**Reference Threshold:** {st.session_state.predictor.threshold}")
        st.info(f"**Current Batch Size:** {st.session_state.batch_size}")
    else:
        st.warning("⚠️ No ensemble models loaded")
        st.info("Please configure and load ensemble models in the sidebar.")

def render_scoring_results(result):
    """Render ensemble scoring results with detailed analysis"""
    st.markdown("---")
    st.header("📊 Ensemble Scoring Results")
    
    ensemble_score = result['ensemble_score']
    ensemble_prediction = result['ensemble_prediction']
    stats = result['statistics']
    
    # Main score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Ensemble Score", 
            f"{ensemble_score:.4f}",
            help="Mean score across all 5 folds (0.0 = Low relevance, 1.0 = High relevance)"
        )
    
    with col2:
        st.metric(
            "📈 Score Range", 
            f"{stats['min_score']:.3f} - {stats['max_score']:.3f}",
            help="Minimum and maximum scores across folds"
        )
    
    with col3:
        st.metric(
            "📊 Score Stability", 
            f"σ = {stats['std_score']:.4f}",
            help="Standard deviation - lower values indicate more stable predictions"
        )
    
    # Score interpretation
    st.subheader("🔍 Score Interpretation")
    
    if ensemble_score >= 0.8:
        score_interpretation = "🟢 **High Relevance** - Strong biodiversity research content"
        score_color = "success"
    elif ensemble_score >= 0.6:
        score_interpretation = "🟡 **Medium-High Relevance** - Likely biodiversity-related"
        score_color = "warning"
    elif ensemble_score >= 0.4:
        score_interpretation = "🟠 **Medium Relevance** - Mixed or unclear biodiversity content"
        score_color = "warning"
    elif ensemble_score >= 0.2:
        score_interpretation = "🔴 **Low Relevance** - Unlikely to be biodiversity-focused"
        score_color = "error"
    else:
        score_interpretation = "⚫ **Very Low Relevance** - Not biodiversity research"
        score_color = "error"
    
    if score_color == "success":
        st.success(score_interpretation)
    elif score_color == "warning":
        st.warning(score_interpretation)
    else:
        st.error(score_interpretation)
    
    # Detailed statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Score", f"{stats['mean_score']:.4f}")
    with col2:
        st.metric("Median Score", f"{stats['median_score']:.4f}")
    with col3:
        st.metric("Std Deviation", f"{stats['std_score']:.4f}")
    with col4:
        consensus_strength = stats['consensus_strength']
        st.metric("Consensus", f"{consensus_strength:.1%}")
    
    # Reference binary prediction
    st.subheader("📋 Reference Classification")
    col1, col2 = st.columns(2)
    with col1:
        if ensemble_prediction == 1:
            st.success("✅ Above threshold → Biodiversity-related")
        else:
            st.info("❌ Below threshold → Not biodiversity-related")
    with col2:
        st.info(f"Reference threshold: {st.session_state.predictor.threshold}")
    
    # Individual fold scores
    with st.expander("🔍 Individual Fold Scores"):
        fold_df = pd.DataFrame(result['fold_results'])
        fold_df['score'] = fold_df['score'].round(4)
        fold_df = fold_df.sort_values('score', ascending=False)  # Sort by score
        
        st.dataframe(fold_df[['fold', 'score']], use_container_width=True)
        
        # Fold scores visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {row['fold']}" for _, row in fold_df.iterrows()],
            y=[row['score'] for _, row in fold_df.iterrows()],
            name="Fold Scores",
            text=[f"{row['score']:.3f}" for _, row in fold_df.iterrows()],
            textposition='auto',
            marker_color=['#1f77b4' for _ in range(len(fold_df))]  # Consistent color for scoring
        ))
        
        # Add threshold line for reference
        fig.add_hline(y=st.session_state.predictor.threshold, 
                     line_dash="dash", line_color="red",
                     annotation_text="Reference Threshold")
        
        # Add ensemble mean line
        fig.add_hline(y=ensemble_score, 
                     line_dash="dot", line_color="green",
                     annotation_text="Ensemble Mean")
        
        fig.update_layout(
            title="Individual Fold Scores (Ranked)",
            xaxis_title="Fold",
            yaxis_title="Score",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def process_batch_scoring(abstracts):
    """Process batch of abstracts using optimized ensemble scoring"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get batch size from session state (default to 16)
    batch_size = getattr(st.session_state, 'batch_size', 16)
    
    status_text.text(f"🚀 Processing {len(abstracts)} texts with optimized batch scoring (batch_size={batch_size})...")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Use optimized batch scoring - much faster!
        full_results = st.session_state.predictor.score_batch_optimized(abstracts, batch_size=batch_size)
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Simplify results for batch display
        results = []
        for result in full_results:
            simplified_result = {
                "abstract": result["abstract"],
                "ensemble_score": result["ensemble_score"],
                "std_score": result["statistics"]["std_score"],
                "min_score": result["statistics"]["min_score"],
                "max_score": result["statistics"]["max_score"],
                "consensus_strength": result["statistics"]["consensus_strength"],
                "positive_folds": result["statistics"]["positive_folds"]
            }
            results.append(simplified_result)
        
        progress_bar.progress(1.0)
        
        # Show performance metrics
        texts_per_second = len(abstracts) / processing_time
        status_text.text(f"✅ Batch scoring complete! Processed {len(abstracts)} texts in {processing_time:.2f}s ({texts_per_second:.1f} texts/sec)")
        
        # Performance info box
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"⚡ Processing Time: {processing_time:.2f}s")
        with col2:
            st.success(f"🚀 Speed: {texts_per_second:.1f} texts/sec")
        with col3:
            st.success(f"🎯 Batch Size: {batch_size}")
        
    except Exception as e:
        st.error(f"Batch scoring failed: {str(e)}")
        return []
    
    # Display batch results
    st.header("📊 Batch Scoring & Ranking Results")
    
    # Sort by ensemble score (highest first)
    results.sort(key=lambda x: x['ensemble_score'], reverse=True)
    
    # Summary stats
    all_scores = [r['ensemble_score'] for r in results]
    high_relevance = sum(1 for score in all_scores if score >= 0.6)
    avg_score = np.mean(all_scores)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scored", len(results))
    with col2:
        st.metric("High Relevance (≥0.6)", high_relevance)
    with col3:
        st.metric("Average Score", f"{avg_score:.3f}")
    with col4:
        highest_score = max(all_scores) if all_scores else 0
        st.metric("Highest Score", f"{highest_score:.3f}")
    
    # Scoring distribution
    st.subheader("📈 Score Distribution")
    fig = px.histogram(
        x=all_scores,
        nbins=20,
        title="Distribution of Ensemble Scores",
        labels={"x": "Ensemble Score", "y": "Count"}
    )
    fig.add_vline(x=avg_score, line_dash="dash", line_color="red", 
                  annotation_text="Average")
    st.plotly_chart(fig, use_container_width=True)
    
    # Ranking controls
    st.subheader("🏆 Ranked Results")
    
    col1, col2 = st.columns(2)
    with col1:
        sort_order = st.selectbox(
            "Sort Order",
            ["Highest to Lowest Score", "Lowest to Highest Score"],
            help="Choose ranking order"
        )
    with col2:
        score_filter = st.slider(
            "Minimum Score Filter",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Show only results above this score"
        )
    
    # Apply filters and sorting
    filtered_results = [r for r in results if r['ensemble_score'] >= score_filter]
    if sort_order == "Lowest to Highest Score":
        filtered_results.sort(key=lambda x: x['ensemble_score'])
    
    # Results table with ranking
    df_results = pd.DataFrame([
        {
            "Rank": i + 1,
            "Abstract": r['abstract'][:150] + "..." if len(r['abstract']) > 150 else r['abstract'],
            "Ensemble Score": f"{r['ensemble_score']:.4f}",
            "Score Range": f"{r['min_score']:.3f}-{r['max_score']:.3f}",
            "Stability (σ)": f"{r['std_score']:.3f}",
            "Consensus": f"{r['consensus_strength']:.1%}"
        }
        for i, r in enumerate(filtered_results)
    ])
    
    st.dataframe(df_results, use_container_width=True)
    
    # Download results
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download Scoring Results (JSON)",
        data=results_json,
        file_name="biomoqa_scoring_results.json",
        mime="application/json"
    )
    
    # Download as CSV with ranking
    df_download = pd.DataFrame([
        {
            "rank": i + 1,
            "abstract": r['abstract'],
            "ensemble_score": r['ensemble_score'],
            "std_score": r['std_score'],
            "min_score": r['min_score'],
            "max_score": r['max_score'],
            "consensus_strength": r['consensus_strength'],
            "positive_folds": r['positive_folds']
        }
        for i, r in enumerate(results)
    ])
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="📥 Download Ranked Results (CSV)",
        data=csv,
        file_name="biomoqa_ranked_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()