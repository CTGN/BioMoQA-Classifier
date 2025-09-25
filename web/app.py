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
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from web.utils import get_example_texts, format_confidence_score
from src.models.biomoqa.folds_ensemble_predictor import CrossValidationPredictor, validate_model_path

# Page config
st.set_page_config(
    page_title="BioMoQA Scorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


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
    if 'auto_load_attempted' not in st.session_state:
        st.session_state.auto_load_attempted = False
    if 'batch_processing' not in st.session_state:
        st.session_state.batch_processing = False
    if 'cancel_processing' not in st.session_state:
        st.session_state.cancel_processing = False
    if 'processing_id' not in st.session_state:
        st.session_state.processing_id = None
    if 'current_batch_data' not in st.session_state:
        st.session_state.current_batch_data = None
    
    # Auto-load roberta-base model on first visit
    if not st.session_state.auto_load_attempted and not st.session_state.model_loaded:
        st.session_state.auto_load_attempted = True
        load_ensemble_models("roberta-base", "BCE", "results/final_model", 0.5, None, False)
    
    # Sidebar for model configuration
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Input mode selection
        input_mode = st.radio(
            "Select input mode:",
            ["Batch Scoring & Ranking","Single Text Scoring", "Example Texts"],
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
        value="results/final_model",
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
    
    # Advanced Performance Settings
    with st.sidebar.expander("⚡ Advanced Performance Settings"):
        use_ultra_optimization = st.checkbox(
            "Enable Ultra-Optimization",
            value=True,
            help="Use dynamic batching + optimized processing for ~3-5x speedup"
        )
        
        use_dynamic_batching = st.checkbox(
            "Dynamic Length-Based Batching",
            value=True,
            help="Group texts by similar length for better GPU utilization"
        )
        
        max_workers = st.slider(
            "Processing Mode",
            min_value=1,
            max_value=2,
            value=1,
            help="1 = Sequential (stable), 2 = Experimental parallel"
        )
        
        enable_compilation = st.checkbox(
            "Enable Model Compilation (Experimental)",
            value=False,
            help="⚠️ May cause errors with threading. Only enable if you experience no issues."
        )
    
    # Store advanced settings in session state
    st.session_state.use_ultra_optimization = use_ultra_optimization
    st.session_state.use_dynamic_batching = use_dynamic_batching
    st.session_state.max_workers = max_workers
    st.session_state.enable_compilation = enable_compilation
    
    # Store batch size in session state
    st.session_state.batch_size = batch_size
    
    # GPU memory info and optimization status
    if device_option in ["auto", "cuda"] and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.sidebar.info(f"🔥 GPU Memory: {gpu_memory:.1f} GB")
        if gpu_memory >= 15:  # A100 or similar
            st.sidebar.success("🚀 High-end GPU detected! Consider batch_size=32-64")
        elif gpu_memory >= 8:
            st.sidebar.info("💡 Mid-range GPU: batch_size=16-32 recommended")
        else:
            st.sidebar.warning("⚠️ Low GPU memory: use batch_size=8 or less")
        
        # Show optimization features
        st.sidebar.subheader("⚡ GPU Optimizations")
        st.sidebar.info("🎯 **FP16 Mixed Precision**: 2x memory savings + speed boost")
        if enable_compilation:
            st.sidebar.info("🚀 **Model Compilation**: PyTorch 2.0+ optimization (EXPERIMENTAL)")
        else:
            st.sidebar.info("🚀 **Model Compilation**: Disabled for stability")
        st.sidebar.info("📊 **Dynamic Batching**: Length-based grouping for efficiency")
        st.sidebar.info("⚡ **Sequential Processing**: Stable fold processing")
        
        if use_ultra_optimization:
            st.sidebar.success("🚀 **ULTRA-OPTIMIZATION ENABLED** (~3-5x faster!)")
        else:
            st.sidebar.info("💡 Enable Ultra-Optimization for maximum speed")
    
    # Show model info
    st.sidebar.subheader("📋 Model Configuration")
    st.sidebar.info(f"**Model:** {model_type}")
    st.sidebar.info(f"**Loss:** {loss_type}")
    st.sidebar.info(f"**Folds:** 5 models")
    st.sidebar.info(f"**Batch Size:** {batch_size}")
    
    # Load model button
    if st.sidebar.button("🚀 Load Ensemble Models", type="primary", use_container_width=True):
        load_ensemble_models(model_type, loss_type, base_path, threshold, device, enable_compilation)
    
    # Example paths
    st.sidebar.subheader("💡 Example Model Configurations")
    st.sidebar.code("Model: BiomedBERT-abs\nLoss: BCE")
    st.sidebar.code("Model: bert-base\nLoss: focal")

def load_ensemble_models(model_type: str, loss_type: str, base_path: str, threshold: float, device: str, enable_compilation: bool = False):
    """Load the ensemble of fold models with GPU optimizations"""
    with st.spinner("Loading ensemble models with GPU optimizations..."):
        try:
            # Validate base path
            if not os.path.exists(base_path):
                st.sidebar.error("Base path does not exist. Please check the path.")
                return
            
            # Load predictor with GPU optimizations enabled
            predictor = CrossValidationPredictor(
                model_type=model_type,
                loss_type=loss_type,
                base_path=base_path,
                threshold=threshold,
                device=device,
                use_fp16=True,  # Enable FP16 for GPU acceleration
                use_compile=enable_compilation  # User-configurable compilation
            )
            
            # Store in session state
            st.session_state.predictor = predictor
            st.session_state.model_loaded = True
            
            # Show optimization status
            optimization_info = []
            if predictor.use_fp16:
                optimization_info.append("FP16 Mixed Precision")
            if predictor.use_compile:
                optimization_info.append("Model Compilation")
            
            success_msg = f"✅ Loaded {predictor.num_folds} fold models successfully!"
            if optimization_info:
                success_msg += f"\n🚀 Optimizations: {', '.join(optimization_info)}"
            
            st.sidebar.success(success_msg)
            
        except Exception as e:
            st.sidebar.error(f"Failed to load ensemble models: {str(e)}")
            st.session_state.model_loaded = False

def render_single_text_input():
    """Render the single text input interface"""
    # Title input
    title = st.text_input(
        "Title",
        placeholder="Enter the research title here (optional)...",
        help="The title text to use alongside the abstract"
    )
    
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
                result = st.session_state.predictor.score_text(abstract, title)
                render_scoring_results(result)
                
            except Exception as e:
                st.error(f"Scoring failed: {str(e)}")

def render_batch_upload():
    """Render the batch upload interface"""
    st.info("🚀 Upload a JSON or CSV file with multiple texts for GPU-accelerated batch scoring and ranking.")
    
    # Show current optimization status
    use_ultra = getattr(st.session_state, 'use_ultra_optimization', True)
    enable_comp = getattr(st.session_state, 'enable_compilation', False)
    if use_ultra:
        comp_text = " + compilation" if enable_comp else ""
        st.success(f"⚡ **ULTRA-OPTIMIZATION ENABLED**: Dynamic batching + FP16{comp_text} = ~2-4x faster!")
    else:
        st.success("⚡ **GPU Optimizations**: FP16 mixed precision + dynamic batching for maximum speed!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['json', 'csv'],
        help="Upload a JSON file (array of strings or objects with 'abstract'/'text' field) or CSV file with 'abstract' column"
    )
    
    if uploaded_file:
        try:
            # Parse file based on type
            if uploaded_file.name.endswith('.json'):
                json_data = json.load(uploaded_file)
                if not isinstance(json_data, list):
                    st.error("JSON file must contain an array.")
                    return
                
                # Handle different JSON formats
                texts_data = []
                abstracts = []
                
                for i, item in enumerate(json_data):
                    if isinstance(item, str):
                        # Array of strings - treat each string as abstract
                        abstracts.append(item)
                        texts_data.append({"abstract": item, "index": i + 1})
                    elif isinstance(item, dict):
                        # Array of objects - extract abstract field
                        abstract = item.get('abstract', item.get('text', ''))
                        if not abstract:
                            # If no 'abstract' or 'text' field, try to find the first string field
                            string_fields = [v for v in item.values() if isinstance(v, str)]
                            abstract = string_fields[0] if string_fields else ''
                        abstracts.append(abstract)
                        texts_data.append(item)
                    else:
                        st.error(f"Unsupported item type in JSON array: {type(item)}")
                        return
            elif uploaded_file.name.endswith('.csv'):
                # Read CSV and handle potential index column
                df = pd.read_csv(uploaded_file)
                
                # Remove unnamed index columns that pandas sometimes creates
                unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
                if unnamed_cols:
                    df = df.drop(columns=unnamed_cols)
                
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
            
            # Batch processing buttons
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Main processing button
                if st.session_state.batch_processing:
                    process_button = st.button("⏳ Processing...", type="primary", disabled=True)
                else:
                    process_button = st.button("🚀 Score & Rank Batch", type="primary")
            
            with col2:
                # Cancel button - always show if processing
                if st.session_state.batch_processing:
                    cancel_button = st.button("❌ Cancel", type="secondary")
                else:
                    cancel_button = False
            
            # Handle cancel button
            if cancel_button:
                st.session_state.cancel_processing = True
                st.session_state.batch_processing = False
                st.session_state.processing_id = None
                st.session_state.current_batch_data = None
                st.warning("🛑 Processing cancelled by user.")
                st.rerun()
            
            # Handle process button
            if process_button and not st.session_state.batch_processing:
                if not st.session_state.model_loaded:
                    st.error("Please load ensemble models first.")
                    return
                
                # Start new processing - cancel any existing one
                processing_id = str(uuid.uuid4())
                
                # Reset all processing state
                st.session_state.cancel_processing = False
                st.session_state.batch_processing = True
                st.session_state.processing_id = processing_id
                st.session_state.current_batch_data = texts_data
                st.session_state.batch_progress = 0
                st.session_state.batch_results = []
                st.session_state.batch_total = 0
                
                # Immediately rerun to show processing state
                st.rerun()
            
            # Continue processing if we're in the middle of it
            if (st.session_state.batch_processing and 
                st.session_state.current_batch_data is not None and
                not st.session_state.cancel_processing):
                
                try:
                    process_batch_scoring_chunked(st.session_state.current_batch_data, st.session_state.processing_id)
                except Exception as e:
                    st.error(f"Batch processing failed: {str(e)}")
                    st.session_state.batch_processing = False
                    st.session_state.processing_id = None
                    st.session_state.current_batch_data = None
                
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
                result = st.session_state.predictor.score_text(example['abstract'], example.get('title'))
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
        
        # Show GPU optimizations status
        if hasattr(st.session_state.predictor, 'use_fp16') or hasattr(st.session_state.predictor, 'use_compile'):
            st.subheader("⚡ GPU Optimizations")
            if getattr(st.session_state.predictor, 'use_fp16', False):
                st.success("🎯 **FP16 Mixed Precision**: Enabled")
            else:
                st.info("🎯 **FP16 Mixed Precision**: Disabled")
                
            if getattr(st.session_state.predictor, 'use_compile', False):
                st.success("🚀 **Model Compilation**: Enabled")
            else:
                st.info("🚀 **Model Compilation**: Disabled")
                
            st.success("📊 **Dynamic Batching**: Enabled")
        
        # Show batch processing status
        if st.session_state.batch_processing:
            st.warning("⏳ **Batch Processing Active**")
            st.info("Use the Cancel button to stop current processing.")
        else:
            st.success("🟢 **Ready for Processing**")
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

def process_batch_scoring_chunked(texts_data, processing_id):
    """Process batch of abstracts with chunked processing and proper cancellation support"""
    
    # Check if this is still the current processing job
    if st.session_state.processing_id != processing_id:
        return  # This job has been superseded
    
    # Check for cancellation
    if st.session_state.cancel_processing:
        st.session_state.batch_processing = False
        st.session_state.processing_id = None
        st.session_state.current_batch_data = None
        return
    
    # Initialize processing state if not exists
    if 'batch_progress' not in st.session_state:
        st.session_state.batch_progress = 0
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'batch_total' not in st.session_state:
        st.session_state.batch_total = 0
    
    # If this is the start of processing, prepare the data
    if st.session_state.batch_progress == 0:
        # Extract abstracts and titles for scoring
        abstracts = []
        titles = []
        for item in texts_data:
            if isinstance(item, str):
                abstracts.append(item if item is not None else None)
                titles.append(None)
            elif isinstance(item, dict):
                abstract = item.get('abstract', item.get('text', None))
                title = item.get('title', None)
                abstracts.append(abstract)
                titles.append(title)
            else:
                abstracts.append(None)
                titles.append(None)
        
        # Store prepared data in session state
        st.session_state.batch_abstracts = abstracts
        st.session_state.batch_titles = titles
        st.session_state.batch_texts_data = texts_data
        st.session_state.batch_total = len(abstracts)
        st.session_state.batch_results = []
        st.session_state.batch_start_time = time.time()
    
    # Show progress
    progress_bar = st.progress(st.session_state.batch_progress / max(st.session_state.batch_total, 1))
    status_text = st.empty()
    
    batch_size = getattr(st.session_state, 'batch_size', 16)
    
    # Process one batch at a time
    if st.session_state.batch_progress < st.session_state.batch_total:
        batch_start = st.session_state.batch_progress
        batch_end = min(batch_start + batch_size, st.session_state.batch_total)
        
        status_text.text(f"🚀 Processing batch {batch_start//batch_size + 1}/{(st.session_state.batch_total-1)//batch_size + 1} (items {batch_start+1}-{batch_end})")
        
        # Prepare batch data
        batch_abstracts = st.session_state.batch_abstracts[batch_start:batch_end]
        batch_titles = st.session_state.batch_titles[batch_start:batch_end]
        
        # Prepare data for predictor
        predictor_data = []
        valid_indices = []
        for i, (abstract, title) in enumerate(zip(batch_abstracts, batch_titles)):
            if abstract is not None and str(abstract).strip():
                predictor_data.append({"abstract": str(abstract), "title": title, "index": batch_start + i})
                valid_indices.append(i)
        
        # Process this batch using ultra-optimization if enabled
        if predictor_data:
            try:
                # Get optimization settings
                use_ultra = getattr(st.session_state, 'use_ultra_optimization', True)
                use_dynamic = getattr(st.session_state, 'use_dynamic_batching', True)
                max_workers = getattr(st.session_state, 'max_workers', 5)
                
                if use_ultra:
                    # Use ultra-optimized method with parallel processing
                    batch_results = st.session_state.predictor.score_batch_ultra_optimized(
                        predictor_data, 
                        base_batch_size=len(predictor_data),
                        max_workers=max_workers,
                        use_dynamic_batching=use_dynamic
                    )
                else:
                    # Use standard optimized method
                    batch_results = st.session_state.predictor.score_batch_optimized(predictor_data, batch_size=len(predictor_data))
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")
                st.session_state.batch_processing = False
                st.session_state.processing_id = None
                st.session_state.current_batch_data = None
                return
        else:
            batch_results = []
        
        # Process results for this batch
        batch_final_results = []
        valid_result_idx = 0
        
        for i, (abstract, title) in enumerate(zip(batch_abstracts, batch_titles)):
            original_data = st.session_state.batch_texts_data[batch_start + i]
            
            if abstract is not None and str(abstract).strip() and valid_result_idx < len(batch_results):
                prediction_result = batch_results[valid_result_idx]
                valid_result_idx += 1
            else:
                # Create None result for invalid abstracts
                prediction_result = {
                    "ensemble_score": None,
                    "statistics": {
                        "std_score": None,
                        "min_score": None,
                        "max_score": None,
                        "consensus_strength": None,
                        "positive_folds": None
                    }
                }
            
            # Combine original data with prediction results
            combined_result = {
                **original_data,
                "original_index": batch_start + i + 1,
                "ensemble_score": prediction_result["ensemble_score"],
                "std_score": prediction_result["statistics"]["std_score"],
                "min_score": prediction_result["statistics"]["min_score"],
                "max_score": prediction_result["statistics"]["max_score"],
                "consensus_strength": prediction_result["statistics"]["consensus_strength"],
                "positive_folds": prediction_result["statistics"]["positive_folds"]
            }
            batch_final_results.append(combined_result)
        
        # Add batch results to overall results
        st.session_state.batch_results.extend(batch_final_results)
        st.session_state.batch_progress = batch_end
        
        # Update progress
        progress_bar.progress(st.session_state.batch_progress / st.session_state.batch_total)
        
        # Continue processing if not done
        if st.session_state.batch_progress < st.session_state.batch_total:
            # Schedule next batch
            time.sleep(0.1)  # Small delay to allow UI updates
            st.rerun()
        else:
            # Processing complete
            end_time = time.time()
            processing_time = end_time - st.session_state.batch_start_time
            texts_per_second = st.session_state.batch_total / processing_time
            
            status_text.text(f"✅ Batch scoring complete! Processed {st.session_state.batch_total} texts in {processing_time:.2f}s ({texts_per_second:.1f} texts/sec)")
            
            # Show results
            display_batch_results(st.session_state.batch_results, processing_time, texts_per_second, batch_size)
            
            # Reset processing state
            st.session_state.batch_processing = False
            st.session_state.processing_id = None
            st.session_state.current_batch_data = None
            # Keep results for display
    else:
        # Processing already complete, just show results
        if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results:
            processing_time = time.time() - st.session_state.batch_start_time
            texts_per_second = st.session_state.batch_total / processing_time
            display_batch_results(st.session_state.batch_results, processing_time, texts_per_second, batch_size)


def display_batch_results(results, processing_time, texts_per_second, batch_size):
    """Display batch scoring results with performance metrics and download options"""
    
    # Performance info box
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"⚡ Processing Time: {processing_time:.2f}s")
    with col2:
        st.success(f"🚀 Speed: {texts_per_second:.1f} texts/sec")
    with col3:
        st.success(f"🎯 Batch Size: {batch_size}")
    
    # Display batch results
    st.header("📊 Batch Scoring & Ranking Results")
    
    # Sort by ensemble score (highest first)
    results.sort(key=lambda x: x['ensemble_score'] if x['ensemble_score'] is not None else 0, reverse=True)
    
    # Summary stats (filter out None scores)
    valid_scores = [r['ensemble_score'] for r in results if r['ensemble_score'] is not None]
    if valid_scores:
        high_relevance = sum(1 for score in valid_scores if score >= 0.6)
        avg_score = np.mean(valid_scores)
        highest_score = max(valid_scores)
    else:
        high_relevance = 0
        avg_score = 0
        highest_score = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scored", len(results))
    with col2:
        st.metric("High Relevance (≥0.6)", high_relevance)
    with col3:
        st.metric("Average Score", f"{avg_score:.3f}")
    with col4:
        st.metric("Highest Score", f"{highest_score:.3f}")
    
    # Scoring distribution
    if valid_scores:
        st.subheader("📈 Score Distribution")
        fig = px.histogram(
            x=valid_scores,
            nbins=20,
            title="Distribution of Ensemble Scores",
            labels={"x": "Ensemble Score", "y": "Count"}
        )
        fig.add_vline(x=avg_score, line_dash="dash", line_color="red", 
                      annotation_text="Average")
        st.plotly_chart(fig, use_container_width=True)
    
    # Ranking controls
    st.subheader("🏆 Results with Rankings")
    st.info("📋 **Original_Index**: Position in uploaded dataset | **Score_Rank**: Rank by ensemble score (1 = highest)")
    
    col1, col2 = st.columns(2)
    with col1:
        sort_order = st.selectbox(
            "Sort Order",
            ["Original Order", "Highest to Lowest Score", "Lowest to Highest Score"],
            help="Choose display order - Original preserves input order"
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
    
    # Calculate score ranks (1 = highest score)
    valid_results = [r for r in results if r['ensemble_score'] is not None]
    sorted_by_score = sorted(valid_results, key=lambda x: x['ensemble_score'], reverse=True)
    for rank, result in enumerate(sorted_by_score, 1):
        result['score_rank'] = rank
    
    # Set score_rank to None for invalid results
    for result in results:
        if result['ensemble_score'] is None:
            result['score_rank'] = None
    
    # Apply filters and sorting for display
    filtered_results = [r for r in results if r['ensemble_score'] is None or r['ensemble_score'] >= score_filter]
    
    if sort_order == "Highest to Lowest Score":
        filtered_results.sort(key=lambda x: x['ensemble_score'] if x['ensemble_score'] is not None else -1, reverse=True)
    elif sort_order == "Lowest to Highest Score":
        filtered_results.sort(key=lambda x: x['ensemble_score'] if x['ensemble_score'] is not None else float('inf'))
    # If "Original Order", keep the original order (no sorting needed)
    
    # Create display dataframe with original data + simple ranking columns
    display_data = []
    for r in filtered_results:
        # Create row with original data + simple ranking columns
        row = {
            "Original_Index": r['original_index'],  # Original position in dataset
            "Score_Rank": r['score_rank'],  # Rank by ensemble score
            **r,  # All original columns + prediction scores
            "Ensemble_Score_Formatted": f"{r['ensemble_score']:.4f}" if r['ensemble_score'] is not None else "N/A",
            "Score_Range": f"{r['min_score']:.3f}-{r['max_score']:.3f}" if r['min_score'] is not None else "N/A",
            "Stability": f"{r['std_score']:.3f}" if r['std_score'] is not None else "N/A",
            "Consensus": f"{r['consensus_strength']:.1%}" if r['consensus_strength'] is not None else "N/A"
        }
        display_data.append(row)
    
    # Create DataFrame and reorder columns to show ranking info first
    df_results = pd.DataFrame(display_data)
    
    # Reorder columns: ranking columns first, then original data, then detailed scores
    ranking_cols = ["Original_Index", "Score_Rank"]
    score_cols = ["Ensemble_Score_Formatted", "Score_Range", "Stability", "Consensus"]
    
    # Get original columns (excluding the prediction scores and ranking fields we added)
    prediction_score_cols = ["ensemble_score", "std_score", "min_score", "max_score", 
                           "consensus_strength", "positive_folds", "original_index", "score_rank"]
    original_cols = [col for col in df_results.columns 
                    if col not in ranking_cols + score_cols + prediction_score_cols]
    
    # Final column order: ranking info, original data, formatted scores
    column_order = ranking_cols + original_cols + score_cols
    df_results = df_results[column_order]
    
    st.dataframe(df_results, use_container_width=True)
    
    # Download results
    results_json = json.dumps(results, indent=2, default=str)  # Handle None values
    st.download_button(
        label="📥 Download Scoring Results (JSON)",
        data=results_json,
        file_name="biomoqa_scoring_results.json",
        mime="application/json"
    )
    
    # Download as CSV with original data and rankings
    # Create download dataframe with same structure as display
    df_download = pd.DataFrame([
        {
            "original_index": r['original_index'],
            "score_rank": r['score_rank'],
            **{k: v for k, v in r.items() if k not in ['original_index', 'score_rank']},  # All other columns
        }
        for r in results
    ])
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="📥 Download Results with Rankings (CSV)",
        data=csv,
        file_name="biomoqa_results_with_rankings.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()