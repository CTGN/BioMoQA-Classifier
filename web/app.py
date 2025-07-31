import streamlit as st
import sys
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time

# Add src to path to import your modules
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/leandre/Projects/BioMoQA_Playground"))

# Add it to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.models.biomoqa.instantiation import load_predictor, BioMoQAPredictor

# Import custom components
from web.components.sidebar import render_sidebar
from web.components.results import render_results
from utils import get_example_texts, validate_model_path, format_confidence_score

# Page config
st.set_page_config(
    page_title="BioMoQA Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 BioMoQA Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Binary classifcation for Biodiversity on islands**")
    st.markdown("---")
    
    # Sidebar configuration
    model_config = render_sidebar()
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Input mode selection
        input_mode = st.radio(
            "Select input mode:",
            ["Single Text", "Batch Upload", "Example Texts"],
            horizontal=True
        )
        
        if input_mode == "Single Text":
            render_single_text_input(model_config)
        elif input_mode == "Batch Upload":
            render_batch_upload(model_config)
        else:
            render_example_texts(model_config)
    
    with col2:
        st.header("⚙️ Model Status")
        render_model_status()
        
        # Model performance info
        if st.session_state.model_loaded:
            st.header("📊 Quick Stats")
            render_model_info()

def render_single_text_input(model_config):
    """Render the single text input interface"""
    
    # Text inputs based on model configuration
    abstract = st.text_area(
        "Abstract*",
        height=200,
        placeholder="Enter the research abstract here...",
        help="The main text content to classify"
    )
    
    title = None
    keywords = None
    
    if model_config['with_title']:
        title = st.text_input(
            "Title*",
            placeholder="Enter the research title...",
            help="Required when model was trained with titles"
        )
    
    if model_config['with_keywords']:
        keywords = st.text_input(
            "Keywords*",
            placeholder="keyword1, keyword2, keyword3...",
            help="Comma-separated keywords. Required when model was trained with keywords"
        )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔍 Classify Text",
            type="primary",
            use_container_width=True
        )
    
    # Validation and prediction
    if predict_button:
        if not abstract.strip():
            st.error("Please enter an abstract.")
            return
        
        if model_config['with_title'] and not title:
            st.error("Title is required for this model.")
            return
            
        if model_config['with_keywords'] and not keywords:
            st.error("Keywords are required for this model.")
            return
        
        if not st.session_state.model_loaded:
            st.error("Please load a model first in the sidebar.")
            return
        
        # Make prediction
        with st.spinner("Classifying..."):
            try:
                result = st.session_state.predictor.evaluate_text(
                    abstract=abstract,
                    title=title if model_config['with_title'] else None,
                    keywords=keywords if model_config['with_keywords'] else None,
                    return_binary=True
                )
                
                render_results(result, model_config)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def render_batch_upload(model_config):
    """Render the batch upload interface"""
    st.info("Upload a JSON file with multiple texts for batch classification.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=['json'],
        help="Upload a JSON file containing an array of objects with 'abstract', 'title', and 'keywords' fields"
    )
    
    if uploaded_file:
        try:
            texts_data = json.load(uploaded_file)
            
            if not isinstance(texts_data, list):
                st.error("JSON file must contain an array of objects.")
                return
            
            st.success(f"Loaded {len(texts_data)} texts for classification.")
            
            # Show preview
            with st.expander("Preview uploaded data"):
                st.json(texts_data[:3])  # Show first 3 items
            
            # Batch processing
            if st.button("🚀 Process Batch", type="primary"):
                if not st.session_state.model_loaded:
                    st.error("Please load a model first.")
                    return
                
                process_batch(texts_data, model_config)
                
        except json.JSONDecodeError:
            st.error("Invalid JSON file format.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_example_texts(model_config):
    """Render the example texts interface"""
    st.info("Try the classifier with pre-loaded example texts.")
    
    examples = get_example_texts()
    
    selected_example = st.selectbox(
        "Choose an example:",
        range(len(examples)),
        format_func=lambda x: f"Example {x+1}: {examples[x]['title'][:50]}..."
    )
    
    example = examples[selected_example]
    
    # Display example
    st.markdown('<div class="example-card">', unsafe_allow_html=True)
    st.markdown(f"**Title:** {example['title']}")
    st.markdown(f"**Keywords:** {example['keywords']}")
    st.markdown(f"**Abstract:** {example['abstract'][:200]}...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Classify example
    if st.button("🔍 Classify This Example", type="primary"):
        if not st.session_state.model_loaded:
            st.error("Please load a model first.")
            return
        
        with st.spinner("Classifying example..."):
            try:
                result = st.session_state.predictor.evaluate_text(
                    abstract=example['abstract'],
                    title=example['title'] if model_config['with_title'] else None,
                    keywords=example['keywords'] if model_config['with_keywords'] else None,
                    return_binary=True
                )
                
                render_results(result, model_config)
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")

def render_model_status():
    """Render model loading status"""
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully!")
        st.info(f"**Model path:** {st.session_state.predictor.model_path}")
        st.info(f"**Device:** {st.session_state.predictor.device}")
        st.info(f"**Threshold:** {st.session_state.predictor.threshold}")
    else:
        st.warning("⚠️ No model loaded")
        st.info("Please configure and load a model in the sidebar.")

def render_model_info():
    """Render additional model information"""
    if st.session_state.model_loaded:
        # Create a simple performance placeholder
        st.metric("Threshold", f"{st.session_state.predictor.threshold:.2f}")
        
        # Model configuration
        config_info = {
            "With Title": st.session_state.predictor.with_title,
            "With Keywords": st.session_state.predictor.with_keywords,
        }
        
        for key, value in config_info.items():
            if value:
                st.success(f"✅ {key}")
            else:
                st.info(f"➖ {key}")

def process_batch(texts_data, model_config):
    """Process batch of texts"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text_data in enumerate(texts_data):
        status_text.text(f"Processing {i+1}/{len(texts_data)}...")
        
        try:
            result = st.session_state.predictor.evaluate_text(
                abstract=text_data.get('abstract', ''),
                title=text_data.get('title') if model_config['with_title'] else None,
                keywords=text_data.get('keywords') if model_config['with_keywords'] else None,
                return_binary=True
            )
            results.append(result)
            
        except Exception as e:
            st.warning(f"Failed to process item {i+1}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(texts_data))
    
    status_text.text("Processing complete!")
    
    # Display batch results
    st.header("📊 Batch Results")
    
    # Summary stats
    positive_count = sum(1 for r in results if r.get('prediction') == 1)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Processed", len(results))
    with col2:
        st.metric("Positive Classifications", positive_count)
    with col3:
        st.metric("Negative Classifications", len(results) - positive_count)
    
    # Results table
    import pandas as pd
    
    df_results = pd.DataFrame([
        {
            "Index": i,
            "Title": r.get('title', 'N/A')[:50] + "..." if r.get('title') else 'N/A',
            "Score": f"{r['score']:.3f}",
            "Prediction": "Positive" if r.get('prediction') == 1 else "Negative"
        }
        for i, r in enumerate(results)
    ])
    
    st.dataframe(df_results, use_container_width=True)
    
    # Download results
    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="📥 Download Results (JSON)",
        data=results_json,
        file_name="biomoqa_batch_results.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()