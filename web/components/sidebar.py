import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path to import your modules
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/leandre/Projects/BioMoQA_Playground"))

# Add it to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.models.biomoqa.instantiation import load_predictor
from web.utils import validate_model_path

def render_sidebar():
    """Render the sidebar for model configuration"""
    
    st.sidebar.header("🔧 Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        placeholder="/path/to/your/trained/model",
        help="Path to your trained BioMoQA model checkpoint"
    )
    
    # Model configuration
    st.sidebar.subheader("Model Settings")
    
    with_title = st.sidebar.checkbox(
        "Model trained with titles",
        help="Check if your model was trained using paper titles"
    )
    
    with_keywords = st.sidebar.checkbox(
        "Model trained with keywords", 
        help="Check if your model was trained using keywords"
    )
    
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Threshold for binary classification"
    )
    
    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        help="Select computation device"
    )
    
    device = None if device_option == "auto" else device_option
    
    # Load model button
    load_button = st.sidebar.button(
        "🚀 Load Model",
        type="primary",
        use_container_width=True
    )
    
    if load_button:
        if not model_path:
            st.sidebar.error("Please provide a model path.")
        else:
            load_model(model_path, with_title, with_keywords, threshold, device)
    
    # Example model paths (if you want to provide some)
    st.sidebar.subheader("💡 Example Paths")
    st.sidebar.code("results/biomoqa/final_model/best_model_fold-1")
    st.sidebar.code("path/to/your/checkpoint")
    
    return {
        'model_path': model_path,
        'with_title': with_title,
        'with_keywords': with_keywords,
        'threshold': threshold,
        'device': device
    }

def load_model(model_path, with_title, with_keywords, threshold, device):
    """Load the model with given configuration"""
    
    with st.spinner("Loading model..."):
        try:
            # Validate path
            if not validate_model_path(model_path):
                st.sidebar.error("Invalid model path. Please check the path exists.")
                return
            
            # Load predictor
            predictor = load_predictor(
                model_path=model_path,
                with_title=with_title,
                with_keywords=with_keywords,
                device=device,
                threshold=threshold
            )
            
            # Store in session state
            st.session_state.predictor = predictor
            st.session_state.model_loaded = True
            
            st.sidebar.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {str(e)}")
            st.session_state.model_loaded = False