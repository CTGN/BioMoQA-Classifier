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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src to path to import your modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/leandre/Projects/BioMoQA_Playground"))
sys.path.insert(0, str(project_root))

from web.utils import get_example_texts, validate_model_path, format_confidence_score

# Page config
st.set_page_config(
    page_title="BioMoQA Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleBioMoQAPredictor:
    """Simplified BioMoQA predictor using Hugging Face transformers"""
    
    def __init__(self, model_path: str, threshold: float = 0.5, device: str = None):
        self.model_path = model_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, abstract: str) -> dict:
        """Predict on a single abstract text"""
        # Tokenize input
        inputs = self.tokenizer(
            abstract,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).squeeze().cpu().item()
        
        # Return result
        prediction = int(score > self.threshold)
        return {
            "abstract": abstract,
            "score": score,
            "prediction": prediction
        }
    
    def predict_batch(self, abstracts: list) -> list:
        """Predict on a batch of abstracts"""
        results = []
        for abstract in abstracts:
            result = self.predict(abstract)
            results.append(result)
        return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 BioMoQA Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Binary classification for Biodiversity research questions**")
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
    st.sidebar.header("🔧 Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="",
        placeholder="path/to/your/trained/model",
        help="Path to your trained BioMoQA model checkpoint (Hugging Face format)"
    )
    
    # Threshold
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
    if st.sidebar.button("🚀 Load Model", type="primary", use_container_width=True):
        if not model_path:
            st.sidebar.error("Please provide a model path.")
        else:
            load_model(model_path, threshold, device)
    
    # Example paths
    st.sidebar.subheader("💡 Example Paths")
    st.sidebar.code("results/biomoqa/best_model")
    st.sidebar.code("path/to/your/checkpoint")

def load_model(model_path: str, threshold: float, device: str):
    """Load the model with given configuration"""
    with st.sidebar.spinner("Loading model..."):
        try:
            # Validate path
            if not os.path.exists(model_path):
                st.sidebar.error("Model path does not exist. Please check the path.")
                return
            
            # Load predictor
            predictor = SimpleBioMoQAPredictor(
                model_path=model_path,
                threshold=threshold,
                device=device
            )
            
            # Store in session state
            st.session_state.predictor = predictor
            st.session_state.model_loaded = True
            
            st.sidebar.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {str(e)}")
            st.session_state.model_loaded = False

def render_single_text_input():
    """Render the single text input interface"""
    # Abstract input
    abstract = st.text_area(
        "Abstract*",
        height=200,
        placeholder="Enter the research abstract here...",
        help="The abstract text to classify"
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
        
        if not st.session_state.model_loaded:
            st.error("Please load a model first in the sidebar.")
            return
        
        # Make prediction
        with st.spinner("Classifying..."):
            try:
                result = st.session_state.predictor.predict(abstract)
                render_results(result)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def render_batch_upload():
    """Render the batch upload interface"""
    st.info("Upload a JSON or CSV file with multiple texts for batch classification.")
    
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
            
            st.success(f"Loaded {len(abstracts)} texts for classification.")
            
            # Show preview
            with st.expander("Preview uploaded data"):
                if uploaded_file.name.endswith('.json'):
                    st.json(texts_data[:3])
                else:
                    st.dataframe(df.head(3))
            
            # Batch processing
            if st.button("🚀 Process Batch", type="primary"):
                if not st.session_state.model_loaded:
                    st.error("Please load a model first.")
                    return
                
                process_batch(abstracts)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def render_example_texts():
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
    st.markdown("**Title:** " + example['title'])
    st.markdown("**Keywords:** " + example['keywords'])
    st.markdown("**Abstract:** " + example['abstract'][:200] + "...")
    
    # Classify example
    if st.button("🔍 Classify This Example", type="primary"):
        if not st.session_state.model_loaded:
            st.error("Please load a model first.")
            return
        
        with st.spinner("Classifying example..."):
            try:
                result = st.session_state.predictor.predict(example['abstract'])
                render_results(result)
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")

def render_model_status():
    """Render model loading status"""
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully!")
        st.info(f"**Device:** {st.session_state.predictor.device}")
        st.info(f"**Threshold:** {st.session_state.predictor.threshold}")
    else:
        st.warning("⚠️ No model loaded")
        st.info("Please configure and load a model in the sidebar.")

def render_results(result):
    """Render prediction results"""
    st.markdown("---")
    st.header("🎯 Classification Results")
    
    score = result['score']
    prediction = result['prediction']
    
    # Main result
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success("🎯 **POSITIVE** - Biomedical Research Question")
        else:
            st.info("❌ **NEGATIVE** - Not a Biomedical Research Question")
    
    with col2:
        st.metric("Confidence Score", f"{score:.1%}")
    
    # Confidence gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': st.session_state.predictor.threshold
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed info
    with st.expander("📊 Detailed Analysis"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raw Score", f"{score:.4f}")
            st.metric("Threshold", f"{st.session_state.predictor.threshold:.2f}")
        with col2:
            st.metric("Binary Prediction", prediction)
            margin = abs(score - st.session_state.predictor.threshold)
            st.metric("Decision Margin", f"{margin:.4f}")

def process_batch(abstracts):
    """Process batch of abstracts"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, abstract in enumerate(abstracts):
        status_text.text(f"Processing {i+1}/{len(abstracts)}...")
        
        try:
            result = st.session_state.predictor.predict(abstract)
            results.append(result)
        except Exception as e:
            st.warning(f"Failed to process item {i+1}: {str(e)}")
            
        progress_bar.progress((i + 1) / len(abstracts))
    
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
    df_results = pd.DataFrame([
        {
            "Index": i,
            "Abstract": r['abstract'][:100] + "..." if len(r['abstract']) > 100 else r['abstract'],
            "Score": f"{r['score']:.3f}",
            "Prediction": "Positive" if r['prediction'] == 1 else "Negative"
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
    
    # Download as CSV
    df_download = pd.DataFrame(results)
    csv = df_download.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="biomoqa_batch_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()