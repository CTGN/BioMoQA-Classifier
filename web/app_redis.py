import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import numpy as np
from typing import Dict, Any, Optional, List
import requests

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from web.utils import get_example_texts

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="BioMoQA Scorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def main():
    st.markdown('<h1 class="main-header">🧬 BioMoQA Scoring & Ranking System</h1>', unsafe_allow_html=True)
    st.markdown("**Score and rank research abstracts using ensemble cross-validation models via FastAPI + Redis**")
    st.markdown("---")

    init_session_state()

    render_sidebar()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Text Input")

        input_mode = st.radio(
            "Select input mode:",
            ["Batch Scoring & Ranking", "Single Text Scoring", "Example Texts"],
            horizontal=True
        )

        if input_mode == "Single Text Scoring":
            render_single_text_input()
        elif input_mode == "Batch Scoring & Ranking":
            render_batch_upload()
        else:
            render_example_texts()

    with col2:
        st.header("⚙️ System Status")
        render_system_status()


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'model_type': 'roberta-base',
        'loss_type': 'BCE',
        'base_path': 'results/final_model',
        'threshold': 0.5,
        'device': None,
        'batch_size': 16,
        'use_ultra_optimization': True,
        'use_dynamic_batching': True,
        'current_task_id': None,
        'task_type': None,
        'last_result': None,
        'batch_processing': False,
        'task_start_time': None,
        'task_end_time': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar for model configuration"""
    st.sidebar.header("🔧 Model Configuration")

    model_types = [
        "bert-base",
        "biobert-v1",
        "BiomedBERT-abs",
        "BiomedBERT-abs-ft",
        "roberta-base"
    ]

    loss_types = ["BCE", "focal"]

    st.session_state.model_type = st.sidebar.selectbox(
        "Model Type",
        model_types,
        index=model_types.index(st.session_state.model_type),
        help="Select the base model architecture"
    )

    st.session_state.loss_type = st.sidebar.selectbox(
        "Loss Type",
        loss_types,
        index=loss_types.index(st.session_state.loss_type),
        help="Select the loss function used during training"
    )

    st.session_state.base_path = st.sidebar.text_input(
        "Models Base Path",
        value=st.session_state.base_path,
        help="Base directory containing the fold model checkpoints"
    )

    st.session_state.threshold = st.sidebar.slider(
        "Reference Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.01,
        help="Reference threshold for binary classification"
    )

    device_option = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        help="Select computation device"
    )
    st.session_state.device = None if device_option == "auto" else device_option

    st.sidebar.subheader("🚀 GPU Optimization")
    st.session_state.batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=64,
        value=st.session_state.batch_size,
        step=1,
        help="Number of texts to process together"
    )

    with st.sidebar.expander("⚡ Advanced Performance Settings"):
        st.session_state.use_ultra_optimization = st.checkbox(
            "Enable Ultra-Optimization",
            value=st.session_state.use_ultra_optimization,
            help="Use dynamic batching + optimized processing for ~3-5x speedup"
        )

        st.session_state.use_dynamic_batching = st.checkbox(
            "Dynamic Length-Based Batching",
            value=st.session_state.use_dynamic_batching,
            help="Group texts by similar length for better GPU utilization"
        )

    st.sidebar.subheader("📋 Current Configuration")
    st.sidebar.info(f"**Model:** {st.session_state.model_type}")
    st.sidebar.info(f"**Loss:** {st.session_state.loss_type}")
    st.sidebar.info(f"**Batch Size:** {st.session_state.batch_size}")

    st.sidebar.subheader("🔌 API Connection")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success(f"✅ Connected to API")
            health_data = response.json()
            if health_data.get("redis") == "connected":
                st.sidebar.success("✅ Redis connected")
            else:
                st.sidebar.error(f"❌ Redis: {health_data.get('redis')}")
        else:
            st.sidebar.error(f"❌ API error: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"❌ API connection error: {str(e)}")

    st.sidebar.subheader("📚 API Documentation")
    with st.sidebar.expander("View API Docs", expanded=False):
        st.markdown(f"""
        ### Interactive API Documentation

        FastAPI provides auto-generated interactive documentation:

        - **Swagger UI:** [{API_BASE_URL}/docs]({API_BASE_URL}/docs)
        - **ReDoc:** [{API_BASE_URL}/redoc]({API_BASE_URL}/redoc)

        ### Available Endpoints

        #### **POST** `/score/single`
        Score a single research abstract.

        **Request Body:**
        ```json
        {{
          "abstract": "text to score",
          "title": "optional title",
          "model_type": "roberta-base",
          "loss_type": "BCE",
          "threshold": 0.5
        }}
        ```

        **Returns:** Task ID for tracking

        ---

        #### **POST** `/score/batch/upload`
        Upload and score a batch file (CSV/JSON).

        **Form Data:**
        - `file`: CSV or JSON file
        - `model_type`, `loss_type`, etc.

        **Returns:** Task ID for tracking

        ---

        #### **GET** `/tasks/{{task_id}}`
        Check task status and get results.

        **Returns:**
        ```json
        {{
          "task_id": "...",
          "status": "SUCCESS",
          "result": {{...}},
          "progress": {{...}}
        }}
        ```

        ---

        #### **DELETE** `/tasks/{{task_id}}/cancel`
        Cancel a running task.

        ---

        #### **GET** `/health`
        Check API and Redis health status.

        #### **GET** `/models/available`
        Get list of available models and loss types.
        """)

        st.markdown("---")
        st.markdown("**API Base URL:**")
        st.code(API_BASE_URL, language="text")


def render_single_text_input():
    """Render the single text input interface"""
    title = st.text_input(
        "Title",
        placeholder="Enter the research title here (optional)...",
        help="The title text to use alongside the abstract"
    )

    abstract = st.text_area(
        "Abstract*",
        height=200,
        placeholder="Enter the research abstract here...",
        help="The abstract text to score using cross-validation ensemble"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score_button = st.button(
            "📊 Score Text",
            type="primary",
            use_container_width=True
        )

    if score_button:
        if not abstract.strip():
            st.error("Please enter an abstract.")
            return

        with st.spinner("Submitting task to API..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/score/single",
                    json={
                        "abstract": abstract,
                        "title": title if title.strip() else None,
                        "model_type": st.session_state.model_type,
                        "loss_type": st.session_state.loss_type,
                        "base_path": st.session_state.base_path,
                        "threshold": st.session_state.threshold,
                        "device": st.session_state.device,
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_task_id = result["task_id"]
                    st.session_state.task_type = "single"
                    st.session_state.task_start_time = time.time()
                    st.session_state.task_end_time = None
                    st.success(f"✅ Task submitted! Task ID: {result['task_id'][:8]}...")
                else:
                    st.error(f"Failed to submit task: {response.text}")
                    return

            except Exception as e:
                st.error(f"Failed to submit task: {str(e)}")
                return

    if st.session_state.current_task_id and st.session_state.task_type == "single":
        check_and_display_single_result()


def render_batch_upload():
    """Render the batch upload interface"""
    st.info("🚀 Upload a JSON or CSV file with multiple texts for GPU-accelerated batch scoring via API.")

    if st.session_state.use_ultra_optimization:
        st.success(f"⚡ **ULTRA-OPTIMIZATION ENABLED**: Dynamic batching for maximum speed!")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['json', 'csv'],
        help="Upload a JSON file (array of strings or objects with 'abstract'/'text' field) or CSV file with 'abstract' column"
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)

        estimated_upload_size_mb = file_size_mb * 1.33
        max_size_mb = 500

        if estimated_upload_size_mb > max_size_mb:
            st.error(f"📄 **File**: {uploaded_file.name} ({file_size_mb:.2f} MB) - **TOO LARGE**")
            st.error(f"❌ File too large! Estimated upload size: {estimated_upload_size_mb:.0f}MB (limit: {max_size_mb}MB)")
            st.warning("⚠️ HTTP multipart encoding adds ~33% overhead to file size.")
            st.info("💡 **Solutions:**\n"
                   "- Split your file into smaller batches (recommended: under 300MB raw file size)\n"
                   "- Filter your data to fewer records\n"
                   "- Process in multiple smaller uploads")
            return
        elif estimated_upload_size_mb > max_size_mb * 0.8:
            st.warning(f"📄 **File**: {uploaded_file.name} ({file_size_mb:.2f} MB) - **Close to limit**")
            st.warning(f"⚠️ File is close to upload limit. Estimated: {estimated_upload_size_mb:.0f}MB / {max_size_mb}MB")
        else:
            st.info(f"📄 **File**: {uploaded_file.name} ({file_size_mb:.2f} MB)")

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.session_state.batch_processing:
                process_button = st.button("⏳ Processing...", type="primary", disabled=True)
            else:
                process_button = st.button("🚀 Upload & Score Batch", type="primary")

        with col2:
            if st.session_state.batch_processing:
                cancel_button = st.button("❌ Cancel", type="secondary")
            else:
                cancel_button = False

        if cancel_button and st.session_state.current_task_id:
            try:
                response = requests.delete(
                    f"{API_BASE_URL}/tasks/{st.session_state.current_task_id}/cancel",
                    timeout=5
                )
                st.session_state.batch_processing = False
                st.session_state.current_task_id = None
                st.warning("🛑 Task cancelled.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to cancel: {str(e)}")

        if process_button and not st.session_state.batch_processing:
            with st.spinner("Uploading file and submitting task..."):
                try:
                    files = {
                        'file': (uploaded_file.name, file_bytes, uploaded_file.type)
                    }
                    data = {
                        'model_type': st.session_state.model_type,
                        'loss_type': st.session_state.loss_type,
                        'base_path': st.session_state.base_path,
                        'threshold': st.session_state.threshold,
                        'device': st.session_state.device or '',
                        'batch_size': st.session_state.batch_size,
                        'use_ultra_optimization': st.session_state.use_ultra_optimization,
                        'use_dynamic_batching': st.session_state.use_dynamic_batching,
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/score/batch/upload",
                        files=files,
                        data=data,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.current_task_id = result["task_id"]
                        st.session_state.task_type = "batch"
                        st.session_state.batch_processing = True
                        st.session_state.task_start_time = time.time()
                        st.session_state.task_end_time = None
                        st.success(f"✅ File uploaded ({file_size_mb:.2f}MB)! Task submitted: {result['task_id'][:8]}...")
                        st.rerun()
                    else:
                        st.error(f"Failed to submit batch task: {response.text}")
                        return

                except Exception as e:
                    error_msg = str(e)
                    if "413" in error_msg or "too large" in error_msg.lower():
                        st.error(f"❌ Upload failed: File too large")
                        st.info(f"💡 Your file ({file_size_mb:.2f}MB) exceeds the server limit. Split it into smaller batches.")
                    else:
                        st.error(f"Failed to submit batch task: {error_msg}")
                    return

        if st.session_state.batch_processing and st.session_state.current_task_id:
            check_and_display_batch_result()


def render_example_texts():
    """Render the example texts interface"""
    st.info("Try the ensemble scorer with pre-loaded example texts via API.")

    examples = get_example_texts()

    selected_example = st.selectbox(
        "Choose an example:",
        range(len(examples)),
        format_func=lambda x: f"Example {x+1}: {examples[x]['title'][:50]}..."
    )

    example = examples[selected_example]

    st.markdown("**Title:** " + example['title'])
    st.markdown("**Keywords:** " + example['keywords'])
    st.markdown("**Abstract:** " + example['abstract'][:200] + "...")

    if st.button("📊 Score This Example", type="primary"):
        with st.spinner("Submitting task..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/score/single",
                    json={
                        "abstract": example['abstract'],
                        "title": example.get('title'),
                        "model_type": st.session_state.model_type,
                        "loss_type": st.session_state.loss_type,
                        "base_path": st.session_state.base_path,
                        "threshold": st.session_state.threshold,
                        "device": st.session_state.device,
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_task_id = result["task_id"]
                    st.session_state.task_type = "single"
                    st.session_state.task_start_time = time.time()
                    st.session_state.task_end_time = None
                    st.success(f"✅ Task submitted! Task ID: {result['task_id'][:8]}...")
                else:
                    st.error(f"Failed to submit task: {response.text}")
                    return

            except Exception as e:
                st.error(f"Failed to submit task: {str(e)}")
                return

    if st.session_state.current_task_id and st.session_state.task_type == "single":
        check_and_display_single_result()


def render_system_status():
    """Render system status information"""
    st.subheader("🔌 API Connection")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Connected to API")
            health_data = response.json()
            if health_data.get("redis") == "connected":
                st.success("✅ Redis connected")
        else:
            st.error(f"❌ API error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")

    st.subheader("📋 Current Configuration")
    st.info(f"**Model:** {st.session_state.model_type}")
    st.info(f"**Loss:** {st.session_state.loss_type}")
    st.info(f"**Device:** {st.session_state.device or 'auto'}")
    st.info(f"**Batch Size:** {st.session_state.batch_size}")

    if st.session_state.current_task_id:
        st.subheader("🔄 Current Task")
        st.info(f"**Task ID:** {st.session_state.current_task_id[:8]}...")
        st.info(f"**Type:** {st.session_state.task_type}")

        try:
            response = requests.get(
                f"{API_BASE_URL}/tasks/{st.session_state.current_task_id}",
                timeout=2
            )
            if response.status_code == 200:
                status_data = response.json()
                st.info(f"**Status:** {status_data['status']}")
        except:
            pass
    else:
        st.success("🟢 **Ready for Processing**")


def check_and_display_single_result():
    """Check task status and display single text result when ready"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/tasks/{st.session_state.current_task_id}",
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to get task status: {response.text}")
            st.session_state.current_task_id = None
            st.session_state.task_type = None
            return

        status_data = response.json()
        status = status_data["status"]

        status_placeholder = st.empty()

        if status == 'PENDING':
            status_placeholder.info("⏳ Task is queued...")
            time.sleep(1)
            st.rerun()
        elif status == 'STARTED':
            status_placeholder.info("🔄 Task is being processed...")
            time.sleep(1)
            st.rerun()
        elif status == 'PROGRESS':
            progress_info = status_data.get('progress', {})
            status_msg = progress_info.get('message', 'Processing...')
            status_placeholder.info(f"🔄 {status_msg}")
            time.sleep(1)
            st.rerun()
        elif status == 'SUCCESS':
            status_placeholder.success("✅ Task completed!")

            if st.session_state.task_end_time is None:
                st.session_state.task_end_time = time.time()

            result = status_data.get('result', {})

            if result.get('status') == 'success':
                elapsed_time = None
                if st.session_state.task_start_time and st.session_state.task_end_time:
                    elapsed_time = st.session_state.task_end_time - st.session_state.task_start_time

                render_scoring_results(result['result'], elapsed_time)
            else:
                st.error(f"Scoring failed: {result.get('error', 'Unknown error')}")

            if st.button("🔄 Score Another Text"):
                st.session_state.current_task_id = None
                st.session_state.task_type = None
                st.session_state.task_start_time = None
                st.session_state.task_end_time = None
                st.rerun()

        elif status == 'FAILURE':
            status_placeholder.error(f"❌ Task failed: {status_data.get('error', 'Unknown error')}")
            st.session_state.current_task_id = None
            st.session_state.task_type = None

    except Exception as e:
        st.error(f"Error checking task status: {str(e)}")
        st.session_state.current_task_id = None
        st.session_state.task_type = None


def check_and_display_batch_result():
    """Check batch task status and display results when ready"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/tasks/{st.session_state.current_task_id}",
            timeout=5
        )

        if response.status_code != 200:
            st.error(f"Failed to get task status: {response.text}")
            st.session_state.batch_processing = False
            st.session_state.current_task_id = None
            st.session_state.task_type = None
            return

        status_data = response.json()
        status = status_data["status"]

        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        if status == 'PENDING':
            status_placeholder.info("⏳ Batch task is queued...")
            progress_placeholder.progress(0.0)
            time.sleep(2)
            st.rerun()
        elif status in ['STARTED', 'PROGRESS']:
            progress_info = status_data.get('progress', {})

            if progress_info and 'current' in progress_info and 'total' in progress_info:
                current = progress_info.get('current', 0)
                total = progress_info.get('total', 1)
                status_msg = progress_info.get('status', 'Processing...')

                progress = min(1.0, max(0.0, current / max(total, 1)))

                status_placeholder.info(f"🔄 {status_msg}")
                progress_placeholder.progress(progress, text=f"{current}/{total} records processed ({progress*100:.1f}%)")
            else:
                status_placeholder.info("🔄 Loading and processing file...")
                progress_placeholder.progress(0.0)

            time.sleep(1)
            st.rerun()
        elif status == 'SUCCESS':
            progress_placeholder.empty()
            status_placeholder.success("✅ Batch processing completed!")

            if st.session_state.task_end_time is None:
                st.session_state.task_end_time = time.time()

            result = status_data.get('result', {})

            if result and result.get('status') == 'success':
                elapsed_time = None
                if st.session_state.task_start_time and st.session_state.task_end_time:
                    elapsed_time = st.session_state.task_end_time - st.session_state.task_start_time

                merged_results = result.get('results', [])

                st.info(f"📊 Processed {result.get('scored_records', 0)} valid records out of {result.get('total_records', 0)} total")

                display_batch_results(merged_results, elapsed_time)
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                st.error(f"Batch scoring failed: {error_msg}")

            st.session_state.batch_processing = False

            if st.button("🔄 Process Another Batch"):
                st.session_state.current_task_id = None
                st.session_state.task_type = None
                st.session_state.task_start_time = None
                st.session_state.task_end_time = None
                st.rerun()

        elif status == 'FAILURE':
            progress_placeholder.empty()
            error_msg = status_data.get('error', 'Unknown error')
            status_placeholder.error(f"❌ Batch task failed: {error_msg}")
            st.session_state.batch_processing = False
            st.session_state.current_task_id = None
            st.session_state.task_type = None

    except Exception as e:
        st.error(f"Error checking task status: {str(e)}")
        st.session_state.batch_processing = False
        st.session_state.current_task_id = None
        st.session_state.task_type = None


def render_scoring_results(result: Dict[str, Any], elapsed_time: Optional[float] = None):
    """Render single text scoring results"""
    st.markdown("---")
    st.header("📊 Ensemble Scoring Results")

    if elapsed_time is not None:
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f} seconds"
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        st.success(f"⏱️ **Processing Time:** {time_str}")

    ensemble_score = result['ensemble_score']
    stats = result['statistics']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎯 Ensemble Score",
            f"{ensemble_score:.4f}",
            help="Mean score across all 5 folds"
        )

    with col2:
        st.metric(
            "📈 Score Range",
            f"{stats['min_score']:.3f} - {stats['max_score']:.3f}",
            help="Min and max scores across folds"
        )

    with col3:
        st.metric(
            "📊 Stability",
            f"σ = {stats['std_score']:.4f}",
            help="Standard deviation"
        )

    st.subheader("🔍 Score Interpretation")

    if ensemble_score >= 0.8:
        st.success("🟢 **High Relevance** - Strong biodiversity research content")
    elif ensemble_score >= 0.6:
        st.warning("🟡 **Medium-High Relevance** - Likely biodiversity-related")
    elif ensemble_score >= 0.4:
        st.warning("🟠 **Medium Relevance** - Mixed or unclear biodiversity content")
    elif ensemble_score >= 0.2:
        st.error("🔴 **Low Relevance** - Unlikely to be biodiversity-focused")
    else:
        st.error("⚫ **Very Low Relevance** - Not biodiversity research")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Positive Folds", stats['positive_folds'])
    with col2:
        st.metric("Consensus", f"{stats['consensus_strength']:.1%}")
    with col3:
        st.metric("Mean Score", f"{stats['mean_score']:.4f}")


def display_batch_results(results: List[Dict], elapsed_time: Optional[float] = None):
    """Display batch scoring results"""
    st.header("📊 Batch Scoring & Ranking Results")

    if elapsed_time is not None:
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f} seconds"
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"

        num_records = len([r for r in results if r['ensemble_score'] is not None])
        if num_records > 0 and elapsed_time > 0:
            throughput = num_records / elapsed_time
            st.success(f"⏱️ **Processing Time:** {time_str} | **Throughput:** {throughput:.2f} records/sec")
        else:
            st.success(f"⏱️ **Processing Time:** {time_str}")

    results.sort(key=lambda x: x['ensemble_score'] if x['ensemble_score'] is not None else -1, reverse=True)

    valid_scores = [r['ensemble_score'] for r in results if r['ensemble_score'] is not None]
    if valid_scores:
        high_relevance = sum(1 for s in valid_scores if s >= 0.6)
        avg_score = np.mean(valid_scores)
        highest_score = max(valid_scores)
    else:
        high_relevance = 0
        avg_score = 0
        highest_score = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scored", len([r for r in results if r['ensemble_score'] is not None]))
    with col2:
        st.metric("High Relevance (≥0.6)", high_relevance)
    with col3:
        st.metric("Average Score", f"{avg_score:.3f}")
    with col4:
        st.metric("Highest Score", f"{highest_score:.3f}")

    if valid_scores:
        st.subheader("📈 Score Distribution")
        fig = px.histogram(
            x=valid_scores,
            nbins=20,
            title="Distribution of Ensemble Scores",
            labels={"x": "Ensemble Score", "y": "Count"}
        )
        fig.add_vline(x=avg_score, line_dash="dash", line_color="red", annotation_text="Average")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Results with Rankings")

    valid_results = [r for r in results if r['ensemble_score'] is not None]
    for rank, r in enumerate(valid_results, 1):
        r['score_rank'] = rank

    df = pd.DataFrame(results)

    rank_cols = ['original_index', 'score_rank']
    score_cols = ['ensemble_score', 'std_score', 'min_score', 'max_score', 'consensus_strength']
    other_cols = [c for c in df.columns if c not in rank_cols + score_cols]

    df = df[rank_cols + other_cols + score_cols]
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📥 Download JSON",
            data=results_json,
            file_name="biomoqa_results.json",
            mime="application/json"
        )
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name="biomoqa_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
