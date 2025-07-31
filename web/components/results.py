import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def render_results(result, model_config):
    """Render prediction results with rich visualizations"""
    
    st.markdown("---")
    st.header("🎯 Classification Results")
    
    score = result['score']
    prediction = result.get('prediction', 0)
    
    # Main result card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction result
        if prediction == 1:
            st.success("🎯 **POSITIVE** - Biomedical Research Question")
        else:
            st.info("❌ **NEGATIVE** - Not a Biomedical Research Question")
    
    with col2:
        # Confidence score
        confidence_class = get_confidence_class(score)
        st.markdown(f'<p class="{confidence_class}">Confidence: {score:.1%}</p>', 
                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence gauge
    create_confidence_gauge(score)
    
    # Detailed breakdown
    with st.expander("📊 Detailed Analysis"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Raw Score", f"{score:.4f}")
            st.metric("Threshold Used", f"{st.session_state.predictor.threshold:.2f}")
            
        with col2:
            st.metric("Binary Prediction", prediction)
            margin = abs(score - st.session_state.predictor.threshold)
            st.metric("Decision Margin", f"{margin:.4f}")
        
        # Input summary
        st.subheader("Input Summary")
        st.write(f"**Abstract length:** {len(result['abstract'])} characters")
        
        if 'title' in result:
            st.write(f"**Title:** {result['title']}")
        
        if 'keywords' in result:
            st.write(f"**Keywords:** {result['keywords']}")
    
    # Score interpretation
    render_score_interpretation(score)

def create_confidence_gauge(score):
    """Create a confidence gauge visualization"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        delta = {'reference': 0.5},
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

def render_score_interpretation(score):
    """Render score interpretation guide"""
    
    st.subheader("🔍 Score Interpretation")
    
    if score >= 0.8:
        st.success("**High Confidence**: The model is very confident this is a biomedical research question.")
    elif score >= 0.6:
        st.info("**Medium-High Confidence**: The model thinks this is likely a biomedical research question.")
    elif score >= 0.4:
        st.warning("**Medium-Low Confidence**: The model is uncertain about the classification.")
    else:
        st.error("**Low Confidence**: The model thinks this is likely NOT a biomedical research question.")

def get_confidence_class(score):
    """Get CSS class for confidence score"""
    if score >= 0.7:
        return "confidence-high"
    elif score >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"