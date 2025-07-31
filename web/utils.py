import os
from pathlib import Path

def validate_model_path(model_path):
    """Validate that the model path exists"""
    return os.path.exists(model_path)

def get_example_texts():
    """Get example texts for demonstration"""
    return [
        {
            "abstract": """
            This study investigates the effects of climate change on biodiversity patterns 
            in marine ecosystems. We analyzed species composition data from coral reefs 
            across multiple geographical locations over a 10-year period. Our findings 
            show significant shifts in species distribution correlating with temperature 
            increases and ocean acidification levels.
            """,
            "title": "Climate Change Impacts on Marine Biodiversity",
            "keywords": "climate change, biodiversity, marine ecosystems, coral reefs"
        },
        {
            "abstract": """
            A novel machine learning approach for protein structure prediction is presented.
            The method combines deep learning architectures with evolutionary information
            to achieve state-of-the-art accuracy. We tested our approach on multiple
            benchmark datasets and compared it with existing methods.
            """,
            "title": "Deep Learning for Protein Structure Prediction",
            "keywords": "machine learning, protein structure, deep learning, bioinformatics"
        },
        {
            "abstract": """
            We describe a new surgical technique for minimally invasive cardiac procedures.
            The technique reduces patient recovery time and shows improved outcomes
            compared to traditional open-heart surgery. A randomized controlled trial
            with 200 patients demonstrates the safety and efficacy of this approach.
            """,
            "title": "Minimally Invasive Cardiac Surgery Technique",
            "keywords": "cardiac surgery, minimally invasive, medical technique"
        }
    ]

def format_confidence_score(score):
    """Format confidence score for display"""
    return f"{score:.1%}"