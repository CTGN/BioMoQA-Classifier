import os
from pathlib import Path

def validate_model_path(model_path):
    """Validate that the model path exists and contains model files"""
    if not os.path.exists(model_path):
        return False
    
    # Check if it's a valid Hugging Face model directory
    required_files = ['config.json']
    has_model_file = any(
        os.path.exists(os.path.join(model_path, f))
        for f in ['pytorch_model.bin', 'model.safetensors']
    )
    
    return (
        all(os.path.exists(os.path.join(model_path, f)) for f in required_files) and
        has_model_file
    )

def get_example_texts():
    """Get example texts for demonstration - focused on biodiversity research"""
    return [
        {
            "abstract": """
            This study investigates the effects of climate change on biodiversity patterns 
            in marine ecosystems. We analyzed species composition data from coral reefs 
            across multiple geographical locations over a 10-year period. Our findings 
            show significant shifts in species distribution correlating with temperature 
            increases and ocean acidification levels. The results indicate substantial 
            threats to marine biodiversity from anthropogenic climate change.
            """,
            "title": "Climate Change Impacts on Marine Biodiversity",
            "keywords": "climate change, biodiversity, marine ecosystems, coral reefs"
        },
        {
            "abstract": """
            We conducted a comprehensive analysis of endemic bird species on tropical islands
            to understand patterns of species richness and endemism. Using field surveys and
            molecular phylogenetic analysis, we documented the evolutionary relationships
            among island bird communities. Our research reveals that island biogeography
            theory accurately predicts patterns of species diversity in these isolated
            ecosystems, with implications for conservation strategies.
            """,
            "title": "Island Biogeography and Avian Endemism in Tropical Archipelagos",
            "keywords": "biogeography, island species, endemic birds, biodiversity conservation"
        },
        {
            "abstract": """
            A novel machine learning approach for protein structure prediction is presented.
            The method combines deep learning architectures with evolutionary information
            to achieve state-of-the-art accuracy. We tested our approach on multiple
            benchmark datasets and compared it with existing methods. The results show
            significant improvements in prediction accuracy across different protein families.
            """,
            "title": "Deep Learning for Protein Structure Prediction",
            "keywords": "machine learning, protein structure, deep learning, bioinformatics"
        },
        {
            "abstract": """
            Forest fragmentation poses significant threats to tropical biodiversity. We
            examined the effects of habitat fragmentation on mammalian communities in
            the Amazon rainforest using camera trap surveys across fragmented and
            continuous forest areas. Our 3-year study reveals that fragment size and
            connectivity are critical factors determining species persistence in
            fragmented landscapes.
            """,
            "title": "Habitat Fragmentation Effects on Amazonian Mammals",
            "keywords": "habitat fragmentation, Amazon, mammals, biodiversity conservation"
        }
    ]

def format_confidence_score(score):
    """Format confidence score for display"""
    return f"{score:.1%}"