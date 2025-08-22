import os
from pathlib import Path

def validate_model_path(model_path):
    """Validate that the model path exists and contains model files"""
    if not os.path.exists(model_path):
        return False
    
    
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
    """Get example texts for demonstration - focused on biodiversity research with varying relevance scores"""
    return [
        {
            "abstract": """
            This study investigates the effects of climate change on biodiversity patterns 
            in marine ecosystems. We analyzed species composition data from coral reefs 
            across multiple geographical locations over a 10-year period. Our findings 
            show significant shifts in species distribution correlating with temperature 
            increases and ocean acidification levels. The results indicate substantial 
            threats to marine biodiversity from anthropogenic climate change, with 
            implications for conservation strategies and ecosystem management.
            """,
            "title": "Climate Change Impacts on Marine Biodiversity",
            "keywords": "climate change, biodiversity, marine ecosystems, coral reefs",
            "expected_score": "High (>0.8) - Strong biodiversity focus"
        },
        {
            "abstract": """
            We conducted a comprehensive analysis of endemic bird species on tropical islands
            to understand patterns of species richness and endemism. Using field surveys and
            molecular phylogenetic analysis, we documented the evolutionary relationships
            among island bird communities. Our research reveals that island biogeography
            theory accurately predicts patterns of species diversity in these isolated
            ecosystems, with implications for conservation strategies and understanding
            evolutionary processes in fragmented habitats.
            """,
            "title": "Island Biogeography and Avian Endemism in Tropical Archipelagos",
            "keywords": "biogeography, island species, endemic birds, biodiversity conservation",
            "expected_score": "High (>0.8) - Core biodiversity research"
        },
        {
            "abstract": """
            A novel machine learning approach for protein structure prediction is presented.
            The method combines deep learning architectures with evolutionary information
            to achieve state-of-the-art accuracy. We tested our approach on multiple
            benchmark datasets and compared it with existing methods. The results show
            significant improvements in prediction accuracy across different protein families,
            with potential applications in drug discovery and structural biology.
            """,
            "title": "Deep Learning for Protein Structure Prediction",
            "keywords": "machine learning, protein structure, deep learning, bioinformatics",
            "expected_score": "Low-Medium (0.2-0.4) - Computational, not biodiversity"
        },
        {
            "abstract": """
            Forest fragmentation poses significant threats to tropical biodiversity. We
            examined the effects of habitat fragmentation on mammalian communities in
            the Amazon rainforest using camera trap surveys across fragmented and
            continuous forest areas. Our 3-year study reveals that fragment size and
            connectivity are critical factors determining species persistence in
            fragmented landscapes, with direct implications for conservation planning.
            """,
            "title": "Habitat Fragmentation Effects on Amazonian Mammals",
            "keywords": "habitat fragmentation, Amazon, mammals, biodiversity conservation",
            "expected_score": "High (>0.8) - Direct biodiversity conservation"
        },
        {
            "abstract": """
            The economic impact of tourism on small island states has been extensively studied.
            This research examines the relationship between tourism revenue and local economic 
            development indicators across Caribbean nations over the past decade. We found 
            significant correlations between tourism growth and GDP increases, though with 
            notable variations across different island economies and seasonal patterns.
            """,
            "title": "Tourism Economics in Caribbean Island States",
            "keywords": "tourism economics, Caribbean, economic development, island states",
            "expected_score": "Very Low (<0.2) - Economic focus, no biodiversity"
        }
    ]

def format_confidence_score(score):
    """Format confidence score for display"""
    return f"{score:.1%}"

def get_score_interpretation(score: float) -> dict:
    """Get interpretation for a given score"""
    if score >= 0.8:
        return {
            "level": "High Relevance",
            "icon": "🟢",
            "description": "Strong biodiversity research content",
            "color": "success"
        }
    elif score >= 0.6:
        return {
            "level": "Medium-High Relevance", 
            "icon": "🟡",
            "description": "Likely biodiversity-related research",
            "color": "warning"
        }
    elif score >= 0.4:
        return {
            "level": "Medium Relevance",
            "icon": "🟠", 
            "description": "Mixed or unclear biodiversity content",
            "color": "warning"
        }
    elif score >= 0.2:
        return {
            "level": "Low Relevance",
            "icon": "🔴",
            "description": "Unlikely to be biodiversity-focused",
            "color": "error"
        }
    else:
        return {
            "level": "Very Low Relevance",
            "icon": "⚫",
            "description": "Not biodiversity research",
            "color": "error"
        }

def rank_by_score(results: list, ascending: bool = False) -> list:
    """Rank results by ensemble score"""
    return sorted(results, key=lambda x: x['ensemble_score'], reverse=not ascending)

def filter_by_score(results: list, min_score: float = 0.0, max_score: float = 1.0) -> list:
    """Filter results by score range"""
    return [
        r for r in results 
        if min_score <= r['ensemble_score'] <= max_score
    ]

def get_score_statistics(scores: list) -> dict:
    """Calculate comprehensive statistics for a list of scores"""
    if not scores:
        return {}
    
    import numpy as np
    
    return {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "q25": np.percentile(scores, 25),
        "q75": np.percentile(scores, 75),
        "count": len(scores),
        "high_relevance": sum(1 for s in scores if s >= 0.6),
        "medium_relevance": sum(1 for s in scores if 0.4 <= s < 0.6),
        "low_relevance": sum(1 for s in scores if s < 0.4)
    }