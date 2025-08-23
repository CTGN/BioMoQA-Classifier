
import sys
import os
import argparse
import logging
from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.biomoqa.instantiation import load_predictor, BioMoQAPredictor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_prediction(
    model_name: str,
    abstract: str,
    loss_type: str = "BCE",
    title: str = None,
    keywords: str = None,
    with_title: bool = False,
    with_keywords: bool = False,
    threshold: float = 0.5,
    weights_parent_dir: str = "results/final_model"
) -> Dict[str, Any]:
    """
    Run prediction on a single text.
    
    Args:
        model_name: Model name (e.g., BiomedBERT-abs)
        loss_type: Loss type used during training
        abstract: Abstract text
        title: Title text (optional)
        keywords: Keywords (optional)
        with_title: Whether model was trained with titles
        with_keywords: Whether model was trained with keywords
        threshold: Classification threshold
        weights_parent_dir: Directory containing model checkpoints
        
    Returns:
        Dictionary with prediction results
    """
    logger.info("Loading BioMoQA predictor...")
    predictor = load_predictor(
        model_name=model_name,
        loss_type=loss_type,
        with_title=with_title,
        with_keywords=with_keywords,
        threshold=threshold,
        weights_parent_dir=weights_parent_dir
    )
    
    logger.info("Making prediction...")
    result = predictor.evaluate_text(
        abstract=abstract,
        title=title,
        keywords=keywords,
        return_binary=True
    )
    
    return result


def run_batch_predictions(
    model_name: str,
    texts: List[Dict[str, str]],
    loss_type: str = "BCE",
    with_title: bool = False,
    with_keywords: bool = False,
    threshold: float = 0.5,
    weights_parent_dir: str = "results/final_model"
) -> List[Dict[str, Any]]:
    """
    Run predictions on a batch of texts.
    
    Args:
        model_name: Model name (e.g., BiomedBERT-abs)
        loss_type: Loss type used during training
        texts: List of dictionaries containing text data
        with_title: Whether model was trained with titles
        with_keywords: Whether model was trained with keywords
        threshold: Classification threshold
        weights_parent_dir: Directory containing model checkpoints
        
    Returns:
        List of prediction results
    """
    logger.info("Loading BioMoQA predictor...")
    predictor = load_predictor(
        model_name=model_name,
        loss_type=loss_type,
        with_title=with_title,
        with_keywords=with_keywords,
        threshold=threshold,
        weights_parent_dir=weights_parent_dir
    )
    
    results = []
    for i, text_data in enumerate(texts):
        logger.info(f"Processing text {i+1}/{len(texts)}")
        
        result = predictor.evaluate_text(
            abstract=text_data.get('abstract', ''),
            title=text_data.get('title'),
            keywords=text_data.get('keywords'),
            return_binary=True
        )
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with BioMoQA models")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., BiomedBERT-abs)"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="BCE",
        help="Loss type used during training (default: BCE)"
    )
    parser.add_argument(
        "--weights_parent_dir",
        type=str,
        default="results/final_model",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--abstract",
        type=str,
        help="Abstract text for single prediction"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title text (optional)"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        help="Keywords (optional)"
    )
    parser.add_argument(
        "--with_title",
        action="store_true",
        help="Whether the model was trained with titles"
    )
    parser.add_argument(
        "--with_keywords", 
        action="store_true",
        help="Whether the model was trained with keywords"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with example texts"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Demo with example texts
        logger.info("Running demo with example texts...")
        
        example_texts = [
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
        
        results = run_batch_predictions(
            model_name=args.model_name,
            texts=example_texts,
            loss_type=args.loss_type,
            with_title=args.with_title,
            with_keywords=args.with_keywords,
            threshold=args.threshold,
            weights_parent_dir=args.weights_parent_dir
        )
        
        logger.info("\n" + "="*50)
        logger.info("DEMO RESULTS")
        logger.info("="*50)
        
        for i, result in enumerate(results):
            logger.info(f"\nText {i+1}:")
            if 'title' in result:
                logger.info(f"Title: {result['title']}")
            logger.info(f"Abstract: {result['abstract'][:100]}...")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
            
    elif args.abstract:
        # Single prediction
        result = run_single_prediction(
            model_name=args.model_name,
            abstract=args.abstract,
            loss_type=args.loss_type,
            title=args.title,
            keywords=args.keywords,
            with_title=args.with_title,
            with_keywords=args.with_keywords,
            threshold=args.threshold,
            weights_parent_dir=args.weights_parent_dir
        )
        
        logger.info("\n" + "="*50)
        logger.info("PREDICTION RESULT")
        logger.info("="*50)
        logger.info(f"Score: {result['score']:.4f}")
        logger.info(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
        
    else:
        logger.error("Either provide --abstract for single prediction or use --demo for examples")
        parser.print_help()


if __name__ == "__main__":
    main()
