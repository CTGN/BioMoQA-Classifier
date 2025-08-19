#!/usr/bin/env python3
"""
Inference script for BioMoQA models using the instantiation module.

This script demonstrates how to use trained BioMoQA models to make predictions
on new texts using the instantiation.py module.
"""

import sys
import os
import argparse
import logging
from typing import List, Dict, Any

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/leandre/Projects/BioMoQA_Playground/src/.."))

# Add it to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.models.biomoqa.instantiation import load_predictor, BioMoQAPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_prediction(
    model_path: str,
    abstract: str,
    title: str = None,
    keywords: str = None,
    with_title: bool = False,
    with_keywords: bool = False,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run prediction on a single text.
    
    Args:
        model_path: Path to the trained model
        abstract: Abstract text
        title: Title text (optional)
        keywords: Keywords (optional)
        with_title: Whether model was trained with titles
        with_keywords: Whether model was trained with keywords
        threshold: Classification threshold
        
    Returns:
        Dictionary with prediction results
    """
    logger.info("Loading BioMoQA predictor...")
    predictor = load_predictor(
        model_path=model_path,
        with_title=with_title,
        with_keywords=with_keywords,
        threshold=threshold
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
    model_path: str,
    texts: List[Dict[str, str]],
    with_title: bool = False,
    with_keywords: bool = False,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Run predictions on a batch of texts.
    
    Args:
        model_path: Path to the trained model
        texts: List of dictionaries containing text data
        with_title: Whether model was trained with titles
        with_keywords: Whether model was trained with keywords
        threshold: Classification threshold
        
    Returns:
        List of prediction results
    """
    logger.info("Loading BioMoQA predictor...")
    predictor = load_predictor(
        model_path=model_path,
        with_title=with_title,
        with_keywords=with_keywords,
        threshold=threshold
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
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
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
            model_path=args.model_path,
            texts=example_texts,
            with_title=args.with_title,
            with_keywords=args.with_keywords,
            threshold=args.threshold
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
            model_path=args.model_path,
            abstract=args.abstract,
            title=args.title,
            keywords=args.keywords,
            with_title=args.with_title,
            with_keywords=args.with_keywords,
            threshold=args.threshold
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