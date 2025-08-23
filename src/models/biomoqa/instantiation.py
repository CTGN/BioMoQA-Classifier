import os
import torch
import numpy as np
import logging
from typing import Union, List, Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from src.config import CONFIG
from src.utils import map_name

logger = logging.getLogger(__name__)

#Instantiation code :
#Here I need to take a model's checkpoint and use it to evaluate a given text
#I should then return the given text with the predicted score
#For the code to be modular I shall give as argument the path to a model's weights
#Possible upgrade : infere the model given pids or dois () using the fetch api or something else)

class BioMoQAPredictor:
    
    def __init__(
        self,
        model_name: str,
        loss_type: str = "BCE",
        with_title: bool = False,
        with_keywords: bool = False,
        device: Optional[str] = None,
        weights_parent_dir: str = "results/final_model",
        threshold: float = 0.5
    ):
        self.model_name=model_name
        self.with_title = with_title
        self.loss_type=loss_type
        self.with_keywords = with_keywords
        self.threshold = threshold
        self.weights_parent_dir = weights_parent_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(self.weights_parent_dir):
            raise FileNotFoundError(f"Checkpoints parent directory does not exist : {self.weights_parent_dir}")
        
        self.model_paths  = [
            os.path.join(self.weights_parent_dir, dirname)
            for dirname in os.listdir(self.weights_parent_dir)
            if dirname.startswith( "best_model_cross_val_"+str(self.loss_type)+"_" +str(map_name(self.model_name))) and os.path.isdir(os.path.join(self.weights_parent_dir, dirname))
        ]
        
        self._load_model()
        
    def _load_model(self):
        if not self.model_paths:
            raise FileNotFoundError(f"No model checkpoints found in directory: {self.weights_parent_dir}")
            
        try:
            try:
                self.tokenizer_per_fold = [AutoTokenizer.from_pretrained(model_path) for model_path in self.model_paths]
            except:
                logger.warning("Tokenizer not found in model path, using default BERT tokenizer")
                self.tokenizer_per_fold = [AutoTokenizer.from_pretrained("google-bert/bert-base-uncased") for _ in range(len(self.model_paths)) ]
            
            self.models_per_fold = [AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=CONFIG["num_labels"]
            ) for model_path in self.model_paths]

            # Move models to device and set to eval mode
            for model in self.models_per_fold:
                model.to(self.device)
                model.eval()
            
            self.data_collators = [DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True
            ) for tokenizer in self.tokenizer_per_fold]
            
            logger.info(f"Successfully loaded {len(self.model_paths)} models from: {self.weights_parent_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _tokenize_text(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        if self.with_title and self.with_keywords:
            if title is None or keywords is None:
                raise ValueError("Model requires both title and keywords, but one or both are missing")
            
            sep_tok = self.tokenizer_per_fold[0].sep_token or "[SEP]"
            combined = title + sep_tok + keywords
            
            tokens = self.tokenizer_per_fold[0](
                combined,
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        elif self.with_title:
            if title is None:
                raise ValueError("Model requires title, but it's missing")
                
            tokens = self.tokenizer_per_fold[0](
                title,
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        elif self.with_keywords:
            if keywords is None:
                raise ValueError("Model requires keywords, but they're missing")
                
            tokens = self.tokenizer_per_fold[0](
                abstract,
                keywords,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        else:
            tokens = self.tokenizer_per_fold[0](
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
        
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        return tokens
    
    def predict_score(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None
    ) -> float:
        """Predict score using ensemble of all fold models"""
        fold_scores = []
        
        for i, (model, tokenizer) in enumerate(zip(self.models_per_fold, self.tokenizer_per_fold)):
            # Tokenize with specific fold tokenizer
            tokens = self._tokenize_text_with_tokenizer(abstract, title, keywords, tokenizer)
            
            with torch.no_grad():
                outputs = model(**tokens)
                logits = outputs.logits.squeeze()
                score = torch.sigmoid(logits).cpu().item()
                fold_scores.append(score)
        
        # Return ensemble average
        return np.mean(fold_scores)
    
    def _tokenize_text_with_tokenizer(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None,
        tokenizer: AutoTokenizer = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with specific tokenizer"""
        if tokenizer is None:
            tokenizer = self.tokenizer_per_fold[0]
            
        if self.with_title and self.with_keywords:
            if title is None or keywords is None:
                raise ValueError("Model requires both title and keywords, but one or both are missing")
            
            sep_tok = tokenizer.sep_token or "[SEP]"
            combined = title + sep_tok + keywords
            
            tokens = tokenizer(
                combined,
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        elif self.with_title:
            if title is None:
                raise ValueError("Model requires title, but it's missing")
                
            tokens = tokenizer(
                title,
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        elif self.with_keywords:
            if keywords is None:
                raise ValueError("Model requires keywords, but they're missing")
                
            tokens = tokenizer(
                abstract,
                keywords,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
        else:
            tokens = tokenizer(
                abstract,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
        
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        return tokens
    
    def predict_batch(
        self,
        abstracts: List[str],
        titles: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None
    ) -> List[float]:
        if titles is not None and len(titles) != len(abstracts):
            raise ValueError("Number of titles must match number of abstracts")
        if keywords is not None and len(keywords) != len(abstracts):
            raise ValueError("Number of keywords must match number of abstracts")
            
        scores = []
        for i, abstract in enumerate(abstracts):
            title = titles[i] if titles is not None else None
            keyword = keywords[i] if keywords is not None else None
            
            score = self.predict_score(abstract, title, keyword)
            scores.append(score)
            
        return scores
    
    def predict_binary(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None
    ) -> int:
        score = self.predict_score(abstract, title, keywords)
        return int(score > self.threshold)
    
    def evaluate_text(
        self,
        abstract: str,
        title: Optional[str] = None,
        keywords: Optional[str] = None,
        return_binary: bool = False
    ) -> Dict[str, Any]:
        score = self.predict_score(abstract, title, keywords)
        
        result = {
            "abstract": abstract,
            "score": score
        }
        
        if title is not None:
            result["title"] = title
        if keywords is not None:
            result["keywords"] = keywords
            
        if return_binary:
            result["prediction"] = int(score > self.threshold)
            
        return result


def load_predictor(
    model_name: str,
    loss_type: str = "BCE",
    with_title: bool = False,
    with_keywords: bool = False,
    device: Optional[str] = None,
    weights_parent_dir: str = "results/final_model",
    threshold: float = 0.5
) -> BioMoQAPredictor:
    return BioMoQAPredictor(
        model_name=model_name,
        loss_type=loss_type,
        with_title=with_title,
        with_keywords=with_keywords,
        device=device,
        weights_parent_dir=weights_parent_dir,
        threshold=threshold
    )


def example_usage():
    predictor = load_predictor(
        model_name="BiomedBERT-abs",
        loss_type="BCE",
        with_title=False,
        with_keywords=False,
        threshold=0.5
    )
    
    abstract = """
    This study investigates the effects of climate change on biodiversity patterns 
    in marine ecosystems. We analyzed species composition data from coral reefs 
    across multiple geographical locations over a 10-year period.
    """
    
    score = predictor.predict_score(abstract)
    print(f"Prediction score: {score:.4f}")
    
    binary_pred = predictor.predict_binary(abstract)
    print(f"Binary prediction: {binary_pred}")
    
    result = predictor.evaluate_text(abstract, return_binary=True)
    print(f"Full evaluation: {result}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BioMoQA Model Instantiation and Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default example
  python instantiation.py --model_name BiomedBERT-abs --example
  
  # Single prediction
  python instantiation.py --model_name BiomedBERT-abs --abstract "Your abstract text"
  
  # With title and keywords
  python instantiation.py --model_name BiomedBERT-abs --with_title --with_keywords \\
  --abstract "Your abstract" --title "Your title" --keywords "keyword1, keyword2"
  
  # Batch prediction from file
  python instantiation.py --model_name BiomedBERT-abs --input_file texts.txt
        """
    )
    
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
        help="Abstract text for prediction"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title text (optional, used if --with_title is set)"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        help="Keywords (optional, used if --with_keywords is set)"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to file containing texts for batch prediction (JSON format)"
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
        help="Classification threshold for binary prediction (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save prediction results (JSON format)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with the built-in example"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    device = None if args.device == "auto" else args.device
    
    try:
        if args.example:
            logger.info("Running built-in example...")
            example_usage()
            return
            
        logger.info(f"Loading model: {args.model_name} with {args.loss_type} loss")
        predictor = load_predictor(
            model_name=args.model_name,
            loss_type=args.loss_type,
            with_title=args.with_title,
            with_keywords=args.with_keywords,
            device=device,
            weights_parent_dir=args.weights_parent_dir,
            threshold=args.threshold
        )
        logger.info("Model loaded successfully!")
        
        results = []
        
        if args.input_file:
            logger.info(f"Loading texts from: {args.input_file}")
            import json
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                if args.input_file.endswith('.json'):
                    texts_data = json.load(f)
                else:
                    texts_data = []
                    for line in f:
                        if line.strip():
                            texts_data.append(json.loads(line.strip()))
            
            logger.info(f"Processing {len(texts_data)} texts...")
            for i, text_data in enumerate(texts_data):
                if args.verbose:
                    logger.info(f"Processing text {i+1}/{len(texts_data)}")
                
                result = predictor.evaluate_text(
                    abstract=text_data.get('abstract', ''),
                    title=text_data.get('title') if args.with_title else None,
                    keywords=text_data.get('keywords') if args.with_keywords else None,
                    return_binary=True
                )
                results.append(result)
                
        elif args.abstract:
            logger.info("Making single prediction...")
            
            if args.with_title and not args.title:
                parser.error("--title is required when --with_title is set")
            if args.with_keywords and not args.keywords:
                parser.error("--keywords is required when --with_keywords is set")
            
            result = predictor.evaluate_text(
                abstract=args.abstract,
                title=args.title if args.with_title else None,
                keywords=args.keywords if args.with_keywords else None,
                return_binary=True
            )
            results.append(result)
            
        else:
            parser.error("Either --abstract, --input_file, or --example must be provided")
        
        logger.info("\n" + "="*60)
        logger.info("PREDICTION RESULTS")
        logger.info("="*60)
        
        for i, result in enumerate(results):
            if len(results) > 1:
                logger.info(f"\nResult {i+1}:")
                logger.info("-" * 20)
            
            if 'title' in result:
                logger.info(f"Title: {result['title']}")
            if 'keywords' in result:
                logger.info(f"Keywords: {result['keywords']}")
            
            abstract_display = result['abstract']
            if len(abstract_display) > 200:
                abstract_display = abstract_display[:200] + "..."
            logger.info(f"Abstract: {abstract_display}")
            
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
            
            if args.threshold != 0.5:
                logger.info(f"Threshold used: {args.threshold}")
        
        if args.output_file:
            logger.info(f"\nSaving results to: {args.output_file}")
            import json
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Results saved successfully!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())