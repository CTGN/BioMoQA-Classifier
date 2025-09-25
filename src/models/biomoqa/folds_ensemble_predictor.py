#!/usr/bin/env python3
"""
Ensemble prediction module for BioMoQA models.
This module provides cross-validation ensemble scoring using multiple fold models.
"""

import os
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class CrossValidationPredictor:
    """Cross-validation predictor using ensemble of 5 fold models for scoring"""
    
    def __init__(self, model_type: str, loss_type: str, base_path: str = "results/final_model", 
                 threshold: float = 0.5, device: str = None):
        self.model_type = model_type
        self.loss_type = loss_type
        self.base_path = base_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store fold models and tokenizers
        self.fold_models = {}
        self.fold_tokenizers = {}
        self.num_folds = 5
        
        self._load_fold_models()
    
    def _load_fold_models(self):
        """Load all 5 fold models for the specified model type and loss type"""
        logger.info(f"Loading {self.num_folds} fold models for {self.model_type} with {self.loss_type} loss...")
        
        for fold in range(1, self.num_folds + 1):
            fold_path = os.path.join(
                self.base_path, 
                f"best_model_cross_val_{self.loss_type}_{self.model_type}_fold-{fold}"
            )
            
            if not os.path.exists(fold_path):
                raise FileNotFoundError(f"Fold model not found: {fold_path}")
            
            try:
                # Load tokenizer (try from model path, fallback to default)
                tokenizer = AutoTokenizer.from_pretrained(fold_path)
                
                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(fold_path)
                model.to(self.device)
                model.eval()
                
                self.fold_tokenizers[fold] = tokenizer
                self.fold_models[fold] = model
                
                logger.info(f"✅ Loaded fold {fold} successfully")
                
            except Exception as e:
                raise Exception(f"Failed to load fold {fold}: {str(e)}")
        
        logger.info(f"Successfully loaded all {self.num_folds} fold models!")
    
    def predict_single_fold(self, abstract: str, fold: int, title: str = None) -> dict:
        """Score using a single fold model"""
        tokenizer = self.fold_tokenizers[fold]
        model = self.fold_models[fold]
        
        # Tokenize input - use title and abstract if title is provided
        if title is not None:
            inputs = tokenizer(
                str(title),
                str(abstract),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = tokenizer(
                str(abstract),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).squeeze().cpu().item()
        
        prediction = int(score > self.threshold)
        return {
            "fold": fold,
            "score": score,
            "prediction": prediction
        }
    
    def score_text(self, abstract: str, title: str = None) -> dict:
        """Score a text using ensemble of all fold models"""
        fold_results = []
        fold_scores = []
        
        # Get predictions from all folds
        for fold in range(1, self.num_folds + 1):
            result = self.predict_single_fold(abstract, fold, title)
            fold_results.append(result)
            fold_scores.append(result["score"])
        
        # Calculate ensemble statistics
        ensemble_score = np.mean(fold_scores)
        ensemble_std = np.std(fold_scores)
        ensemble_prediction = int(ensemble_score > self.threshold)
        
        return {
            "abstract": abstract,
            "fold_results": fold_results,
            "fold_scores": fold_scores,
            "ensemble_score": ensemble_score,
            "ensemble_std": ensemble_std,
            "ensemble_prediction": ensemble_prediction,
            "confidence": 1.0 - ensemble_std,  # Lower std = higher confidence
            "threshold": self.threshold,
            "statistics": {
                "mean_score": ensemble_score,
                "median_score": np.median(fold_scores),
                "std_score": ensemble_std,
                "min_score": np.min(fold_scores),
                "max_score": np.max(fold_scores),
                "positive_folds": sum(1 for result in fold_results if result["prediction"] == 1),
                "negative_folds": self.num_folds - sum(1 for result in fold_results if result["prediction"] == 1),
                "consensus_strength": max(
                    sum(1 for result in fold_results if result["prediction"] == 1),
                    self.num_folds - sum(1 for result in fold_results if result["prediction"] == 1)
                ) / self.num_folds
            }
        }
    
    def score_batch_optimized(self, data, batch_size: int = 16) -> list:
        """
        Optimized batch scoring using GPU acceleration
        Process multiple abstracts and titles simultaneously for better performance
        """
        # Extract abstracts and titles, filter out items with None abstracts
        valid_items = []
        for item in data:
            abstract = item.get('abstract')
            if abstract is not None and abstract.strip():  # Only process items with valid, non-empty abstracts
                title = item.get('title', None)
                valid_items.append({'abstract': abstract, 'title': title})
        
        if not valid_items:
            return []
        
        abstracts = [item['abstract'] for item in valid_items]
        titles = [item['title'] for item in valid_items]
        
        logger.info(f"Processing {len(abstracts)} abstracts with batch_size={batch_size}")
        
        final_results = []
        
        # Process in batches
        for batch_start in range(0, len(abstracts), batch_size):
            batch_end = min(batch_start + batch_size, len(abstracts))
            batch_abstracts = abstracts[batch_start:batch_end]
            batch_titles = titles[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(abstracts)-1)//batch_size + 1}")
            
            # Store results for this batch
            batch_fold_results = {fold: [] for fold in range(1, self.num_folds + 1)}
            
            # Process each fold
            for fold in range(1, self.num_folds + 1):
                tokenizer = self.fold_tokenizers[fold]
                model = self.fold_models[fold]
                
                # Tokenize entire batch - handle titles
                # Check if any titles are not None in this batch
                has_titles = any(title is not None for title in batch_titles)
                
                if has_titles:
                    # Use title-abstract pairs, handling None titles
                    title_inputs = [str(title) if title is not None else "" for title in batch_titles]
                    abstract_inputs = [str(abstract) if abstract is not None else "" for abstract in batch_abstracts]
                    inputs = tokenizer(
                        title_inputs,
                        abstract_inputs,
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_tensors="pt"
                    )
                else:
                    # No titles in this batch, use abstracts only
                    abstract_inputs = [str(abstract) if abstract is not None else "" for abstract in batch_abstracts]
                    inputs = tokenizer(
                        abstract_inputs,
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_tensors="pt"
                    )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions for entire batch
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    scores = torch.sigmoid(logits).squeeze().cpu().numpy()
                    
                    # Handle single item case
                    if len(batch_abstracts) == 1:
                        scores = [scores.item()]
                    else:
                        scores = scores.tolist()
                
                # Store fold results
                for i, score in enumerate(scores):
                    batch_fold_results[fold].append({
                        "fold": fold,
                        "score": score,
                        "prediction": int(score > self.threshold)
                    })
            
            # Compile final results for this batch
            for i, (abstract, title) in enumerate(zip(batch_abstracts, batch_titles)):
                fold_results = [batch_fold_results[fold][i] for fold in range(1, self.num_folds + 1)]
                fold_scores = [result["score"] for result in fold_results]
                
                ensemble_score = np.mean(fold_scores)
                ensemble_std = np.std(fold_scores)
                ensemble_prediction = int(ensemble_score > self.threshold)
                
                result = {
                    "abstract": abstract,
                    "fold_results": fold_results,
                    "fold_scores": fold_scores,
                    "ensemble_score": ensemble_score,
                    "ensemble_std": ensemble_std,
                    "ensemble_prediction": ensemble_prediction,
                    "confidence": 1.0 - ensemble_std,
                    "threshold": self.threshold,
                    "statistics": {
                        "mean_score": ensemble_score,
                        "median_score": np.median(fold_scores),
                        "std_score": ensemble_std,
                        "min_score": np.min(fold_scores),
                        "max_score": np.max(fold_scores),
                        "positive_folds": sum(1 for result in fold_results if result["prediction"] == 1),
                        "negative_folds": self.num_folds - sum(1 for result in fold_results if result["prediction"] == 1),
                        "consensus_strength": max(
                            sum(1 for result in fold_results if result["prediction"] == 1),
                            self.num_folds - sum(1 for result in fold_results if result["prediction"] == 1)
                        ) / self.num_folds
                    }
                }
                
                final_results.append(result)
            
            # Clean up GPU memory
            torch.cuda.empty_cache()
        
        return final_results


def load_ensemble_predictor(
    model_type: str,
    loss_type: str = "BCE", 
    base_path: str = "results/final_model",
    threshold: float = 0.5,
    device: str = None
) -> CrossValidationPredictor:
    """Load an ensemble predictor with the specified configuration"""
    return CrossValidationPredictor(
        model_type=model_type,
        loss_type=loss_type,
        base_path=base_path,
        threshold=threshold,
        device=device
    )


def validate_model_path(model_path: str) -> bool:
    """Validate that the model path exists and contains required model files"""
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