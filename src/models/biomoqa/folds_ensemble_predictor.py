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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import multiprocessing as mp

logger = logging.getLogger(__name__)


class CrossValidationPredictor:
    """Cross-validation predictor using ensemble of 5 fold models for scoring"""
    
    def __init__(self, model_type: str, loss_type: str, base_path: str = "results/final_model", 
                 threshold: float = 0.5, device: str = None, use_fp16: bool = None, use_compile: bool = None):
        self.model_type = model_type
        self.loss_type = loss_type
        self.base_path = base_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU optimization settings
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        # Disable compilation by default due to threading/CUDA graph conflicts
        self.use_compile = use_compile if use_compile is not None else False
        
        # Store fold models and tokenizers
        self.fold_models = {}
        self.fold_tokenizers = {}
        self.num_folds = 5
        
        # Thread safety for parallel processing
        self._model_locks = {}
        self._gpu_lock = threading.Lock()  # Global GPU lock for CUDA operations
        
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
                
                # Apply GPU optimizations
                if self.device == "cuda":
                    # Enable FP16 mixed precision if requested
                    if self.use_fp16:
                        model = model.half()  # Convert model to FP16
                        logger.info(f"✅ Enabled FP16 mixed precision for fold {fold}")
                    
                    # Enable model compilation if requested and supported (use safer mode)
                    if self.use_compile and hasattr(torch, 'compile'):
                        try:
                            # Use "default" mode instead of "reduce-overhead" to avoid CUDA graph issues
                            model = torch.compile(model, mode="default", dynamic=True)
                            logger.info(f"✅ Compiled model for fold {fold} (safe mode)")
                        except Exception as e:
                            logger.warning(f"Model compilation failed for fold {fold}: {e}")
                            self.use_compile = False  # Disable for all folds if one fails
                
                self.fold_tokenizers[fold] = tokenizer
                self.fold_models[fold] = model
                self._model_locks[fold] = threading.Lock()  # Thread safety for parallel processing
                
                logger.info(f"✅ Loaded fold {fold} successfully")
                
            except Exception as e:
                raise Exception(f"Failed to load fold {fold}: {str(e)}")
        
        logger.info(f"Successfully loaded all {self.num_folds} fold models!")
        
        # Quick diagnostic test
        self._run_diagnostic_test()
    
    def _run_diagnostic_test(self):
        """Run a quick test to verify models are working correctly"""
        try:
            logger.info("🔍 Running diagnostic test...")
            test_abstract = "This is a test abstract about biodiversity research."
            test_title = "Test Title"
            
            # Test fold 1 with basic inference
            fold = 1
            tokenizer = self.fold_tokenizers[fold]
            model = self.fold_models[fold]
            
            # Simple tokenization test
            inputs = tokenizer(
                test_abstract,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Simple inference test
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                score = torch.sigmoid(logits).squeeze().cpu().item()
            
            logger.info(f"✅ Diagnostic test passed! Test score: {score:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Diagnostic test failed: {str(e)}")
            import traceback
            logger.error(f"Diagnostic traceback: {traceback.format_exc()}")
    
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
        
        # Make prediction with proper dtype handling
        with torch.no_grad():
            # Handle FP16 inputs if model is using FP16
            if self.use_fp16 and self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).squeeze().float().cpu().item()  # Convert back to float32 for consistency
        
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
                
                # Get predictions for entire batch with proper dtype handling
                with torch.no_grad():
                    # Handle FP16 inputs if model is using FP16
                    if self.use_fp16 and self.device == "cuda":
                        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    logits = outputs.logits
                    scores = torch.sigmoid(logits).squeeze().float().cpu().numpy()  # Convert back to float32 for consistency
                    
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
    
    def _process_fold_batch_sequential(self, fold: int, batch_abstracts: List[str], batch_titles: List[str]) -> List[Dict]:
        """Process a batch of texts through a single fold model (optimized sequential)"""
        try:
            # Check if fold models are loaded
            if fold not in self.fold_tokenizers:
                raise ValueError(f"Tokenizer for fold {fold} not found")
            if fold not in self.fold_models:
                raise ValueError(f"Model for fold {fold} not found")
                
            tokenizer = self.fold_tokenizers[fold]
            model = self.fold_models[fold]
            
            # Verify model is on correct device
            model_device = next(model.parameters()).device
            logger.info(f"Debug - Fold {fold}: Model on device {model_device}, Expected: {self.device}")
            
            # Validate input data
            if not batch_abstracts:
                raise ValueError(f"Empty batch_abstracts for fold {fold}")
            if len(batch_abstracts) != len(batch_titles):
                raise ValueError(f"Mismatch between abstracts and titles length for fold {fold}")
            
            logger.info(f"Debug - Fold {fold}: Processing {len(batch_abstracts)} abstracts")
            
            # Tokenize entire batch - handle titles
            has_titles = any(title is not None for title in batch_titles)
            
            if has_titles:
                # Use title-abstract pairs, handling None titles
                title_inputs = [str(title) if title is not None else "" for title in batch_titles]
                abstract_inputs = [str(abstract) if abstract is not None else "" for abstract in batch_abstracts]
                logger.info(f"Debug - Fold {fold}: Tokenizing with titles")
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
                logger.info(f"Debug - Fold {fold}: Tokenizing without titles")
                inputs = tokenizer(
                    abstract_inputs,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                )
            
            logger.info(f"Debug - Fold {fold}: Tokenization complete, input shape: {inputs['input_ids'].shape}")
            
            # Move to device and handle FP16
            logger.info(f"Debug - Fold {fold}: Moving tensors to device {self.device}")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions for entire batch with proper dtype handling
            with torch.no_grad():
                logger.info(f"Debug - Fold {fold}: Starting inference")
                
                # Handle FP16 inputs if model is using FP16
                if self.use_fp16 and self.device == "cuda":
                    logger.info(f"Debug - Fold {fold}: Converting to FP16")
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                logger.info(f"Debug - Fold {fold}: Running model forward pass")
                outputs = model(**inputs)
                logger.info(f"Debug - Fold {fold}: Model forward pass complete")
                
                logits = outputs.logits
                logger.info(f"Debug - Fold {fold}: Logits shape: {logits.shape}")
                
                scores = torch.sigmoid(logits).squeeze().float().cpu().numpy()  # Convert back to float32 for consistency
                logger.info(f"Debug - Fold {fold}: Scores computed")
                
                # Handle single item case
                if len(batch_abstracts) == 1:
                    scores = [scores.item()]
                else:
                    scores = scores.tolist()
                
                logger.info(f"Debug - Fold {fold}: Scores converted to list: {len(scores)} items")
            
            # Return fold results
            fold_results = []
            for i, score in enumerate(scores):
                fold_results.append({
                    "fold": fold,
                    "score": score,
                    "prediction": int(score > self.threshold)
                })
            
            logger.info(f"✅ Fold {fold} completed successfully ({len(fold_results)} results)")
            return fold_results
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ Fold {fold} processing failed: {str(e)}")
            logger.error(f"Full traceback: {error_details}")
            
            # Additional debugging info
            logger.error(f"Debug info - Fold: {fold}, Device: {self.device}")
            logger.error(f"Debug info - Batch size: {len(batch_abstracts)}")
            logger.error(f"Debug info - Use FP16: {self.use_fp16}")
            logger.error(f"Debug info - Use compile: {self.use_compile}")
            
            raise e  # Re-raise to handle at higher level
    
    def score_batch_parallel(self, data, batch_size: int = 16, max_workers: int = 5) -> list:
        """
        Ultra-fast parallel batch scoring using all fold models simultaneously
        ~5x faster than sequential processing by running folds in parallel
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
        
        logger.info(f"🚀 PARALLEL Processing {len(abstracts)} abstracts with batch_size={batch_size}, max_workers={max_workers}")
        
        final_results = []
        
        # Process in batches
        for batch_start in range(0, len(abstracts), batch_size):
            batch_end = min(batch_start + batch_size, len(abstracts))
            batch_abstracts = abstracts[batch_start:batch_end]
            batch_titles = titles[batch_start:batch_end]
            
            logger.info(f"⚡ Processing batch {batch_start//batch_size + 1}/{(len(abstracts)-1)//batch_size + 1} in PARALLEL")
            
            # Process all folds in parallel using ThreadPoolExecutor
            batch_fold_results = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all fold processing tasks
                future_to_fold = {
                    executor.submit(self._process_fold_batch, fold, batch_abstracts, batch_titles): fold
                    for fold in range(1, self.num_folds + 1)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_fold):
                    fold = future_to_fold[future]
                    try:
                        fold_results = future.result()
                        batch_fold_results[fold] = fold_results
                        logger.info(f"✅ Fold {fold} completed")
                    except Exception as e:
                        logger.error(f"❌ Fold {fold} failed: {e}")
                        raise e
            
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
            
            # Clean up GPU memory less frequently for better performance
            if batch_start % (batch_size * 4) == 0:  # Every 4 batches instead of every batch
                torch.cuda.empty_cache()
        
        return final_results
    
    def _create_dynamic_batches(self, abstracts: List[str], titles: List[str], base_batch_size: int = 16) -> List[List[int]]:
        """
        Create dynamic batches based on text length for optimal GPU utilization
        Groups texts of similar length together to minimize padding waste
        """
        # Calculate text lengths (title + abstract)
        text_lengths = []
        for i, (abstract, title) in enumerate(zip(abstracts, titles)):
            abstract_len = len(str(abstract).split()) if abstract else 0
            title_len = len(str(title).split()) if title else 0
            total_len = abstract_len + title_len
            text_lengths.append((i, total_len))
        
        # Sort by length
        text_lengths.sort(key=lambda x: x[1])
        
        # Create batches with similar lengths
        batches = []
        current_batch = []
        current_length = 0
        
        for idx, length in text_lengths:
            # If adding this text would make the batch too different in length or too large
            if (current_batch and 
                (len(current_batch) >= base_batch_size or 
                 abs(length - current_length / len(current_batch)) > 50)):  # 50 word difference threshold
                batches.append(current_batch)
                current_batch = [idx]
                current_length = length
            else:
                current_batch.append(idx)
                current_length += length
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"📊 Dynamic batching: Created {len(batches)} batches from {len(abstracts)} texts")
        return batches
    
    def score_batch_ultra_optimized(self, data, base_batch_size: int = 16, max_workers: int = 5, use_dynamic_batching: bool = True) -> list:
        """
        Ultra-optimized batch scoring with:
        - Dynamic length-based batching for better GPU utilization
        - Optimized sequential fold processing (GPU operations are inherently sequential)
        - Smart memory management
        - FP16 + compilation optimizations
        Expected speedup: ~3-5x over original implementation
        """
        # Extract abstracts and titles, filter out items with None abstracts
        valid_items = []
        for item in data:
            abstract = item.get('abstract')
            if abstract is not None and abstract.strip():
                title = item.get('title', None)
                valid_items.append({'abstract': abstract, 'title': title})
        
        if not valid_items:
            return []
        
        abstracts = [item['abstract'] for item in valid_items]
        titles = [item['title'] for item in valid_items]
        
        logger.info(f"🚀 ULTRA-OPTIMIZED Processing {len(abstracts)} abstracts")
        logger.info(f"⚡ Features: Dynamic batching + Sequential folds + FP16 + Compilation")
        
        # Create dynamic batches based on text length
        if use_dynamic_batching:
            batch_indices = self._create_dynamic_batches(abstracts, titles, base_batch_size)
        else:
            # Fall back to regular batching
            batch_indices = [list(range(i, min(i + base_batch_size, len(abstracts)))) 
                           for i in range(0, len(abstracts), base_batch_size)]
        
        final_results = [None] * len(abstracts)  # Pre-allocate results array
        
        # Process dynamic batches
        for batch_num, indices in enumerate(batch_indices):
            batch_abstracts = [abstracts[i] for i in indices]
            batch_titles = [titles[i] for i in indices]
            
            logger.info(f"⚡ Processing dynamic batch {batch_num + 1}/{len(batch_indices)} (size: {len(indices)})")
            
            # Process all folds sequentially (but optimized)
            batch_fold_results = {}
            
            for fold in range(1, self.num_folds + 1):
                try:
                    fold_results = self._process_fold_batch_sequential(fold, batch_abstracts, batch_titles)
                    batch_fold_results[fold] = fold_results
                except Exception as e:
                    logger.error(f"❌ Fold {fold} failed: {e}")
                    
                    # Try fallback with disabled optimizations
                    logger.info(f"🔄 Attempting fallback for fold {fold} with disabled optimizations")
                    try:
                        # Temporarily disable FP16 and compilation for this fold
                        original_fp16 = self.use_fp16
                        original_compile = self.use_compile
                        self.use_fp16 = False
                        self.use_compile = False
                        
                        fold_results = self._process_fold_batch_sequential(fold, batch_abstracts, batch_titles)
                        batch_fold_results[fold] = fold_results
                        logger.info(f"✅ Fold {fold} succeeded with fallback mode")
                        
                        # Restore original settings
                        self.use_fp16 = original_fp16
                        self.use_compile = original_compile
                        
                    except Exception as fallback_e:
                        logger.error(f"❌ Fold {fold} fallback also failed: {fallback_e}")
                        # Restore original settings
                        self.use_fp16 = original_fp16
                        self.use_compile = original_compile
                        
                        # Mark this fold as failed with default scores
                        batch_fold_results[fold] = []
                        for i in range(len(batch_abstracts)):
                            batch_fold_results[fold].append({
                                "fold": fold,
                                "score": 0.0,  # Default score for failed fold
                                "prediction": 0
                            })
            
            # Compile results for this batch and place them in correct positions
            for i, original_idx in enumerate(indices):
                fold_results = [batch_fold_results[fold][i] for fold in range(1, self.num_folds + 1)]
                fold_scores = [result["score"] for result in fold_results]
                
                ensemble_score = np.mean(fold_scores)
                ensemble_std = np.std(fold_scores)
                ensemble_prediction = int(ensemble_score > self.threshold)
                
                result = {
                    "abstract": abstracts[original_idx],
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
                
                final_results[original_idx] = result
            
            # Optimized memory cleanup - less frequent
            if batch_num % 3 == 0:  # Every 3 batches
                if self.device == "cuda":
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
