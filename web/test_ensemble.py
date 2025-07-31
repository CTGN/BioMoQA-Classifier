#!/usr/bin/env python3
"""
Test script for the BioMoQA Cross-Validation Ensemble scoring pipeline.
This script tests the ensemble scoring and ranking functionality.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

class CrossValidationPredictor:
    """Cross-validation predictor using ensemble of 5 fold models for scoring"""
    
    def __init__(self, model_type: str, loss_type: str, base_path: str = "results/biomoqa/final_model", 
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
        print(f"Loading {self.num_folds} fold models for {self.model_type} with {self.loss_type} loss...")
        
        for fold in range(1, self.num_folds + 1):
            fold_path = os.path.join(
                self.base_path, 
                f"best_model_cross_val_{self.loss_type}_{self.model_type}_fold-{fold}"
            )
            
            if not os.path.exists(fold_path):
                raise FileNotFoundError(f"Fold model not found: {fold_path}")
            
            try:
                # Load tokenizer (try from model path, fallback to default)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(fold_path)
                except:
                    print(f"Using default BERT tokenizer for fold {fold}")
                    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
                
                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(fold_path)
                model.to(self.device)
                model.eval()
                
                self.fold_tokenizers[fold] = tokenizer
                self.fold_models[fold] = model
                
                print(f"✅ Loaded fold {fold} successfully")
                
            except Exception as e:
                raise Exception(f"Failed to load fold {fold}: {str(e)}")
        
        print(f"Successfully loaded all {self.num_folds} fold models!")
    
    def predict_single_fold(self, abstract: str, fold: int) -> dict:
        """Score using a single fold model"""
        tokenizer = self.fold_tokenizers[fold]
        model = self.fold_models[fold]
        
        # Tokenize input
        inputs = tokenizer(
            abstract,
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
    
    def score_text(self, abstract: str) -> dict:
        """Score text using ensemble of all fold models"""
        fold_results = []
        scores = []
        
        # Get scores from all folds
        for fold in range(1, self.num_folds + 1):
            fold_result = self.predict_single_fold(abstract, fold)
            fold_results.append(fold_result)
            scores.append(fold_result["score"])
        
        # Calculate ensemble statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        median_score = np.median(scores)
        
        # Optional ensemble prediction for reference
        ensemble_prediction = int(mean_score > self.threshold)
        
        # Count individual fold predictions for consensus
        positive_folds = sum(1 for result in fold_results if result["prediction"] == 1)
        negative_folds = self.num_folds - positive_folds
        
        # Consensus strength (percentage of folds agreeing with ensemble)
        consensus_strength = max(positive_folds, negative_folds) / self.num_folds
        
        return {
            "abstract": abstract,
            "ensemble_score": mean_score,
            "ensemble_prediction": ensemble_prediction,
            "fold_results": fold_results,
            "statistics": {
                "mean_score": mean_score,
                "median_score": median_score,
                "std_score": std_score,
                "min_score": min_score,
                "max_score": max_score,
                "positive_folds": positive_folds,
                "negative_folds": negative_folds,
                "consensus_strength": consensus_strength
            }
        }

def get_score_interpretation(score: float) -> dict:
    """Get interpretation for a given score"""
    if score >= 0.8:
        return {
            "level": "High Relevance",
            "icon": "🟢",
            "description": "Strong biodiversity research content"
        }
    elif score >= 0.6:
        return {
            "level": "Medium-High Relevance", 
            "icon": "🟡",
            "description": "Likely biodiversity-related research"
        }
    elif score >= 0.4:
        return {
            "level": "Medium Relevance",
            "icon": "🟠", 
            "description": "Mixed or unclear biodiversity content"
        }
    elif score >= 0.2:
        return {
            "level": "Low Relevance",
            "icon": "🔴",
            "description": "Unlikely to be biodiversity-focused"
        }
    else:
        return {
            "level": "Very Low Relevance",
            "icon": "⚫",
            "description": "Not biodiversity research"
        }

def test_ensemble_scoring():
    """Test the ensemble scoring pipeline with sample texts"""
    
    # Sample test abstracts with expected score ranges
    test_abstracts = [
        {
            "text": """This study investigates the effects of climate change on biodiversity patterns 
            in marine ecosystems. We analyzed species composition data from coral reefs 
            across multiple geographical locations over a 10-year period. Our findings 
            show significant shifts in species distribution correlating with temperature 
            increases and ocean acidification levels. Conservation implications are discussed.""",
            "expected_range": "High (>0.8)",
            "category": "Core biodiversity research"
        },
        {
            "text": """We conducted a comprehensive analysis of endemic bird species on tropical islands
            to understand patterns of species richness and endemism. Using field surveys and
            molecular phylogenetic analysis, we documented the evolutionary relationships
            among island bird communities. Results reveal critical conservation priorities.""",
            "expected_range": "High (>0.8)",
            "category": "Biodiversity conservation"
        },
        {
            "text": """A novel machine learning approach for protein structure prediction is presented.
            The method combines deep learning architectures with evolutionary information
            to achieve state-of-the-art accuracy. We tested our approach on multiple
            benchmark datasets and compared it with existing methods.""",
            "expected_range": "Low-Medium (0.2-0.4)",
            "category": "Computational biology"
        },
        {
            "text": """Forest fragmentation poses significant threats to tropical biodiversity. We
            examined the effects of habitat fragmentation on mammalian communities in
            the Amazon rainforest using camera trap surveys across fragmented and
            continuous forest areas. Fragment connectivity is critical for species persistence.""",
            "expected_range": "High (>0.8)",
            "category": "Habitat fragmentation"
        },
        {
            "text": """The economic impact of tourism on small island states has been extensively studied.
            This research examines the relationship between tourism revenue and local economic 
            development indicators across Caribbean nations over the past decade. GDP correlations
            are analyzed with respect to seasonal tourism patterns.""",
            "expected_range": "Very Low (<0.2)",
            "category": "Economic research"
        }
    ]
    
    # Available model configurations
    available_configs = [
        {"model_type": "bert-base", "loss_type": "BCE"},
        {"model_type": "biobert-v1", "loss_type": "BCE"},
        {"model_type": "BiomedBERT-abs", "loss_type": "BCE"},
        {"model_type": "BiomedBERT-abs-ft", "loss_type": "BCE"},
        {"model_type": "roberta-base", "loss_type": "BCE"},
        {"model_type": "bert-base", "loss_type": "focal"},
        {"model_type": "biobert-v1", "loss_type": "focal"},
        {"model_type": "BiomedBERT-abs", "loss_type": "focal"},
        {"model_type": "BiomedBERT-abs-ft", "loss_type": "focal"},
        {"model_type": "roberta-base", "loss_type": "focal"}
    ]
    
    # Test with the first available model configuration
    base_path = "results/biomoqa/final_model"
    
    # Find first available configuration
    selected_config = None
    for config in available_configs:
        # Check if at least one fold exists for this configuration
        test_fold_path = os.path.join(
            base_path, 
            f"best_model_cross_val_{config['loss_type']}_{config['model_type']}_fold-1"
        )
        if os.path.exists(test_fold_path):
            selected_config = config
            break
    
    if not selected_config:
        print("❌ No model configurations found!")
        print("\nAvailable configurations to test:")
        for config in available_configs:
            fold_path = os.path.join(
                base_path, 
                f"best_model_cross_val_{config['loss_type']}_{config['model_type']}_fold-1"
            )
            status = "✅" if os.path.exists(fold_path) else "❌"
            print(f"{status} {config['model_type']} with {config['loss_type']} loss")
        
        print(f"\nExpected directory structure:")
        print(f"{base_path}/")
        print(f"├── best_model_cross_val_BCE_bert-base_fold-1/")
        print(f"├── best_model_cross_val_BCE_bert-base_fold-2/")
        print(f"├── ... (folds 3-5)")
        print(f"├── best_model_cross_val_focal_bert-base_fold-1/")
        print(f"└── ... (other model types)")
        return
    
    try:
        print(f"\n🚀 Testing ensemble scoring with {selected_config['model_type']} ({selected_config['loss_type']} loss)")
        print("="*80)
        
        # Load ensemble predictor
        predictor = CrossValidationPredictor(
            model_type=selected_config['model_type'],
            loss_type=selected_config['loss_type'],
            base_path=base_path
        )
        
        print(f"\n🔬 ENSEMBLE SCORING TESTING")
        print("="*80)
        
        results = []
        
        for i, test_case in enumerate(test_abstracts, 1):
            print(f"\n📝 Test {i}: {test_case['category']} (Expected: {test_case['expected_range']})")
            print(f"Abstract: {test_case['text'][:100]}...")
            print("-" * 60)
            
            # Get ensemble score
            result = predictor.score_text(test_case['text'])
            
            # Display results
            ensemble_score = result['ensemble_score']
            stats = result['statistics']
            
            # Get score interpretation
            interpretation = get_score_interpretation(ensemble_score)
            
            print(f"🎯 Ensemble Score: {ensemble_score:.4f}")
            print(f"{interpretation['icon']} {interpretation['level']}: {interpretation['description']}")
            print(f"📊 Score Statistics: μ={stats['mean_score']:.3f}, σ={stats['std_score']:.3f}, range=[{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
            print(f"🤝 Consensus: {stats['consensus_strength']:.1%}")
            
            # Show individual fold scores (sorted)
            fold_scores = [(r['fold'], r['score']) for r in result['fold_results']]
            fold_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"🔍 Fold Scores (ranked): {', '.join([f'F{fold}:{score:.3f}' for fold, score in fold_scores])}")
            
            results.append({
                "test_case": test_case,
                "score": ensemble_score,
                "interpretation": interpretation
            })
            
            print("=" * 60)
        
        # Ranking demonstration
        print(f"\n🏆 RANKING DEMONSTRATION")
        print("="*80)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("Research abstracts ranked by biodiversity relevance score:")
        for rank, result in enumerate(results, 1):
            score = result['score']
            interp = result['interpretation']
            category = result['test_case']['category']
            print(f"{rank}. {interp['icon']} {score:.4f} - {category} ({interp['level']})")
        
        # Summary statistics
        all_scores = [r['score'] for r in results]
        print(f"\n📊 SCORING SUMMARY")
        print("="*80)
        print(f"Total texts scored: {len(all_scores)}")
        print(f"Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"Average score: {np.mean(all_scores):.3f}")
        print(f"Score std deviation: {np.std(all_scores):.3f}")
        print(f"High relevance (≥0.6): {sum(1 for s in all_scores if s >= 0.6)}/{len(all_scores)}")
        print(f"Low relevance (<0.4): {sum(1 for s in all_scores if s < 0.4)}/{len(all_scores)}")
        
        print(f"\n✅ Ensemble scoring test completed successfully!")
        print(f"📊 Model Configuration: {selected_config['model_type']} with {selected_config['loss_type']} loss")
        print(f"🎯 Ensemble Size: {predictor.num_folds} fold models")
        print(f"💻 Device: {predictor.device}")
        
        # Ranking validation
        print(f"\n🎯 RANKING VALIDATION")
        print("="*40)
        biodiversity_papers = [r for r in results if 'biodiversity' in r['test_case']['category'].lower() 
                              or 'conservation' in r['test_case']['category'].lower()
                              or 'fragmentation' in r['test_case']['category'].lower()]
        non_biodiversity_papers = [r for r in results if r not in biodiversity_papers]
        
        if biodiversity_papers and non_biodiversity_papers:
            avg_bio_score = np.mean([r['score'] for r in biodiversity_papers])
            avg_non_bio_score = np.mean([r['score'] for r in non_biodiversity_papers])
            print(f"✅ Biodiversity papers average score: {avg_bio_score:.3f}")
            print(f"✅ Non-biodiversity papers average score: {avg_non_bio_score:.3f}")
            if avg_bio_score > avg_non_bio_score:
                print("✅ Ranking validation PASSED: Biodiversity papers scored higher on average")
            else:
                print("⚠️ Ranking validation: Unexpected score distribution")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nThis might indicate:")
        print("- Missing model files in the expected directories")
        print("- Incorrect model checkpoint format")
        print("- Insufficient system resources (GPU/CPU memory)")
        print(f"\nThe web application will work once you have trained models in:")
        print(f"{base_path}/")

def check_model_availability():
    """Check which model configurations are available"""
    base_path = "results/biomoqa/final_model"
    
    if not os.path.exists(base_path):
        print(f"❌ Base path does not exist: {base_path}")
        return
    
    print(f"🔍 Checking model availability in: {base_path}")
    print("="*60)
    
    model_types = ["bert-base", "biobert-v1", "BiomedBERT-abs", "BiomedBERT-abs-ft", "roberta-base"]
    loss_types = ["BCE", "focal"]
    
    available_count = 0
    
    for loss_type in loss_types:
        print(f"\n📁 {loss_type} Loss Models:")
        for model_type in model_types:
            fold_paths = []
            missing_folds = []
            
            for fold in range(1, 6):
                fold_path = os.path.join(
                    base_path, 
                    f"best_model_cross_val_{loss_type}_{model_type}_fold-{fold}"
                )
                if os.path.exists(fold_path):
                    fold_paths.append(fold_path)
                else:
                    missing_folds.append(fold)
            
            if len(fold_paths) == 5:
                print(f"   ✅ {model_type}: All 5 folds available")
                available_count += 1
            elif len(fold_paths) > 0:
                print(f"   ⚠️  {model_type}: {len(fold_paths)}/5 folds (missing: {missing_folds})")
            else:
                print(f"   ❌ {model_type}: No folds found")
    
    print("="*60)
    print(f"📊 Summary: {available_count}/{len(model_types) * len(loss_types)} complete configurations available")

if __name__ == "__main__":
    print("🧬 BioMoQA Cross-Validation Ensemble Scoring Test Script")
    print("="*60)
    
    # Check model availability first
    check_model_availability()
    
    print("\n" + "="*60)
    
    # Run ensemble scoring test
    test_ensemble_scoring() 