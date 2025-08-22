#!/usr/bin/env python3
"""
Test script for the BioMoQA Cross-Validation Ensemble scoring pipeline.
This script tests the ensemble scoring and ranking functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.models.biomoqa.ensemble import CrossValidationPredictor, load_ensemble_predictor


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
            "text": """
            This study investigates the effects of climate change on biodiversity patterns 
            in marine ecosystems. We analyzed species composition data from coral reefs 
            across multiple geographical locations over a 10-year period. Our findings 
            show significant shifts in species distribution correlating with temperature 
            increases and ocean acidification levels. The results indicate substantial 
            threats to marine biodiversity from anthropogenic climate change, with 
            implications for conservation strategies and ecosystem management.
            """,
            "expected_range": "High (>0.8)",
            "category": "Strong biodiversity focus"
        },
        {
            "text": """
            We conducted a comprehensive analysis of endemic bird species on tropical islands
            to understand patterns of species richness and endemism. Using field surveys and
            molecular phylogenetic analysis, we documented the evolutionary relationships
            among island bird communities. Our research reveals that island biogeography
            theory accurately predicts patterns of species diversity in these isolated
            ecosystems, with implications for conservation strategies and understanding
            evolutionary processes in fragmented habitats.
            """,
            "expected_range": "High (>0.8)",
            "category": "Core biodiversity research"
        },
        {
            "text": """
            A novel machine learning approach for protein structure prediction is presented.
            The method combines deep learning architectures with evolutionary information
            to achieve state-of-the-art accuracy. We tested our approach on multiple
            benchmark datasets and compared it with existing methods. The results show
            significant improvements in prediction accuracy across different protein families,
            with potential applications in drug discovery and structural biology.
            """,
            "expected_range": "Low-Medium (0.2-0.4)",
            "category": "Computational, not biodiversity"
        },
        {
            "text": """
            Forest fragmentation poses significant threats to tropical biodiversity. We
            examined the effects of habitat fragmentation on mammalian communities in
            the Amazon rainforest using camera trap surveys across fragmented and
            continuous forest areas. Our 3-year study reveals that fragment size and
            connectivity are critical factors determining species persistence in
            fragmented landscapes, with direct implications for conservation planning.
            """,
            "expected_range": "High (>0.8)",
            "category": "Direct biodiversity conservation"
        },
        {
            "text": """
            The economic impact of tourism on small island states has been extensively studied.
            This research examines the relationship between tourism revenue and local economic 
            development indicators across Caribbean nations over the past decade. We found 
            significant correlations between tourism growth and GDP increases, though with 
            notable variations across different island economies and seasonal patterns.
            """,
            "expected_range": "Very Low (<0.2)",
            "category": "Economic focus, no biodiversity"
        }
    ]
    
    print("🧬 BioMoQA Ensemble Scoring Test")
    print("=" * 60)
    
    try:
        # Initialize the ensemble predictor
        print("Loading ensemble models...")
        predictor = load_ensemble_predictor(
            model_type="BiomedBERT-abs",
            loss_type="BCE",
            base_path="results/final_model",
            threshold=0.5
        )
        print("✅ Models loaded successfully!")
        print()
        
        # Test each abstract
        for i, test_case in enumerate(test_abstracts, 1):
            print(f"📝 Test Case {i}: {test_case['category']}")
            print(f"Expected: {test_case['expected_range']}")
            print("-" * 40)
            
            # Get ensemble prediction
            result = predictor.score_text(test_case["text"])
            
            # Extract key metrics
            ensemble_score = result["ensemble_score"]
            ensemble_std = result["ensemble_std"]
            confidence = result["confidence"]
            fold_scores = result["fold_scores"]
            
            # Get interpretation
            interpretation = get_score_interpretation(ensemble_score)
            
            # Display results
            print(f"🎯 Ensemble Score: {ensemble_score:.4f}")
            print(f"📊 Standard Deviation: {ensemble_std:.4f}")
            print(f"💪 Confidence: {confidence:.4f}")
            print(f"📈 Score Range: {min(fold_scores):.3f} - {max(fold_scores):.3f}")
            print(f"{interpretation['icon']} {interpretation['level']}")
            print(f"   {interpretation['description']}")
            
            # Show individual fold scores
            print("🔍 Individual Fold Scores:")
            for j, (fold_result, score) in enumerate(zip(result["fold_results"], fold_scores), 1):
                print(f"   Fold {j}: {score:.4f}")
            
            print()
            
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please ensure model checkpoints are available in the specified directory.")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    print("✅ Ensemble scoring test completed successfully!")
    return True


def test_batch_scoring():
    """Test batch scoring functionality"""
    print("🚀 Testing Batch Scoring")
    print("=" * 40)
    
    try:
        # Initialize predictor
        predictor = load_ensemble_predictor(
            model_type="BiomedBERT-abs",
            loss_type="BCE",
            base_path="results/final_model"
        )
        
        # Simple batch of abstracts
        batch_abstracts = [
            "Climate change impacts on marine biodiversity in coral reef ecosystems.",
            "Machine learning algorithms for protein structure prediction and analysis.",
            "Economic development in small island developing states and tourism growth."
        ]
        
        print(f"Processing batch of {len(batch_abstracts)} abstracts...")
        
        # Test optimized batch scoring
        results = predictor.score_batch_optimized(batch_abstracts, batch_size=2)
        
        print("📊 Batch Results:")
        for i, result in enumerate(results, 1):
            score = result["ensemble_score"]
            print(f"  Abstract {i}: {score:.4f}")
        
        print("✅ Batch scoring test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Batch scoring test failed: {e}")
        return False


def test_model_validation():
    """Test model path validation"""
    print("🔍 Testing Model Validation")
    print("=" * 40)
    
    from src.models.biomoqa.ensemble import validate_model_path
    
    # Test valid and invalid paths
    test_paths = [
        "results/final_model/best_model_cross_val_BCE_BiomedBERT-abs_fold-1",
        "nonexistent/path",
        "results/final_model"
    ]
    
    for path in test_paths:
        is_valid = validate_model_path(path)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"{status}: {path}")
    
    return True


if __name__ == "__main__":
    print("🧬 BioMoQA Ensemble Testing Suite")
    print("=" * 60)
    print()
    
    # Run all tests
    tests = [
        ("Single Text Ensemble Scoring", test_ensemble_scoring),
        ("Batch Scoring", test_batch_scoring),
        ("Model Validation", test_model_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
        
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        exit(0)
    else:
        print("⚠️  Some tests failed.")
        exit(1)