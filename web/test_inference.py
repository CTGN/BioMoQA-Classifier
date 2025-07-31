#!/usr/bin/env python3
"""
Test script for the simplified BioMoQA inference pipeline.
This script tests the core functionality without the web interface.
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

class SimpleBioMoQAPredictor:
    """Simplified BioMoQA predictor using Hugging Face transformers"""
    
    def __init__(self, model_path: str, threshold: float = 0.5, device: str = None):
        self.model_path = model_path
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, abstract: str) -> dict:
        """Predict on a single abstract text"""
        # Tokenize input
        inputs = self.tokenizer(
            abstract,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).squeeze().cpu().item()
        
        # Return result
        prediction = int(score > self.threshold)
        return {
            "abstract": abstract,
            "score": score,
            "prediction": prediction
        }

def test_inference():
    """Test the inference pipeline with sample texts"""
    
    # Sample test abstracts
    test_abstracts = [
        """This study investigates the effects of climate change on biodiversity patterns 
        in marine ecosystems. We analyzed species composition data from coral reefs 
        across multiple geographical locations over a 10-year period.""",
        
        """A novel machine learning approach for protein structure prediction is presented.
        The method combines deep learning architectures with evolutionary information
        to achieve state-of-the-art accuracy.""",
        
        """Forest fragmentation poses significant threats to tropical biodiversity. We
        examined the effects of habitat fragmentation on mammalian communities in
        the Amazon rainforest using camera trap surveys."""
    ]
    
    # Test with mock model (for demonstration - replace with actual model path)
    try:
        # Replace this with your actual model path
        model_path = "path/to/your/biomoqa/model"
        
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist.")
            print("Please update the model_path variable with your actual model path.")
            print("\nExample paths might be:")
            print("- results/biomoqa/best_model_fold_1/")
            print("- models/biomoqa_bert_base/")
            return
        
        predictor = SimpleBioMoQAPredictor(model_path)
        
        print("\n" + "="*60)
        print("TESTING INFERENCE PIPELINE")
        print("="*60)
        
        for i, abstract in enumerate(test_abstracts, 1):
            print(f"\nTest {i}:")
            print(f"Abstract: {abstract[:100]}...")
            
            result = predictor.predict(abstract)
            
            print(f"Score: {result['score']:.4f}")
            print(f"Prediction: {'Positive (Biodiversity)' if result['prediction'] == 1 else 'Negative (Not Biodiversity)'}")
            print("-" * 40)
        
        print("\nInference pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("\nThis is normal if you don't have a trained model yet.")
        print("The web application will work once you provide a valid model path.")

if __name__ == "__main__":
    test_inference() 