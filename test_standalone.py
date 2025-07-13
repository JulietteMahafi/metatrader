#!/usr/bin/env python3
"""
Test script for standalone predictor
Tests the predictor with sample data to ensure it works correctly
"""

import json
import tempfile
import os
from pathlib import Path
import sys

# Add current directory to path to import standalone_predictor
sys.path.append(os.getcwd())

def create_sample_data():
    """Create sample feature data for testing"""
    features = []
    
    # Create 90 bars of sample data
    for i in range(90):
        bar = {
            "Body": 0.001 * (i % 3 - 1),  # Vary between -0.001, 0, 0.001
            "Close": 1.1000 + (i * 0.0001),
            "High": 1.1005 + (i * 0.0001),
            "High-Low": 0.0010,
            "Is_Doji": i % 5 == 0,  # Every 5th bar is doji
            "Is_Long_Shadow": i % 7 == 0,  # Every 7th bar has long shadow
            "Is_Spike": i % 11 == 0,  # Every 11th bar is spike
            "Low": 1.0995 + (i * 0.0001),
            "Open": 1.0999 + (i * 0.0001),
            "day_cos": 0.5,
            "day_sin": 0.866,
            "gap": 0.0001 * (i % 3 - 1),
            "hour_cos": 0.707,
            "hour_sin": 0.707,
            "minute_cos": 1.0,
            "minute_sin": 0.0
        }
        features.append(bar)
    
    return {"features": features}

def create_sample_model_files(temp_dir):
    """Create sample model files for testing"""
    import torch
    import torch.nn as nn
    import joblib
    import numpy as np
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    
    # Create sample model metadata
    metadata = {
        "num_features": 16,
        "num_classes": 3,
        "numerical_features": [
            "Body", "Close", "High", "High-Low", "Low", "Open", 
            "day_cos", "day_sin", "gap", "hour_cos", "hour_sin", 
            "minute_cos", "minute_sin"
        ]
    }
    
    # Create sample scaler
    scaler = RobustScaler()
    sample_data = np.random.randn(100, 13)  # 13 numerical features
    scaler.fit(sample_data)
    
    # Create sample label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(["BUY", "KEEP", "SELL"])
    
    # Create sample model
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(16, 64)
            self.fc = nn.Linear(64, 3)
        
        def forward(self, x):
            x = self.input_projection(x)
            x = torch.relu(x)
            x = x.mean(dim=1)  # Global average pooling
            return self.fc(x)
    
    model = SimpleTransformer()
    
    # Create sample checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "args": {
            "d_model": 64,
            "n_heads": 4,
            "num_layers": 2,
            "d_ff": 128,
            "dropout": 0.1
        }
    }
    
    # Save files
    torch.save(checkpoint, temp_dir / "model_best.pt")
    joblib.dump(scaler, temp_dir / "scaler.joblib")
    joblib.dump(label_encoder, temp_dir / "label_encoder.joblib")
    
    with open(temp_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    return temp_dir

def test_standalone_predictor():
    """Test the standalone predictor"""
    print("=== Testing Standalone Predictor ===")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("Creating sample model files...")
        model_dir = create_sample_model_files(temp_path)
        
        print("Creating sample input data...")
        sample_data = create_sample_data()
        input_file = temp_path / "test_input.json"
        output_file = temp_path / "test_output.json"
        
        with open(input_file, 'w') as f:
            json.dump(sample_data, f)
        
        print("Testing predictor...")
        try:
            # Import and test the predictor
            from standalone_predictor import load_model, predict
            
            # Load model
            model, scaler, label_encoder, model_metadata, device = load_model(model_dir)
            
            if model is None:
                print("ERROR: Failed to load model")
                return False
            
            print(f"Model loaded successfully on device: {device}")
            
            # Make prediction
            result = predict(sample_data["features"], model, scaler, label_encoder, model_metadata, device)
            
            if "error" in result:
                print(f"ERROR: Prediction failed - {result['error']}")
                return False
            
            print("Prediction successful!")
            print(f"Signal: {result['signal']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: {result['probabilities']}")
            
            # Test file I/O
            print("\nTesting file I/O...")
            
            # Write input file
            with open(input_file, 'w') as f:
                json.dump(sample_data, f)
            
            # Simulate command line arguments
            import argparse
            args = argparse.Namespace()
            args.input = str(input_file)
            args.output = str(output_file)
            args.model_dir = str(model_dir)
            
            # Test main function
            from standalone_predictor import main
            
            # Temporarily modify sys.argv for testing
            original_argv = sys.argv
            sys.argv = ['standalone_predictor.py', 
                       '--input', str(input_file),
                       '--output', str(output_file),
                       '--model-dir', str(model_dir)]
            
            try:
                main()
                
                # Check if output file was created
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        output_result = json.load(f)
                    
                    print("File I/O test successful!")
                    print(f"Output: {output_result}")
                    return True
                else:
                    print("ERROR: Output file not created")
                    return False
                    
            finally:
                sys.argv = original_argv
            
        except Exception as e:
            print(f"ERROR: Test failed - {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main test function"""
    print("Starting standalone predictor tests...")
    
    success = test_standalone_predictor()
    
    if success:
        print("\n✅ All tests passed!")
        print("The standalone predictor is working correctly.")
        print("You can now build the executable with: python build_standalone.py")
    else:
        print("\n❌ Tests failed!")
        print("Please check the errors above and fix them before building.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())