#!/usr/bin/env python3
"""
Test script for JulietteV2 standalone predictor
Tests the predictor with sample data before building the executable
"""

import json
import subprocess
import sys
from pathlib import Path
import tempfile

def test_predictor():
    # Create test input similar to your input.txt
    test_data = {
        "features": [
            {
                "Body": 0.1, "Close": 1.0, "High": 1.1, "High-Low": 0.2,
                "Is_Doji": 0, "Is_Long_Shadow": 1, "Is_Spike": 0,
                "Low": 0.9, "Open": 0.95, "day_cos": 0.5, "day_sin": 0.866,
                "gap": 0.05, "hour_cos": 0.707, "hour_sin": 0.707,
                "minute_cos": 1.0, "minute_sin": 0.0
            }
        ] * 90  # Repeat to get 90 bars
    }
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(test_data, input_file)
        input_path = input_file.name
    
    output_path = Path(tempfile.gettempdir()) / "test_output.json"
    
    try:
        # Test the predictor script directly
        print("Testing standalone_predictor.py...")
        result = subprocess.run(
            [sys.executable, "standalone_predictor.py", input_path, str(output_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Script executed successfully")
            
            # Read and display output
            with open(output_path, 'r') as f:
                output = json.load(f)
            
            print("\nPrediction Results:")
            print(f"Signal: {output.get('signal', 'N/A')}")
            print(f"Confidence: {output.get('confidence', 0):.4f}")
            print(f"Probabilities: {output.get('probabilities', {})}")
            
            if 'error' not in output:
                print("\n✓ Test PASSED!")
                return True
            else:
                print(f"\n✗ Error in prediction: {output['error']}")
                return False
        else:
            print(f"✗ Script failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False
    finally:
        # Cleanup
        Path(input_path).unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

def check_requirements():
    """Check if all required files are present"""
    required_files = [
        "model_best.pt",
        "scaler.joblib", 
        "label_encoder.joblib",
        "model_metadata.json"
    ]
    
    print("Checking required model files...")
    all_present = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} NOT FOUND")
            all_present = False
    
    return all_present

if __name__ == "__main__":
    print("JulietteV2 Predictor Test")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n⚠️  Missing required model files!")
        print("Please ensure all model files are in the current directory")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Run test
    if test_predictor():
        print("\nAll tests passed! Ready to build executable.")
        print("\nNext step: Run 'build_executable.bat' to create the .exe file")
    else:
        print("\nTests failed. Please check the errors above.")
        sys.exit(1)