#!/usr/bin/env python3
"""
Build script for JulietteV2 standalone predictor executable
"""

import subprocess
import sys
import os
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("PyInstaller already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_executable():
    """Build the standalone executable"""
    print("Building JulietteV2 standalone executable...")
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",                           # Single executable file
        "--name=JulietteV2Predictor",         # Output name
        "--hidden-import=sklearn.preprocessing._data",  # Hidden sklearn imports
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=sklearn.neighbors._partition_nodes",
        "--hidden-import=joblib",
        "--hidden-import=torch",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--collect-data=torch",                # Include torch data
        "--collect-data=sklearn",             # Include sklearn data
        "--collect-data=pandas",              # Include pandas data
        "--exclude-module=matplotlib",        # Exclude unnecessary modules
        "--exclude-module=IPython",
        "--exclude-module=jupyter",
        "--noupx",                           # Don't compress (faster execution)
        "--console",                         # Console application
        "standalone_predictor.py"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("Build completed successfully!")
        print("Executable created: dist/JulietteV2Predictor.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False

def create_test_script():
    """Create a test script to verify the executable works"""
    test_script = """
import json
import subprocess
import sys
from pathlib import Path

# Test data (same format as input.json)
test_data = {
    "features": [
        {
            "Body": 0.1, "Close": 1.0, "High": 1.1, "High-Low": 0.2,
            "Is_Doji": 0, "Is_Long_Shadow": 1, "Is_Spike": 0,
            "Low": 0.9, "Open": 0.95, "day_cos": 0.5, "day_sin": 0.866,
            "gap": 0.05, "hour_cos": 0.707, "hour_sin": 0.707,
            "minute_cos": 1.0, "minute_sin": 0.0
        }
        # Add more test data as needed
    ]
}

def test_executable():
    exe_path = Path("dist/JulietteV2Predictor.exe")
    if not exe_path.exists():
        print("Executable not found. Run build first.")
        return False
    
    # Test with JSON input
    test_json = json.dumps(test_data)
    
    try:
        result = subprocess.run([
            str(exe_path),
            "--model-dir", "model_artifacts",  # Adjust path as needed
            "--json", test_json
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Test passed!")
            print("Output:", result.stdout)
            return True
        else:
            print("✗ Test failed!")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

if __name__ == "__main__":
    test_executable()
"""
    
    with open("test_executable.py", "w") as f:
        f.write(test_script)
    
    print("Test script created: test_executable.py")

def main():
    print("JulietteV2 Executable Builder")
    print("=" * 40)
    
    # Check if standalone_predictor.py exists
    if not Path("standalone_predictor.py").exists():
        print("Error: standalone_predictor.py not found!")
        return False
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Build executable
    success = build_executable()
    
    if success:
        # Create test script
        create_test_script()
        
        print("\n" + "=" * 40)
        print("BUILD COMPLETE")
        print("=" * 40)
        print("Next steps:")
        print("1. Copy your model artifacts to 'model_artifacts' directory")
        print("2. Run: python test_executable.py")
        print("3. Copy dist/JulietteV2Predictor.exe to your MT5 directory")
        print("4. Update your MT5 EA to call the executable")
        
    return success

if __name__ == "__main__":
    main()