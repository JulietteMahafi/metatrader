#!/usr/bin/env python3
"""
Build script for standalone MT5 predictor
Compiles standalone_predictor.py to executable using PyInstaller
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=== Building Standalone MT5 Predictor ===")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Build command
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable
        "--noconsole",  # No console window
        "--hidden-import", "sklearn.preprocessing._data",
        "--hidden-import", "sklearn.preprocessing._label",
        "--hidden-import", "torch.nn.modules.transformer",
        "--hidden-import", "torch.nn.modules.activation",
        "--hidden-import", "torch.nn.modules.normalization",
        "--hidden-import", "torch.nn.modules.linear",
        "--hidden-import", "torch.nn.modules.dropout",
        "--hidden-import", "torch.nn.functional",
        "--hidden-import", "torch.serialization",
        "--hidden-import", "_pickle",
        "--name", "standalone_predictor",
        "standalone_predictor.py"
    ]
    
    print("Building executable...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        print("Output:", result.stdout)
        
        # Check if executable was created
        exe_path = Path("dist/standalone_predictor.exe")
        if exe_path.exists():
            print(f"Executable created: {exe_path.absolute()}")
            print(f"Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Copy to MT5 Files directory if APPDATA is available
            appdata_dir = os.getenv('APPDATA')
            if appdata_dir:
                mt5_files_dir = Path(appdata_dir) / "MetaQuotes/Terminal/Common/MQL5/Files"
                mt5_files_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = mt5_files_dir / "standalone_predictor.exe"
                import shutil
                shutil.copy2(exe_path, target_path)
                print(f"Copied to MT5 Files directory: {target_path}")
            else:
                print("APPDATA not found. Please manually copy the executable to your MT5 Files directory.")
        else:
            print("ERROR: Executable not found after build")
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    
    print("=== Build Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())