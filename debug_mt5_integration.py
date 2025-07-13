import sys
import os
import json
from pathlib import Path

# Create a debug log in the MT5 Files directory
mt5_files_dir = Path(os.getenv('APPDATA')) / 'MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files'
debug_log = mt5_files_dir / 'mt5_python_debug.json'

debug_info = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "working_directory": os.getcwd(),
    "python_path": sys.path,
    "environment_vars": dict(os.environ),
    "package_imports": {},
    "file_checks": {}
}

# Test package imports
packages = ['torch', 'sklearn', 'pandas', 'numpy', 'joblib']
for pkg in packages:
    try:
        __import__(pkg)
        debug_info['package_imports'][pkg] = "SUCCESS"
    except Exception as e:
        debug_info['package_imports'][pkg] = f"FAILED: {str(e)}"

# Check model files
model_files = ['model_best.pt', 'scaler.joblib', 'label_encoder.joblib', 'model_metadata.json']
for file in model_files:
    file_path = mt5_files_dir / file
    debug_info['file_checks'][file] = {
        "exists": file_path.exists(),
        "size": file_path.stat().st_size if file_path.exists() else 0,
        "readable": os.access(file_path, os.R_OK) if file_path.exists() else False
    }

# Write debug info
with open(debug_log, 'w') as f:
    json.dump(debug_info, f, indent=2)

print(f"Debug info written to: {debug_log}") 