import sys
import os
import json
from pathlib import Path

# -- Resolve MT5 Files directory cross-platform --
# • On Windows with APPDATA available: use the real MT5 path (Terminal-specific or Common).
# • Otherwise (Linux / macOS / missing APPDATA): fall back to ./comm_dir next to this script

if os.name == 'nt' and os.getenv('APPDATA'):
    appdata_dir: str = os.getenv('APPDATA')  # type: ignore[assignment]
    # If you need a terminal-specific folder, replace below; Common is fine for dev
    mt5_files_dir = Path(appdata_dir) / 'MetaQuotes/Terminal/Common/MQL5/Files'
else:
    print("[WARN] APPDATA not found or non-Windows OS detected – using local 'comm_dir' directory.")
    mt5_files_dir = Path(__file__).resolve().parent / 'comm_dir'

mt5_files_dir.mkdir(parents=True, exist_ok=True)

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