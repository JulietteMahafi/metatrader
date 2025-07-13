#!/usr/bin/env python3
"""
Cleanup script for JulietteV2 project
Removes failed approach files and organizes the workspace
"""

import os
import shutil
from pathlib import Path

def main():
    print("=== Cleaning up JulietteV2 Project ===")
    
    # Files to remove (failed approaches)
    files_to_remove = [
        "FIXED_model_server.py",
        "debug_mt5_integration.py", 
        "input.json",
        "input.txt"
    ]
    
    # Create backup directory
    backup_dir = Path("legacy_backup")
    backup_dir.mkdir(exist_ok=True)
    
    print("Moving failed approach files to legacy_backup/...")
    
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                shutil.move(str(file_path), str(backup_dir / file_name))
                print(f"  Moved: {file_name}")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
        else:
            print(f"  Not found: {file_name}")
    
    # Create organized directory structure
    print("\nCreating organized directory structure...")
    
    dirs_to_create = [
        "mt5_files",
        "model_artifacts", 
        "docs"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"  Created: {dir_name}/")
    
    # Move documentation files
    docs_to_move = [
        "FAILED_APPROACHES_SUMMARY.md",
        "Full_prompt.txt"
    ]
    
    for doc_name in docs_to_move:
        doc_path = Path(doc_name)
        if doc_path.exists():
            try:
                shutil.move(str(doc_path), str(Path("docs") / doc_name))
                print(f"  Moved to docs/: {doc_name}")
            except Exception as e:
                print(f"  Error moving {doc_name}: {e}")
    
    # Create .gitignore for Python artifacts
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# PyTorch
*.pt
*.pth

# MT5
*.ex5
*.mq5~

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("  Created: .gitignore")
    
    # Create requirements.txt
    requirements_content = """torch>=2.0.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.21.0
joblib>=1.1.0
pyinstaller>=5.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("  Created: requirements.txt")
    
    print("\n=== Cleanup Complete ===")
    print("\nCurrent project structure:")
    print("├── JulietteV2_Standalone.mq5    # Main MT5 EA")
    print("├── standalone_predictor.py       # ML inference engine")
    print("├── build_standalone.py           # Build script")
    print("├── cleanup.py                    # This script")
    print("├── JAson.mqh                     # JSON library")
    print("├── README.md                     # Documentation")
    print("├── requirements.txt              # Python dependencies")
    print("├── .gitignore                    # Git ignore rules")
    print("├── legacy_backup/                # Failed approaches")
    print("├── mt5_files/                    # MT5 files directory")
    print("├── model_artifacts/              # Model files")
    print("└── docs/                         # Documentation")
    
    print("\nNext steps:")
    print("1. Run: python build_standalone.py")
    print("2. Copy model files to mt5_files/")
    print("3. Install EA in MT5")
    print("4. Test the system")

if __name__ == "__main__":
    main()