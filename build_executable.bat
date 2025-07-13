@echo off
echo Building JulietteV2 Standalone Predictor...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required packages
echo Installing required packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn joblib pyinstaller

REM Build executable
echo.
echo Building executable...
pyinstaller --onefile ^
    --hidden-import sklearn.preprocessing._data ^
    --hidden-import sklearn.utils._typedefs ^
    --hidden-import sklearn.neighbors._partition_nodes ^
    --add-data "model_best.pt;." ^
    --add-data "scaler.joblib;." ^
    --add-data "label_encoder.joblib;." ^
    --add-data "model_metadata.json;." ^
    --name JulietteV2_Predictor ^
    standalone_predictor.py

echo.
if exist dist\JulietteV2_Predictor.exe (
    echo SUCCESS: Executable created at dist\JulietteV2_Predictor.exe
    echo.
    echo Next steps:
    echo 1. Copy dist\JulietteV2_Predictor.exe to your desired location
    echo 2. Update the ExecutablePath in JulietteV2_Standalone.mq5
    echo 3. Copy JulietteV2_Standalone.mq5 to MT5\MQL5\Experts\
    echo 4. Compile and run the EA in MT5
) else (
    echo ERROR: Failed to create executable
)

echo.
pause