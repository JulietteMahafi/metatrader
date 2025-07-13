# JulietteV2 Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Prepare Model Files
Place these files in the project directory:
- `model_best.pt` (your trained model)
- `scaler.joblib` (data scaler)
- `label_encoder.joblib` (label encoder)
- `model_metadata.json` (model configuration)

### Step 2: Test the System
```bash
python test_predictor.py
```

### Step 3: Build Executable
```bash
build_executable.bat
```

### Step 4: Install in MT5
1. Copy `dist/JulietteV2_Predictor.exe` to `C:\Trading\JulietteV2\`
2. Copy `JulietteV2_Standalone.mq5` to MT5's Expert folder
3. Compile the EA in MetaEditor (F7)

### Step 5: Configure & Run
1. Attach EA to a chart
2. Set the executable path in EA settings
3. Enable auto-trading
4. Watch the profits roll in! 💰

## 📊 Quick Business Model

**Fastest Path to Profit:**
1. **Week 1-2**: Test with small live account ($500-1000)
2. **Week 3-4**: Document performance metrics
3. **Month 2**: Launch signal service at $99/month
4. **Month 3**: Scale to 50+ subscribers = $5000/month

## ⚡ Troubleshooting

**"Model files not found"**
→ Run `python test_predictor.py` to check files

**"Executable not found in MT5"**
→ Check the path in EA settings matches your .exe location

**"No trades executing"**
→ Lower MinConfidence setting (try 0.5)
→ Check if market is open

## 📞 Need Help?

Check the full README.md for detailed instructions and business model options.

---
*Ready to trade? Let's make this profitable! 🎯*