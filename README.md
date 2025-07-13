# JulietteV2 MetaTrader 5 Integration

## 🎯 Project Overview
Automated forex trading system using PyTorch Transformer model integrated with MetaTrader 5. Features R/R 1:2 risk management (10 pips SL, 20 pips TP).

## 🚀 Current Working Solution: Standalone Executable

### Architecture
- **MT5 Expert Advisor**: `JulietteV2_Standalone.mq5` - Handles trading logic and feature generation
- **Standalone Predictor**: `standalone_predictor.py` - ML inference engine (compiled to .exe)
- **Communication**: File-based JSON communication between MT5 and Python

### Key Features
- ✅ **No Python Environment Issues** - Standalone executable includes all dependencies
- ✅ **R/R 1:2 Risk Management** - 10 pips SL, 20 pips TP (non-negotiable)
- ✅ **Real-time Signal Generation** - Processes new bars automatically
- ✅ **Local Execution Only** - No external connections required

## 📁 File Structure

### Core Files (Keep These)
```
├── JulietteV2_Standalone.mq5      # Main MT5 Expert Advisor
├── standalone_predictor.py         # ML inference engine
├── build_standalone.py            # Build script for executable
├── JAson.mqh                      # JSON library for MT5
├── README.md                      # This file
└── model_artifacts/               # Model files (place in MT5 Files directory)
    ├── model_best.pt
    ├── scaler.joblib
    ├── label_encoder.joblib
    └── model_metadata.json
```

### Legacy Files (Can Be Removed)
```
├── FIXED_model_server.py          # Failed file-based server approach
├── _3.py                          # Training script (keep for reference)
├── debug_mt5_integration.py       # Debug script (no longer needed)
├── FAILED_APPROACHES_SUMMARY.md   # Documentation of failed approaches
├── Full_prompt.txt                # Original project requirements
├── input.json                     # Test data
└── input.txt                      # Test data
```

## 🛠️ Setup Instructions

### 1. Build the Standalone Executable
```bash
python build_standalone.py
```
This will:
- Install PyInstaller if needed
- Compile `standalone_predictor.py` to `standalone_predictor.exe`
- Copy the executable to your MT5 Files directory

### 2. Prepare Model Files
Copy your model artifacts to the MT5 Files directory:
```
%APPDATA%\MetaQuotes\Terminal\Common\MQL5\Files\
├── standalone_predictor.exe
├── model_best.pt
├── scaler.joblib
├── label_encoder.joblib
└── model_metadata.json
```

### 3. Install in MetaTrader 5
1. Copy `JulietteV2_Standalone.mq5` to your MT5 Experts directory
2. Copy `JAson.mqh` to your MT5 Include directory
3. Compile the Expert Advisor in MetaEditor
4. Attach to a chart

## ⚙️ Configuration

### Expert Advisor Parameters
- **StopLoss**: 10 pips (default)
- **TakeProfit**: 20 pips (default, R/R 1:2)
- **LotSize**: 0.1 (adjust as needed)
- **MagicNumber**: 12345 (unique identifier for trades)
- **EnableTrading**: true/false (for testing)
- **PredictorPath**: Path to executable (auto-detected if empty)

### Model Specifications
- **Input**: 90-bar sequences with 16 features
- **Features**: OHLC, time features, candle patterns, technical indicators
- **Output**: 3 classes (BUY, KEEP, SELL)
- **Architecture**: Transformer (512-dim, 8-head, 6-layer)

## 🔧 Troubleshooting

### Common Issues

1. **"Predictor executable not found"**
   - Run `build_standalone.py` to create the executable
   - Ensure it's in the MT5 Files directory

2. **"Model file not found"**
   - Copy all model artifacts to MT5 Files directory
   - Check file names match exactly

3. **"ShellExecute failed"**
   - Check Windows permissions
   - Ensure executable path is correct
   - Try running executable manually first

4. **"Response timeout"**
   - Increase timeout in EA code (default: 5 seconds)
   - Check if antivirus is blocking execution

### Debug Mode
Set `EnableTrading = false` to test predictions without placing trades.

## 📊 Performance Notes

### Current Model Performance
- **Validation Accuracy**: ~40%
- **F1 Score**: 0.36
- **Status**: Not yet profitable

### Improvement Suggestions
1. **Feature Engineering**: Add more technical indicators
2. **Data Quality**: Use higher quality data sources
3. **Model Architecture**: Try different architectures (LSTM, GRU)
4. **Hyperparameter Tuning**: Optimize model parameters
5. **Ensemble Methods**: Combine multiple models

## 💡 Profitable Model Suggestions

### 1. **Multi-Timeframe Analysis**
- Combine signals from multiple timeframes (M1, M5, M15, H1)
- Weight signals based on timeframe importance

### 2. **Market Regime Detection**
- Identify trending vs ranging markets
- Use different strategies for different regimes

### 3. **Sentiment Integration**
- Add news sentiment analysis
- Include economic calendar events

### 4. **Risk-Adjusted Position Sizing**
- Scale position size based on signal confidence
- Use Kelly Criterion for optimal sizing

### 5. **Alternative Models**
- **LSTM/GRU**: Better for sequential data
- **Random Forest**: More interpretable
- **XGBoost**: Often better performance
- **Ensemble**: Combine multiple models

## 🚨 Important Notes

- **Risk Management**: Always use proper position sizing
- **Backtesting**: Test thoroughly before live trading
- **Monitoring**: Monitor system performance continuously
- **Updates**: Keep model and system updated

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review MT5 logs for error messages
3. Test components individually
4. Ensure all dependencies are properly installed

---

**Status**: ✅ Working solution implemented  
**Next Steps**: Improve model performance for profitability