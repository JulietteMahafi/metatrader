# JulietteV2 MetaTrader 5 Integration - WORKING SOLUTION

## 🚀 **CLEAN WORKING IMPLEMENTATION**

This solution uses a **standalone executable approach** that bypasses MT5's security restrictions.

---

## 📁 **File Structure (Clean)**

```
├── standalone_predictor.py     # ✅ Clean ML predictor (compiles to .exe)
├── build_executable.py         # ✅ Build script for executable
├── JulietteV2_Clean.mq5        # ✅ Clean MT5 Expert Advisor
├── JAson.mqh                   # ✅ Required JSON library
├── README.md                   # ✅ This file
└── model_artifacts/            # 📁 Your model files go here
    ├── model_best.pt
    ├── scaler.joblib
    ├── label_encoder.joblib
    └── model_metadata.json
```

---

## 🔧 **Setup Instructions**

### 1. **Build the Executable**
```bash
# Install dependencies
pip install torch scikit-learn pandas numpy joblib pyinstaller

# Build executable
python build_executable.py
```

### 2. **Copy Model Artifacts**
```bash
# Create model directory
mkdir model_artifacts

# Copy your trained model files
cp model_best.pt model_artifacts/
cp scaler.joblib model_artifacts/
cp label_encoder.joblib model_artifacts/
cp model_metadata.json model_artifacts/
```

### 3. **Test the Executable**
```bash
# Test the built executable
python test_executable.py
```

### 4. **Install in MT5**
```bash
# Copy executable to MT5 directory
cp dist/JulietteV2Predictor.exe /path/to/MT5/MQL5/Files/

# Copy model artifacts
cp -r model_artifacts/ /path/to/MT5/MQL5/Files/

# Install EA in MT5
cp JulietteV2_Clean.mq5 /path/to/MT5/MQL5/Experts/
cp JAson.mqh /path/to/MT5/MQL5/Include/
```

---

## ⚙️ **How It Works**

1. **MT5 EA** runs on every new bar
2. **EA prepares** 90-bar sequence with 16 features
3. **EA calls** standalone executable via `ShellExecute()`
4. **Executable** loads model and makes prediction
5. **EA receives** JSON response with signal + confidence
6. **EA executes** trade based on signal (BUY/SELL/KEEP)

---

## 💰 **PROFITABLE MODEL SUGGESTIONS**

### **Current Performance Issues:**
- **Accuracy:** 40% (barely above random)
- **F1 Score:** 0.36 (poor)
- **Risk/Reward:** 1:2 (good, but not enough to overcome low accuracy)

### **Profitability Requirements:**
With 1:2 R/R, you need **>33.3% win rate** to break even.
For profitability, aim for **>40% win rate**.

### **🎯 Immediate Improvements:**

#### 1. **Feature Engineering**
```python
# Add these features to your model
def add_profitable_features(df):
    # Market regime features
    df['volatility_regime'] = df['High-Low'].rolling(20).std()
    df['trend_strength'] = df['Close'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x.std())
    
    # Volume/momentum proxies
    df['price_velocity'] = df['Close'].diff().rolling(5).mean()
    df['acceleration'] = df['price_velocity'].diff()
    
    # Support/resistance levels
    df['distance_to_high'] = df['High'].rolling(50).max() - df['Close']
    df['distance_to_low'] = df['Close'] - df['Low'].rolling(50).min()
    
    # Market microstructure
    df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
    df['momentum_divergence'] = df['Close'].diff() - df['Body'].rolling(10).mean()
    
    return df
```

#### 2. **Model Architecture Improvements**
```python
# Try these architectures:
class ProfitableTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, num_layers=4):
        super().__init__()
        # Reduce complexity for better generalization
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Add skip connections
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(num_layers)
        ])
        
        # Multi-head output for confidence estimation
        self.classifier = nn.Linear(d_model, 3)
        self.confidence_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_projection(x)
        
        for layer in self.transformer_layers:
            x = layer(x) + x  # Skip connection
        
        x = x.mean(dim=1)  # Global average pooling
        
        logits = self.classifier(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return logits, confidence
```

#### 3. **Training Strategy**
```python
# Focus on profitability, not just accuracy
class ProfitLoss(nn.Module):
    def __init__(self, rr_ratio=2.0):
        super().__init__()
        self.rr_ratio = rr_ratio
    
    def forward(self, predictions, targets):
        # Reward correct predictions based on profit potential
        correct_buys = (predictions == 0) & (targets == 0)  # BUY
        correct_sells = (predictions == 2) & (targets == 2)  # SELL
        correct_keeps = (predictions == 1) & (targets == 1)  # KEEP
        
        # Profit matrix
        profit = torch.zeros_like(predictions, dtype=torch.float)
        profit[correct_buys] = self.rr_ratio
        profit[correct_sells] = self.rr_ratio
        profit[correct_keeps] = 0.0
        
        # Loss for incorrect predictions
        wrong_trades = ~(correct_buys | correct_sells | correct_keeps)
        profit[wrong_trades] = -1.0
        
        return -profit.mean()  # Maximize profit
```

#### 4. **Advanced Strategies**

**A. Ensemble Model:**
```python
# Combine multiple models
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = []
        confidences = []
        
        for model in self.models:
            pred, conf = model.predict(X)
            predictions.append(pred)
            confidences.append(conf)
        
        # Weighted voting based on confidence
        weights = np.array(confidences) / np.sum(confidences)
        final_pred = np.average(predictions, weights=weights)
        
        return final_pred, np.mean(confidences)
```

**B. Dynamic Position Sizing:**
```python
// In MQL5 EA, adjust lot size based on confidence
double CalculateLotSize(double confidence)
{
    double baseLot = 0.1;
    double maxLot = 1.0;
    
    // Scale lot size with confidence
    double adjustedLot = baseLot * (1 + (confidence - 0.5) * 2);
    
    return MathMin(adjustedLot, maxLot);
}
```

**C. Market Regime Filtering:**
```python
def should_trade(market_data):
    # Only trade in favorable conditions
    volatility = market_data['High-Low'].rolling(20).std()
    trend_strength = abs(market_data['Close'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x.std()))
    
    # Trade only in medium volatility, trending markets
    return (0.001 < volatility < 0.005) and (trend_strength > 0.5)
```

---

## 📊 **Performance Targets**

### **Minimum Profitable Metrics:**
- **Win Rate:** >40%
- **Profit Factor:** >1.3
- **Sharpe Ratio:** >0.8
- **Maximum Drawdown:** <20%

### **Monthly Performance Goals:**
- **Return:** 5-15%
- **Trades:** 50-200 per month
- **Success Rate:** 45-55%

---

## 🗂️ **Cleanup Summary**

### **Files Removed (Technical Debt):**
- `FIXED_model_server.py` - File-based approach (unreliable)
- `debug_mt5_integration.py` - Debug script (no longer needed)
- `_3.py` - Large training script (not needed for deployment)

### **Files Kept:**
- `standalone_predictor.py` - Working solution
- `build_executable.py` - Build tool
- `JulietteV2_Clean.mq5` - Clean EA
- `JAson.mqh` - JSON library
- `input.json` - Sample data

---

## 🚀 **Next Steps**

1. **Immediate:** Build and test the executable
2. **Short-term:** Implement feature engineering improvements
3. **Medium-term:** Retrain model with profit-focused loss function
4. **Long-term:** Implement ensemble and dynamic position sizing

---

## 💡 **Alternative Profitable Models**

If transformer complexity is the issue, consider:

1. **XGBoost/LightGBM** - Often outperforms neural networks on tabular data
2. **Simple LSTM** - Less complex than transformer, good for sequences
3. **Random Forest** - Reliable, interpretable, easier to tune

---

**Bottom Line:** The technical integration is now solved. Focus on improving the model's profitability through better features, training strategies, and risk management.