# JulietteV2 Trading System

## Overview
JulietteV2 is an advanced forex trading system that uses a transformer-based machine learning model to predict market movements. The system analyzes 90-bar sequences with 16 technical features to generate BUY, SELL, or HOLD signals with confidence scores.

## Key Features
- **Transformer Architecture**: State-of-the-art deep learning model for sequence analysis
- **Risk Management**: Built-in 1:2 risk/reward ratio with customizable stop-loss
- **Confidence Filtering**: Only trades when model confidence exceeds threshold
- **MT5 Integration**: Seamless integration with MetaTrader 5 platform

## Setup Instructions

### Prerequisites
1. MetaTrader 5 terminal installed
2. Python 3.8 or higher
3. Your trained model files:
   - `model_best.pt`
   - `scaler.joblib`
   - `label_encoder.joblib`
   - `model_metadata.json`

### Installation Steps

1. **Clone or download this repository**

2. **Place your model files in the project directory**

3. **Build the standalone executable**:
   ```bash
   build_executable.bat
   ```
   This will create `dist/JulietteV2_Predictor.exe`

4. **Copy the executable** to a permanent location, e.g.:
   ```
   C:\Trading\JulietteV2\JulietteV2_Predictor.exe
   ```

5. **Install the MT5 Expert Advisor**:
   - Copy `JulietteV2_Standalone.mq5` to `[MT5_Directory]\MQL5\Experts\`
   - Open MT5 and compile the EA (F7 in MetaEditor)

6. **Configure the EA**:
   - Attach to a chart
   - Set `ExecutablePath` to your predictor location
   - Adjust trading parameters as needed

## Configuration Parameters

- **ExecutablePath**: Full path to the predictor executable
- **LotSize**: Trading volume (default: 0.1)
- **RiskRewardRatio**: Take profit multiplier (default: 2.0)
- **StopLossPips**: Stop loss distance in pips (default: 50)
- **MinConfidence**: Minimum model confidence to trade (default: 0.6)
- **EnableTrading**: Toggle live trading on/off

## Business Model Suggestions

### 1. **Signal Service Subscription** 💰
- **Model**: Monthly/quarterly subscriptions for trading signals
- **Pricing**: $99-299/month based on features
- **Revenue Potential**: 100 subscribers × $199/month = $19,900/month
- **Benefits**: Recurring revenue, scalable, low operational cost
- **Implementation**: 
  - Web dashboard showing signals
  - Email/Telegram alerts
  - Performance tracking

### 2. **Managed Account Service** 💎
- **Model**: Trade client accounts for performance fee
- **Pricing**: 20-30% performance fee + 2% management fee
- **Revenue Potential**: $1M AUM × 20% annual return × 30% fee = $60,000/year
- **Benefits**: High-value clients, performance-based income
- **Requirements**: 
  - Regulatory compliance (CTA registration)
  - Risk management protocols
  - Transparent reporting

### 3. **White Label Solution** 🏢
- **Model**: License the system to brokers/prop firms
- **Pricing**: $10,000-50,000 setup + monthly licensing
- **Revenue Potential**: 5 clients × $25,000 + $2,000/month = $235,000 first year
- **Benefits**: B2B sales, higher ticket value
- **Features**:
  - Custom branding
  - Integration support
  - Performance metrics

### 4. **Educational Platform** 📚
- **Model**: Teach algorithmic trading with your system
- **Pricing**: $497-1,997 course + $97/month community
- **Revenue Potential**: 200 students × $997 + ongoing = $199,400+
- **Benefits**: Knowledge product, community building
- **Content**:
  - ML trading fundamentals
  - System architecture
  - Live trading sessions

### 5. **Copy Trading Platform** 📈
- **Model**: Allow users to copy your trades automatically
- **Pricing**: Performance fee or subscription
- **Revenue Potential**: Volume-based commissions + fees
- **Benefits**: Network effects, platform economics
- **Tech Stack**:
  - Trade copier software
  - User dashboard
  - Risk controls

### 6. **Proprietary Trading** 🚀
- **Model**: Trade your own capital
- **Revenue**: Direct trading profits
- **Scaling**: Seek prop firm funding or investors
- **Benefits**: Keep 100% of profits (minus funding costs)
- **Path**:
  - Build track record
  - Apply to prop firms
  - Scale with OPM (Other People's Money)

## Recommended Monetization Strategy

**Phase 1 (Months 1-3)**: Start with proprietary trading to build track record
- Trade small account to verify live performance
- Document all trades and statistics
- Refine system based on real market conditions

**Phase 2 (Months 4-6)**: Launch signal service
- Create simple website with performance stats
- Start with beta users at discounted price
- Build email list and social proof

**Phase 3 (Months 7-12)**: Expand offerings
- Add educational content
- Explore managed accounts (if compliant)
- Consider white label opportunities

**Phase 4 (Year 2+)**: Scale and diversify
- Build copy trading platform
- Seek institutional partnerships
- Expand to multiple markets/timeframes

## Risk Disclaimers
- Past performance does not guarantee future results
- Trading forex involves substantial risk of loss
- Only trade with capital you can afford to lose
- Ensure compliance with local regulations

## Technical Support

### Common Issues

1. **"Executable not found" error**:
   - Verify the path in EA settings
   - Ensure executable has proper permissions

2. **"Failed to get prediction" error**:
   - Check if model files are bundled correctly
   - Verify Python dependencies

3. **No trades executing**:
   - Check MinConfidence setting
   - Verify EnableTrading is true
   - Review MT5 journal for errors

### Performance Optimization

- Run backtests on historical data
- Optimize confidence threshold
- Adjust position sizing for risk management
- Monitor drawdown and win rate

## License
Proprietary - All rights reserved

---

**Ready to revolutionize your trading? Let's make those profits! 🚀💰**