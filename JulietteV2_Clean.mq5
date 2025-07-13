//+------------------------------------------------------------------+
//|                                             JulietteV2_Clean.mq5 |
//|                                  Clean MT5 EA for JulietteV2 ML |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "JulietteV2 Trading System"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <JAson.mqh>

//--- Input parameters
input double LotSize = 0.1;                    // Lot size
input int    StopLoss = 10;                     // Stop loss in pips
input int    TakeProfit = 20;                   // Take profit in pips (R/R 1:2)
input int    SequenceLength = 90;               // Sequence length for prediction
input double MinConfidence = 0.6;              // Minimum confidence for trade
input string ExecutablePath = "JulietteV2Predictor.exe";  // Path to executable
input string ModelDir = "model_artifacts";      // Model artifacts directory
input bool   EnableLogging = true;              // Enable detailed logging

//--- Global variables
CTrade trade;
datetime lastBarTime = 0;
string logFile = "JulietteV2_EA.log";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Log EA start
    if(EnableLogging)
    {
        string msg = StringFormat("JulietteV2 EA Started - %s", TimeToString(TimeCurrent()));
        Print(msg);
        LogToFile(msg);
    }
    
    // Validate inputs
    if(TakeProfit != StopLoss * 2)
    {
        Print("ERROR: TakeProfit must be exactly 2x StopLoss for R/R 1:2");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    if(LotSize <= 0)
    {
        Print("ERROR: LotSize must be positive");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    // Check if executable exists
    if(!FileIsExist(ExecutablePath))
    {
        Print("ERROR: Executable not found: ", ExecutablePath);
        return INIT_FAILED;
    }
    
    Print("JulietteV2 EA initialized successfully");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    string msg = StringFormat("JulietteV2 EA Stopped - Reason: %d", reason);
    Print(msg);
    if(EnableLogging) LogToFile(msg);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if new bar
    if(Time[0] == lastBarTime)
        return;
    
    lastBarTime = Time[0];
    
    // Get prediction
    string prediction = GetMLPrediction();
    
    if(prediction == "")
    {
        if(EnableLogging) LogToFile("No prediction received");
        return;
    }
    
    // Parse prediction and execute trade
    ExecuteTradeFromPrediction(prediction);
}

//+------------------------------------------------------------------+
//| Get ML prediction using standalone executable                    |
//+------------------------------------------------------------------+
string GetMLPrediction()
{
    // Prepare input data
    string inputJson = PrepareInputData();
    if(inputJson == "")
    {
        Print("ERROR: Failed to prepare input data");
        return "";
    }
    
    // Create temporary files
    string inputFile = "ml_input_" + IntegerToString(GetTickCount()) + ".json";
    string outputFile = "ml_output_" + IntegerToString(GetTickCount()) + ".json";
    
    // Write input to file
    int handle = FileOpen(inputFile, FILE_WRITE|FILE_TXT);
    if(handle == INVALID_HANDLE)
    {
        Print("ERROR: Cannot create input file");
        return "";
    }
    
    FileWriteString(handle, inputJson);
    FileClose(handle);
    
    // Build command
    string command = StringFormat("%s --model-dir %s --input %s --output %s",
                                  ExecutablePath, ModelDir, inputFile, outputFile);
    
    // Execute command
    int result = ShellExecuteW(0, "open", "cmd.exe", "/c " + command, NULL, 0);
    
    if(result <= 32)
    {
        Print("ERROR: Failed to execute ML predictor. Error code: ", result);
        FileDelete(inputFile);
        return "";
    }
    
    // Wait for output file (max 10 seconds)
    int timeout = 100; // 10 seconds in 100ms increments
    while(timeout > 0 && !FileIsExist(outputFile))
    {
        Sleep(100);
        timeout--;
    }
    
    if(!FileIsExist(outputFile))
    {
        Print("ERROR: ML prediction timed out");
        FileDelete(inputFile);
        return "";
    }
    
    // Read output
    handle = FileOpen(outputFile, FILE_READ|FILE_TXT);
    if(handle == INVALID_HANDLE)
    {
        Print("ERROR: Cannot read output file");
        FileDelete(inputFile);
        FileDelete(outputFile);
        return "";
    }
    
    string output = FileReadString(handle);
    FileClose(handle);
    
    // Clean up
    FileDelete(inputFile);
    FileDelete(outputFile);
    
    return output;
}

//+------------------------------------------------------------------+
//| Prepare input data for ML model                                 |
//+------------------------------------------------------------------+
string PrepareInputData()
{
    if(Bars < SequenceLength)
    {
        Print("ERROR: Not enough bars for prediction");
        return "";
    }
    
    CJAson json;
    CJAson *features = new CJAson();
    features.m_type = 1; // Array type
    
    // Create feature array for the last SequenceLength bars
    for(int i = SequenceLength - 1; i >= 0; i--)
    {
        CJAson *bar = new CJAson();
        
        // OHLC features
        bar.Set("Open", Open[i]);
        bar.Set("High", High[i]);
        bar.Set("Low", Low[i]);
        bar.Set("Close", Close[i]);
        
        // Derived features
        double body = Close[i] - Open[i];
        double highLow = High[i] - Low[i];
        double gap = (i < Bars - 1) ? Open[i] - Close[i + 1] : 0.0;
        
        bar.Set("Body", body);
        bar.Set("High-Low", highLow);
        bar.Set("gap", gap);
        
        // Pattern features
        bool isDoji = (MathAbs(body) < highLow * 0.1);
        bool isLongShadow = ((High[i] - MathMax(Open[i], Close[i])) > body * 2) ||
                           ((MathMin(Open[i], Close[i]) - Low[i]) > body * 2);
        bool isSpike = (highLow > body * 5);
        
        bar.Set("Is_Doji", isDoji ? 1 : 0);
        bar.Set("Is_Long_Shadow", isLongShadow ? 1 : 0);
        bar.Set("Is_Spike", isSpike ? 1 : 0);
        
        // Time features
        MqlDateTime dt;
        TimeToStruct(Time[i], dt);
        
        double hourAngle = 2 * M_PI * dt.hour / 24;
        double minuteAngle = 2 * M_PI * dt.min / 60;
        double dayAngle = 2 * M_PI * dt.day_of_week / 7;
        
        bar.Set("hour_cos", MathCos(hourAngle));
        bar.Set("hour_sin", MathSin(hourAngle));
        bar.Set("minute_cos", MathCos(minuteAngle));
        bar.Set("minute_sin", MathSin(minuteAngle));
        bar.Set("day_cos", MathCos(dayAngle));
        bar.Set("day_sin", MathSin(dayAngle));
        
        features.Add(bar);
    }
    
    json.Set("features", features);
    return json.Serialize();
}

//+------------------------------------------------------------------+
//| Execute trade based on ML prediction                            |
//+------------------------------------------------------------------+
void ExecuteTradeFromPrediction(string predictionJson)
{
    CJAson json;
    if(!json.Deserialize(predictionJson))
    {
        Print("ERROR: Failed to parse prediction JSON");
        return;
    }
    
    // Check for error in prediction
    if(json["error"] != NULL)
    {
        Print("ERROR: ML prediction error: ", json["error"].ToStr());
        return;
    }
    
    string signal = json["signal"].ToStr();
    double confidence = json["confidence"].ToDbl();
    
    if(EnableLogging)
    {
        string msg = StringFormat("ML Prediction - Signal: %s, Confidence: %.3f", signal, confidence);
        LogToFile(msg);
    }
    
    // Check confidence threshold
    if(confidence < MinConfidence)
    {
        if(EnableLogging) LogToFile("Confidence below threshold, skipping trade");
        return;
    }
    
    // Close existing positions first
    CloseAllPositions();
    
    // Calculate SL and TP
    double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    double sl_distance = StopLoss * point * 10;
    double tp_distance = TakeProfit * point * 10;
    
    double price, sl, tp;
    
    // Execute trade based on signal
    if(signal == "BUY")
    {
        price = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
        sl = price - sl_distance;
        tp = price + tp_distance;
        
        if(trade.Buy(LotSize, Symbol(), price, sl, tp, "JulietteV2 BUY"))
        {
            string msg = StringFormat("BUY order executed: Price=%.5f, SL=%.5f, TP=%.5f", price, sl, tp);
            Print(msg);
            if(EnableLogging) LogToFile(msg);
        }
    }
    else if(signal == "SELL")
    {
        price = SymbolInfoDouble(Symbol(), SYMBOL_BID);
        sl = price + sl_distance;
        tp = price - tp_distance;
        
        if(trade.Sell(LotSize, Symbol(), price, sl, tp, "JulietteV2 SELL"))
        {
            string msg = StringFormat("SELL order executed: Price=%.5f, SL=%.5f, TP=%.5f", price, sl, tp);
            Print(msg);
            if(EnableLogging) LogToFile(msg);
        }
    }
    // KEEP signal = no trade
}

//+------------------------------------------------------------------+
//| Close all positions                                             |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetString(POSITION_SYMBOL) == Symbol())
            {
                trade.PositionClose(PositionGetTicket(i));
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Log message to file                                             |
//+------------------------------------------------------------------+
void LogToFile(string message)
{
    int handle = FileOpen(logFile, FILE_WRITE|FILE_READ|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        FileSeek(handle, 0, SEEK_END);
        FileWriteString(handle, TimeToString(TimeCurrent()) + " - " + message + "\n");
        FileClose(handle);
    }
}