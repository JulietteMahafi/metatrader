//+------------------------------------------------------------------+
//|                                          JulietteV2_Standalone.mq5|
//|                                     JulietteV2 Standalone EA      |
//|                       Using executable for ML predictions         |
//+------------------------------------------------------------------+
#property copyright "JulietteV2 Trading System"
#property version   "1.0"
#property strict

//--- Input parameters
input string   ExecutablePath = "C:\\Trading\\JulietteV2\\predictor.exe";  // Path to predictor executable
input double   LotSize = 0.1;           // Trade lot size
input double   RiskRewardRatio = 2.0;   // Risk/Reward ratio (1:2 default)
input int      StopLossPips = 50;       // Stop loss in pips
input double   MinConfidence = 0.6;     // Minimum confidence to take trade
input int      MagicNumber = 123456;    // Magic number for orders
input bool     EnableTrading = true;    // Enable live trading

//--- Global variables
datetime lastBarTime = 0;
string tempDir;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Get temporary directory for file communication
   tempDir = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";
   
   // Check if executable exists
   if(!FileIsExist(ExecutablePath))
   {
      Print("ERROR: Predictor executable not found at: ", ExecutablePath);
      Print("Please compile standalone_predictor.py and place at specified location");
      return(INIT_FAILED);
   }
   
   Print("JulietteV2 Standalone EA initialized");
   Print("Executable path: ", ExecutablePath);
   Print("Temp directory: ", tempDir);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Cleanup temporary files
   string inputFile = tempDir + "mt5_input.json";
   string outputFile = tempDir + "mt5_output.json";
   
   if(FileIsExist(inputFile)) FileDelete(inputFile);
   if(FileIsExist(outputFile)) FileDelete(outputFile);
   
   Print("JulietteV2 EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;
   
   // Collect last 90 bars of data
   MqlRates rates[];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 90, rates) != 90)
   {
      Print("Failed to copy 90 bars of data");
      return;
   }
   
   // Prepare features
   string features = PrepareFeatures(rates);
   
   // Get prediction
   string signal;
   double confidence;
   if(!GetPrediction(features, signal, confidence))
   {
      Print("Failed to get prediction");
      return;
   }
   
   Print("Prediction: ", signal, " (Confidence: ", confidence, ")");
   
   // Execute trade if enabled and confidence is high enough
   if(EnableTrading && confidence >= MinConfidence)
   {
      ExecuteTrade(signal);
   }
}

//+------------------------------------------------------------------+
//| Prepare features for model                                       |
//+------------------------------------------------------------------+
string PrepareFeatures(MqlRates &rates[])
{
   string json = "{\"features\":[";
   
   for(int i = 0; i < ArraySize(rates); i++)
   {
      if(i > 0) json += ",";
      
      // Calculate features
      double body = rates[i].close - rates[i].open;
      double highLow = rates[i].high - rates[i].low;
      int isDoji = (MathAbs(body) < highLow * 0.1) ? 1 : 0;
      int isLongShadow = ((rates[i].high - MathMax(rates[i].open, rates[i].close)) > body * 2 ||
                          (MathMin(rates[i].open, rates[i].close) - rates[i].low) > body * 2) ? 1 : 0;
      int isSpike = (highLow > Average(rates, i, 20) * 2) ? 1 : 0;
      double gap = (i > 0) ? rates[i].open - rates[i-1].close : 0;
      
      // Time features
      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);
      double hourAngle = 2 * M_PI * dt.hour / 24;
      double dayAngle = 2 * M_PI * dt.day_of_week / 7;
      double minuteAngle = 2 * M_PI * dt.min / 60;
      
      json += "{";
      json += "\"Body\":" + DoubleToString(body, 5) + ",";
      json += "\"Close\":" + DoubleToString(rates[i].close, 5) + ",";
      json += "\"High\":" + DoubleToString(rates[i].high, 5) + ",";
      json += "\"High-Low\":" + DoubleToString(highLow, 5) + ",";
      json += "\"Is_Doji\":" + IntegerToString(isDoji) + ",";
      json += "\"Is_Long_Shadow\":" + IntegerToString(isLongShadow) + ",";
      json += "\"Is_Spike\":" + IntegerToString(isSpike) + ",";
      json += "\"Low\":" + DoubleToString(rates[i].low, 5) + ",";
      json += "\"Open\":" + DoubleToString(rates[i].open, 5) + ",";
      json += "\"day_cos\":" + DoubleToString(MathCos(dayAngle), 5) + ",";
      json += "\"day_sin\":" + DoubleToString(MathSin(dayAngle), 5) + ",";
      json += "\"gap\":" + DoubleToString(gap, 5) + ",";
      json += "\"hour_cos\":" + DoubleToString(MathCos(hourAngle), 5) + ",";
      json += "\"hour_sin\":" + DoubleToString(MathSin(hourAngle), 5) + ",";
      json += "\"minute_cos\":" + DoubleToString(MathCos(minuteAngle), 5) + ",";
      json += "\"minute_sin\":" + DoubleToString(MathSin(minuteAngle), 5);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Calculate average for spike detection                           |
//+------------------------------------------------------------------+
double Average(MqlRates &rates[], int current, int period)
{
   double sum = 0;
   int count = 0;
   
   for(int i = MathMax(0, current - period); i < current; i++)
   {
      sum += rates[i].high - rates[i].low;
      count++;
   }
   
   return count > 0 ? sum / count : rates[current].high - rates[current].low;
}

//+------------------------------------------------------------------+
//| Get prediction from executable                                   |
//+------------------------------------------------------------------+
bool GetPrediction(string features, string &signal, double &confidence)
{
   string inputFile = tempDir + "mt5_input.json";
   string outputFile = tempDir + "mt5_output.json";
   
   // Write input file
   int handle = FileOpen(inputFile, FILE_WRITE|FILE_TXT);
   if(handle == INVALID_HANDLE)
   {
      Print("Failed to create input file");
      return false;
   }
   FileWriteString(handle, features);
   FileClose(handle);
   
   // Prepare command
   string cmd = "\"" + ExecutablePath + "\" \"" + inputFile + "\" \"" + outputFile + "\"";
   
   // Execute predictor
   // Note: ShellExecuteW is used as it's allowed by MT5
   #import "shell32.dll"
      int ShellExecuteW(int hwnd, string lpOperation, string lpFile, string lpParameters, string lpDirectory, int nShowCmd);
   #import
   
   int result = ShellExecuteW(0, "open", ExecutablePath, "\"" + inputFile + "\" \"" + outputFile + "\"", NULL, 0);
   
   if(result <= 32)
   {
      Print("Failed to execute predictor. Error code: ", result);
      return false;
   }
   
   // Wait for output file (max 5 seconds)
   int attempts = 0;
   while(!FileIsExist(outputFile) && attempts < 50)
   {
      Sleep(100);
      attempts++;
   }
   
   if(!FileIsExist(outputFile))
   {
      Print("Timeout waiting for prediction");
      return false;
   }
   
   // Read output
   handle = FileOpen(outputFile, FILE_READ|FILE_TXT);
   if(handle == INVALID_HANDLE)
   {
      Print("Failed to read output file");
      return false;
   }
   
   string response = FileReadString(handle, FileSize(handle));
   FileClose(handle);
   
   // Parse JSON response (simple parsing)
   signal = ExtractJsonValue(response, "signal");
   confidence = StringToDouble(ExtractJsonValue(response, "confidence"));
   
   // Cleanup
   FileDelete(inputFile);
   FileDelete(outputFile);
   
   return true;
}

//+------------------------------------------------------------------+
//| Simple JSON value extractor                                      |
//+------------------------------------------------------------------+
string ExtractJsonValue(string json, string key)
{
   int start = StringFind(json, "\"" + key + "\":");
   if(start == -1) return "";
   
   start = StringFind(json, ":", start) + 1;
   while(StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '"')
      start++;
   
   int end = start;
   while(end < StringLen(json) && 
         StringGetCharacter(json, end) != ',' && 
         StringGetCharacter(json, end) != '}' &&
         StringGetCharacter(json, end) != '"')
      end++;
   
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Execute trade based on signal                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(string signal)
{
   // Check if we already have an open position
   if(PositionsTotal() > 0)
   {
      for(int i = 0; i < PositionsTotal(); i++)
      {
         if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            Print("Already have an open position");
            return;
         }
      }
   }
   
   double price, sl, tp;
   ENUM_ORDER_TYPE orderType;
   
   if(signal == "BUY")
   {
      orderType = ORDER_TYPE_BUY;
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - StopLossPips * _Point * 10;
      tp = price + (StopLossPips * RiskRewardRatio) * _Point * 10;
   }
   else if(signal == "SELL")
   {
      orderType = ORDER_TYPE_SELL;
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + StopLossPips * _Point * 10;
      tp = price - (StopLossPips * RiskRewardRatio) * _Point * 10;
   }
   else
   {
      // HOLD signal - do nothing
      return;
   }
   
   // Send order
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = orderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.magic = MagicNumber;
   request.comment = "JulietteV2 " + signal;
   
   if(!OrderSend(request, result))
   {
      Print("OrderSend failed. Error: ", GetLastError());
      Print("Result code: ", result.retcode);
   }
   else
   {
      Print("Order placed successfully. Ticket: ", result.order);
   }
}
//+------------------------------------------------------------------+