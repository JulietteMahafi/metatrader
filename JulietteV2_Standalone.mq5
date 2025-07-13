//+------------------------------------------------------------------+
//|                                           JulietteV2_Standalone.mq5 |
//|                                                                    |
//|                                    Copyright 2024, JulietteV2 Team |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, JulietteV2 Team"
#property link      ""
#property version   "1.00"
#property strict

#include <JAson.mqh>

//--- Input Parameters
input int      StopLoss = 10;           // Stop Loss in pips
input int      TakeProfit = 20;         // Take Profit in pips (R/R 1:2)
input double   LotSize = 0.1;           // Lot size
input int      MagicNumber = 12345;     // Magic number for trades
input bool     EnableTrading = true;    // Enable actual trading
input string   PredictorPath = "";      // Path to standalone predictor executable

//--- Global Variables
CJAson request_json;
CJAson response_json;
string request_file;
string response_file;
string model_dir;
bool is_new_bar = false;
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== JulietteV2 Standalone EA Initializing ===");
    
    // Set up file paths
    string terminal_data_path = TerminalInfoString(TERMINAL_DATA_PATH);
    string files_path = terminal_data_path + "\\MQL5\\Files\\";
    
    request_file = files_path + "mt5_request.json";
    response_file = files_path + "mt5_response.json";
    model_dir = files_path;
    
    // If predictor path not specified, use default
    if(PredictorPath == "")
    {
        PredictorPath = files_path + "standalone_predictor.exe";
    }
    
    Print("Request file: ", request_file);
    Print("Response file: ", response_file);
    Print("Model directory: ", model_dir);
    Print("Predictor executable: ", PredictorPath);
    
    // Check if predictor executable exists
    if(!FileIsExist(PredictorPath))
    {
        Print("ERROR: Predictor executable not found at: ", PredictorPath);
        Print("Please compile standalone_predictor.py to .exe and place it in the Files directory");
        return INIT_FAILED;
    }
    
    // Check if model files exist
    string model_files[] = {"model_best.pt", "scaler.joblib", "label_encoder.joblib", "model_metadata.json"};
    for(int i = 0; i < ArraySize(model_files); i++)
    {
        if(!FileIsExist(model_dir + model_files[i]))
        {
            Print("ERROR: Model file not found: ", model_files[i]);
            return INIT_FAILED;
        }
    }
    
    Print("=== JulietteV2 Standalone EA Initialized Successfully ===");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("=== JulietteV2 Standalone EA Deinitialized ===");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check for new bar
    if(!is_new_bar())
        return;
    
    // Get current bar time
    datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(current_bar_time == last_bar_time)
        return;
    
    last_bar_time = current_bar_time;
    Print("New bar detected at: ", TimeToString(current_bar_time));
    
    // Generate features and get prediction
    string signal = get_prediction();
    
    if(signal == "ERROR")
    {
        Print("Failed to get prediction");
        return;
    }
    
    Print("Signal received: ", signal);
    
    // Execute trading logic
    if(EnableTrading)
    {
        execute_trading_logic(signal);
    }
}

//+------------------------------------------------------------------+
//| Check for new bar                                                |
//+------------------------------------------------------------------+
bool is_new_bar()
{
    static datetime last_time = 0;
    datetime current_time = iTime(_Symbol, PERIOD_CURRENT, 0);
    
    if(current_time != last_time)
    {
        last_time = current_time;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Generate features and get prediction from standalone executable  |
//+------------------------------------------------------------------+
string get_prediction()
{
    // Generate features (90 bars, 16 features each)
    request_json.Clear();
    CJAson features_array;
    
    for(int i = 0; i < 90; i++)
    {
        CJAson bar_features;
        
        // OHLC data
        double open = iOpen(_Symbol, PERIOD_CURRENT, i);
        double high = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low = iLow(_Symbol, PERIOD_CURRENT, i);
        double close = iClose(_Symbol, PERIOD_CURRENT, i);
        
        // Basic features
        double body = close - open;
        double high_low = high - low;
        double gap = (i > 0) ? (open - iClose(_Symbol, PERIOD_CURRENT, i-1)) : 0;
        
        // Candle pattern features
        bool is_doji = MathAbs(body) < (high_low * 0.1);
        bool is_long_shadow = (high - MathMax(open, close)) > (high_low * 0.6) || 
                             (MathMin(open, close) - low) > (high_low * 0.6);
        bool is_spike = high_low > (iATR(_Symbol, PERIOD_CURRENT, 14, i) * 2);
        
        // Time features
        datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, i);
        MqlDateTime dt;
        TimeToStruct(bar_time, dt);
        
        double day_cos = MathCos(2 * M_PI * dt.day_of_week / 7);
        double day_sin = MathSin(2 * M_PI * dt.day_of_week / 7);
        double hour_cos = MathCos(2 * M_PI * dt.hour / 24);
        double hour_sin = MathSin(2 * M_PI * dt.hour / 24);
        double minute_cos = MathCos(2 * M_PI * dt.min / 60);
        double minute_sin = MathSin(2 * M_PI * dt.min / 60);
        
        // Add features to bar object
        bar_features.Set("Body", body);
        bar_features.Set("Close", close);
        bar_features.Set("High", high);
        bar_features.Set("High-Low", high_low);
        bar_features.Set("Is_Doji", is_doji);
        bar_features.Set("Is_Long_Shadow", is_long_shadow);
        bar_features.Set("Is_Spike", is_spike);
        bar_features.Set("Low", low);
        bar_features.Set("Open", open);
        bar_features.Set("day_cos", day_cos);
        bar_features.Set("day_sin", day_sin);
        bar_features.Set("gap", gap);
        bar_features.Set("hour_cos", hour_cos);
        bar_features.Set("hour_sin", hour_sin);
        bar_features.Set("minute_cos", minute_cos);
        bar_features.Set("minute_sin", minute_sin);
        
        features_array.Add(bar_features);
    }
    
    request_json.Set("features", features_array);
    
    // Write request to file
    int file_handle = FileOpen(request_file, FILE_WRITE | FILE_TXT);
    if(file_handle == INVALID_HANDLE)
    {
        Print("ERROR: Cannot open request file for writing");
        return "ERROR";
    }
    
    FileWriteString(file_handle, request_json.Serialize());
    FileClose(file_handle);
    
    // Call standalone executable
    string command = PredictorPath + " --input " + request_file + " --output " + response_file + " --model-dir " + model_dir;
    Print("Executing command: ", command);
    
    int result = ShellExecute(0, "open", PredictorPath, 
                            "--input " + request_file + " --output " + response_file + " --model-dir " + model_dir, 
                            "", 1);
    
    if(result <= 32) // ShellExecute error codes are 32 or less
    {
        Print("ERROR: ShellExecute failed with code: ", result);
        return "ERROR";
    }
    
    // Wait for response (with timeout)
    int timeout = 5000; // 5 seconds
    int elapsed = 0;
    while(!FileIsExist(response_file) && elapsed < timeout)
    {
        Sleep(100);
        elapsed += 100;
    }
    
    if(!FileIsExist(response_file))
    {
        Print("ERROR: Response file not created within timeout");
        return "ERROR";
    }
    
    // Read response
    file_handle = FileOpen(response_file, FILE_READ | FILE_TXT);
    if(file_handle == INVALID_HANDLE)
    {
        Print("ERROR: Cannot open response file for reading");
        return "ERROR";
    }
    
    string response_text = FileReadString(file_handle);
    FileClose(file_handle);
    
    // Parse response
    if(!response_json.Deserialize(response_text))
    {
        Print("ERROR: Failed to parse response JSON");
        return "ERROR";
    }
    
    // Check for error
    if(response_json["error"] != NULL)
    {
        Print("ERROR: Predictor returned error: ", response_json["error"].ToStr());
        return "ERROR";
    }
    
    // Get signal
    string signal = response_json["signal"].ToStr();
    double confidence = response_json["confidence"].ToDbl();
    
    Print("Prediction: ", signal, " (confidence: ", DoubleToString(confidence, 4), ")");
    
    // Clean up response file
    FileDelete(response_file);
    
    return signal;
}

//+------------------------------------------------------------------+
//| Execute trading logic based on signal                            |
//+------------------------------------------------------------------+
void execute_trading_logic(string signal)
{
    // Close existing positions if signal is KEEP or opposite
    if(signal == "KEEP")
    {
        close_all_positions();
        return;
    }
    
    // Close opposite positions
    if(signal == "BUY")
        close_positions_by_type(ORDER_TYPE_SELL);
    else if(signal == "SELL")
        close_positions_by_type(ORDER_TYPE_BUY);
    
    // Open new position if no existing position of same type
    if(signal == "BUY" && !has_position(ORDER_TYPE_BUY))
    {
        open_position(ORDER_TYPE_BUY);
    }
    else if(signal == "SELL" && !has_position(ORDER_TYPE_SELL))
    {
        open_position(ORDER_TYPE_SELL);
    }
}

//+------------------------------------------------------------------+
//| Open a new position                                              |
//+------------------------------------------------------------------+
void open_position(ENUM_ORDER_TYPE order_type)
{
    double price = (order_type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double sl = (order_type == ORDER_TYPE_BUY) ? price - StopLoss * _Point : price + StopLoss * _Point;
    double tp = (order_type == ORDER_TYPE_BUY) ? price + TakeProfit * _Point : price - TakeProfit * _Point;
    
    ulong ticket = OrderSend(_Symbol, order_type, LotSize, price, 3, sl, tp, 
                           "JulietteV2", MagicNumber, 0, clrNONE);
    
    if(ticket > 0)
    {
        Print("Position opened: ", EnumToString(order_type), " Ticket: ", ticket);
    }
    else
    {
        Print("ERROR: Failed to open position. Error: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Check if position of given type exists                           |
//+------------------------------------------------------------------+
bool has_position(ENUM_ORDER_TYPE order_type)
{
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(PositionSelectByIndex(i))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
               PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
               PositionGetInteger(POSITION_TYPE) == order_type)
            {
                return true;
            }
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Close positions by type                                          |
//+------------------------------------------------------------------+
void close_positions_by_type(ENUM_ORDER_TYPE order_type)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByIndex(i))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
               PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
               PositionGetInteger(POSITION_TYPE) == order_type)
            {
                ulong ticket = PositionGetInteger(POSITION_TICKET);
                if(OrderDelete(ticket))
                {
                    Print("Position closed: Ticket ", ticket);
                }
                else
                {
                    Print("ERROR: Failed to close position. Error: ", GetLastError());
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void close_all_positions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByIndex(i))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                ulong ticket = PositionGetInteger(POSITION_TICKET);
                if(OrderDelete(ticket))
                {
                    Print("Position closed: Ticket ", ticket);
                }
                else
                {
                    Print("ERROR: Failed to close position. Error: ", GetLastError());
                }
            }
        }
    }
}