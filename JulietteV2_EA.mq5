//+------------------------------------------------------------------+
//|                                                   JulietteV2_EA |
//|                       Simple ShellExecute bridge to predictor  |
//|  Requirements: standalone_predictor.exe must be placed in the  |
//|                \MQL5\Experts\Files (or Common\MQL5\Files)      |
//|  Strategy: on every new bar, build JSON with last 90 bars of   |
//|            features, call EXE, parse response, place orders.    |
//+------------------------------------------------------------------+
#property copyright "2025"
#property version   "0.1"
#property strict

#include <JAson.mqh>

// Windows ShellExecuteW import
#import "shell32.dll"
int  ShellExecuteW(int hwnd,LPCWSTR lpOperation,LPCWSTR lpFile,LPCWSTR lpParameters,LPCWSTR lpDirectory,int nCmdShow);
#import

#define PIP         _Point
#define SL_PIPS     10
#define TP_PIPS     20

input string PredictorExe = "standalone_predictor.exe"; // Name of the EXE in Files dir
input int    SequenceLen  = 90;    // number of bars per sample

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime lastBarTime=0;
   datetime currBarTime=iTime(_Symbol,_Period,0);
   if(currBarTime==lastBarTime)
      return; // Not a new bar
   lastBarTime=currBarTime;

   string jsonRequest=BuildFeaturesJson();
   if(jsonRequest=="")
      return;

   string reqFile=TimeToString(TimeCurrent(),TIME_DATE|TIME_MINUTES|TIME_SECONDS)+"_req.json";
   string respFile=reqFile+"_resp.json";

   // Write request JSON
   int handle=FileOpen(reqFile,FILE_WRITE|FILE_TXT|FILE_COMMON);
   if(handle==INVALID_HANDLE)
     {
      Print("Failed to open request file: ",GetLastError());
      return;
     }
   FileWriteString(handle,jsonRequest);
   FileClose(handle);

   // Build command line parameters
   string params=StringFormat(" --input \"%s\" --output \"%s\"",reqFile,respFile);

   // Call predictor
   int res=ShellExecuteW(0,L"open",StringToUnicode(PredictorExe),StringToUnicode(params),NULL,0);
   if(res<=32)
     {
      Print("ShellExecuteW failed: ",res);
      return;
     }

   // Wait for output (simple polling, max 2 seconds)
   datetime start=TimeCurrent();
   while(TimeCurrent()-start<2)
     {
      if(FileIsExist(respFile,FILE_COMMON))
         break;
      Sleep(100);
     }

   if(!FileIsExist(respFile,FILE_COMMON))
     {
      Print("Response file not found");
      return;
     }

   // Read response
   handle=FileOpen(respFile,FILE_READ|FILE_TXT|FILE_COMMON);
   if(handle==INVALID_HANDLE)
     {
      Print("Failed to open response file: ",GetLastError());
      return;
     }
   string jsonResp=FileReadString(handle);
   FileClose(handle);
   CJAson respObj;
   if(!respObj.Deserialize(jsonResp))
     {
      Print("JSON parse failed");
      return;
     }
   string signal=respObj["signal"].ToStr();
   double conf=respObj["confidence"].ToDbl();

   // Basic trading logic (R/R 1:2)
   double sl=SL_PIPS*PIP;
   double tp=TP_PIPS*PIP;
   if(signal=="BUY")
      OrderSend(_Symbol,OP_BUY,0.1,Ask,0,Ask-sl,Ask+tp,"JulietteV2",0,0,clrGreen);
   else if(signal=="SELL")
      OrderSend(_Symbol,OP_SELL,0.1,Bid,0,Bid+sl,Bid-tp,"JulietteV2",0,0,clrRed);
  }
//+------------------------------------------------------------------+
//| Build JSON with recent features                                  |
//+------------------------------------------------------------------+
string BuildFeaturesJson()
  {
   if(Bars(_Symbol,_Period)<SequenceLen)
      return "";
   CJAson root;
   CJAson arr;
   for(int i=SequenceLen-1;i>=0;i--)
     {
      CJAson row;
      row.Set("open",DoubleToString(iOpen(_Symbol,_Period,i),_Digits));
      row.Set("high",DoubleToString(iHigh(_Symbol,_Period,i),_Digits));
      row.Set("low", DoubleToString(iLow(_Symbol,_Period,i),_Digits));
      row.Set("close",DoubleToString(iClose(_Symbol,_Period,i),_Digits));
      arr.Add(row);
     }
   root.Set("features",arr);
   return root.Serialize();
  }
//+------------------------------------------------------------------+