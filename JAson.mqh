//+------------------------------------------------------------------+
//|                                                        JAson.mqh |
//|                                      Copyright 2018, Lirico Software |
//|                                             https://www.mql5.com/en/users/lirico |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, Lirico Software"
#property link      "https://www.mql5.com/en/users/lirico"
#property version   "1.01"
#property strict

class CJAVal;
class CJAson;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CJAVal
  {
private:
   string            m_val;
   int               m_type;
   CJAson           *m_obj;

public:
                     CJAVal(void);
                    ~CJAVal(void);
   //---
   void              Set(const string val);
   void              Set(const double val);
   void              Set(const int val);
   void              Set(const bool val);
   void              Set(CJAson *val);
   //---
   string            ToStr(void);
   double            ToDbl(void);
   int               ToInt(void);
   bool              ToBool(void);
   CJAson           *ToObj(void);
   //---
   int               GetType(void);
   //---
   bool              Deserialize(string &str);
   string            Serialize(void);

protected:
   string            Escape(string str);
   string            UnEscape(string str);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CJAson
  {
private:
   CJAVal           *m_vals[];
   string            m_keys[];
   int               m_type;

public:
                     CJAson(void);
                    ~CJAson(void);
   //---
   void              Set(const string key,const string val);
   void              Set(const string key,const double val);
   void              Set(const string key,const int val);
   void              Set(const string key,const bool val);
   void              Set(const string key,CJAson *val);
   //---
   void              Add(const string val);
   void              Add(const double val);
   void              Add(const int val);
   void              Add(const bool val);
   void              Add(CJAson *val);
   //---
   CJAVal            *operator[](const string key);
   CJAVal            *operator[](const int index);
   //---
   int               Size(void);
   int               GetType(void);
   //---
   bool              Deserialize(string &str);
   string            Serialize(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAVal::CJAVal(void) : m_type(-1)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAVal::~CJAVal(void)
  {
   if(m_type==5 && CheckPointer(m_obj)==POINTER_DYNAMIC)
      delete m_obj;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAVal::Set(const string val)
  {
   m_val=val;
   m_type=1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAVal::Set(const double val)
  {
   m_val=DoubleToString(val,-1);
   m_type=2;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAVal::Set(const int val)
  {
   m_val=IntegerToString(val);
   m_type=3;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAVal::Set(const bool val)
  {
   m_val=(val)?"true":"false";
   m_type=4;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAVal::Set(CJAson* val)
  {
   m_obj=val;
   m_type=5;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CJAVal::ToStr(void)
  {
   return m_val;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CJAVal::ToDbl(void)
  {
   return(StringToDouble(m_val));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CJAVal::ToInt(void)
  {
   return((int)StringToInteger(m_val));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CJAVal::ToBool(void)
  {
   return(m_val=="true");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAson *CJAVal::ToObj(void)
  {
   return m_obj;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CJAVal::GetType(void)
  {
   return m_type;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CJAVal::Deserialize(string &str)
  {
   int p=0;
//---
   ushort s=StringGetCharacter(str,p);
   if(s=='"')
     {
      m_type=1;
      p++;
      int p2=p;
      while((p2=StringFind(str,"\"",p2))>=0)
        {
         if(StringGetCharacter(str,p2-1)!='\\')
            break;
         p2++;
        }
      if(p2<0)
         return(false);
      m_val=UnEscape(StringSubstr(str,p,p2-p));
      str=StringSubstr(str,p2+1);
      return(true);
     }
//---
   if((s>='0' && s<='9') || s=='-') // CORRECTED: Operator precedence
     {
      m_type=3;
      int p2=p;
      while(p2<StringLen(str))
        {
         s=StringGetCharacter(str,p2);
         if(s=='.')
            m_type=2;
         if((s<'0' || s>'9') && s!='.' && s!='-')
            break;
         p2++;
        }
      m_val=StringSubstr(str,p,p2-p);
      str=StringSubstr(str,p2);
      return(true);
     }
//---
   if(StringFind(str,"true",p)==p)
     {
      m_type=4;
      m_val="true";
      str=StringSubstr(str,p+4);
      return(true);
     }
//---
   if(StringFind(str,"false",p)==p)
     {
      m_type=4;
      m_val="false";
      str=StringSubstr(str,p+5);
      return(true);
     }
//---
   if(s=='{' || s=='[')
     {
      m_type=5;
      m_obj=new CJAson();
      if(!m_obj.Deserialize(str))
         return(false);
      return(true);
     }
//---
   return(false);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CJAVal::Serialize(void)
  {
   switch(m_type)
     {
      case 1:
         return "\""+Escape(m_val)+"\"";
      case 2:
      case 3:
      case 4:
         return m_val;
      case 5:
         if(CheckPointer(m_obj)==POINTER_DYNAMIC)
            return m_obj.Serialize();
     }
   return "null";
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CJAVal::Escape(string str)
  {
   StringReplace(str,"\\","\\\\");
   StringReplace(str,"\"","\\\"");
   StringReplace(str,"/","\\/");
   StringReplace(str,"\x08","\\b"); // CORRECTED: Use hex code for backspace
   StringReplace(str,"\f","\\f");
   StringReplace(str,"\n","\\n");
   StringReplace(str,"\r","\\r");
   StringReplace(str,"\t","\\t");
   return str;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CJAVal::UnEscape(string str)
  {
   StringReplace(str,"\\\\","\\");
   StringReplace(str,"\\\"","\"");
   StringReplace(str,"\\/","/");
   StringReplace(str,"\\b","\x08"); // CORRECTED: Use hex code for backspace
   StringReplace(str,"\\f","\f");
   StringReplace(str,"\\n","\n");
   StringReplace(str,"\\r","\r");
   StringReplace(str,"\\t","\t");
   return str;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAson::CJAson(void) : m_type(0)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAson::~CJAson(void)
  {
   for(int i=ArraySize(m_vals)-1; i>=0; i--)
      if(CheckPointer(m_vals[i])==POINTER_DYNAMIC)
         delete m_vals[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Set(const string key,const string val)
  {
   m_type=1;
   int i=ArraySize(m_keys);
   ArrayResize(m_keys,i+1);
   m_keys[i]=key;
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Set(const string key,const double val)
  {
   m_type=1;
   int i=ArraySize(m_keys);
   ArrayResize(m_keys,i+1);
   m_keys[i]=key;
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Set(const string key,const int val)
  {
   m_type=1;
   int i=ArraySize(m_keys);
   ArrayResize(m_keys,i+1);
   m_keys[i]=key;
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Set(const string key,const bool val)
  {
   m_type=1;
   int i=ArraySize(m_keys);
   ArrayResize(m_keys,i+1);
   m_keys[i]=key;
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Set(const string key,CJAson* val)
  {
   m_type=1;
   int i=ArraySize(m_keys);
   ArrayResize(m_keys,i+1);
   m_keys[i]=key;
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Add(const string val)
  {
   m_type=2;
   int i=ArraySize(m_vals);
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Add(const double val)
  {
   m_type=2;
   int i=ArraySize(m_vals);
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Add(const int val)
  {
   m_type=2;
   int i=ArraySize(m_vals);
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Add(const bool val)
  {
   m_type=2;
   int i=ArraySize(m_vals);
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CJAson::Add(CJAson* val)
  {
   m_type=2;
   int i=ArraySize(m_vals);
//---
   CJAVal *v=new CJAVal();
   v.Set(val);
   ArrayResize(m_vals,i+1);
   m_vals[i]=v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAVal *CJAson::operator[](const string key)
  {
   int i=ArraySize(m_keys)-1;
   for(; i>=0; i--)
      if(m_keys[i]==key)
         break;
//---
   if(i<0)
      return NULL;
   return(m_vals[i]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CJAVal *CJAson::operator[](const int index)
  {
   if(index<0 || index>=ArraySize(m_vals))
      return NULL;
   return(m_vals[index]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CJAson::Size(void)
  {
   return ArraySize(m_vals);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CJAson::GetType(void)
  {
   return m_type;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CJAson::Deserialize(string &str)
  {
   int p=0;
   ushort s=StringGetCharacter(str,p);
//--- object
   if(s=='{')
     {
      m_type=1;
      p++;
      while(p<StringLen(str))
        {
         s=StringGetCharacter(str,p);
         if(s=='}')
           {
            p++;
            break;
           }
         if(s==',')
           {
            p++;
           }
         string key="";
         s=StringGetCharacter(str,p);
         if(s=='"')
           {
            p++;
            int p2=StringFind(str,"\"",p);
            if(p2<0)
               return(false);
            key=StringSubstr(str,p,p2-p);
            p=p2+1;
           }
         else
            return(false);
         s=StringGetCharacter(str,p);
         if(s==':')
            p++;
         else
            return(false);
         string sub=StringSubstr(str,p);
         CJAVal *v=new CJAVal();
         if(!v.Deserialize(sub))
            return(false);
         str=sub;
         p=0;
         int i=ArraySize(m_keys);
         ArrayResize(m_keys,i+1);
         m_keys[i]=key;
         ArrayResize(m_vals,i+1);
         m_vals[i]=v;
        }
      str=StringSubstr(str,p);
      return(true);
     }
//--- array
   if(s=='[')
     {
      m_type=2;
      p++;
      while(p<StringLen(str))
        {
         s=StringGetCharacter(str,p);
         if(s==']')
           {
            p++;
            break;
           }
         if(s==',')
           {
            p++;
           }
         string sub=StringSubstr(str,p);
         CJAVal *v=new CJAVal();
         if(!v.Deserialize(sub))
            return(false);
         str=sub;
         p=0;
         int i=ArraySize(m_vals);
         ArrayResize(m_vals,i+1);
         m_vals[i]=v;
        }
      str=StringSubstr(str,p);
      return(true);
     }
//---
   return(false);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CJAson::Serialize(void)
  {
   string str="";
//--- object
   if(m_type==1)
     {
      str="{";
      for(int i=0; i<ArraySize(m_keys); i++)
        {
         if(i>0)
            str+=",";
         str+="\""+m_keys[i]+"\":"+m_vals[i].Serialize();
        }
      str+="}";
     }
//--- array
   if(m_type==2)
     {
      str="[";
      for(int i=0; i<ArraySize(m_vals); i++)
        {
         if(i>0)
            str+=",";
         str+=m_vals[i].Serialize();
        }
      str+="]";
     }
//---
   return str;
  }
//+------------------------------------------------------------------+
