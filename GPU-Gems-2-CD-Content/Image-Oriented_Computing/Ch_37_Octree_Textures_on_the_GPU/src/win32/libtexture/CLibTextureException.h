#ifndef H_CLibTextureException__
#define H_CLibTextureException__

#include <cstdarg>
#include <cstdio>

class CLibTextureException
{
protected:
  char m_szMsg[512];
public:

  CLibTextureException(){m_szMsg[0]='\0';}

  CLibTextureException(char *msg,...)
  {
    va_list args;
    va_start(args, msg);

    vsprintf(m_szMsg,msg,args);

    va_end(args);
  }

  const char *getMsg() const {return (m_szMsg);}
};

#endif
