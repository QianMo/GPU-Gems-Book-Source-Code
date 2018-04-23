//---------------------------------------------------------------------------
#ifndef __CTEXTUREPPM__
#define __CTEXTUREPPM__
//---------------------------------------------------------------------------
#ifdef WIN32
# include <windows.h>
#endif
//---------------------------------------------------------------------------
#include "CTexture.h"
#include <iostream>
#include <fstream>
//---------------------------------------------------------------------------
class CTexturePPM : public CTexture
{
public:
  
  CTexturePPM(const char *);
  CTexturePPM(const char *n,int w,int h,bool a,unsigned char *d) : CTexture(n,w,h,a,d) {}
  CTexturePPM(const CTexture& t) : CTexture(t) {}
  CTexturePPM(CTexture *t) : CTexture(t) {}
  
  void            load();
};
//------------------------------------------------------------------------
#endif
//------------------------------------------------------------------------
