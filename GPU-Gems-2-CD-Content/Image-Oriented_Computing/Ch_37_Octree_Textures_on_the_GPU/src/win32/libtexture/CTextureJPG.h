//---------------------------------------------------------------------------
#ifndef __CTEXTUREJPG__
#define __CTEXTUREJPG__
//---------------------------------------------------------------------------
#include "CTexture.h"
//---------------------------------------------------------------------------
class CTextureJPG : public CTexture
{
private:
  int m_iStride;
public:
  CTextureJPG(const char *);
  CTextureJPG(const char *n,int w,int h,bool a,unsigned char *d) : CTexture(n,w,h,a,d) {}
  CTextureJPG(const CTexture& t) : CTexture(t) {}
  CTextureJPG(CTexture *t) : CTexture(t) {}
  
  void            load();
  
};
//---------------------------------------------------------------------------
#endif
