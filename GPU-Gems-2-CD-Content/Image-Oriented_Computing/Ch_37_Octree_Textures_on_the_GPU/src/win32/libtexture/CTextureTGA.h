//---------------------------------------------------------------------------
#ifndef __CTEXTURETGA__
#define __CTEXTURETGA__
//---------------------------------------------------------------------------
#ifdef WIN32
# include <windows.h>
#endif
//---------------------------------------------------------------------------
#include "CTexture.h"
//---------------------------------------------------------------------------
class CTextureTGA : public CTexture
{
protected:

public:
  CTextureTGA(const char *);
  CTextureTGA(const char *n,int w,int h,bool a,unsigned char *d) : CTexture(n,w,h,a,d) {}
  CTextureTGA(const CTexture& t) : CTexture(t) {}
  CTextureTGA(CTexture *t) : CTexture(t) {}

  void            load();
  void            save();
};
//------------------------------------------------------------------------
typedef struct s_tga_header
{
  unsigned char id_len;
  unsigned char cmap_type;
  unsigned char img_type;
  short cmap_origin;
  short cmap_len;
  unsigned char cmap_entry_size;
  short x;
  short y;
  short w;
  short h;
  unsigned char pix_size;
  unsigned char img_desc;
}tga_header;
//------------------------------------------------------------------------
#endif
//------------------------------------------------------------------------
