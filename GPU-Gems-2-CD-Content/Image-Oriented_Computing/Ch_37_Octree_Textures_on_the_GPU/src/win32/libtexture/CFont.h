//---------------------------------------------------------------------------
/** \class  CFont
    
    Use this class to load a font and print some text on screen.
    Fonts are images with all characters on a line. There must be at least on pixel of
    pure black between two characters. 
    <BR>&nbsp;<BR>
    The character set is<BR>
    !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~<BR>&nbsp;<BR>
    Note that a default font is provided in libtexture/font/default.tga (this font as a low resolution 
    to have a small memory cost)
 */
//  Creation: 20/07/2001
//
//  Sylvain Lefebvre
//---------------------------------------------------------------------------
#ifndef __CFONT__
#define __CFONT__
//---------------------------------------------------------------------------
#ifdef WIN32
# include <windows.h>
#endif
//---------------------------------------------------------------------------
#define CFONT_FIRST_CAR			33
#define CFONT_LAST_CAR			126
#define CFONT_NB_CAR	 		  (CFONT_LAST_CAR-CFONT_FIRST_CAR+1)
//---------------------------------------------------------------------------
class CTexture;
//---------------------------------------------------------------------------
typedef class CFont
{
private:
  static char  *m_szBuf;
  static int    m_iBufLength;
  double	      m_CarSizeX[256];
  double	      m_CarBeginX[256];
  double	      m_CarEndX[256];
  int		        m_iCarMinY;
  int		        m_iCarMaxY;
  double	      m_dCarMinY;
  double	      m_dCarMaxY;
  int		        m_iMaxCarW;
  double	      m_dMaxCarW;
  unsigned int  m_uiCarRenderLists;
  CTexture     *m_Tex;
  unsigned int  m_uiTexId;

  void	computeWidth();
  void  computeHeight();
  void	distribute();
  void	genRenderLists();
  void	copyCar(int car,int n,unsigned char *ndata);
  void  copyToBuffer(const char *s);

public:
  /**
     Creates a font from an image file. If alpha is <B>true</B> the black background is replaced by transparancy.
   */
  CFont(const char *,bool alpha=true);
  ~CFont();

  /**
     Prints the string s at (x,y) with size t.
     Note that the text is printed in a plane but that nothing prevents using an appropriate modelview
     to draw it in 3d.
   */
  void	printString(double x,double y,double t,const char *s);
  /**
     Prints N first characters of string s at (x,y) with size t.
   */
  void	printStringN(double x,double y,double t,const char *s,int n);
  /**
     Retrieve the width and height needed by printString to print string s with size t.
   */
   void	printStringNeed(double t,const char *s,double *w,double *h);

}CFont;
//------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
