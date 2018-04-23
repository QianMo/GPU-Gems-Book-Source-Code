//---------------------------------------------------------------------------
/**
\class CTexture

\brief The main class for texture loading. It also allows some simple 
       operations on alpha channel.

 */
//---------------------------------------------------------------------------
#ifndef __CTEXTURE__
#define __CTEXTURE__
//---------------------------------------------------------------------------
#define TEXTURE_NAME_SIZE 256
#define TEXTURE_EPSILON 0.0001
//---------------------------------------------------------------------------
#include "CLibTextureException.h"
#include <math.h>
//---------------------------------------------------------------------------
#ifndef ABS
#define ABS(x) ((x)>0.0?(x):-(x))
#endif
//---------------------------------------------------------------------------
class CTexture
{
protected:

  char	          m_szName[TEXTURE_NAME_SIZE];
  unsigned char  *m_Data;
  int		          m_iWidth,m_iHeight;
  bool		        m_bLoaded;
  bool		        m_bAlpha;
  bool            m_bDataOwner;
  
  unsigned char *CTexture::computeNormalsFromRGB(double scale,double norme);

public:

  // -------------------------------------- alpha functors ---
  class alpha_functor
  {
  public:
    virtual unsigned char computeAlpha(unsigned char r,unsigned char g,unsigned char b) const =0;
  };

  class alpha_zero : public alpha_functor
  {
  public:
    unsigned char computeAlpha(unsigned char r,unsigned char g,unsigned char b) const
    {return (0);}
  };

  class alpha_one : public alpha_functor
  {
  public:
    unsigned char computeAlpha(unsigned char r,unsigned char g,unsigned char b) const
    {return (255);}
  };

  class alpha_const : public alpha_functor
  {
    unsigned char m_ucAlpha;
  public:
    alpha_const(double c) {m_ucAlpha=(unsigned char)(c*255);}
    unsigned char computeAlpha(unsigned char r,unsigned char g,unsigned char b) const
    {return (m_ucAlpha);}
  };

  class alpha_color : public alpha_functor
  {
  private:
    int m_iTolerence;
    int m_iR;
    int m_iG;
    int m_iB;
  public:
    alpha_color(unsigned char r,unsigned char g,unsigned char b,int t=0)
    {
      m_iR=(int)r;
      m_iG=(int)g;
      m_iB=(int)b;
      m_iTolerence=t;
    }
    unsigned char computeAlpha(unsigned char r,unsigned char g,unsigned char b) const
    {
      if (   ABS((int)r-m_iR) <= m_iTolerence 
        && ABS((int)g-m_iG) <= m_iTolerence 
        && ABS((int)b-m_iB) <= m_iTolerence)
        return (0);
      else
        return (255);
    }
  };

  // -------------------------------------- MIP-map functors ---
  class mipmap_functor
  {
  public:
    virtual void computeAverage(const unsigned char r[4],
                                const unsigned char g[4],
                                const unsigned char b[4],
                                const unsigned char a[4],
                                unsigned char& _r,unsigned char& _g,unsigned char& _b,unsigned char& _a) const =0;
  };

  class mipmap_std : public mipmap_functor
  {
  public:
    virtual void computeAverage(const unsigned char r[4],
                                const unsigned char g[4],
                                const unsigned char b[4],
                                const unsigned char a[4],
                                unsigned char& _r,unsigned char& _g,unsigned char& _b,unsigned char& _a) const
    {
      _r=(unsigned char)(((int)r[0]+(int)r[1]+(int)r[2]+(int)r[3])/4);
      _g=(unsigned char)(((int)g[0]+(int)g[1]+(int)g[2]+(int)g[3])/4);
      _b=(unsigned char)(((int)b[0]+(int)b[1]+(int)b[2]+(int)b[3])/4);
      _a=(unsigned char)(((int)a[0]+(int)a[1]+(int)a[2]+(int)a[3])/4);
    }
  };

  class mipmap_alpha : public mipmap_functor
  {
  public:
    virtual void computeAverage(const unsigned char r[4],
                                const unsigned char g[4],
                                const unsigned char b[4],
                                const unsigned char a[4],
                                unsigned char& _r,unsigned char& _g,unsigned char& _b,unsigned char& _a) const
    {
      int n=0;
      int ir=0,ig=0,ib=0,ia=0;
      for (int i=0;i<4;i++)
      {
      if (a[i] > 0)
      {
        ir+=r[i];
        ig+=g[i];
        ib+=b[i];
        ia+=a[i];
        n++;
      }
      }
      if (n > 0)
      {
        _r=(unsigned char)(ir/n);
        _g=(unsigned char)(ig/n);
        _b=(unsigned char)(ib/n);
        _a=(unsigned char)(ia/n);
      }
      else
        _r=_g=_b=_a=0;
    }
  };

  // -------------------------------------- copy functors ---
  class copy_functor
  {
  public:
    virtual void copyPixel(unsigned char sr,unsigned char sg,unsigned char sb,unsigned char sa,
                   unsigned char& dr,unsigned char& dg,unsigned char& db,unsigned char& da) const = 0;
  };

  class copy_std : public copy_functor
  {
  public:
    virtual void copyPixel(unsigned char sr,unsigned char sg,unsigned char sb,unsigned char sa,
                   unsigned char& dr,unsigned char& dg,unsigned char& db,unsigned char& da) const
    {
      dr=sr;dg=sg;db=sb;da=sa;
    }
  };

  // -------------------------------------- update functors ---
  class update_functor
  {
  public:
    virtual void updatePixel(unsigned char sr,unsigned char sg,unsigned char sb,unsigned char sa,
                   unsigned char& dr,unsigned char& dg,unsigned char& db,unsigned char& da) const = 0;
  };

  class update_std : public update_functor
  {
  public:
    virtual void updatePixel(unsigned char sr,unsigned char sg,unsigned char sb,unsigned char sa,
                   unsigned char& dr,unsigned char& dg,unsigned char& db,unsigned char& da) const
    {
      dr=sr;dg=sg;db=sb;da=sa;
    }
  };

  class update_premult_alpha : public update_functor
  {
  public:
    virtual void updatePixel(unsigned char sr,unsigned char sg,unsigned char sb,unsigned char sa,
                   unsigned char& dr,unsigned char& dg,unsigned char& db,unsigned char& da) const
    {
      dr=(unsigned char)(((float)sr)*(((float)sa)/255.0f));
      dg=(unsigned char)(((float)sg)*(((float)sa)/255.0f));
      db=(unsigned char)(((float)sb)*(((float)sa)/255.0f));
      da=sa;
    }
  };

public:

  CTexture();
  CTexture(const char *name,int w,int h,bool alpha,unsigned char *data);
  CTexture(int w,int h,bool alpha=false);
  CTexture(const CTexture&);
  CTexture(CTexture *);
  virtual ~CTexture();
  
  virtual void            load(){};
  virtual void            unload();
  void                    save(const char *n);

  /// returns texture width
  int		          getWidth() const {return (m_iWidth);}
  /// returns texture height
  int     	      getHeight() const {return (m_iHeight);}
  /// <b>true</b> if texture is loaded in memory
  bool	          isLoaded() const {return (m_bLoaded);}
  /// <b>true</b> if texture is in RGBA format
  bool            isAlpha() const {return (m_bAlpha);}
  /// returns texture data (see isAlpha to test if the texture format is RGB or RGBA)
  unsigned char  *getData() const {return (m_Data);}
  /// returns the name of the texture (filename in most cases)
  const char     *getName() const {return (m_szName);}
  /**
     Convert texture to a normal texture. Luminance is used as an height field to compute normals
     by finite differencing. The scale factor applies to the height map.
  */
  void computeNormals(double scale=1.0,double norme=1.0);
  /**
     Convert a texture to RGBA format by putting alpha to zero where the texture 
     color is [r,g,b] and one everywhere else. 
     It is usefull to make areas of a particular color transparent.
     The tolerence parameter is the tolerence used when comparing colors.
     Note that the same result can be obtained by using computeAlpha and alpha_color 
  */
  /**
    Same as computeNormals. Deprecated.
  */
  void convertToNormal(double scale=1.0,double norme=1.0);
  /**
    Same as computeAlpha with alpha_color functor. Deprecated.
  */
  virtual void	convert2Alpha(unsigned char r,
			      unsigned char g,
			      unsigned char b,
			      unsigned char tolerence);
  /**
     Load an alpha map from the red component of another CTexture
  */
  virtual void            loadAlphaMap(CTexture *);
  /**
     Load an alpha map from the red component of an image file
  */
  void                    loadAlphaMap(const char *);
  /** 
      Compute alpha map from an alpha_functor (see alpha_zero and alpha_one)
  */
  void                    computeAlpha(const alpha_functor& af);
  /** 
      Update colors with a functor
  */
  void                    updateColors(const update_functor& af);
  /** 
      Compute next MIP-map level map from a mipmap_functor.
      Works only on textures with width and height that are power
      of 2.
  */
  CTexture               *computeNextMIPMapLevel(const mipmap_functor& mip);
  /** 
      Copy texture tex at coords x,y. If the texture goes outside of borders,
      it is clipped.
  */
  void                    copy(int x,int y,CTexture *tex,const copy_functor& cpy=copy_std());
  /**
      Extracts a part of a texture as a new texture (with a copy of the data).
  */
  CTexture               *extract(int x,int y,int w,int h);
  /**
     Load a CTexture from an image file. The format is automaticaly deduced from extension.
  */
  static CTexture        *loadTexture(const char *);
  /**
     Save a CTexture to an image file. The format is automaticaly deduced from extension.
  */
  static void             saveTexture(CTexture *,const char *);
  /**
     Returns the size of texture data in memory.
  */
  int                 memsize() const;

  /** 
      Apply an horizontal flip.
  */
  void                flipY();

  /** 
      Apply a vertical flip.
  */
  void                flipX();

  /**
      Tell if the instance should erase its RGB(A) data when deleted.
  */
  void                setDataOwner(bool b) {m_bDataOwner=b;}

  /**
    Retrieve color component.
  */
  unsigned char get(int i,int j,int c) const {return (m_Data[(i+j*m_iWidth)*(isAlpha()?4:3)+c]);}
  

//  virtual void            setLuminance(bool b){m_bLuminance=b;}

//---------------------------------------------------------------------------
private:

  // simple vertex for normal computation
  class vertex
  {
  public:
    double x,y,z;
    
    vertex ()
      {
	x=0;
	y=0;
	z=0;
      }

    vertex(double _x,double _y,double _z)
      {
	x=_x;
	y=_y;
	z=_z;
      }

    void normalize()
      {
	double l=x*x+y*y+z*z;
	if (l < TEXTURE_EPSILON*TEXTURE_EPSILON)
	  CLibTextureException("trying to normalize null vertex !");
	l=sqrt(l);
	x/=l;
	y/=l;
	z/=l;
      }

    vertex cross(const vertex& v)
      {
	return (vertex(y*v.z - z*v.y,z*v.x - x*v.z,x*v.y - y*v.x));
      }
  };

};
//---------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
