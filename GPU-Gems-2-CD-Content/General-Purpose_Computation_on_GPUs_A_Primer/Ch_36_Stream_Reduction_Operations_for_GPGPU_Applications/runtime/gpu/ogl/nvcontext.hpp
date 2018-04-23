// nvcontext.hpp

#ifndef NVCONTEXT_H
#define NVCONTEXT_H

#include "oglcontext.hpp"
#include "ogltexture.hpp"

namespace brook {

  class NVTexture : public OGLTexture {
  public:

    NVTexture ( unsigned int inWidth, 
                unsigned int inHeight, 
                GPUContext::TextureFormat inFormat);
    
    virtual int nativeFormat() const { return _nativeFormat; }

  private:
    int _nativeFormat;
  };


  class NVContext : public OGLContext
  {
  public:

    static NVContext * create();
    
    TextureHandle 
    createTexture2D( unsigned int inWidth, 
                     unsigned int inHeight, 
                     TextureFormat inFormat);

    int getShaderFormatRank (const char *name) const;

    static bool
    isCompatibleContext ();

    static bool
    isVendorContext ();


  protected:
    NVContext();

    bool supportsFP40;
  };
}

#endif

