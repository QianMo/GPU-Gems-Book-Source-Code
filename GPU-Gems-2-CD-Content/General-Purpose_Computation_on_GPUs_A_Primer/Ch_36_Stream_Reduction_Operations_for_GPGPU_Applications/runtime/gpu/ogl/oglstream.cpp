
#include "oglfunc.hpp"
#include "oglcontext.hpp"
#include "ogltexture.hpp"
#include "oglwindow.hpp"
#include "oglfunc.hpp"
#include "oglcheckgl.hpp"

using namespace brook;

void 
OGLContext::copy_to_pbuffer(OGLTexture *texture) {
  int w = texture->width();
  int h = texture->height();

  OGLPixelShader *passthrough = (OGLPixelShader *)
    getPassthroughPixelShader();

  _wnd->bindPbuffer(w, h, 1, texture->components());
  
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB,  
                   passthrough->id);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_NV, texture->id());

  glViewport (0, 0, w, h);
     
  if (w == 1 && h == 1) {
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0.5f, 0.5f);
    glVertex2f(-1.0f, -1.0f);
    glVertex2f(3.0f, -1.0f);
    glVertex2f(-1.0f, 3.0f);
    glEnd();
  } else if (h == 1) {
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0.0f, 0.5f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(w*2.0f, 0.5f);
    glVertex2f(3.0f, -1.0f);
    glTexCoord2f(0.0f, 0.5f);
    glVertex2f(-1.0f, 3.0f);
    glEnd();
  } else if (w == 1) {
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0.5f, 0.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(0.5f, 0.0f);
    glVertex2f(3.0f, -1.0f);
    glTexCoord2f(0.5f, h*2.0f);
    glVertex2f(-1.0f, 3.0f);
    glEnd();
  } else {
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(w*2.0f, 0.0f);
    glVertex2f(3.0f, -1.0f);
    glTexCoord2f(0.0f, h*2.0f);
    glVertex2f(-1.0f, 3.0f);
    glEnd();
  }
  glFinish();
  CHECK_GL();
}

void 
OGLContext::setTextureData(TextureHandle inTexture, 
                           const float* inData,
                           unsigned int inStrideBytes,
                           unsigned int inComponentCount,
                           unsigned int inRank,
                           const unsigned int* inDomainMin,
                           const unsigned int* inDomainMax,
                           const unsigned int* inExtents, bool inUsesAddressTranslation ) {
  void *t;
  
  OGLTexture *oglTexture = (OGLTexture *) inTexture;
  
  int minX, minY, maxX, maxY;
  size_t baseOffset;
  bool fullStream;
  oglTexture->getRectToCopy( inRank, inDomainMin, inDomainMax, inExtents,
    minX, minY, maxX, maxY, baseOffset, fullStream, inUsesAddressTranslation );
  int rectW = maxX - minX;
  int rectH = maxY - minY;

  bool fastPath = oglTexture->isFastSetPath( inStrideBytes, 
                                             rectW, rectH,
                                             inComponentCount );
  fastPath = fastPath && !inUsesAddressTranslation;

  glBindTexture (GL_TEXTURE_RECTANGLE_NV, oglTexture->id());
 
  if (fastPath) {
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, minX, minY, 
                    rectW, //oglTexture->width(), 
                    rectH, //oglTexture->height(), 
                    oglTexture->nativeFormat(),
                    oglTexture->elementType()==OGLTexture::OGL_FIXED?GL_UNSIGNED_BYTE:GL_FLOAT, inData);
    return;
  }
  
  // TIM: could improve this in the domain case
  // by only allocating as much memory as the
  // domain needs
  t = malloc (oglTexture->bytesize());
  if( !fullStream && inUsesAddressTranslation )
  {
    // TIM: hack to get the texture data into our buffer
    int texW = oglTexture->width();
    int texH = oglTexture->height();
    unsigned int texDomainMin[] = {0,0};
    unsigned int texDomainMax[] = { texH, texW };
    unsigned int texExtents[] = { texH, texW };
    getTextureData( oglTexture,(float*) t, inStrideBytes, texW*texH, 2,
      texDomainMin, texDomainMax, texExtents, false );

    oglTexture->setATData(
      inData, inStrideBytes, inRank, inDomainMin, inDomainMax, inExtents, t );

    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, 
      texW, //oglTexture->width(), 
      texH, //oglTexture->height(),  
      oglTexture->nativeFormat(),
      oglTexture->elementType()==OGLTexture::OGL_FIXED?GL_UNSIGNED_BYTE:GL_FLOAT, t);

  }
  else
  {
    oglTexture->copyToTextureFormat(inData, 
                                    inStrideBytes, 
                                    inComponentCount,
                                    t);

    glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, minX, minY, 
                    maxX - minX, //oglTexture->width(), 
                    maxY - minY, //oglTexture->height(),  
                    oglTexture->nativeFormat(),
                    oglTexture->elementType()==OGLTexture::OGL_FIXED?GL_UNSIGNED_BYTE:GL_FLOAT, t);
  }
  
  free(t);
  CHECK_GL();
}

void 
OGLContext::getTextureData( TextureHandle inTexture,
                            float* outData,
                            unsigned int inStrideBytes,
                            unsigned int inComponentCount,
                            unsigned int inRank,
                            const unsigned int* inDomainMin,
                            const unsigned int* inDomainMax,
                            const unsigned int* inExtents, bool inUsesAddressTranslation ) {
   void *t = outData;

   OGLTexture *oglTexture = (OGLTexture *) inTexture;

   int minX, minY, maxX, maxY;
   size_t baseOffset;
   bool fullStream;
   oglTexture->getRectToCopy( inRank, inDomainMin, inDomainMax, inExtents,
     minX, minY, maxX, maxY, baseOffset, fullStream, inUsesAddressTranslation );
   int rectW = maxX - minX;
   int rectH = maxY - minY;
   
   bool fastPath = oglTexture->isFastGetPath( inStrideBytes, 
                                              rectW, rectH,
                                              inComponentCount); 
   if (!fastPath)
     t = malloc (oglTexture->bytesize());

   copy_to_pbuffer(oglTexture);
   CHECK_GL();
   glPixelStorei(GL_PACK_ALIGNMENT,1);   
   // read back the whole thing, 
   unsigned int elemsize=oglTexture->numInternalComponents();//we're always reading from a float pbuffer, therefore we have to give it a reasonable constant for FLOAT, not for BYTE... luminance is wrong here.

   glReadPixels (minX, minY,
              rectW,
              rectH, 
              elemsize==1?GL_RED:(elemsize==3?GL_RGB:GL_RGBA),
              oglTexture->elementType()==OGLTexture::OGL_FIXED?GL_UNSIGNED_BYTE:GL_FLOAT, t);
   CHECK_GL();

   if (!fastPath) {
     if( !inUsesAddressTranslation || fullStream)
     {
       oglTexture->copyFromTextureFormat(t, 
         inStrideBytes, 
         inComponentCount,
         outData);
     }
     else
     {
       oglTexture->getATData(outData, inStrideBytes,
         inRank, inDomainMin, inDomainMax, inExtents, t );
     }
     free(t);
   }
  CHECK_GL();
}


void 
OGLContext::releaseTexture( TextureHandle inTexture ) {
  delete (OGLTexture *) inTexture;
}



