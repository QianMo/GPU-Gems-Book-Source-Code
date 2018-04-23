
#include "../gpucontext.hpp"

#include "oglfunc.hpp"
#include "ogltexture.hpp"
#include "oglcheckgl.hpp"

using namespace brook;
/* Allocates a floating point texture */ 
OGLTexture::OGLTexture (unsigned int width,
                        unsigned int height,
                        GPUContext::TextureFormat format,
                        const unsigned int glFormat[4][OGL_NUMFORMATS],
                        const unsigned int glType[4][OGL_NUMFORMATS],
                        const unsigned int sizeFactor[4][OGL_NUMFORMATS],
                        const unsigned int atomSize[4][OGL_NUMFORMATS]):
   _width(width), _height(height), _format(format) {
   _elementType=OGL_FLOAT;
   
   switch (_format) {
   case GPUContext::kTextureFormat_Float1:
   case GPUContext::kTextureFormat_Float2:
   case GPUContext::kTextureFormat_Float3:
   case GPUContext::kTextureFormat_Float4:
       _elementType=OGL_FLOAT;
       break;
   case GPUContext::kTextureFormat_Fixed1:
   case GPUContext::kTextureFormat_Fixed2:
   case GPUContext::kTextureFormat_Fixed3:
   case GPUContext::kTextureFormat_Fixed4:
       _elementType=OGL_FIXED;
       break;
   case GPUContext::kTextureFormat_Half1:
   case GPUContext::kTextureFormat_Half2:
   case GPUContext::kTextureFormat_Half3:
   case GPUContext::kTextureFormat_Half4:
       _elementType=OGL_HALF;
       break;
   }
   switch (_format) {
   case GPUContext::kTextureFormat_Float1:
   case GPUContext::kTextureFormat_Fixed1:
   case GPUContext::kTextureFormat_Half1:
      _components = 1;
      break;
   case GPUContext::kTextureFormat_Float2:
   case GPUContext::kTextureFormat_Fixed2:
   case GPUContext::kTextureFormat_Half2:
      _components = 2;
      break;
   case GPUContext::kTextureFormat_Float3:
   case GPUContext::kTextureFormat_Half3:
   case GPUContext::kTextureFormat_Fixed3:
      _components = 3;
      break;
   case GPUContext::kTextureFormat_Float4:
   case GPUContext::kTextureFormat_Half4:
   case GPUContext::kTextureFormat_Fixed4:
      _components = 4;
      break;
   default: 
      GPUError("Unkown Texture Format");
   }
   _atomsize = atomSize[_components-1][_elementType];
   _bytesize = _width*_height*sizeFactor[_components-1][_elementType]*_atomsize;
   _elemsize = sizeFactor[_components-1][_elementType];
   _nativeFormat = glFormat[_components-1][_elementType];

   glGenTextures(1, &_id);
   glActiveTextureARB(GL_TEXTURE0_ARB);
   glBindTexture (GL_TEXTURE_RECTANGLE_NV, _id);
   CHECK_GL();
   glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   glPixelStorei(GL_PACK_ALIGNMENT,1);
   // Create a texture with NULL data
   glTexImage2D (GL_TEXTURE_RECTANGLE_NV, 0, 
                 glType[_components-1][_elementType],
                 width, height, 0,
                 glFormat[_components-1][_elementType],
                 _elementType==OGL_FIXED?GL_UNSIGNED_BYTE:GL_FLOAT, NULL);
   CHECK_GL();
   
   glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_S, GL_CLAMP);
   glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_WRAP_T, GL_CLAMP);
   glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   CHECK_GL();
}

OGLTexture::~OGLTexture () {
  glDeleteTextures (1, &_id);
  CHECK_GL();
}


bool
OGLTexture::isFastSetPath( unsigned int inStrideBytes, 
                           unsigned int inWidth,
                           unsigned int inHeight,
                           unsigned int inElemCount ) const {
   return (inStrideBytes == _elemsize*_atomsize &&
           inElemCount   == inWidth*inHeight);
}

bool
OGLTexture::isFastGetPath( unsigned int inStrideBytes, 
                           unsigned int inWidth,
                           unsigned int inHeight,
                           unsigned int inElemCount ) const {
   return (inStrideBytes == _elemsize*_atomsize &&
           inElemCount   == inWidth*inHeight);
}


void
OGLTexture::copyToTextureFormat(const void *src, 
                                unsigned int srcStrideBytes, 
                                unsigned int srcElemCount,
                                void *dst) const {
   unsigned int i;
   
   switch (_components) {
   case 1:
   case 2:
   case 3:
   case 4:
      for (i=0; i<srcElemCount; i++) {
         memcpy(dst,src,_atomsize*_components);
         src = (((unsigned char *) (src)) + srcStrideBytes);
         dst = ((unsigned char *)dst) + _elemsize*_atomsize;
      }
      break;
   default: 
      GPUError("Unkown Texture Format");
   }
}


void
OGLTexture::copyFromTextureFormat(const void *src, 
                                  unsigned int dstStrideBytes, 
                                  unsigned int dstElemCount,
                                  void *dst) const {
   unsigned int i;
   
   switch (_components) {
   case 1:
   case 2: 
   case 3:
   case 4:
      for (i=0; i<dstElemCount; i++) {
         memcpy(dst,src,_atomsize*_components);
         dst = (((unsigned char *) (dst)) + dstStrideBytes);
         src = ((unsigned char *)src) + _elemsize*_atomsize;
      }
      break;
   default: 
      GPUError("Unknown Texture Format");
   }
}

void OGLTexture::getRectToCopy(
  unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax, const unsigned int* inExtents,
  int& outMinX, int& outMinY, int& outMaxX, int& outMaxY, size_t& outBaseOffset, bool& outFullStream, bool inUsesAddressTranslation )
{
  size_t r;
  size_t rank = inRank;
  size_t domainMin[4] = {0,0,0,0};
  size_t domainMax[4] = {1,1,1,1};
  size_t domainExtents[4] = {0,0,0,0};
  size_t minIndex = 0;
  size_t maxIndex = 0;
  size_t stride = 1;

  bool fullStream = true;
  for( r = 0; r < rank; r++ )
  {
    size_t d = rank - (r+1);

    domainMin[r] = inDomainMin[d];
    domainMax[r] = inDomainMax[d];
    domainExtents[r] = domainMax[r] - domainMin[r];
    size_t streamExtent = inExtents[d];
    if( streamExtent != domainExtents[r] )
      fullStream = false;

    minIndex += domainMin[r]*stride;
    maxIndex += (domainMax[r]-1)*stride;
    stride *= streamExtent;
  }

  if( inUsesAddressTranslation )
  {
    size_t minX = minIndex % _width;
    size_t minY = minIndex / _width;
    size_t maxX = maxIndex % _width;
    size_t maxY = maxIndex / _width;

    size_t baseOffset = minX;

    if( minY != maxY )
    {
      minX = 0;
      maxX = _width-1;
    }

    outMinX = minX;
    outMinY = minY;
    outMaxX = maxX+1;
    outMaxY = maxY+1;
    outBaseOffset = baseOffset;
  }
  else
  {
    outMinX = domainMin[0];
    outMinY = domainMin[1];
    outMaxX = domainMax[0];
    outMaxY = domainMax[1];
    outBaseOffset = 0;
  }
  outFullStream = fullStream;
}

void OGLTexture::setATData(
  const void* inStreamData, unsigned int inStrideBytes, unsigned int inRank,
  const unsigned int* inDomainMin, const unsigned int* inDomainMax, const unsigned int* inExtents,
  void* ioTextureData )
{
  // TIM: get all the fun information out of our streams
  size_t r;
  size_t rank = inRank;
  size_t domainMin[4] = {0,0,0,0};
  size_t domainMax[4] = {1,1,1,1};
  size_t strides[4] = {0,0,0,0};
  size_t stride = 1;
  const void *streamElement=inStreamData;
  for( r = 0; r < rank; r++ )
  {
    size_t d = rank - (r+1);

    domainMin[r] = inDomainMin[d];
    domainMax[r] = inDomainMax[d];
    size_t streamExtent = inExtents[d];
    strides[r] = stride;
    stride *= streamExtent;
  }
  
  
  const size_t componentSize = _atomsize*_elemsize;
  size_t x, y, z, w;
  for( w = domainMin[3]; w < domainMax[3]; w++ )
  {
    size_t offsetW = w*strides[3];
    for( z = domainMin[2]; z < domainMax[2]; z++ )
    {
      size_t offsetZ = offsetW + z*strides[2];
      for( y = domainMin[1]; y < domainMax[1]; y++ )
      {
        size_t offsetY = offsetZ + y*strides[1];
        for( x = domainMin[0]; x < domainMax[0]; x++ )
        {
          size_t streamIndex = offsetY + x*strides[0];

          void* textureElement = ((unsigned char*)ioTextureData) + streamIndex*_components*_atomsize;
          memcpy(textureElement,streamElement,componentSize);
          streamElement=((unsigned char *)streamElement)+componentSize;
          
        }
      }
    }
  }
}

void OGLTexture::getATData(
  void* outStreamData, unsigned int inStrideBytes, unsigned int inRank,
  const unsigned int* inDomainMin, const unsigned int* inDomainMax, const unsigned int* inExtents,
  const void* inTextureData )
{
  // TIM: get all the fun information out of our streams
  size_t r;
  size_t rank = inRank;
  size_t domainMin[4] = {0,0,0,0};
  size_t domainMax[4] = {1,1,1,1};
  size_t strides[4] = {0,0,0,0};
  size_t stride = 1;

  for( r = 0; r < rank; r++ )
  {
    size_t d = rank - (r+1);

    domainMin[r] = inDomainMin[d];
    domainMax[r] = inDomainMax[d];
    size_t streamExtent = inExtents[d];
    strides[r] = stride;
    stride *= streamExtent;
  }

  const size_t streamElementSize = inStrideBytes;
  const size_t copySize =  _atomsize*_elemsize;
  void* streamElement = outStreamData;
  size_t x, y, z, w;
  for( w = domainMin[3]; w < domainMax[3]; w++ )
  {
    size_t offsetW = w*strides[3];
    for( z = domainMin[2]; z < domainMax[2]; z++ )
    {
      size_t offsetZ = offsetW + z*strides[2];
      for( y = domainMin[1]; y < domainMax[1]; y++ )
      {
        size_t offsetY = offsetZ + y*strides[1];
        for( x = domainMin[0]; x < domainMax[0]; x++ )
        {
          size_t streamIndex = offsetY + x*strides[0];
          const void* textureElement = ((unsigned char*)inTextureData) + streamIndex*copySize;
          memcpy(streamElement,textureElement,copySize);
          streamElement=((unsigned char*)streamElement)+streamElementSize;
        }
      }
    }
  }
}
