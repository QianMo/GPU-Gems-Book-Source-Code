// dx9texture.cpp
#include "dx9texture.hpp"

using namespace brook;

static const int kComponentSizes[] =
{
  sizeof(float),
  sizeof(unsigned char),
  sizeof(unsigned short)
};

DX9Texture::DX9Texture( GPUContextDX9* inContext, int inWidth, int inHeight, int inComponents, ComponentType inComponentType )
	: width(inWidth),
  height(inHeight),
  components(inComponents),
  internalComponents(inComponents),
  componentType(inComponentType),
  componentSize(0),
  dirtyFlags(0),
  device(NULL),
  textureHandle(NULL),
  surfaceHandle(NULL),
  shadowSurface(NULL)
{
  componentSize = kComponentSizes[ componentType ];

    _context = inContext;
	device = _context->getDevice();
    device->AddRef();
}

static D3DFORMAT getFormatForComponentCount( int inComponentCount, DX9Texture::ComponentType inType )
{
  if( inType == DX9Texture::kComponentType_Float )
  {
    switch( inComponentCount )
    {
    case 1:
      return D3DFMT_R32F;
    case 2:
      return D3DFMT_G32R32F;
    case 4:
      return D3DFMT_A32B32G32R32F;
    default:
      return D3DFMT_UNKNOWN;
    }
  }
  else if( inType == DX9Texture::kComponentType_Half )
  {
    switch( inComponentCount )
    {
    case 1:
      return D3DFMT_R16F;
    case 2:
      return D3DFMT_G16R16F;
    case 4:
      return D3DFMT_A16B16G16R16F;
    default:
      return D3DFMT_UNKNOWN;
    }
  }
  else if( inType == DX9Texture::kComponentType_Fixed )
  {
    switch( inComponentCount )
    {
    case 1:
      return D3DFMT_L8;
    case 4:
      return D3DFMT_A8R8G8B8;
    default:
      return D3DFMT_UNKNOWN;
    }
  }
  return D3DFMT_UNKNOWN;
}

bool DX9Texture::initialize()
{
  HRESULT result;

  D3DFORMAT dxFormat;
  bool validFormat = false;
  for( int i = components; i <= 4 && !validFormat; i++ )
  {
      dxFormat = getFormatForComponentCount( i, componentType );
      if( dxFormat != D3DFMT_UNKNOWN
          && _context->isRenderTextureFormatValid( dxFormat ) )
      {
          validFormat = true;
          internalComponents = i;
      }
  }
  if( !validFormat )
  {
    DX9WARN << "Could not find supported texture format." << std::endl;
    return false;
  }

	result = device->CreateTexture( width, height, 1, D3DUSAGE_RENDERTARGET, dxFormat, D3DPOOL_DEFAULT, &textureHandle, NULL );
  if( FAILED( result ) )
  {
    DX9WARN << "Unable to create render target texture of size "
      << width << " by " << height << " by " << (componentType == kComponentType_Float ? "float" : "ubyte") << components << ".";
    return false;
  }
	result = textureHandle->GetSurfaceLevel( 0, &surfaceHandle );
	DX9AssertResult( result, "GetSurfaceLevel failed" );

	result = device->CreateOffscreenPlainSurface( width, height, dxFormat, D3DPOOL_SYSTEMMEM, &shadowSurface, NULL );
  if( FAILED( result ) )
  {
    DX9WARN << "Unable to create floating-point plain surface of size "
      << width << " by " << height << ".";
    return false;
  }
	return true;
}

DX9Texture::~DX9Texture()
{
  DX9LOG(2) << "~DX9Texture";
  if( shadowSurface != NULL )
    shadowSurface->Release();
  if( surfaceHandle != NULL )
    surfaceHandle->Release();
  if( textureHandle != NULL )
    textureHandle->Release();
  if( device != NULL )
    device->Release();
}

DX9Texture* DX9Texture::create( GPUContextDX9* inContext, int inWidth, int inHeight, int inComponents, ComponentType inType  )
{
  DX9PROFILE("DX9Texture::create")
  DX9Texture* result = new DX9Texture( inContext, inWidth, inHeight, inComponents, inType );
  if( result->initialize() )
    return result;
  delete result;
  return NULL;
}

void DX9Texture::setData( const float* inData, unsigned int inStride, unsigned int inCount,
                         unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
                         const unsigned int* inExtents, bool inUsesAddressTranslation  )
{
  DX9PROFILE("DX9Texture::setData")

	setShadowData( inData, inStride, inCount, inRank, inDomainMin, inDomainMax, inExtents, inUsesAddressTranslation );
  markShadowDataChanged();
}

void DX9Texture::getData( float* outData, unsigned int inStride, unsigned int inCount,
                         unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
                         const unsigned int* inExtents, bool inUsesAddressTranslation  )
{
  DX9PROFILE("DX9Texture::getData")

  if( dirtyFlags & kShadowDataDirty )
    flushCachedToShadow();
  getShadowData( outData, inStride, inCount, inRank, inDomainMin, inDomainMax, inExtents, inUsesAddressTranslation );
}

void DX9Texture::markCachedDataChanged()
{
  dirtyFlags = kShadowDataDirty;
}

void DX9Texture::markShadowDataChanged()
{
  dirtyFlags = kCachedDataDirty;
}

void DX9Texture::validateCachedData()
{
  if( !(dirtyFlags & kCachedDataDirty) ) return;
  flushShadowToCached();
}

void DX9Texture::validateShadowData()
{
  if( !(dirtyFlags & kShadowDataDirty) ) return;
  flushCachedToShadow();
}

void DX9Texture::flushCachedToShadow()
{
  DX9PROFILE("DX9Texture::flushCachedToShadow")

  HRESULT result = device->GetRenderTargetData( surfaceHandle, shadowSurface );
	DX9AssertResult( result, "Failed to copy floating-point render target to plain surface." );
  dirtyFlags &= ~kShadowDataDirty;
}

void DX9Texture::flushShadowToCached()
{
  DX9PROFILE("DX9Texture::flushShadowToCached")

  HRESULT result = device->UpdateSurface( shadowSurface, NULL, surfaceHandle, NULL );
	DX9AssertResult( result, "Failed to copy floating-point plain surface to render target." );
  dirtyFlags &= ~kCachedDataDirty;
}

void DX9Texture::getShadowData( void* outData, unsigned int inStride, unsigned int inCount,
                               unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
                               const unsigned int* inExtents, bool inUsesAddressTranslation )
{
  DX9PROFILE("DX9Texture::getShadowData")

  if( !inUsesAddressTranslation )
  {
    HRESULT result;

    RECT rectToLock;
    bool isWholeBuffer;
    findRectForCopy( inRank, inDomainMin, inDomainMax, inExtents,
      inUsesAddressTranslation, rectToLock, isWholeBuffer );

    int domainWidth = rectToLock.right - rectToLock.left;
    int domainHeight = rectToLock.bottom - rectToLock.top;

    D3DLOCKED_RECT info;
    result = shadowSurface->LockRect( &info, &rectToLock, D3DLOCK_READONLY );
    DX9AssertResult( result, "LockRect failed" );

    copyData( outData, domainWidth*inStride, inStride,
              info.pBits, info.Pitch, internalComponents*componentSize,
              domainWidth, domainHeight, components,componentSize );

    result = shadowSurface->UnlockRect();
    DX9AssertResult( result, "UnlockRect failed" );
  }
  else // using address translation
  {
    HRESULT result;

    RECT rectToLock;
    bool isWholeBuffer;
    size_t baseX, baseY;
    findRectForCopyAT( inRank, inDomainMin, inDomainMax, inExtents,
      inUsesAddressTranslation, rectToLock, isWholeBuffer,
      width, height, baseX, baseY );

    int rectWidth = rectToLock.right - rectToLock.left;
    int rectHeight = rectToLock.bottom - rectToLock.top;

    D3DLOCKED_RECT info;
    result = shadowSurface->LockRect( &info, &rectToLock, D3DLOCK_READONLY );
    DX9AssertResult( result, "LockRect failed" );

    if( isWholeBuffer )
    {
      copyAllDataAT( outData, width*inStride, inStride,
                     info.pBits, info.Pitch, internalComponents*componentSize,
                     width, height, components,componentSize, inRank, inExtents  );
    }
    else
    {
      getDataAT( outData, inRank, inDomainMin, inDomainMax, inExtents, components*componentSize,
        info.pBits, info.Pitch, rectWidth, rectHeight, internalComponents*componentSize, baseX, baseY );
    }

    result = shadowSurface->UnlockRect();
    DX9AssertResult( result, "UnlockRect failed" );
  }
}

void DX9Texture::setShadowData( const void* inData, unsigned int inStride, unsigned int inCount,
                               unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
                               const unsigned int* inExtents, bool inUsesAddressTranslation )
{
  DX9PROFILE("DX9Texture::setShadowData")

  if( !inUsesAddressTranslation )
  {
    RECT rectToLock;
    bool isWholeBuffer;

    findRectForCopy( inRank, inDomainMin, inDomainMax, inExtents,
      inUsesAddressTranslation, rectToLock, isWholeBuffer );

    int domainWidth = rectToLock.right - rectToLock.left;
    int domainHeight = rectToLock.bottom - rectToLock.top;

    if( !isWholeBuffer )
    {
      // we are only writing to a portion of the buffer,
      // and so we must be sure to validate the rest of it
      validateShadowData();
    }

    HRESULT result;
	  D3DLOCKED_RECT info;

	  result = shadowSurface->LockRect( &info, &rectToLock, 0 );
	  DX9AssertResult( result, "LockRect failed" );

          copyData( info.pBits, info.Pitch, internalComponents*componentSize,
                    inData, domainWidth*inStride, inStride,
                    domainWidth, domainHeight, components,componentSize );

	  result = shadowSurface->UnlockRect();
	  DX9AssertResult( result, "UnlockRect failed" );
  }
  else // using address translation
  {
    RECT rectToLock;
    bool isWholeBuffer;
    size_t baseX, baseY;

    findRectForCopyAT( inRank, inDomainMin, inDomainMax, inExtents,
      inUsesAddressTranslation, rectToLock, isWholeBuffer,
      width, height, baseX, baseY );

    int rectWidth = rectToLock.right - rectToLock.left;
    int rectHeight = rectToLock.bottom - rectToLock.top;

    if( !isWholeBuffer )
    {
      // we are only writing to a portion of the buffer,
      // and so we must be sure to validate the rest of it
      validateShadowData();
    }

    HRESULT result;
    D3DLOCKED_RECT info;

    result = shadowSurface->LockRect( &info, &rectToLock, 0 );
    DX9AssertResult( result, "LockRect failed" );

    if( isWholeBuffer )
    {
      copyAllDataAT( info.pBits, info.Pitch, internalComponents*componentSize,
                     inData, width*inStride, inStride,
                     width, height, components,componentSize, inRank, inExtents  );
    }
    else
    {
      setDataAT( inData, inRank, inDomainMin, inDomainMax, inExtents, components*componentSize,
        info.pBits, info.Pitch, rectWidth, rectHeight, internalComponents*componentSize, baseX, baseY );
    }

    result = shadowSurface->UnlockRect();
    DX9AssertResult( result, "UnlockRect failed" );
  }
}

void DX9Texture::getPixelAt( int x, int y, float4& outResult ) {

  unsigned int domainMin[2] = { y, x };
  unsigned int domainMax[2] = { y+1, x+1 };
  unsigned int extents[2] = { height, width };

  getData( (float*) &outResult, components*sizeof(float), 2,
    2, domainMin, domainMax, extents, false );
}

void DX9Texture::findRectForCopy( unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
  const unsigned int* inExtents, bool inUsesAddressTranslation, RECT& outRect, bool& outFullBuffer )
{
  RECT rect;
  if( inRank == 1 )
  {
    rect.left = inDomainMin[0];
    rect.top = 0;
    rect.right = inDomainMax[0];
    rect.bottom = 1;
  }
  else
  {
    rect.left = inDomainMin[1];
    rect.top = inDomainMin[0];
    rect.right = inDomainMax[1];
    rect.bottom = inDomainMax[0];
  }
  outRect = rect;

  int domainWidth = rect.right - rect.left;
  int domainHeight = rect.bottom - rect.top;

  if( domainWidth != width || domainHeight != height )
  {
    outFullBuffer = false;
  }
  else
    outFullBuffer = true;
}

void DX9Texture::copyData( void* toBuffer, size_t toRowStride,  size_t toElementStride,
                           const void* fromBuffer, size_t fromRowStride, size_t fromElementStride,
                           size_t columnCount, size_t rowCount, size_t numElements,size_t elementSize )
{
  char* outputLine = (char*)toBuffer;
  const char* inputLine = (const char*)fromBuffer;

//  size_t componentCount = elementSize / sizeof(float);

  for( size_t y = 0; y < rowCount; y++ )
  {
    char* outputPixel = outputLine;
    const char* inputPixel = inputLine;
    for( size_t x = 0; x < columnCount; x++ )
    {
      // TIM: for now we assume floating-point components
      char* output = outputPixel;
      const char* input = inputPixel;
      if (elementSize!=1)
        for( size_t i = 0; i < numElements*elementSize; i++ )
          *output++ = *input++;
      else {
          if (fromElementStride!=toElementStride&&toElementStride==1)
            input+=(fromElementStride==4?fromElementStride-2:fromElementStride-1);//offset the input if writing to it
          for( size_t i =((output+=(toElementStride==1?1:toElementStride-1)),0); i <(numElements>3?3:numElements); i++ )//offset the output, but remember the alpha channel is separate
              *--output = *input++;// I've always wanted to do that
        if(numElements>3)
            output[3]=*input++;//now to copy alpha
      }

      inputPixel += fromElementStride;
      outputPixel += toElementStride;
    }
    inputLine += fromRowStride;
    outputLine += toRowStride;
  }
}

void DX9Texture::findRectForCopyAT( unsigned int inRank, const unsigned int* inDomainMin, const unsigned int* inDomainMax,
  const unsigned int* inExtents, bool inUsesAddressTranslation, RECT& outRect, bool& outFullBuffer,
  size_t inWidth, size_t inHeight, size_t& outBaseX, size_t& outBaseY )
{
  size_t r;
  size_t rank = inRank;

  size_t stride = 1;
  size_t minIndex = 0;
  size_t maxIndex = 0;
  bool wholeBuffer = true;
  for( r = 0; r < rank; r++ )
  {
    size_t d = rank - (r+1);

    size_t domainMin = inDomainMin[d];
    size_t domainMax = inDomainMax[d];
    size_t domainExtent = domainMax - domainMin;
    size_t streamExtent = inExtents[d];

    if( streamExtent != domainExtent )
      wholeBuffer = false;

    minIndex += domainMin * stride;
    maxIndex += (domainMax-1) * stride;
    stride *= streamExtent;
  }

  size_t minX, minY, maxX, maxY;

  minX = minIndex % inWidth;
  minY = minIndex / inWidth;
  maxX = maxIndex % inWidth;
  maxY = maxIndex / inWidth;

  RECT rect;
  rect.left = (minY == maxY) ? minX : 0;
  rect.top = minY;
  rect.right = (minY == maxY) ? maxX+1 : inWidth;
  rect.bottom = maxY+1;

  outRect = rect;
  outFullBuffer = wholeBuffer;
  outBaseX = minX;
  outBaseY = minY;
}

void DX9Texture::copyAllDataAT( void* toBuffer, size_t toRowStride, size_t toElementStride,
                                const void* fromBuffer, size_t fromRowStride, size_t fromElementStride,
                                size_t columnCount, size_t rowCount, size_t numElements,size_t elementSize, size_t inRank, const unsigned int* inExtents )
{
  size_t elementCount = 1;
  for( size_t r = 0; r < inRank; r++ )
    elementCount *= inExtents[r];

  char* outputLine = (char*)toBuffer;
  const char* inputLine = (const char*)fromBuffer;

//  size_t componentCount = elementSize / sizeof(float);

  size_t copiedElementCount = 0;
  for( size_t y = 0; y < rowCount; y++ )
  {
    char* outputPixel = outputLine;
    const char* inputPixel = inputLine;
    for( size_t x = 0; x < columnCount; x++ )
    {

      // TIM: for now we assume floating-point components
      char* output = outputPixel;
      const char* input = inputPixel;
      if (elementSize!=1)
        for( size_t i = 0; i < numElements*elementSize; i++ )
          *output++ = *input++;
      else {
          if (fromElementStride!=toElementStride&&toElementStride==1)
            input+=(fromElementStride==4?fromElementStride-2:fromElementStride-1);//offset the input if writing to it
          for( size_t i =((output+=(toElementStride==1?1:toElementStride-1)),0); i <(numElements>3?3:numElements); i++ )//offset the output, but remember the alpha channel is separate
              *--output = *input++;// I've always wanted to do that
        if(numElements>3)
            output[3]=*input++;//now to copy alpha
      }

      inputPixel += fromElementStride;
      outputPixel += toElementStride;

      copiedElementCount++;
      if( copiedElementCount == elementCount )
        return;
    }
    inputLine += fromRowStride;
    outputLine += toRowStride;
  }
}

void DX9Texture::getDataAT( void* streamData, unsigned int streamRank,
  const unsigned int* streamDomainMin, const unsigned int* streamDomainMax,
  const unsigned int* streamExtents, size_t streamElementStride,
  const void* textureData, size_t textureLineStride, size_t textureWidth, size_t textureHeight,
  size_t textureElementStride, size_t textureBaseX, size_t textureBaseY )
{
  size_t r;
  size_t rank = streamRank;
  size_t domainMin[4] = {0,0,0,0};
  size_t domainMax[4] = {1,1,1,1};
  size_t domainExtents[4] = {1,1,1,1};
  size_t extents[4] = {1,1,1,1};
  size_t strides[4] = {0,0,0,0};

  size_t stride = 1;
  for( r = 1; r <= rank; r++ )
  {
    size_t d = rank - r;

    domainMin[d] = streamDomainMin[r];
    domainMax[d] = streamDomainMax[r];
    extents[d] = streamExtents[r];

    domainExtents[d] = domainMax[d] - domainMin[d];
    strides[d] = stride;
    stride *= domainExtents[d];
  }

  const char* textureBuffer = (const char*) textureData;
  char* streamBuffer = (char*) streamData;

  size_t x, y, z, w;
  for( w = domainMin[3]; w < domainMax[3]; w++ )
  {
    size_t offsetW = w * strides[3];
    for( z = domainMin[2]; z < domainMax[2]; z++ )
    {
      size_t offsetZ = offsetW + z * strides[2];
      for( y = domainMin[1]; y < domainMax[1]; y++ )
      {
        size_t offsetY = offsetZ + y * strides[1];
        for( x = domainMin[0]; x < domainMax[0]; x++ )
        {
          size_t offsetX = offsetY + x * strides[0];
          size_t streamOffset = offsetX * streamElementStride;

          size_t textureX = (offsetX % textureWidth) - textureBaseX;
          size_t textureY = (offsetX / textureWidth) - textureBaseY;
          size_t textureOffset = textureY * textureLineStride + textureX * textureElementStride;

          const char* textureElement = (textureBuffer + textureOffset);
          char* streamElement = (streamBuffer + streamOffset);

          for( int i = 0; i < components*componentSize; i++ )
            *streamElement++ = *textureElement++;
        }
      }
    }
  }
}

void DX9Texture::setDataAT( const void* streamData, unsigned int streamRank,
                           const unsigned int* streamDomainMin, const unsigned int* streamDomainMax,
                           const unsigned int* streamExtents, size_t streamElementStride,
                           void* textureData, size_t textureLineStride, size_t textureWidth, size_t textureHeight,
                           size_t textureElementStride, size_t textureBaseX, size_t textureBaseY )
{
  size_t r;
  size_t rank = streamRank;
  size_t domainMin[4] = {0,0,0,0};
  size_t domainMax[4] = {1,1,1,1};
  size_t domainExtents[4] = {1,1,1,1};
  size_t extents[4] = {1,1,1,1};
  size_t strides[4] = {0,0,0,0};

  size_t stride = 1;
  for( r = 1; r <= rank; r++ )
  {
    size_t d = rank - r;

    domainMin[d] = streamDomainMin[r];
    domainMax[d] = streamDomainMax[r];
    extents[d] = streamExtents[r];

    domainExtents[d] = domainMax[d] - domainMin[d];
    strides[d] = stride;
    stride *= domainExtents[d];
  }

  char* textureBuffer = (char*) textureData;
  const char* streamBuffer = (const char*) streamData;

  size_t x, y, z, w;
  for( w = domainMin[3]; w < domainMax[3]; w++ )
  {
    size_t offsetW = w * strides[3];
    for( z = domainMin[2]; z < domainMax[2]; z++ )
    {
      size_t offsetZ = offsetW + z * strides[2];
      for( y = domainMin[1]; y < domainMax[1]; y++ )
      {
        size_t offsetY = offsetZ + y * strides[1];
        for( x = domainMin[0]; x < domainMax[0]; x++ )
        {
          size_t offsetX = offsetY + x * strides[0];
          size_t streamOffset = offsetX * streamElementStride;

          size_t textureX = (offsetX % textureWidth) - textureBaseX;
          size_t textureY = (offsetX / textureWidth) - textureBaseY;
          size_t textureOffset = textureY * textureLineStride + textureX * textureElementStride;

          char* textureElement = (textureBuffer + textureOffset);
          const char* streamElement = (streamBuffer + streamOffset);

          for( int c = 0; c < components*componentSize; c++ )
            *textureElement++ = *streamElement++;
        }
      }
    }
  }
}
