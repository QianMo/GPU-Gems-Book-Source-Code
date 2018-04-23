#include "gpuruntime.hpp"
#include "gpubase.hpp"

#include "gpucontext.hpp"
#include "gpukernel.hpp"
#include "gpustream.hpp"
#include "gpuiterator.hpp"

namespace brook
{

  GPURuntime::GPURuntime()
    : _context(NULL)
  {
    for( size_t i = 0; i < kGPUReductionTempBufferCount; i++ )
    {
      _reductionTempBuffers[i] = NULL;
      _reductionTempBufferWidths[i] = 0;
      _reductionTempBufferHeights[i] = 0;
      _reductionTempBufferComponents[i] = 0;
    }
    _reductionTargetBuffer = NULL;
    _reductionTargetBufferComponents = 0;
  }

  bool GPURuntime::initialize( GPUContext* inContext )
  {
    GPUAssert( inContext, "No context provided for gpu runtime." );

    _context = inContext;
    return true;
  }

  GPURuntime::~GPURuntime()
  {
    // TODO: release passthroughs and such
    if( _context )
      delete _context;
  }

  Kernel* GPURuntime::CreateKernel(const void* source[]) {
    Kernel* result = GPUKernel::create( this, source );
    GPUAssert( result != NULL, "Unable to allocate a kernel, exiting." );
    return result;
  }

  Stream* GPURuntime::CreateStream(
    unsigned int fieldCount, const StreamType fieldTypes[],
    unsigned int dims, const unsigned int extents[])
  {
    Stream* result = GPUStream::create( this, fieldCount, fieldTypes, dims, extents );
    GPUAssert( result != NULL, "Unable to allocate a stream, exiting." );
    return result;
  }

  Iter* GPURuntime::CreateIter(
    StreamType type, unsigned int dims, 
    const unsigned int extents[], const float r[] )
  {
    Iter* result = GPUIterator::create( this, type, dims, extents , r );
    GPUAssert( result != NULL, "Unable to allocate an iterator, exiting." );
    return result;
  }

  GPUContext::TextureHandle GPURuntime::getReductionTempBuffer(
    GPUReductionTempBufferID inBufferID,
    size_t inMinWidth, size_t inMinHeight, size_t inMinComponents,
    size_t* outWidth, size_t* outHeight, size_t* outComponents )
  {
    GPUAssert( inBufferID >= 0 && inBufferID < kGPUReductionTempBufferCount,
      "Invalid reduction temp buffer requested." );

    GPUContext::TextureHandle& buffer = _reductionTempBuffers[inBufferID];
    size_t& width = _reductionTempBufferWidths[inBufferID];
    size_t& height = _reductionTempBufferHeights[inBufferID];
    size_t& components = _reductionTempBufferComponents[inBufferID];
    
    if( buffer == NULL || width < inMinWidth || height < inMinHeight || components < inMinComponents )
    {
      if( buffer != NULL )
        _context->releaseTexture( buffer );

      if( inMinWidth > width )
        width = inMinWidth;
      if( inMinHeight > height )
        height = inMinHeight;
      if( inMinComponents > components )
        components = inMinComponents;

      GPUContext::TextureFormat format = (GPUContext::TextureFormat)(GPUContext::kTextureFormat_Float1 + (inMinComponents-1));
      buffer = _context->createTexture2D( width, height, format );
      GPUAssert( buffer != NULL, "Failed to allocate reduction temp buffer." );
    }
    
    *outWidth = width;
    *outHeight = height;
    return buffer;
  }

  GPUContext::TextureHandle GPURuntime::getReductionTargetBuffer( size_t inMinComponents )
  {
    if( inMinComponents > _reductionTargetBufferComponents )
    {
        _context->releaseTexture( _reductionTargetBuffer );
        _reductionTargetBuffer = NULL;
    }
    if( _reductionTargetBuffer == NULL )
    {
      GPUContext::TextureFormat format = (GPUContext::TextureFormat)(GPUContext::kTextureFormat_Float1 + (inMinComponents-1));
      _reductionTargetBuffer = _context->createTexture2D( 1, 1, format );
      _reductionTargetBufferComponents = inMinComponents;
      GPUAssert( _reductionTargetBuffer != NULL, "Failed to allocate reduction target buffer." );
    }
    return _reductionTargetBuffer;
  }

  // TIM: hacky magick for raytracer
  void GPURuntime::hackEnableWriteMask() {
    _context->hackEnableWriteMask();
  }

  void GPURuntime::hackDisableWriteMask() {
    _context->hackDisableWriteMask();
  }

  void GPURuntime::hackSetWriteMask( Stream* inStream )
  {
    GPUAssert( inStream, "NULL stream provided to hackSetWriteMask" );
    GPUAssert( inStream->getFieldCount() == 1, "only one field allowed for write mask" );
    GPUStream* gpuStream = (GPUStream*) inStream;
    GPUContext::TextureHandle textureHandle = gpuStream->getIndexedFieldTexture( 0 );
    _context->hackSetWriteMask( textureHandle );
  }

  void GPURuntime::hackBeginWriteQuery() {
    _context->hackBeginWriteQuery();
  }

  int GPURuntime::hackEndWriteQuery() {
    return _context->hackEndWriteQuery();
  }

  void GPURuntime::hackRestoreContext() {
    _context->hackRestoreContext();
  }
}


