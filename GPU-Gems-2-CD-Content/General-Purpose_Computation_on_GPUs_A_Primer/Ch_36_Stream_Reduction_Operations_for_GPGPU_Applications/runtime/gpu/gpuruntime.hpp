// gpuruntime.hpp
#ifndef GPU_RUNTIME_HPP
#define GPU_RUNTIME_HPP

#include "gpucontext.hpp"

namespace brook
{
  enum
  {
    kGPUReductionTempBuffer_Swap0 = 0,
    kGPUReductionTempBuffer_Swap1,
    kGPUReductionTempBuffer_Slop,
    kGPUReductionTempBufferCount
  };
  typedef unsigned int GPUReductionTempBufferID;

  class GPURuntime : public Runtime
  {
  public:

    virtual Kernel* CreateKernel(const void*[]);
    virtual Stream* CreateStream(unsigned int fieldCount,
                                 const StreamType fieldTypes[],
                                 unsigned int dims, 
                                 const unsigned int extents[]);
    virtual Iter* CreateIter(StreamType type, 
                             unsigned int dims, 
                             const unsigned int extents[],
                             const float range[]);
    virtual ~GPURuntime();

    // TIM: hacky magick for raytracer
    virtual void hackEnableWriteMask();
    virtual void hackDisableWriteMask();
    virtual void hackSetWriteMask( Stream* );
    virtual void hackBeginWriteQuery();
    virtual int hackEndWriteQuery();

    virtual void hackRestoreContext();

    // internal GPURuntime methods (not in Runtime interface)
    GPUContext* getContext() { return _context; }
    GPUContext::TextureHandle getReductionTempBuffer(
      GPUReductionTempBufferID inBufferID,
      size_t inMinWidth, size_t inMinHeight, size_t inMinComponents,
      size_t* outWidth, size_t* outHeight, size_t* outComponents );
    GPUContext::TextureHandle getReductionTargetBuffer( size_t inMinComponents );

  protected:
    GPURuntime();
    bool initialize( GPUContext* inContext );

  private:
    GPUContext* _context;
    GPUContext::TextureHandle _reductionTempBuffers[kGPUReductionTempBufferCount];
    size_t _reductionTempBufferWidths[kGPUReductionTempBufferCount];
    size_t _reductionTempBufferHeights[kGPUReductionTempBufferCount];
    size_t _reductionTempBufferComponents[kGPUReductionTempBufferCount];
    GPUContext::TextureHandle _reductionTargetBuffer;
    size_t _reductionTargetBufferComponents;
  };
}

#endif
