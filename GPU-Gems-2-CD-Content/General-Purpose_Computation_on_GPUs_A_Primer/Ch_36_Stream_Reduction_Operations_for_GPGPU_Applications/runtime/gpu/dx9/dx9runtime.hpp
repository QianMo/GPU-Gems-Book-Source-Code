// dx9runtime.hpp
#ifndef GPU_RUNTIME_DX9_HPP
#define GPU_RUNTIME_DX9_HPP

#include "../gpubase.hpp"
#include "../gpuruntime.hpp"

#define DX9_RUNTIME_STRING "dx9"

namespace brook
{

  class GPURuntimeDX9 : public GPURuntime
  {
  public:
    static GPURuntimeDX9* create( void* inContextValue = 0 );

  private:
    GPURuntimeDX9() {}
  };

}

#endif
