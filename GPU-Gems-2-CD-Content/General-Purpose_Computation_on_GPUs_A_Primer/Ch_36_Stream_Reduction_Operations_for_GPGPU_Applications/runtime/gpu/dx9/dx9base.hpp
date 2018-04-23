// dx9base.hpp
#pragma once

#include <assert.h>
#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "../gpubase.hpp"
#include "../gpucontext.hpp"

namespace brook
{
    
  class GPUContextDX9 : public GPUContext
  {
  public:
      virtual bool isRenderTextureFormatValid( D3DFORMAT inFormat ) = 0;
      virtual IDirect3DDevice9* getDevice() = 0;
  };

  #define DX9PROFILE( __name ) \
    GPUPROFILE( __name )

  #define DX9LOG( __level ) \
    GPULOG( __level )

  #define DX9LOGPRINT( __level ) \
    GPULOGPRINT( __level )

  #define DX9WARN GPUWARN

  #define DX9AssertResult( _result, _message ) \
    if(SUCCEEDED(_result)) {} else GPUError( _message )

  #define DX9Assert( _condition, _message ) \
    GPUAssert( _condition, _message )
}