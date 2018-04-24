#include "DXUT.h"
#include "SAT.hpp"

//--------------------------------------------------------------------------------------
SAT::SAT(ID3D10Device *d3dDevice,
         ID3D10Effect* Effect,
         int Width, int Height,
         PostProcess* PostProcess,
         bool FPDistribute)
  : VSM(d3dDevice, Effect, Width, Height,
        FPDistribute ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32G32_FLOAT,
        false)
  , m_PostProcess(PostProcess), m_FPDistribute(FPDistribute)
{
  
}

//--------------------------------------------------------------------------------------
SAT::~SAT()
{
}
