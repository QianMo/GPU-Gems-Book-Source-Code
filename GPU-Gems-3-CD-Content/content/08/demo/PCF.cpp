#include "DXUT.h"
#include "PCF.hpp"

//--------------------------------------------------------------------------------------
PCF::PCF(ID3D10Device* d3dDevice,
         ID3D10Effect* Effect,
         int Width, int Height)
  : Point(d3dDevice, Effect, Width, Height)
{
  HRESULT hr;

  // Setup effect
  m_EffectConeBias = m_Effect->GetVariableByName("g_ConeBias")->AsScalar();
  assert(m_EffectConeBias && m_EffectConeBias->IsValid());

  // PCF requires fairly large biasing
  V(m_EffectDepthBias->SetFloat(0.006f));
  V(m_EffectConeBias->SetFloat(1.0f));
}

//--------------------------------------------------------------------------------------
PCF::~PCF()
{
}
