#include "DXUT.h"
#include "Hardware.hpp"
#include <cmath>

//--------------------------------------------------------------------------------------
Hardware::Hardware(ID3D10Device* d3dDevice,
                   ID3D10Effect* Effect,
                   int Width, int Height,
                   PostProcess* PostProcess,
                   const DXGI_SAMPLE_DESC* SampleDesc)
  : VSM(d3dDevice, Effect, Width, Height, PostProcess,
        DXGI_FORMAT_R32G32_FLOAT, SampleDesc, DXGI_FORMAT_R32G32_FLOAT, true)
{
  // Create two shadow maps (for ping-ponging) and put them in the cache
  AddFullTextures(d3dDevice, 2);

  // Setup effect
  m_EffectShadowMap = m_Effect->GetVariableByName("texShadow")->AsShaderResource();
  assert(m_EffectShadowMap && m_EffectShadowMap->IsValid());
}

//--------------------------------------------------------------------------------------
bool Hardware::EndShadowMap(ID3D10EffectTechnique* Technique)
{
  VSM::EndShadowMap(Technique);

  // Blur if required
  if (m_MinFilterWidth > 1.0f) {
    RenderableTexture2D* Temp = m_FullTextures.Get();
    BoxBlur(m_ShadowMap, Temp, D3DXVECTOR2(m_MinFilterWidth, m_MinFilterWidth));
    m_FullTextures.Add(Temp);
  }

  // Generate mipmaps
  m_d3dDevice->GenerateMips(m_ShadowMap->GetShaderResource());

  // Done
  return false;
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  Hardware::BeginShading()
{
  // Bind shadow texture
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(m_ShadowMap->GetShaderResource()));

  return Filtering::BeginShading();
}

//--------------------------------------------------------------------------------------
void Hardware::EndShading(ID3D10EffectTechnique* Technique)
{
  // Unbind shadow texture
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(0));

  VSM::EndShading(Technique);
}
