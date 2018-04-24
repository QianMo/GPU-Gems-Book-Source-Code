#include "DXUT.h"
#include "Point.hpp"

//--------------------------------------------------------------------------------------
Point::Point(ID3D10Device* d3dDevice,
             ID3D10Effect* Effect,
             int Width, int Height)
  : Filtering(d3dDevice, Effect, Width, Height)
  , m_ShadowMap(0)
{
  HRESULT hr;
  
  // Create the shadow map
  m_ShadowMap = new RenderableTexture2D(d3dDevice, Width, Height, 1, DXGI_FORMAT_R32_FLOAT);

  // Setup effect
  m_EffectShadowMap = m_Effect->GetVariableByName("texShadow")->AsShaderResource();
  assert(m_EffectShadowMap && m_EffectShadowMap->IsValid());
  m_EffectDepthBias = m_Effect->GetVariableByName("g_DepthBias")->AsScalar();
  assert(m_EffectDepthBias && m_EffectDepthBias->IsValid());

  V(m_EffectDepthBias->SetFloat(0.001f));
}

//--------------------------------------------------------------------------------------
Point::~Point()
{
  SAFE_DELETE(m_ShadowMap);
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  Point::BeginShadowMap(D3DXMATRIXA16& LightView,
                                              D3DXMATRIXA16& LightProj,
                                              const D3DXVECTOR3& LightPos,
                                              float LightLinFar,
                                              const CBaseCamera& Camera)
{
  ID3D10EffectTechnique* Technique =
    Filtering::BeginShadowMap(LightView, LightProj, LightPos, LightLinFar, Camera);

  ID3D10RenderTargetView *RT = m_ShadowMap->GetRenderTarget();

  // Clear shadow map and bind it for rendering
  m_d3dDevice->ClearRenderTargetView(RT, D3DXVECTOR4(1.0f, 1.0f, 1.0f, 1.0f));
  m_d3dDevice->ClearDepthStencilView(m_DepthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0);

  // Setup shadow render target
  m_d3dDevice->OMSetRenderTargets(1, &RT, m_DepthStencilView);
  m_d3dDevice->RSSetViewports(1, &m_ShadowViewport);

  return Technique;
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  Point::BeginShading()
{
  // Bind shadow texture
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(m_ShadowMap->GetShaderResource()));

  return Filtering::BeginShading();
}

//--------------------------------------------------------------------------------------
void Point::EndShading(ID3D10EffectTechnique* Technique)
{
  // Unbind shadow texture
  HRESULT hr;
  V(m_EffectShadowMap->SetResource(0));
  
  Filtering::EndShading(Technique);
}