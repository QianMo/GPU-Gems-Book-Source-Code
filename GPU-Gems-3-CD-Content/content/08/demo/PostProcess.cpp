#include "DXUT.h"
#include <algorithm>
#include "PostProcess.hpp"

//--------------------------------------------------------------------------------------
PostProcess::PostProcess(ID3D10Device* d3dDevice, ID3D10Effect* Effect)
  : m_d3dDevice(0), m_Effect(Effect)
  , m_Dest(0), m_Src(0), m_OrigSrc(0)
{
  // Setup default shadow viewport
  m_Viewport.Width    = 0;
  m_Viewport.Height   = 0;
  m_Viewport.MinDepth = 0.0f;
  m_Viewport.MaxDepth = 1.0f;
  m_Viewport.TopLeftX = 0;
  m_Viewport.TopLeftY = 0;

  // Grab uniforms
  // Don't assert in case PostProcess is never actually used...
  m_EffectSourceTexture = m_Effect->GetVariableByName("texPPSource")->AsShaderResource();
  m_EffectSourceTextureArray = m_Effect->GetVariableByName("texPPSourceArray")->AsShaderResource();
  m_EffectSourceTextureUint = m_Effect->GetVariableByName("texPPSourceUint")->AsShaderResource();
  m_EffectSourceTextureSize = m_Effect->GetVariableByName("g_PPSourceSize")->AsVector();
}

//--------------------------------------------------------------------------------------
PostProcess::~PostProcess()
{
}

//--------------------------------------------------------------------------------------
void PostProcess::Begin(ID3D10Device* d3dDevice, int Width, int Height,
                        RenderableTexture2D* Dest, RenderableTexture2D* Src, 
                        RenderableTexture2D* Temp)
{
  HRESULT hr;

  // If no temporary is provided, use the source texture
  if (!Temp) {
    Temp = Src;
  }

  // Save state
  m_d3dDevice = d3dDevice;
  m_OrigSrc = Src;
  m_Src = Temp;
  m_Dest = Dest;
   
  // Update viewport
  m_Viewport.Width  = Width;
  m_Viewport.Height = Height;
  m_d3dDevice->RSSetViewports(1, &m_Viewport);

  // Setup the effect
  V(m_EffectSourceTextureSize->SetFloatVector(D3DXVECTOR4(static_cast<float>(Width),
                                                          static_cast<float>(Height), 0, 0)));
  
  SetupDrawState(m_d3dDevice);

  m_FirstApply = true;
}

//--------------------------------------------------------------------------------------
void PostProcess::SetupDrawState(ID3D10Device* d3dDevice) const
{
  UINT Stride = 0;
  UINT Offset = 0;
  ID3D10Buffer* VertexBuffer = 0;

  d3dDevice->IASetInputLayout(0);
  d3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  d3dDevice->IASetVertexBuffers(0, 1, &VertexBuffer, &Stride, &Offset);
}

//--------------------------------------------------------------------------------------
void PostProcess::FillScreen(ID3D10Device* d3dDevice,
                             ID3D10EffectTechnique* RenderTechnique,
                             bool SetState) const
{
  if (SetState) {
    SetupDrawState(d3dDevice);
  }

  // Loop over passes
  D3D10_TECHNIQUE_DESC TechDesc;
  RenderTechnique->GetDesc(&TechDesc);
  for (unsigned int p = 0; p < TechDesc.Passes; ++p) {
    // Render a full-screen triangle
    RenderTechnique->GetPassByIndex(p)->Apply(0);
    d3dDevice->Draw(3, 0);
  }
}

//--------------------------------------------------------------------------------------
void PostProcess::Apply(ID3D10EffectTechnique* RenderTechnique, const D3D10_RECT &DestRegion)
{
  // Setup the effect
  // Use the original source for the very first pass
  if (m_FirstApply) {
    SetSourceTexture(m_OrigSrc->GetShaderResource());
    m_FirstApply = false;
  } else {
    SetSourceTexture(m_Src->GetShaderResource());
  }

  // Lame and unnecessary, but avoids D3D10 warnings...
  //*************************************************************************************
  m_d3dDevice->OMSetRenderTargets(0, 0, 0);
  RenderTechnique->GetPassByIndex(0)->Apply(0);
  //*************************************************************************************

  // Set render target
  ID3D10RenderTargetView *RT = m_Dest->GetRenderTarget();
  m_d3dDevice->OMSetRenderTargets(1, &RT, 0);
  
  // Setup the scissor region
  m_d3dDevice->RSSetScissorRects(1, &DestRegion);

  FillScreen(m_d3dDevice, RenderTechnique, false);

  // Swap pointers (ping pong)
  std::swap(m_Dest, m_Src);
}

//--------------------------------------------------------------------------------------
RenderableTexture2D * PostProcess::End()
{
  // Restore state
  m_d3dDevice->OMSetRenderTargets(0, 0, 0);

  // Release references
  m_OrigSrc = 0;
  m_Dest = 0;
  m_Effect = 0;
  m_d3dDevice = 0;

  RenderableTexture2D *Result = m_Src;
  m_Src = 0;

  return Result;
}
