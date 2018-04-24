#include "DXUT.h"
#include "Filtering.hpp"
#include <cmath>

//--------------------------------------------------------------------------------------
Filtering::Filtering(ID3D10Device* d3dDevice,
                     ID3D10Effect* Effect,
                     int Width, int Height,
                     const DXGI_SAMPLE_DESC* SampleDesc)
  : m_d3dDevice(0), m_Effect(Effect)
  , m_Width(Width), m_Height(Height)
  , m_MaxLOD(0.0f), m_MinFilterWidth(1.0f)
  , m_DepthStencilView(0)
{
  m_Effect->AddRef();

  // Setup multisampling
  if (SampleDesc) {
    m_SampleDesc = *SampleDesc;
  } else {
    m_SampleDesc.Count = 1;
    m_SampleDesc.Quality = 0;
  }

  // Create the depth-stencil buffer
  InitDepthStencil(d3dDevice);

  // Setup default shadow viewport
  m_ShadowViewport.Width    = m_Width;
  m_ShadowViewport.Height   = m_Height;
  m_ShadowViewport.MinDepth = 0.0f;
  m_ShadowViewport.MaxDepth = 1.0f;
  m_ShadowViewport.TopLeftX = 0;
  m_ShadowViewport.TopLeftY = 0;

  // Techniques
  m_DepthTechnique   = m_Effect->GetTechniqueByName("Depth");
  assert(m_DepthTechnique && m_DepthTechnique->IsValid());
  m_ShadingTechnique = m_Effect->GetTechniqueByName("Shading");
  // NOTE: Don't assert as it's possible that derived classes will use custom techniques

  // Setup effect
  ID3D10EffectVectorVariable* TextureSize = m_Effect->GetVariableByName("g_ShadowTextureSize")->AsVector();
  assert(TextureSize && TextureSize->IsValid());
  m_EffectMinFilterWidth = m_Effect->GetVariableByName("g_MinFilterWidth")->AsScalar();
  assert(m_EffectMinFilterWidth && m_EffectMinFilterWidth->IsValid());

  HRESULT hr;
  V(TextureSize->SetFloatVector(D3DXVECTOR4(static_cast<float>(m_Width),
                                            static_cast<float>(m_Height), 0, 0)));
}

//--------------------------------------------------------------------------------------
Filtering::~Filtering()
{
  SAFE_RELEASE(m_DepthStencilView);
  SAFE_RELEASE(m_DepthStencilTexture);
  SAFE_RELEASE(m_Effect);
}

//--------------------------------------------------------------------------------------
void Filtering::BeginFrame(ID3D10Device* d3dDevice, bool UpdateStats)
{
  m_d3dDevice = d3dDevice;
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  Filtering::BeginShadowMap(D3DXMATRIXA16& LightView,
                                                  D3DXMATRIXA16& LightProj,
                                                  const D3DXVECTOR3& LightPos,
                                                  float LightLinFar,
                                                  const CBaseCamera& Camera)
{
  return m_DepthTechnique;
}

//--------------------------------------------------------------------------------------
bool Filtering::EndShadowMap(ID3D10EffectTechnique* Technique)
{
  // Weird, I know, but this is the only way that it seems possible to convince
  // the Effects framework to actually unbind textures, etc.
  Technique->GetPassByIndex(0)->Apply(0);

  // Assume done
  return false;
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique* Filtering::BeginShading()
{
  return m_ShadingTechnique;
}

//--------------------------------------------------------------------------------------
void Filtering::EndShading(ID3D10EffectTechnique* Technique)
{
  // See note above
  Technique->GetPassByIndex(0)->Apply(0);
}

//--------------------------------------------------------------------------------------
void Filtering::EndFrame()
{
  m_d3dDevice = 0;
}

//--------------------------------------------------------------------------------------
void Filtering::SetMaxLOD(float l)
{
  m_MaxLOD = l;

  // Compute derived settings
  m_MinFilterWidth = std::pow(2.0f, m_MaxLOD);

  // Update effect
  // NOTE: Clamp filter width to 1 here. The negative values are only useful internally
  // when scaling blur sizes (see PSVSM), which will subsequently be clamped themselves.
  HRESULT hr;
  V(m_EffectMinFilterWidth->SetFloat(std::max(1.0f, m_MinFilterWidth)));
}

//--------------------------------------------------------------------------------------
void Filtering::SetMinFilterWidth(float f)
{
  SetMaxLOD(std::log(f) / std::log(2.0f));
}

//--------------------------------------------------------------------------------------
void Filtering::InitDepthStencil(ID3D10Device* d3dDevice)
{
  HRESULT hr;
  SAFE_RELEASE(m_DepthStencilView);

  bool Multisampling = (m_SampleDesc.Count > 1 || m_SampleDesc.Quality > 0);

  // Create the texture
  D3D10_TEXTURE2D_DESC texDesc;
  texDesc.Width              = m_Width;
  texDesc.Height             = m_Height;
  texDesc.MipLevels          = 1;
  texDesc.ArraySize          = 1;
  texDesc.Format             = DXGI_FORMAT_D24_UNORM_S8_UINT;
  texDesc.SampleDesc         = m_SampleDesc;
  texDesc.Usage              = D3D10_USAGE_DEFAULT;
  texDesc.BindFlags          = D3D10_BIND_DEPTH_STENCIL;
  texDesc.CPUAccessFlags     = 0;
  texDesc.MiscFlags          = 0;

  V(d3dDevice->CreateTexture2D(&texDesc, NULL, &m_DepthStencilTexture));

  // Create the depth-stencil view
  D3D10_DEPTH_STENCIL_VIEW_DESC dsDesc;
  dsDesc.Format              = texDesc.Format;
  dsDesc.ViewDimension       = Multisampling ? D3D10_DSV_DIMENSION_TEXTURE2DMS :
                                               D3D10_DSV_DIMENSION_TEXTURE2D;
  dsDesc.Texture2D.MipSlice  = 0;

  V(d3dDevice->CreateDepthStencilView(m_DepthStencilTexture, &dsDesc, &m_DepthStencilView));
}

//--------------------------------------------------------------------------------------
int Filtering::ComputeMipLevels(int Width, int Height)
{
  return 1 + static_cast<int>(std::ceil(std::max(
    std::log(static_cast<double>(Width)) / std::log(2.0),
    std::log(static_cast<double>(Height)) / std::log(2.0))));
}
