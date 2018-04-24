#include "DXUT.h"
#include "Hardware.hpp"
#include <cmath>

//--------------------------------------------------------------------------------------
VSM::VSM(ID3D10Device* d3dDevice,
         ID3D10Effect* Effect,
         int Width, int Height,
         PostProcess* PostProcess,
         DXGI_FORMAT Format,
         const DXGI_SAMPLE_DESC* SampleDesc,
         DXGI_FORMAT MSAAFormat,
         bool Mipmapped)
  : Filtering(d3dDevice, Effect, Width, Height, SampleDesc)
  , m_PostProcess(PostProcess)
  , m_ShadowMap(0), m_ShadowFormat(Format), m_MSAAShadowFormat(MSAAFormat)
  , m_Mipmapped(Mipmapped), m_MSAAShadowMap(0), m_ShadowMapFromCache(false)
{
  // Use the same format for MSAA?
  if (m_MSAAShadowFormat == DXGI_FORMAT_UNKNOWN) {
    m_MSAAShadowFormat = m_ShadowFormat;
  }

  // Do we need a multisampled render target?
  if (m_SampleDesc.Count > 1 || m_SampleDesc.Quality > 0) {
    m_MSAAShadowMap =
      new RenderableTexture2D(d3dDevice, m_Width, m_Height, 1,
                              m_MSAAShadowFormat, &m_SampleDesc);
  }

  // Effect interface
  // NOTE: May not actually be there if not used, so don't assert
  m_BoxBlurTechnique = m_Effect->GetTechniqueByName("BoxBlur");
  m_BoxBlurArrayTechnique = m_Effect->GetTechniqueByName("BoxBlurArray");
  m_EffectBlurDim = m_Effect->GetVariableByName("g_BlurDim")->AsVector();
  m_EffectBlurSamples = m_Effect->GetVariableByName("g_BlurSamples")->AsScalar();
}

//--------------------------------------------------------------------------------------
VSM::~VSM()
{
  SAFE_DELETE(m_MSAAShadowMap);
  // Just in case
  SAFE_DELETE(m_ShadowMap);
}

//--------------------------------------------------------------------------------------
ID3D10EffectTechnique*  VSM::BeginShadowMap(D3DXMATRIXA16& LightView,
                                            D3DXMATRIXA16& LightProj,
                                            const D3DXVECTOR3& LightPos,
                                            float LightLinFar,
                                            const CBaseCamera& Camera)
{
  ID3D10EffectTechnique* Technique = 
    Filtering::BeginShadowMap(LightView, LightProj, LightPos, LightLinFar, Camera);
  
  ID3D10RenderTargetView *RT;
  m_ShadowMapFromCache = false;
  if (m_MSAAShadowMap) {
    RT = m_MSAAShadowMap->GetRenderTarget();
  } else {
    // If necessary, grab an available texture to use as the shadow map
    if (!m_ShadowMap) {
      m_ShadowMap = m_FullTextures.Get();
      m_ShadowMapFromCache = true;
    }
    RT = m_ShadowMap->GetRenderTarget();
  }

  // Clear variance shadow map and bind it for rendering
  // NOTE: FPBias, and potential distribute components (in children)
  // NOTE: Use an extra-high depth^2 so that any texels that get filtered with the
  // background are guaranteed to have a high variance (to simulate "no geometry" there).
  m_d3dDevice->ClearRenderTargetView(RT, D3DXVECTOR4(0.5f, 1.0f, 0.0f, 0.0f));
  m_d3dDevice->ClearDepthStencilView(m_DepthStencilView, D3D10_CLEAR_DEPTH, 1.0f, 0);

  // Setup shadow render target
  m_d3dDevice->OMSetRenderTargets(1, &RT, m_DepthStencilView);
  m_d3dDevice->RSSetViewports(1, &m_ShadowViewport);

  return Technique;
}

//--------------------------------------------------------------------------------------
bool VSM::EndShadowMap(ID3D10EffectTechnique* Technique)
{
  // MSAA resolve if necessary/possible
  if (m_MSAAShadowMap) {
    // If necessary and formats match, grab an available texture to use as the shadow map
    if (!m_ShadowMap && m_MSAAShadowFormat == m_ShadowFormat) {
      m_ShadowMap = m_FullTextures.Get();
      m_ShadowMapFromCache = true;
    }

    // Resolve to our shadow map texture if we've been given one
    // NOTE: If we haven't, the caller should do a custom resolve!
    if (m_ShadowMap) {
      m_d3dDevice->ResolveSubresource(
        m_ShadowMap->GetTexture(),
        D3D10CalcSubresource(0, m_ShadowMap->GetArrayIndex(), m_ShadowMap->GetMipLevels()),
        m_MSAAShadowMap->GetTexture(),
        D3D10CalcSubresource(0, m_MSAAShadowMap->GetArrayIndex(), m_MSAAShadowMap->GetMipLevels()),
        m_ShadowMap->GetFormat());
    }
  }

  Filtering::EndShadowMap(Technique);

  // Done
  return false;
}

//--------------------------------------------------------------------------------------
void VSM::AddFullTextures(ID3D10Device* d3dDevice, int Num)
{
  // Create full-sized textures
  for (int i = 0; i < Num; ++i) {
    m_FullTextures.Add(new RenderableTexture2D(d3dDevice, m_Width, m_Height,
                                               m_Mipmapped ? 0 : 1, m_ShadowFormat));
  }
}

//--------------------------------------------------------------------------------------
void VSM::EndFrame()
{
  // Return the shadow map texture to the cache if that's where it came from
  if (m_ShadowMap && m_ShadowMapFromCache) {
    m_FullTextures.Add(m_ShadowMap);
    m_ShadowMap = 0;
    m_ShadowMapFromCache = false;
  }
  
  Filtering::EndFrame();
}

//--------------------------------------------------------------------------------------
void VSM::BoxBlur(RenderableTexture2D *Src,
                  RenderableTexture2D *Temp,
                  const D3DXVECTOR2& FilterWidth)
{
  HRESULT hr;

  // Round to the nearest integer
  // TODO: Optionally use a non-integer filter width
  int HorizFilterSamples = static_cast<int>(std::floor(FilterWidth.x + 0.5f));
  int VertFilterSamples  = static_cast<int>(std::floor(FilterWidth.y + 0.5f));

  if (HorizFilterSamples <= 1 && VertFilterSamples <= 1) {
    return;
  }

  // Choose the right effect so that the compiler won't complain
  // NOTE: Assumes that Src and Temp have the same view dimension, or else we'd have
  // to swap techniques every pass to avoid D3D10 whining - yikes!
  ID3D10EffectTechnique* Technique = Src->IsArray() ? m_BoxBlurArrayTechnique
                                                    : m_BoxBlurTechnique;

  m_PostProcess->Begin(m_d3dDevice, m_Width, m_Height, Temp, Src);

  // Setup
  D3D10_RECT Region = {0, 0, m_Width, m_Height};

  // Vertical
  V(m_EffectBlurDim->SetFloatVector(D3DXVECTOR4(0, 1, 0, 0)));
  V(m_EffectBlurSamples->SetInt(VertFilterSamples));
  m_PostProcess->Apply(Technique, Region);

  // Horizontal
  V(m_EffectBlurDim->SetFloatVector(D3DXVECTOR4(1, 0, 0, 0)));
  V(m_EffectBlurSamples->SetInt(HorizFilterSamples));
  m_PostProcess->Apply(Technique, Region);

  RenderableTexture2D *Result = m_PostProcess->End();
  assert(Result == Src);
}

//--------------------------------------------------------------------------------------
Filtering::MSAAList VSM::QueryMSAASupport(ID3D10Device* d3dDevice) const
{
  MSAAList Modes;
  DXGI_FORMAT Format = m_MSAAShadowFormat;
  bool NVIDIA = false;

  // Get current adapter info
  IDXGIAdapter *Adapter = 0;
  DXGI_ADAPTER_DESC AdapterDesc;
  if (SUCCEEDED(DXUTGetDXGIFactory()->EnumAdapters(DXUTGetDeviceSettings().d3d10.AdapterOrdinal, &Adapter))) {
    if (SUCCEEDED(Adapter->GetDesc(&AdapterDesc))) {
      NVIDIA = (AdapterDesc.VendorId == 0x10DE);
    }
    SAFE_RELEASE(Adapter);
  }

  if (Format != DXGI_FORMAT_UNKNOWN) {
    for (unsigned int Samples = 2; Samples < D3D10_MAX_MULTISAMPLE_SAMPLE_COUNT; ++Samples) {
      unsigned int MaxQuality;
      unsigned int FormatSupport;

      // Ensure that we can render to, and resolve this multisampled format
      if (SUCCEEDED(d3dDevice->CheckMultisampleQualityLevels(Format, Samples, &MaxQuality)) &&
          MaxQuality > 0 &&
          SUCCEEDED(d3dDevice->CheckFormatSupport(Format, &FormatSupport)) &&
          FormatSupport & D3D10_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET &&
          FormatSupport & D3D10_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE) {

        if (NVIDIA) {
          // Check for CSAA support
          switch (Samples) {
            case 4:
              if (MaxQuality > 4)    Modes.push_back(MSAAMode(L"4x",  4, 4));
              if (MaxQuality > 8)    Modes.push_back(MSAAMode(L"8x",  4, 8));
              if (MaxQuality > 16)   Modes.push_back(MSAAMode(L"16x", 4, 16));
              break;
            case 8:
              if (MaxQuality > 8)    Modes.push_back(MSAAMode(L"8xQ",  8, 8));
              if (MaxQuality > 16)   Modes.push_back(MSAAMode(L"16xQ", 8, 16));
              break;
            default:
              break;
          };
        } else {
          // Add the standard mode (quality = 0)
          std::wostringstream oss;
          oss << Samples << L"x";
          Modes.push_back(MSAAMode(oss.str(), Samples, 0));
        }
      }
    }
  }

  return Modes;
}
