#pragma once

#include <sstream>
#include <vector>
#include <string>
#include "TextureCache.hpp"
#include "DXUTcamera.h"

//--------------------------------------------------------------------------------------
// Base class for various shadow map filtering algorithms
// NOTE: The encapsulation here is far from perfect, but we at least separate out some
// of the more obvious filtering-related functionality, and allow for polymorphic
// behaviour.
// In particular the shader must also be given the corresponding defines to work
// properly with these implementations... unfortunately, but an artifact of the loose
// binding of shaders when using HLSL.
class Filtering
{
public:
  virtual ~Filtering();
  
  // Order of calls should be similar to:
  // BeginFrame
  //   [loop until EndShadowMap returns false] BeginShadowMap
  //     [Render scene from light's POV]
  //   EndShadowMap
  //   BeginShading
  //     [Render scene from camera's POV]
  //     [optional] DisplayShadowMap(ID3D10RenderTargetView* Dest)
  //   EndShading
  // EndFrame

  // Set "UpdateStats" to potentially compute rendering statistics...
  // Only for debugging as this may be quite slow!
  virtual void BeginFrame(ID3D10Device* d3dDevice,
                          bool UpdateStats = false);

  // Returns the technique to use to render the shadow map
  // Might change input matrices if different ones are desired to render the scene
  // NOTE: Some of the extra data could be gleaned from the view/projection matrices,
  // but it's more convenient and efficient just to pass it in directly.
  // NOTE: We could pass camera info separately as well to avoid dependence on DXUT,
  // but this is a bit more convenient.
  virtual ID3D10EffectTechnique* BeginShadowMap(D3DXMATRIXA16& LightView,
                                                D3DXMATRIXA16& LightProj,
                                                const D3DXVECTOR3& LightPos,
                                                float LightLinFar,
                                                const CBaseCamera& Camera);

  // Pass the technique that was returned from BeginShadowMap
  virtual bool EndShadowMap(ID3D10EffectTechnique* Technique);

  // Returns the technique to use to render the scene
  virtual ID3D10EffectTechnique* BeginShading();
  
  // Pass the technique that was returned from BeginShading
  virtual void EndShading(ID3D10EffectTechnique* Technique);

  // Shadow map(s) will be displayed in the given viewport
  virtual void DisplayShadowMap(const D3D10_VIEWPORT& Viewport) {}

  virtual void EndFrame();

  // Shadow multisampling (if supported)
  struct MSAAMode
  {
    std::wstring Name;
    DXGI_SAMPLE_DESC SampleDesc;
    MSAAMode(const std::wstring& Name, unsigned int Count, unsigned int Quality)
      : Name(Name)
    {
      SampleDesc.Count = Count;
      SampleDesc.Quality = Quality;
    }
  };
  typedef std::vector<MSAAMode> MSAAList;
  virtual MSAAList QueryMSAASupport(ID3D10Device* d3dDevice) const { return MSAAList(); }

  // UI (some of these may only applicable to some subclasses)
  virtual DXGI_FORMAT GetShadowFormat() const { return DXGI_FORMAT_UNKNOWN; }

  // The next two change the same setting
  void SetMaxLOD(float l);
  void SetMinFilterWidth(float f);
  // Integer version
  void SetMinFilterWidth(int f) { SetMinFilterWidth(static_cast<float>(f)); }

protected:
  Filtering(ID3D10Device* d3dDevice,
            ID3D10Effect* Effect,
            int Width, int Height,
            const DXGI_SAMPLE_DESC* SampleDesc = 0);

  // (Re)create the depth stencil surface
  void InitDepthStencil(ID3D10Device* d3dDevice);

  // Get the number of mipmaps for given dimensions
  static int ComputeMipLevels(int Width, int Height);

  // Persistent state
  ID3D10Effect*                m_Effect;
  ID3D10EffectScalarVariable*  m_EffectMinFilterWidth;  // Minimum filter width

  ID3D10EffectTechnique*       m_DepthTechnique;
  ID3D10EffectTechnique*       m_ShadingTechnique;
  int                          m_Width;
  int                          m_Height;
  DXGI_SAMPLE_DESC             m_SampleDesc;

  // These next two are equivalent settings
  float                        m_MaxLOD;
  float                        m_MinFilterWidth;

  // Depth stencil surface for rendering shadow map
  ID3D10Texture2D*             m_DepthStencilTexture;
  ID3D10DepthStencilView*      m_DepthStencilView;
  D3D10_VIEWPORT               m_ShadowViewport;
  
  // BeginFrame/EndFrame interior state
  ID3D10Device*                m_d3dDevice;
};