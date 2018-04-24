#pragma once

#include "Filtering.hpp"
#include "PostProcess.hpp"

//--------------------------------------------------------------------------------------
// A basis for variance shadow map filtering implementations
class VSM : public Filtering
{
public:
  virtual ~VSM();

  virtual ID3D10EffectTechnique* BeginShadowMap(D3DXMATRIXA16& LightView,
                                                D3DXMATRIXA16& LightProj,
                                                const D3DXVECTOR3& LightPos,
                                                float LightLinFar,
                                                const CBaseCamera& Camera);

  virtual bool EndShadowMap(ID3D10EffectTechnique* Technique);

  virtual void EndFrame();

  virtual DXGI_FORMAT GetShadowFormat() const { return m_ShadowFormat; }

  // Check for multisampling support of the current shadow format
  virtual MSAAList QueryMSAASupport(ID3D10Device* d3dDevice) const;

protected:
  VSM(ID3D10Device* d3dDevice,
      ID3D10Effect* Effect,
      int Width, int Height,
      PostProcess* PostProcess,
      DXGI_FORMAT Format = DXGI_FORMAT_R32G32_FLOAT,
      const DXGI_SAMPLE_DESC* SampleDesc = 0,
      DXGI_FORMAT MSAAFormat = DXGI_FORMAT_UNKNOWN,
      bool Mipmapped = true);

  // Add new textures to the cache (of type m_ShadowFormat)
  void AddFullTextures(ID3D10Device* d3dDevice, int Num);

  // Blur the given texture, using the given temporary texture for ping-ponging
  // Since this is exactly a two-pass operation, the result will be written
  // back into the source texture.
  void BoxBlur(RenderableTexture2D *Src,
               RenderableTexture2D *Temp,
               const D3DXVECTOR2& FilterWidth);

  DXGI_FORMAT                         m_ShadowFormat;      // Shadow map format
  DXGI_FORMAT                         m_MSAAShadowFormat;  // MSAA render target format
  bool                                m_Mipmapped;         // Mipmap the shadow texture(s)

  PostProcess*                        m_PostProcess;       // Post processing helper
  TextureCache                        m_FullTextures;      // Texture cache
  RenderableTexture2D*                m_ShadowMap;         // Variance shadow map
  RenderableTexture2D*                m_MSAAShadowMap;     // Multisampled render target
  
private:
  ID3D10EffectTechnique*              m_BoxBlurTechnique;  // Box blurring technique
  ID3D10EffectTechnique*              m_BoxBlurArrayTechnique;
  ID3D10EffectVectorVariable*         m_EffectBlurDim;
  ID3D10EffectScalarVariable*         m_EffectBlurSamples;

  bool                                m_ShadowMapFromCache;
};