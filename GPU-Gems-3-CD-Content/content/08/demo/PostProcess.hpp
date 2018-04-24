#pragma once

#include "RenderableTexture2D.hpp"

//--------------------------------------------------------------------------------------
class PostProcess
{
public:
  PostProcess(ID3D10Device* d3dDevice, ID3D10Effect* Effect);
  ~PostProcess();

  // Begin post processing with two textures to ping-pong
  // All textures should be of dimesions Width by Height
  // Optionally provide a temporary texture... if provided, Src will
  // not be modified - otherwise the contents of Src is undefined after
  // applying post-processing operations (it may be used as a temporary).
  // Temp = Src is valid and equivalent to Temp = 0.
  // NOTE: Could extend this to work with non-RenderableTexture2D's, but this
  // interface is pretty convenient for our usage cases.
  void Begin(ID3D10Device* d3dDevice, int Width, int Height,
             RenderableTexture2D* Dest, RenderableTexture2D* Src, 
             RenderableTexture2D* Temp = 0);

  // Apply a post-processing effect to the given source surface.
  // The technique is assumed to have already been set
  // Must be called between Begin/End!
  void Apply(ID3D10EffectTechnique* RenderTechnique, const D3D10_RECT &DestRegion);

  // End post processing (and return the resultant texture)
  RenderableTexture2D* End();

  // Utilities - made public since they are somewhat convenient to use
  // NOTE: Do not maintain current render state...
  void SetupDrawState(ID3D10Device* d3dDevice) const;
  void FillScreen(ID3D10Device* d3dDevice, ID3D10EffectTechnique* RenderTechnique,
                  bool SetState = true) const;

private:
  // Handy utility
  void SetSourceTexture(ID3D10ShaderResourceView* Texture)
  {
    // NOTE: Just set *all* relevant texture types and trust the shader and
    // program to use the right ones
    HRESULT hr;
    V(m_EffectSourceTexture->SetResource(Texture));
    V(m_EffectSourceTextureArray->SetResource(Texture));
    V(m_EffectSourceTextureUint->SetResource(Texture));
  }

  // Effect state
  ID3D10Effect*                       m_Effect;
  ID3D10EffectShaderResourceVariable* m_EffectSourceTexture;
  ID3D10EffectShaderResourceVariable* m_EffectSourceTextureArray;
  ID3D10EffectShaderResourceVariable* m_EffectSourceTextureUint;
  ID3D10EffectVectorVariable*         m_EffectSourceTextureSize;

  D3D10_VIEWPORT                      m_Viewport;

  // Begin/End interior state
  ID3D10Device*                       m_d3dDevice;
  RenderableTexture2D*                m_Dest;
  RenderableTexture2D*                m_Src;
  RenderableTexture2D*                m_OrigSrc;
  bool                                m_FirstApply;
};
