#pragma once

#include "Filtering.hpp"

//--------------------------------------------------------------------------------------
// Impelements point filtering (standard shadow mapping)
class Point : public Filtering
{
public:
  Point(ID3D10Device* d3dDevice,
        ID3D10Effect* Effect,
        int Width, int Height);

  virtual ~Point();

  virtual ID3D10EffectTechnique* BeginShadowMap(D3DXMATRIXA16& LightView,
                                                 D3DXMATRIXA16& LightProj,
                                                 const D3DXVECTOR3& LightPos,
                                                 float LightLinFar,
                                                 const CBaseCamera& Camera);

  virtual ID3D10EffectTechnique* BeginShading();
  virtual void EndShading(ID3D10EffectTechnique* Technique);

  virtual DXGI_FORMAT GetShadowFormat() const { return DXGI_FORMAT_R32_FLOAT; }

protected:
  // Effect interface
  ID3D10EffectShaderResourceVariable* m_EffectShadowMap;
  ID3D10EffectScalarVariable*         m_EffectDepthBias;

  // Shadow map
  RenderableTexture2D*                m_ShadowMap;

private:
};