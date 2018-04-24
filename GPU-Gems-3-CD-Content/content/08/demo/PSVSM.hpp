#pragma once

#include "VSM.hpp"

//--------------------------------------------------------------------------------------
// Impelements parallel-split variance shadow maps using hardware filtering
// Note that we use standard blurred variance shadow maps here, although it would be
// quite possible to use a summed-area variance shadow map for each slice. Actually this
// would probably work really well since the slices in PSSM tend to be smaller than a
// single gigantic shadow map, which plays to all of the strengths of SAVSM.
class PSVSM : public VSM
{
public:
  PSVSM(ID3D10Device* d3dDevice,
        ID3D10Effect* Effect,
        int Width, int Height,
        PostProcess* PostProcess,
        const DXGI_SAMPLE_DESC* SampleDesc,
        unsigned int NumSplits);

  virtual ~PSVSM();

  virtual ID3D10EffectTechnique* BeginShadowMap(D3DXMATRIXA16& LightView,
                                                 D3DXMATRIXA16& LightProj,
                                                 const D3DXVECTOR3& LightPos,
                                                 float LightLinFar,
                                                 const CBaseCamera& Camera);

  virtual bool EndShadowMap(ID3D10EffectTechnique* Technique);
  
  virtual ID3D10EffectTechnique* BeginShading();

  virtual void EndShading(ID3D10EffectTechnique* Technique);

  virtual void DisplayShadowMap(const D3D10_VIEWPORT& Viewport);

  // UI
  void SetSplitLambda(float l);
  void SetVisualizeSplits(bool v);

protected:
  // Recompute split distances using given camera parameters
  void RecomputeSplitDistances(float SplitNear, float SplitFar,
                               float Lambda = 0.5f);

  // Recompute split matrices using current split distances and light parameters
  void RecomputeSplitMatrices(const D3DXMATRIXA16& CameraView,
                              const D3DXMATRIXA16& CameraProj,
                              const D3DXMATRIXA16& LightView,
                              const D3DXMATRIXA16& LightProj);

  ID3D10EffectShaderResourceVariable* m_EffectShadowMap;
  ID3D10EffectTechnique*              m_DisplayTechnique;          // Visualize shadow map
  ID3D10EffectVectorVariable*         m_EffectSplits;
  ID3D10EffectMatrixVariable*         m_EffectSplitMatrices;
  ID3D10EffectScalarVariable*         m_EffectDisplayArrayIndex;
  ID3D10EffectScalarVariable*         m_EffectVisualizeSplits;

  unsigned int                        m_NumSplits;
  unsigned int                        m_MipLevels;
  unsigned int                        m_CurSplit;
  float                               m_SplitLambda;

  typedef std::vector<float>          SplitDistanceList;
  SplitDistanceList                   m_SplitDistances;
  typedef std::vector<D3DXMATRIX>     MatrixList;
  MatrixList                          m_SplitProj;
  MatrixList                          m_SplitViewProj;
  typedef std::vector<D3DXVECTOR2>    ScaleList;
  ScaleList                           m_SplitScales;

  // Texture array and shader resource
  ID3D10Texture2D*                    m_ShadowMapArray;
  ID3D10ShaderResourceView*           m_ShadowMapArrayResource;

  // Also store render target and shader resource views for each slice
  typedef std::vector<RenderableTexture2D*> TextureList;
  TextureList                         m_ShadowMapSlices;

private:
};