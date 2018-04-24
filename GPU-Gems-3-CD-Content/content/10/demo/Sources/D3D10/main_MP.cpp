#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D10.h"
#include "ShadowMap_D3D10.h"

// This file handles multi-pass rendering
//

extern Matrix GetTexScaleBiasMatrix(void);
extern ID3D10Effect *g_pEffect;
extern ID3D10EffectMatrixVariable *g_pWorldVariable;
extern ID3D10EffectMatrixVariable *g_pViewProjVariable;
extern ID3D10EffectVectorVariable *g_pLightDirVariable;
extern ID3D10EffectVectorVariable *g_pLightAmbientVariable;
extern ID3D10EffectVectorVariable *g_pLightColorVariable;
extern ID3D10EffectMatrixVariable *g_pViewVariable;
extern ID3D10EffectMatrixVariable *g_pTextureMatrixVariable;
extern ID3D10EffectScalarVariable *g_pSplitPlaneVariable;
extern ID3D10EffectTechnique *g_pTechniqueShadowMap_Standard;
extern ID3D10EffectTechnique *g_pTechniqueShadows_MP;
extern ID3D10EffectShaderResourceVariable *g_pShadowMapTextureVariable;


// Renders the given scene objects
//
//
static void RenderCasters(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj)
{
  // set constants
  g_pViewProjVariable->SetMatrix((float*)&mViewProj);

  // for each pass in technique
  D3D10_TECHNIQUE_DESC DescTech;
  g_pTechniqueShadowMap_Standard->GetDesc( &DescTech );
  for(UINT p = 0; p < DescTech.Passes; ++p)
  {
    // for each object
    for(unsigned int j = 0; j < Objects.size(); j++)
    {
      SceneObject *pObject = Objects[j];

      // set world matrix
      g_pWorldVariable->SetMatrix((float*)&pObject->m_mWorld);

      // activate pass
      g_pTechniqueShadowMap_Standard->GetPassByIndex(p)->Apply(0);
      // draw
      pObject->m_pMesh->Draw();
    }
  }
}


// Renders the given scene objects
//
//
static void RenderReceivers(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj)
{
  // set constants
  g_pViewProjVariable->SetMatrix((float*)&mViewProj);
  g_pLightDirVariable->SetFloatVector((float*)&g_Light.GetDir());
  g_pLightColorVariable->SetFloatVector((float*)&g_Light.m_vLightDiffuse);
  g_pLightAmbientVariable->SetFloatVector((float*)&g_Light.m_vLightAmbient);

  // for each pass in technique
  D3D10_TECHNIQUE_DESC DescTech;
  g_pTechniqueShadows_MP->GetDesc( &DescTech );
  for(UINT p = 0; p < DescTech.Passes; ++p)
  {
    // for each object
    for(unsigned int j = 0; j < Objects.size(); j++)
    {
      SceneObject *pObject = Objects[j];

      // set world matrix
      g_pWorldVariable->SetMatrix((float*)&pObject->m_mWorld);

      // activate pass
      g_pTechniqueShadows_MP->GetPassByIndex(p)->Apply(0);
      // draw
      pObject->m_pMesh->Draw();
    }
  }
}


// Starts rendering to shadow maps
//
//
static void ActivateShadowMap(void)
{
  // unbind shadow maps
  ID3D10ShaderResourceView *pResources[NUM_SPLITS_IN_SHADER] = {NULL};
  g_pShadowMapTextureVariable->SetResourceArray(pResources, 0, NUM_SPLITS_IN_SHADER);

  // Enable rendering to shadow map
  GetShadowMap<ShadowMap_D3D10>()->EnableRendering();

  // Clear texture
  float ClearSM[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  GetApp()->GetDevice()->ClearDepthStencilView(GetShadowMap<ShadowMap_D3D10>()->m_pDSV, D3D10_CLEAR_DEPTH, 1.0f, 0);
}


// Stops rendering to shadow map
//
//
static void DeactivateShadowMap(void)
{
  // Disable rendering to shadow map
  GetShadowMap<ShadowMap_D3D10>()->DisableRendering();
}


// Sets shader parameters
//
//
static void SetShaderParams(const Matrix &mTextureMatrix)
{
  g_pTextureMatrixVariable->SetMatrixArray((float*)&mTextureMatrix, 0, 1);

  // bind shadow map texture
  ID3D10ShaderResourceView *pResources[1];
  pResources[0] = GetShadowMap<ShadowMap_D3D10>()->m_pSRV;
  g_pShadowMapTextureVariable->SetResourceArray(pResources, 0, 1);
}


// Sets depth range settings for a split
//
//
static D3D10_VIEWPORT _OriginalCameraViewport;
static float _fOriginalCameraNear;
static float _fOriginalCameraFar;
void SetDepthRange(float fNear, float fFar)
{
  D3D10_VIEWPORT CameraViewport;
  unsigned int iNumVP = 1;
  GetApp()->GetDevice()->RSGetViewports(&iNumVP, &CameraViewport);

  // store original values
  _OriginalCameraViewport = CameraViewport;
  _fOriginalCameraNear = g_Camera.m_fNear;
  _fOriginalCameraFar = g_Camera.m_fFar;

  // set new depth value range
  CameraViewport.MinDepth = (fNear - g_Camera.m_fNear) / (g_Camera.m_fFar - g_Camera.m_fNear);
  CameraViewport.MaxDepth = (fFar - g_Camera.m_fNear) / (g_Camera.m_fFar - g_Camera.m_fNear);
  GetApp()->GetDevice()->RSSetViewports(1, &CameraViewport);

  // set new far and near plane
  g_Camera.m_fNear = fNear;
  g_Camera.m_fFar = fFar;
  g_Camera.CalculateMatrices();
}


// Resets original depth range settings
//
//
void RestoreDepthRange(void)
{
  GetApp()->GetDevice()->RSSetViewports(1, &_OriginalCameraViewport);
  g_Camera.m_fNear = _fOriginalCameraNear;
  g_Camera.m_fFar = _fOriginalCameraFar;
}


// Multi-pass rendering function
//
//
void Render_MP(void)
{
  // find receivers
  std::vector<SceneObject *> receivers, casters;
  receivers = g_Camera.FindReceivers();

  // adjust camera planes to contain scene tightly
  g_Camera.AdjustPlanes(receivers);

  // calculate the distances of split planes
  g_Camera.CalculateSplitPositions(g_fSplitPos);

  // for each split part
  for(int i = 0; i < g_iNumSplits; i++)
  {
    // calculate frustum
    Frustum splitFrustum;
    splitFrustum = g_Camera.CalculateFrustum(g_fSplitPos[i], g_fSplitPos[i+1]);
    // find casters
    casters = g_Light.FindCasters(splitFrustum);

    // calculate crop matrix
    Matrix mCropMatrix = g_Light.CalculateCropMatrix(casters, receivers, splitFrustum);
    // calculate view-proj matrix
    Matrix mSplitViewProj = g_Light.m_mView * g_Light.m_mProj * mCropMatrix;
    // calculate texture matrix
    Matrix mTextureMatrix = mSplitViewProj * GetTexScaleBiasMatrix();

    // render shadow map
    ActivateShadowMap();
    RenderCasters(casters, mSplitViewProj);
    DeactivateShadowMap();

    // render scene
    SetDepthRange(g_fSplitPos[i], g_fSplitPos[i+1]);
    SetShaderParams(mTextureMatrix);
    RenderReceivers(receivers, g_Camera.m_mView * g_Camera.m_mProj);
    RestoreDepthRange();
  }
}