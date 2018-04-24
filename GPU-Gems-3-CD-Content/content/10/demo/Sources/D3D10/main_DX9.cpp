#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D10.h"
#include "ShadowMap_D3D10.h"

// This file handles DX9-level rendering
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
extern ID3D10EffectTechnique *g_pTechniqueShadows_DX9;
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
  g_pViewVariable->SetMatrix((float*)&g_Camera.m_mView);
  g_pLightDirVariable->SetFloatVector((float*)&g_Light.GetDir());
  g_pLightColorVariable->SetFloatVector((float*)&g_Light.m_vLightDiffuse);
  g_pLightAmbientVariable->SetFloatVector((float*)&g_Light.m_vLightAmbient);

  // for each pass in technique
  D3D10_TECHNIQUE_DESC DescTech;
  g_pTechniqueShadows_DX9->GetDesc( &DescTech );
  for(UINT p = 0; p < DescTech.Passes; ++p)
  {
    // for each object
    for(unsigned int j = 0; j < Objects.size(); j++)
    {
      SceneObject *pObject = Objects[j];

      // set world matrix
      g_pWorldVariable->SetMatrix((float*)&pObject->m_mWorld);

      // activate pass
      g_pTechniqueShadows_DX9->GetPassByIndex(p)->Apply(0);
      // draw
      pObject->m_pMesh->Draw();
    }
  }
}


// Starts rendering to shadow maps
//
//
static void ActivateShadowMap(int iSM)
{
  // unbind shadow maps
  ID3D10ShaderResourceView *pResources[NUM_SPLITS_IN_SHADER] = {NULL};
  g_pShadowMapTextureVariable->SetResourceArray(pResources, 0, NUM_SPLITS_IN_SHADER);

  // Enable rendering to shadow map
  GetShadowMap<ShadowMap_D3D10>(iSM)->EnableRendering();

  // Clear texture
  float ClearSM[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  GetApp()->GetDevice()->ClearDepthStencilView(GetShadowMap<ShadowMap_D3D10>(iSM)->m_pDSV, D3D10_CLEAR_DEPTH, 1.0f, 0);
}


// Stops rendering to shadow map
//
//
static void DeactivateShadowMap(int iSM)
{
  // Disable rendering to shadow map
  GetShadowMap<ShadowMap_D3D10>(iSM)->DisableRendering();
}


// Sets shader parameters for hardware shadow rendering
//
//
static void SetShaderParams(Matrix *mTextureMatrix)
{
  g_pTextureMatrixVariable->SetMatrixArray((float*)mTextureMatrix, 0, g_iNumSplits);
  delete[] mTextureMatrix;

  // store split end positions in shader
  g_pSplitPlaneVariable->SetFloatArray(&g_fSplitPos[1], 0, g_iNumSplits);

  // bind shadow map textures
  ID3D10ShaderResourceView *pResources[NUM_SPLITS_IN_SHADER];
  for(int i = 0; i < g_iNumSplits; i++)
  {
    pResources[i] = GetShadowMap<ShadowMap_D3D10>(i)->m_pSRV;
  }
  g_pShadowMapTextureVariable->SetResourceArray(pResources, 0, g_iNumSplits);
}

// DX9-level rendering function
//
//
void Render_DX9(void)
{
  // find receivers
  std::vector<SceneObject *> receivers, casters;
  receivers = g_Camera.FindReceivers();

  // adjust camera planes to contain scene tightly
  g_Camera.AdjustPlanes(receivers);

  // calculate the distances of split planes
  g_Camera.CalculateSplitPositions(g_fSplitPos);

  // array of texture matrices
  Matrix *mTextureMatrix = new Matrix[g_iNumSplits];

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
    mTextureMatrix[i] = mSplitViewProj * GetTexScaleBiasMatrix();

    // render shadow map
    ActivateShadowMap(i);
    RenderCasters(casters, mSplitViewProj);
    DeactivateShadowMap(i);
  }

  // render scene
  SetShaderParams(mTextureMatrix);
  RenderReceivers(receivers, g_Camera.m_mView * g_Camera.m_mProj);
}