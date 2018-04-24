#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D10.h"
#include "ShadowMap_D3D10.h"
#include "Mesh_D3D10.h"

// This file handles DX10-level rendering
//

extern Matrix GetTexScaleBiasMatrix(void);
extern ID3D10Effect *g_pEffect;
extern ID3D10EffectTechnique *g_pTechniqueShadowMap_GSC;
extern ID3D10EffectTechnique *g_pTechniqueShadowMap_Inst;
extern ID3D10EffectTechnique *g_pTechniqueShadows;
extern ID3D10EffectMatrixVariable *g_pWorldVariable;
extern ID3D10EffectMatrixVariable *g_pViewProjVariable;
extern ID3D10EffectVectorVariable *g_pLightDirVariable;
extern ID3D10EffectVectorVariable *g_pLightAmbientVariable;
extern ID3D10EffectVectorVariable *g_pLightColorVariable;
extern ID3D10EffectMatrixVariable *g_pViewVariable;
extern ID3D10EffectMatrixVariable *g_pCropMatrixVariable;
extern ID3D10EffectShaderResourceVariable *g_pShadowMapTextureArrayVariable;
extern ID3D10EffectMatrixVariable *g_pTextureMatrixVariable;
extern ID3D10EffectScalarVariable *g_pSplitPlaneVariable;
extern ID3D10EffectScalarVariable *g_pFirstSplitVariable;
extern ID3D10EffectScalarVariable *g_pLastSplitVariable;

// Renders the given scene objects
//
//
static void RenderCasters(std::set<SceneObject *> &Objects, const Matrix &mViewProj, Matrix *mCropMatrix)
{
  // set constants
  g_pViewProjVariable->SetMatrix((float*)&mViewProj);
  g_pCropMatrixVariable->SetMatrixArray((float*)mCropMatrix, 0, g_iNumSplits);
  delete[] mCropMatrix;

  // for each pass in technique
  D3D10_TECHNIQUE_DESC DescTech;
  if(g_iRenderingMethod == METHOD_DX10_INST)
    g_pTechniqueShadowMap_Inst->GetDesc( &DescTech );
  else if(g_iRenderingMethod == METHOD_DX10_GSC)
    g_pTechniqueShadowMap_GSC->GetDesc( &DescTech );

  for(UINT p = 0; p < DescTech.Passes; ++p)
  {
    // for each object
    std::set<SceneObject *>::iterator it;
    for(it = Objects.begin(); it != Objects.end(); it++)
    {
      SceneObject *pObject = (*it);

      // set world matrix
      g_pWorldVariable->SetMatrix((float*)&pObject->m_mWorld);

      // set split range
      g_pFirstSplitVariable->SetInt(pObject->m_iFirstSplit);
      g_pLastSplitVariable->SetInt(pObject->m_iLastSplit);

      // keep triangle count accurate
      g_iTrisPerFrame += pObject->m_pMesh->m_iNumTris * (pObject->m_iLastSplit - pObject->m_iFirstSplit);
      
      if(g_iRenderingMethod == METHOD_DX10_INST)
      {
        // activate pass
        g_pTechniqueShadowMap_Inst->GetPassByIndex(p)->Apply(0);
        // draw instanced
        int iNumInstances = pObject->m_iLastSplit - pObject->m_iFirstSplit + 1;
        ((Mesh_D3D10 *)pObject->m_pMesh)->DrawInstanced(iNumInstances);
      }
      else
      {
        // activate pass
        g_pTechniqueShadowMap_GSC->GetPassByIndex(p)->Apply(0);
        // draw
        pObject->m_pMesh->Draw();
      }

      // reset variables
      pObject->m_iFirstSplit = INT_MAX;
      pObject->m_iLastSplit = INT_MIN;
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
  g_pTechniqueShadows->GetDesc( &DescTech );
  for(UINT p = 0; p < DescTech.Passes; ++p)
  {
    // for each object
    for(unsigned int j = 0; j < Objects.size(); j++)
    {
      SceneObject *pObject = Objects[j];

      // set world matrix
      g_pWorldVariable->SetMatrix((float*)&pObject->m_mWorld);

      // activate pass
      g_pTechniqueShadows->GetPassByIndex(p)->Apply(0);
      // draw
      pObject->m_pMesh->Draw();
    }
  }
}


// Set rendering range for given casters
//
//
static void UpdateSplitRange(const std::vector<SceneObject*> &casters, int iSplit)
{
  for(unsigned int i=0; i < casters.size(); i++)
  {
    SceneObject *pCaster = casters[i];
    if(iSplit < pCaster->m_iFirstSplit) pCaster->m_iFirstSplit = iSplit;
    if(iSplit > pCaster->m_iLastSplit) pCaster->m_iLastSplit = iSplit;
  }
}


// Starts rendering to shadow maps
//
//
static void ActivateShadowMaps(void)
{
  // unbind shadow map
  g_pShadowMapTextureArrayVariable->SetResource(NULL);

  // Enable rendering to shadow map
  GetShadowMap<ShadowMap_D3D10>()->EnableRendering();

  // Clear texture
  float ClearSM[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  GetApp()->GetDevice()->ClearDepthStencilView(GetShadowMap<ShadowMap_D3D10>()->m_pDSV, D3D10_CLEAR_DEPTH, 1.0f, 0);
}


// Stops rendering to shadow maps
//
//
static void DeactivateShadowMaps(void)
{
  // Disable rendering to shadow map
  GetShadowMap<ShadowMap_D3D10>()->DisableRendering();
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

  // bind shadow map texture array
  g_pShadowMapTextureArrayVariable->SetResource(GetShadowMap<ShadowMap_D3D10>()->m_pSRV);
}


// DX10-level rendering function
//
//
void Render_DX10(void)
{
  // find receivers
  std::vector<SceneObject *> receivers;
  std::set<SceneObject *> casters;
  receivers = g_Camera.FindReceivers();

  // adjust camera planes to contain scene tightly
  g_Camera.AdjustPlanes(receivers);

  // calculate the distances of split planes
  g_Camera.CalculateSplitPositions(g_fSplitPos);

  // array of texture matrices
  Matrix *mTextureMatrix = new Matrix[g_iNumSplits];
  // array of crop matrices
  Matrix *mCropMatrix = new Matrix[g_iNumSplits];

  // for each split
  for(int i = 0; i < g_iNumSplits; i++)
  {
    // calculate frustum
    Frustum splitFrustum;
    splitFrustum = g_Camera.CalculateFrustum(g_fSplitPos[i], g_fSplitPos[i+1]);
    // find casters
    std::vector<SceneObject *> castersInSplit;
    castersInSplit = g_Light.FindCasters(splitFrustum);
    UpdateSplitRange(castersInSplit, i);
    casters.insert(castersInSplit.begin(), castersInSplit.end());

    // calculate crop matrix
    mCropMatrix[i] = g_Light.CalculateCropMatrix(castersInSplit, receivers, splitFrustum);
    // calculate texture matrix
    mTextureMatrix[i] = g_Light.m_mView * g_Light.m_mProj * mCropMatrix[i] * GetTexScaleBiasMatrix();
  }

  // render shadow map
  ActivateShadowMaps();
  RenderCasters(casters, g_Light.m_mView * g_Light.m_mProj, mCropMatrix);
  DeactivateShadowMaps();

  // render scene
  SetShaderParams(mTextureMatrix);
  RenderReceivers(receivers, g_Camera.m_mView * g_Camera.m_mProj);
}
