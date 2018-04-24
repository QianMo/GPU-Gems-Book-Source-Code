#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_OGL.h"
#include "ShadowMap_OGL.h"
#include "GLSLShader.h"
#include "Mesh_OGL.h"

// This file handles DX10-level rendering
//

extern Matrix GetTexScaleBiasMatrix(void);
extern void RenderReceivers(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj);
extern GLSLShader g_Shadows_DX10_Shader; // DX10-level shadow rendering
extern GLSLShader g_ShadowMap_DX10_GSC_Shader; // DX10-level (GS cloning) shadow map generation
extern GLSLShader g_ShadowMap_DX10_Inst_Shader; // DX10-level (instancing) shadow map generation
extern GLSLShader *g_pActiveShader;


// Renders shadow casters, for DX10-level rendering
//
//
static void RenderCasters(std::set<SceneObject *> &Objects, const Matrix &mViewProj, Matrix *mCropMatrix)
{
  GLSLShader *pShader = NULL;
  if(g_iRenderingMethod == METHOD_DX10_GSC) pShader = &g_ShadowMap_DX10_GSC_Shader;
  else if(g_iRenderingMethod == METHOD_DX10_INST) pShader = &g_ShadowMap_DX10_Inst_Shader;

  // store crop matrices to the shader
  pShader->SetMatrix("g_mViewProj", mViewProj);
  pShader->SetMatrixArray("g_mCropMatrix", mCropMatrix, g_iNumSplits);
  delete[] mCropMatrix;

  // for each object
  std::set<SceneObject *>::iterator it;
  for(it = Objects.begin(); it != Objects.end(); it++)
  {
    SceneObject *pObject = (*it);
    // set constants
    pShader->SetMatrix("g_mWorld", pObject->m_mWorld);
    pShader->SetInt("g_iFirstSplit", pObject->m_iFirstSplit);
    pShader->SetInt("g_iLastSplit", pObject->m_iLastSplit);

    if(g_iRenderingMethod == METHOD_DX10_INST)
    {
      // draw instanced
      int iNumInstances = pObject->m_iLastSplit - pObject->m_iFirstSplit + 1;
      ((Mesh_OGL *)pObject->m_pMesh)->DrawInstanced(iNumInstances);
    }
    else
    {
      // draw
      pObject->m_pMesh->Draw();
    }

    // keep triangle count accurate
    g_iTrisPerFrame += pObject->m_pMesh->m_iNumTris * (pObject->m_iLastSplit - pObject->m_iFirstSplit);

    // reset counters
    pObject->m_iFirstSplit = INT_MAX;
    pObject->m_iLastSplit = INT_MIN;
  }
}


// Sets shader parameters for DX10-level shadow rendering
//
//
static void SetShaderParams_DX10(Matrix *mTextureMatrix)
{
  // bind texture
  GetShadowMap<ShadowMap_OGL>()->Bind();

  // enable front face culling
  glCullFace(GL_FRONT);

  // activate shaders
  g_Shadows_DX10_Shader.Activate();

  // store shadow map texture matrices to the shader
  g_Shadows_DX10_Shader.SetMatrixArray("g_mTextureMatrix", mTextureMatrix, g_iNumSplits);
  delete[] mTextureMatrix;

  // store split end positions in shader
  g_Shadows_DX10_Shader.SetFloatArray("g_fSplitPlane", &g_fSplitPos[1], NUM_SPLITS_IN_SHADER);
  // store camera's view matrix
  g_Shadows_DX10_Shader.SetMatrix("g_mView", g_Camera.m_mView);

  // set texture for sampler
  g_Shadows_DX10_Shader.SetInt("g_samShadowMap", 0);
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


// Starts rendering to shadow map texture array
//
//
static void ActivateShadowMaps(void)
{
  // make sure no textures are bound
  for(int i=0;i<g_iNumSplits;i++)
  {
    glActiveTextureARB(GL_TEXTURE0_ARB + i);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }
  glActiveTextureARB(GL_TEXTURE0_ARB);
  GetShadowMap<ShadowMap_OGL>()->Bind();

  // set shader
  if(g_iRenderingMethod == METHOD_DX10_GSC) g_ShadowMap_DX10_GSC_Shader.Activate();
  else if(g_iRenderingMethod == METHOD_DX10_INST) g_ShadowMap_DX10_Inst_Shader.Activate();
  glCullFace(GL_BACK);

  // enable rendering to shadow map
  GetShadowMap<ShadowMap_OGL>()->EnableRendering();

  // clear the shadow map
  glClearDepth(1.0f);
  glClear(GL_DEPTH_BUFFER_BIT);
}


// Stops rendering to shadow map texture array
//
//
static void DeactivateShadowMaps(void)
{
  // reset original render target
  GetShadowMap<ShadowMap_OGL>()->DisableRendering();
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

  // for each split part
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
  SetShaderParams_DX10(mTextureMatrix);
  RenderReceivers(receivers, g_Camera.m_mView * g_Camera.m_mProj);
}