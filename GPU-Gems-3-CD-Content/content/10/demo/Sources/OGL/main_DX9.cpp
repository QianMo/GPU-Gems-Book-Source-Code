#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_OGL.h"
#include "ShadowMap_OGL.h"
#include "GLSLShader.h"

// This file handles DX9-level rendering
//

extern Matrix GetTexScaleBiasMatrix(void);
extern void RenderReceivers(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj);
extern GLSLShader g_Shadows_DX9_Shader; // DX9-level shadow rendering
extern GLSLShader *g_pActiveShader;


// Renders shadow casters
//
//
static void RenderCasters(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj)
{
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf((float*)&mViewProj);
  glMatrixMode(GL_MODELVIEW);

  // for each object
  for(unsigned int j = 0; j < Objects.size(); j++)
  {
    SceneObject *pObject = Objects[j];
    // set constants
    glLoadMatrixf((float*)&pObject->m_mWorld);

    // draw
    pObject->m_pMesh->Draw();
  }
}


// Sets shader parameters for DX9-level shadow rendering
//
//
static void SetShaderParams(Matrix *mTextureMatrix)
{
  for(int i = 0;i < g_iNumSplits; i++)
  {
    // bind textures
    glActiveTextureARB(GL_TEXTURE0_ARB + i);
    glEnable(GL_TEXTURE_2D);
    GetShadowMap<ShadowMap_OGL>(i)->Bind();
  }

  // enable front face culling
  glCullFace(GL_FRONT);

  // activate shaders
  g_Shadows_DX9_Shader.Activate();

  g_Shadows_DX9_Shader.SetMatrixArray("g_mTextureMatrix", mTextureMatrix, g_iNumSplits);
  delete[] mTextureMatrix;

  // store split end positions in shader
  g_Shadows_DX9_Shader.SetFloatArray("g_fSplitPlane", &g_fSplitPos[1], NUM_SPLITS_IN_SHADER);
  // store camera's view matrix
  g_Shadows_DX9_Shader.SetMatrix("g_mView", g_Camera.m_mView);

  // set texture registers for samplers
  int pTextureRegisters[NUM_SPLITS_IN_SHADER];
  for(int i=0;i<NUM_SPLITS_IN_SHADER;i++) pTextureRegisters[i] = i;
  g_Shadows_DX9_Shader.SetIntArray("g_samShadowMap", pTextureRegisters, NUM_SPLITS_IN_SHADER);
}


// Starts rendering to shadow map
//
//
static void ActivateShadowMap(int iSM)
{
  // make sure textures are not bound
  for(unsigned int i=0;i<g_ShadowMaps.size();i++)
  {
    glActiveTextureARB(GL_TEXTURE0_ARB + i);
    GetShadowMap<ShadowMap_OGL>(i)->Unbind();
    glDisable(GL_TEXTURE_2D);
  }

  // disable shaders
  if(g_pActiveShader != NULL) g_pActiveShader->Deactivate();
  glCullFace(GL_BACK);

  // enable rendering to shadow map
  GetShadowMap<ShadowMap_OGL>(iSM)->EnableRendering();

  // clear the shadow map
  glClearDepth(1.0f);
  glClear(GL_DEPTH_BUFFER_BIT);
}


// Stops rendering to shadow map
//
//
static void DeactivateShadowMap(int iSM)
{
  // reset original render target
  GetShadowMap<ShadowMap_OGL>(iSM)->DisableRendering();
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