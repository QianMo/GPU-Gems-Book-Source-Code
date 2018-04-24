#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_OGL.h"
#include "ShadowMap_OGL.h"
#include "GLSLShader.h"

// This file handles multi-pass rendering
//

extern Matrix GetTexScaleBiasMatrix(void);
extern void RenderReceivers(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj);
extern GLSLShader g_Shadows_MP_Shader; // multi-pass shadow rendering
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


// Sets shader params for multi-pass shadow rendering
//
//
static void SetShaderParams(const Matrix &mTextureMatrix)
{
  // enable front face culling
  glCullFace(GL_FRONT);

  // bind texture
  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  GetShadowMap<ShadowMap_OGL>()->Bind();

  // activate shaders
  g_Shadows_MP_Shader.Activate();

  // set texture matrix
  g_Shadows_MP_Shader.SetMatrix("g_mTextureMatrix", mTextureMatrix);

  // set texture register for sampler
  g_Shadows_MP_Shader.SetInt("g_samShadowMap", 0);
}


// Sets depth range settings for a split
//
//
static float _fOriginalCameraNear;
static float _fOriginalCameraFar;
static void SetDepthRange(float fNear, float fFar)
{
  _fOriginalCameraNear = g_Camera.m_fNear;
  _fOriginalCameraFar = g_Camera.m_fFar;

  glDepthRange((fNear - g_Camera.m_fNear) / (g_Camera.m_fFar - g_Camera.m_fNear),
                (fFar - g_Camera.m_fNear) / (g_Camera.m_fFar - g_Camera.m_fNear));

  g_Camera.m_fNear = fNear;
  g_Camera.m_fFar = fFar;
  g_Camera.CalculateMatrices();
}


// Resets original depth range settings
//
//
static void RestoreDepthRange(void)
{
  glDepthRange(0, 1);
  g_Camera.m_fNear = _fOriginalCameraNear;
  g_Camera.m_fFar = _fOriginalCameraFar;
  g_Camera.CalculateMatrices();
}


// Starts rendering to shadow map
//
//
static void ActivateShadowMap(void)
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
  GetShadowMap<ShadowMap_OGL>()->EnableRendering();

  // clear the shadow map
  glClearDepth(1.0f);
  glClear(GL_DEPTH_BUFFER_BIT);
}


// Stops rendering to shadow map
//
//
static void DeactivateShadowMap(void)
{
  // reset original render target
  GetShadowMap<ShadowMap_OGL>()->DisableRendering();
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