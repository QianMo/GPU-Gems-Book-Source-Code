#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D9.h"
#include "ShadowMap_D3D9.h"

// This file handles multi-pass rendering
//

extern void RenderSplitOnHUD(int iSplit);
extern Matrix GetTexScaleBiasMatrix(void);
extern void RenderObjects(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj);
extern LPD3DXEFFECT g_pEffect;

// Sets shader params for multi-pass shadow rendering
//
//
static void SetShaderParams(const Matrix &mTextureMatrix)
{
  g_pEffect->SetMatrixArray("g_mTextureMatrix", (D3DXMATRIX*)&mTextureMatrix, 1);
  g_pEffect->SetTechnique("RenderShadows_MP");

  // bind shadowmap as texture
  GetApp()->GetDevice()->SetTexture(0, GetShadowMap<ShadowMap_D3D9>(0)->GetColorTexture());
  
  // set correct filters
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_BORDER);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_BORDER);
  GetApp()->GetDevice()->SetSamplerState(0, D3DSAMP_BORDERCOLOR, 0xFFFFFFFF);
}


// Sets depth range settings for a split
//
//
static D3DVIEWPORT9 _OriginalCameraViewport;
static float _fOriginalCameraNear;
static float _fOriginalCameraFar;
static void SetDepthRange(float fNear, float fFar)
{
  D3DVIEWPORT9 CameraViewport;
  GetApp()->GetDevice()->GetViewport(&CameraViewport);

  // store original values
  _OriginalCameraViewport = CameraViewport;
  _fOriginalCameraNear = g_Camera.m_fNear;
  _fOriginalCameraFar = g_Camera.m_fFar;

  // set new depth value range
  CameraViewport.MinZ = (fNear - g_Camera.m_fNear) / (g_Camera.m_fFar - g_Camera.m_fNear);
  CameraViewport.MaxZ = (fFar - g_Camera.m_fNear) / (g_Camera.m_fFar - g_Camera.m_fNear);
  GetApp()->GetDevice()->SetViewport(&CameraViewport);

  // set new far and near plane
  g_Camera.m_fNear = fNear;
  g_Camera.m_fFar = fFar;
  g_Camera.CalculateMatrices();
}


// Resets original depth range settings
//
//
static void RestoreDepthRange(void)
{
  GetApp()->GetDevice()->SetViewport(&_OriginalCameraViewport);
  g_Camera.m_fNear = _fOriginalCameraNear;
  g_Camera.m_fFar = _fOriginalCameraFar;
}


// Starts rendering to shadow map
//
//
static void ActivateShadowMap(void)
{
  // enable rendering to shadow map
  GetShadowMap<ShadowMap_D3D9>()->EnableRendering();

  // clear the shadowmap
  GetApp()->GetDevice()->Clear(0, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0xFFFFFFFF, 1.0f, 0);

  // set shaders
  g_pEffect->SetTechnique("RenderShadowMap");
}


// Stops rendering to shadow map
//
//
static void DeactivateShadowMap(void)
{
  // reset original render target
  GetShadowMap<ShadowMap_D3D9>()->DisableRendering();
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
    RenderObjects(casters, mSplitViewProj);
    DeactivateShadowMap();

    // render scene
    SetDepthRange(g_fSplitPos[i], g_fSplitPos[i+1]);
    SetShaderParams(mTextureMatrix);
    RenderObjects(receivers, g_Camera.m_mView * g_Camera.m_mProj);
    RestoreDepthRange();

    // render texture preview
    RenderSplitOnHUD(i);
  }
}