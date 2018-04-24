#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D9.h"
#include "ShadowMap_D3D9.h"

// This file handles DX9-level rendering
//

extern void RenderSplitOnHUD(int iSplit);
extern Matrix GetTexScaleBiasMatrix(void);
extern void RenderObjects(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj);
extern LPD3DXEFFECT g_pEffect;

// Sets shader parameters for DX9-level shadow rendering
//
//
static void SetShaderParams(Matrix *mTextureMatrix)
{
  // store shadow map texture matrices to the shader
  g_pEffect->SetMatrixArray("g_mTextureMatrix", (D3DXMATRIX*)mTextureMatrix, g_iNumSplits);
  delete[] mTextureMatrix;

  // store split end positions in shader
  g_pEffect->SetFloatArray("g_fSplitPlane", &g_fSplitPos[1], g_iNumSplits);
  // store camera's view matrix
  g_pEffect->SetMatrix("g_mView", (D3DXMATRIX*)&g_Camera.m_mView);

  for(int i=0;i<g_iNumSplits;i++)
  {
    GetApp()->GetDevice()->SetTexture(i, GetShadowMap<ShadowMap_D3D9>(i)->GetColorTexture());
    // set correct sampler states
    GetApp()->GetDevice()->SetSamplerState(i, D3DSAMP_MINFILTER, D3DTEXF_POINT);
    GetApp()->GetDevice()->SetSamplerState(i, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
    GetApp()->GetDevice()->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
    GetApp()->GetDevice()->SetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_BORDER);
    GetApp()->GetDevice()->SetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_BORDER);
    GetApp()->GetDevice()->SetSamplerState(i, D3DSAMP_BORDERCOLOR, 0xFFFFFFFF);
  }

  // setup shaders
  g_pEffect->SetTechnique("RenderShadows");
}


// Starts rendering to shadow map
//
//
static void ActivateShadowMap(int iSM)
{
  // enable rendering to shadow map
  GetShadowMap<ShadowMap_D3D9>(iSM)->EnableRendering();

  // clear the shadowmap
  GetApp()->GetDevice()->Clear(0, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0xFFFFFFFF, 1.0f, 0);

  // set shaders
  g_pEffect->SetTechnique("RenderShadowMap");
}


// Stops rendering to shadow map
//
//
static void DeactivateShadowMap(int iSM)
{
  // reset original render target
  GetShadowMap<ShadowMap_D3D9>(iSM)->DisableRendering();
}


// Renders all shadow map texture previews
//
//
static void RenderPreviewsToHUD(void)
{
  // draw the shadowmap texture to HUD, just for previewing purposes
  for(int i=0;i<g_iNumSplits;i++)
  {
    GetApp()->GetDevice()->SetTexture(0, GetShadowMap<ShadowMap_D3D9>(i)->GetColorTexture());
    RenderSplitOnHUD(i);
  }

  // unbind textures so we can render into them again
  for(int i=0;i<g_iNumSplits;i++)
  {
    GetApp()->GetDevice()->SetTexture(i, NULL);
  }
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
    RenderObjects(casters, mSplitViewProj);
    DeactivateShadowMap(i);
  }

  // render scene
  SetShaderParams(mTextureMatrix);
  RenderObjects(receivers, g_Camera.m_mView * g_Camera.m_mProj);

  // render texture previews
  RenderPreviewsToHUD();
}
