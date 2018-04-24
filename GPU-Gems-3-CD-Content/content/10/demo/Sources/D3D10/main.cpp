#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D10.h"
#include "ShadowMap_D3D10.h"

extern void Render_DX10(void); // located in main_DX10.cpp
extern void Render_DX9(void); // located in main_DX9.cpp
extern void Render_MP(void); // located in main_MP.cpp

// shader
ID3D10Effect *g_pEffect = NULL;
ID3D10EffectTechnique *g_pTechniqueShadowMap_GSC = NULL;
ID3D10EffectTechnique *g_pTechniqueShadowMap_Inst = NULL;
ID3D10EffectTechnique *g_pTechniqueShadows = NULL;
ID3D10EffectMatrixVariable *g_pWorldVariable = NULL;
ID3D10EffectMatrixVariable *g_pViewProjVariable = NULL;
ID3D10EffectVectorVariable *g_pLightDirVariable = NULL;
ID3D10EffectVectorVariable *g_pLightAmbientVariable = NULL;
ID3D10EffectVectorVariable *g_pLightColorVariable = NULL;
ID3D10EffectMatrixVariable *g_pViewVariable = NULL;
ID3D10EffectMatrixVariable *g_pCropMatrixVariable = NULL;
ID3D10EffectShaderResourceVariable *g_pShadowMapTextureArrayVariable = NULL;
ID3D10EffectMatrixVariable *g_pTextureMatrixVariable = NULL;
ID3D10EffectScalarVariable *g_pSplitPlaneVariable = NULL;
ID3D10EffectScalarVariable *g_pFirstSplitVariable = NULL;
ID3D10EffectScalarVariable *g_pLastSplitVariable = NULL;
ID3D10EffectTechnique *g_pTechniqueShadowMap_Standard = NULL;
ID3D10EffectTechnique *g_pTechniqueShadows_MP = NULL;
ID3D10EffectTechnique *g_pTechniqueShadows_DX9 = NULL;
ID3D10EffectShaderResourceVariable *g_pShadowMapTextureVariable = NULL;

// font
ID3DX10Font *g_pFont = NULL;

#include "HUD.h"

// Creates shadow maps for the given method
//
//
bool ChangeRenderingMethod(int iNewMethod)
{
  // method not supported
  if(!g_bMethodSupported[iNewMethod]) return false;

  // destroy old shadow maps
  DestroyShadowMaps();

  // change to DX10 method
  //
  if(iNewMethod == METHOD_DX10_GSC || iNewMethod == METHOD_DX10_INST)
  {
    // create shadow map texture array
    ShadowMap_D3D10 *pShadowMap = new ShadowMap_D3D10();
    g_ShadowMaps.push_back(pShadowMap);
    //if(!pShadowMap->CreateAsTextureArray(g_iShadowMapSize, NUM_SPLITS_IN_SHADER)) return false;
    if(!pShadowMap->CreateAsTextureCube(g_iShadowMapSize)) return false;

    g_iRenderingMethod = iNewMethod;
    return true;
  }

  // change to DX9 method
  //
  if(iNewMethod == METHOD_DX9)
  {
    // create shadow map textures
    for(int i = 0; i < NUM_SPLITS_IN_SHADER; i++)
    {
      ShadowMap_D3D10 *pShadowMap = new ShadowMap_D3D10();
      g_ShadowMaps.push_back(pShadowMap);
      if(!pShadowMap->Create(g_iShadowMapSize)) return false;
    }

    g_iRenderingMethod = iNewMethod;
    return true;
  }

  // change to multi-pass method
  //
  if(iNewMethod == METHOD_MULTIPASS)
  {
    // create one shadow map texture
    ShadowMap_D3D10 *pShadowMap = new ShadowMap_D3D10();
    g_ShadowMaps.push_back(pShadowMap);
    if(!pShadowMap->Create(g_iShadowMapSize)) return false;

    g_iRenderingMethod = iNewMethod;
    return true;
  }

  return false;
}


// Calculates texture scale bias matrix
//
//
Matrix GetTexScaleBiasMatrix(void)
{
  // Calculate a matrix to transform points to shadow map texture coordinates
  // (this should be exactly like in your standard shadow map implementation)
  //
  float fTexOffset = 0.5f + (0.5f / (float)g_ShadowMaps[0]->GetSize());

  return Matrix(       0.5f,        0.0f,   0.0f,  0.0f,
                       0.0f,       -0.5f,   0.0f,  0.0f,
                       0.0f,        0.0f,   1.0f,  0.0f,
                 fTexOffset,  fTexOffset,   0.0f,  1.0f);
}


// This function is called once per frame
//
//
bool g_bDontRender = false;
double g_fTimePerFrame = 0;
void Render(void)
{
  // Make the app more reference rasterizer friendly
  // 
  if(GetApp()->GetParams().bReferenceRasterizer)
  {
    if(g_bDontRender)
    {
      char pText[1024];
      pText[0]=0;
      _snprintf(pText, 1024, "Frame rendered in %g seconds. Press SPACE to continue.", g_fTimePerFrame);
      SetWindowTextA( GetApp()->GetHWND(), pText);
      if(GetKeyDown(VK_SPACE)) g_bDontRender = false;
      Sleep(10);
      return;
    }
    SetWindowText( GetApp()->GetHWND(), TEXT("Rendering using reference rasterizer...") );
    g_fTimePerFrame = GetAccurateTime();
    g_bDontRender = true;
  }

  // move camera
  if(GetApp()->GetParams().bReferenceRasterizer) g_Camera.CalculateMatrices();
  else g_Camera.DoControls();

  // move light
  g_Light.DoControls();

  DoControls();

  // reset triangle counter
  g_iTrisPerFrame = 0;

  // Clear
  float ClearBG[4] = { 0.25f, 0.25f, 0.25f, 1.0f };
  GetApp()->GetDevice()->ClearRenderTargetView(GetApp()->GetRTV(), ClearBG);
  GetApp()->GetDevice()->ClearDepthStencilView(GetApp()->GetDSV(), D3D10_CLEAR_DEPTH, 1.0f, 0);

  g_iTrisPerFrame = 0;




  if(g_iRenderingMethod == METHOD_MULTIPASS) Render_MP();
  else if(g_iRenderingMethod == METHOD_DX9) Render_DX9();
  else if(g_iRenderingMethod == METHOD_DX10_GSC || g_iRenderingMethod == METHOD_DX10_INST) Render_DX10();

  RenderHUD();

  // present
  GetApp()->GetSwapChain()->Present(0, 0);

  if(GetApp()->GetParams().bReferenceRasterizer)
  {
    g_fTimePerFrame = GetAccurateTime() - g_fTimePerFrame;
  }
}


// Load shaders (this is actually called from Mesh_D3D10::CreateBuffers() because the input layout is needed there)
//
//
bool CreateShaders(void)
{
  ID3D10Blob *pErrors=NULL;

  HRESULT hr = D3DX10CreateEffectFromFile( TEXT("Shaders\\D3D10.fx"), NULL, NULL, "fx_4_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, 
                                         GetApp()->GetDevice(), NULL, NULL, &g_pEffect, &pErrors, NULL);

  // Pre-April 2007 SDK version
  //HRESULT hr = D3DX10CreateEffectFromFile( TEXT("Shaders\\D3D10.fx"), NULL, NULL, D3D10_SHADER_ENABLE_STRICTNESS, 0, 
  //                                         GetApp()->GetDevice(), NULL, NULL, &g_pEffect, &pErrors );

  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Loading effect failed!"), TEXT("Error!"), MB_OK);
    if(pErrors!=NULL)
    {
      MessageBoxA(NULL,(const char *)pErrors->GetBufferPointer(),"Compilation errors",MB_OK);
    }
    return false;
  }
  if(pErrors!=NULL) pErrors->Release();
  g_pTechniqueShadowMap_GSC = g_pEffect->GetTechniqueByName("RenderShadowMap_GSC");
  g_pTechniqueShadowMap_Inst = g_pEffect->GetTechniqueByName("RenderShadowMap_Inst");
  g_pTechniqueShadows = g_pEffect->GetTechniqueByName("RenderShadows");
  g_pWorldVariable = g_pEffect->GetVariableByName("g_mWorld")->AsMatrix();
  g_pViewProjVariable = g_pEffect->GetVariableByName("g_mViewProj")->AsMatrix();
  g_pViewVariable = g_pEffect->GetVariableByName("g_mView")->AsMatrix();
  g_pLightDirVariable = g_pEffect->GetVariableByName("g_vLightDir")->AsVector();
  g_pLightAmbientVariable = g_pEffect->GetVariableByName("g_vAmbient")->AsVector();
  g_pLightColorVariable = g_pEffect->GetVariableByName("g_vLightColor")->AsVector();
  g_pShadowMapTextureArrayVariable = g_pEffect->GetVariableByName("g_txShadowMapArray")->AsShaderResource();
  g_pCropMatrixVariable = g_pEffect->GetVariableByName("g_mCropMatrix")->AsMatrix();
  g_pTextureMatrixVariable = g_pEffect->GetVariableByName("g_mTextureMatrix")->AsMatrix();
  g_pSplitPlaneVariable = g_pEffect->GetVariableByName("g_fSplitPlane")->AsScalar();
  g_pFirstSplitVariable = g_pEffect->GetVariableByName("g_iFirstSplit")->AsScalar();
  g_pLastSplitVariable = g_pEffect->GetVariableByName("g_iLastSplit")->AsScalar();

  g_pTechniqueShadowMap_Standard = g_pEffect->GetTechniqueByName("RenderShadowMap_Standard");
  g_pTechniqueShadows_DX9 = g_pEffect->GetTechniqueByName("RenderShadows_DX9");
  g_pTechniqueShadows_MP = g_pEffect->GetTechniqueByName("RenderShadows_MP");
  g_pShadowMapTextureVariable = g_pEffect->GetVariableByName("g_txShadowMap")->AsShaderResource();

  // Load font
  //
  hr = D3DX10CreateFont(GetApp()->GetDevice(), 14, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                        OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                        TEXT("Verdana"), &g_pFont );
  if(FAILED(hr))
  {
    MessageBox(NULL,TEXT("Loading font failed!"),TEXT("Error!"),MB_OK);
  }

  return true;
}


// Unload everything
//
//
void DestroyShaders(void)
{
  // destroy shaders
  if(g_pEffect != NULL)
  {
    g_pEffect->Release();
    g_pEffect = NULL;
  }

  // unload font
  if(g_pFont!=NULL)
  {
    g_pFont->Release();
    g_pFont=NULL;
  }
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
  g_bMethodSupported[METHOD_DX10_GSC] = true;
  g_bMethodSupported[METHOD_DX10_INST] = true;

  if(CreateAll(lpCmdLine))
  {
    GetApp()->Run(&Render);
  }
  DestroyShaders();
  DestroyAll();
  return 0;
}