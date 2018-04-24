#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_D3D9.h"
#include "ShadowMap_D3D9.h"

extern void Render_DX9(void); // located in main_DX9.cpp
extern void Render_MP(void); // located in main_MP.cpp

// shaders
LPD3DXEFFECT g_pEffect = NULL;

// font
LPD3DXFONT g_pFont = NULL;

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
  
  // change to DX9 method
  //
  if(iNewMethod == METHOD_DX9)
  {
    // create shadow map textures
    for(int i = 0; i < NUM_SPLITS_IN_SHADER; i++)
    {
      ShadowMap_D3D9 *pShadowMap = new ShadowMap_D3D9();
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
    ShadowMap_D3D9 *pShadowMap = new ShadowMap_D3D9();
    g_ShadowMaps.push_back(pShadowMap);
    if(!pShadowMap->Create(g_iShadowMapSize)) return false;

    g_iRenderingMethod = iNewMethod;
    return true;
  }

  return false;
}


// Renders the given scene objects (common for both methods)
//
//
void RenderObjects(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj)
{
  // set constants
  g_pEffect->SetMatrix("g_mViewProj", (D3DXMATRIX*)&mViewProj);
  g_pEffect->SetVector("g_vLightDir", (D3DXVECTOR4*)&Vector4(g_Light.GetDir(),0));
  g_pEffect->SetVector("g_vLightColor", (D3DXVECTOR4*)&g_Light.m_vLightDiffuse);
  g_pEffect->SetVector("g_vAmbient", (D3DXVECTOR4*)&g_Light.m_vLightAmbient);
  g_pEffect->SetFloat("g_fShadowMapSize", (FLOAT)g_ShadowMaps[0]->GetSize());
  g_pEffect->SetFloat("g_fShadowMapTexelSize", 1.0f/(FLOAT)g_ShadowMaps[0]->GetSize());

  // enable effect
  unsigned int iPasses=0;
  if(SUCCEEDED(g_pEffect->Begin(&iPasses, 0)))
  {
    // for each pass in effect 
    for(unsigned int i = 0; i < iPasses; i++)
    {
      // start pass
      if(SUCCEEDED(g_pEffect->BeginPass(i)))
      {
        // for each object
        for(unsigned int j = 0; j < Objects.size(); j++)
        {
          SceneObject *pObject = Objects[j];

          // set world matrix
          g_pEffect->SetMatrix("g_mWorld", (D3DXMATRIX*)&pObject->m_mWorld);
          g_pEffect->CommitChanges();

          // draw
          pObject->m_pMesh->Draw();
        }
        // end pass
        g_pEffect->EndPass();
      }

    }
    // disable effect
    g_pEffect->End();
  }
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

  Matrix mTexScaleBias(       0.5f,        0.0f,   0.0f,  0.0f,
                                  0.0f,       -0.5f,   0.0f,  0.0f,
                                  0.0f,        0.0f,   1.0f,  0.0f,
                            fTexOffset,  fTexOffset,   0.0f,  1.0f);
  return mTexScaleBias;
}


// This function is called once per frame
//
//
void Render(void)
{
  // adjust settings, etc..
  DoControls();

  // reset triangle counter
  g_iTrisPerFrame = 0;

  // move camera
  g_Camera.DoControls();

  // calculate the light position
  g_Light.DoControls();

  // clear the screen
  GetApp()->GetDevice()->Clear(0, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, D3DXCOLOR(0.25f,0.25f,0.25f,0.25f), 1.0f, 0);

  // render multi-pass or DX9
  if(g_iRenderingMethod == METHOD_MULTIPASS) Render_MP();
  else if(g_iRenderingMethod == METHOD_DX9) Render_DX9();

  // render other HUD stuff
  RenderHUD();
}


// Load shaders
//
//
bool CreateShaders(void)
{
  // Load .FX file (shaders)
  //
  LPD3DXBUFFER pErrors=NULL;
  HRESULT hr=D3DXCreateEffectFromFile(GetApp()->GetDevice(),TEXT("Shaders\\D3D9.fx"),NULL,NULL,0,NULL,&g_pEffect,&pErrors);

  if(FAILED(hr))
  {
    MessageBox(NULL,TEXT("Loading effect failed"),TEXT("Error!"),MB_OK);
    if(pErrors!=NULL)
    {
      MessageBoxA(NULL,(const char *)pErrors->GetBufferPointer(),"Compilation errors",MB_OK);
    }
    return false;
  }

  if(pErrors!=NULL) pErrors->Release();

  if(g_pEffect->ValidateTechnique("RenderShadows") != D3D_OK)
  {
    MessageBox(NULL,TEXT("DX9-level shaders not supported. The method will be disabled."),TEXT("Error!"),MB_OK);
    g_bMethodSupported[METHOD_DX9] = false;
  }

  // Load font
  //
  hr = D3DXCreateFont(GetApp()->GetDevice(), 14, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                      OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                      TEXT("Verdana"), &g_pFont );
  if(FAILED(hr))
  {
    MessageBox(NULL,TEXT("Loading font failed!"),TEXT("Error!"),MB_OK);
  }

  return true;
}


// Destroy shaders
//
//
void DestroyShaders(void)
{
  // unload FX file
  if(g_pEffect!=NULL)
  {
    g_pEffect->Release();
    g_pEffect = NULL;
  }

  // unload font
  if(g_pFont!=NULL)
  {
    g_pFont->Release();
    g_pFont = NULL;
  }
}


int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
  if(CreateAll(lpCmdLine) && CreateShaders())
  {
    GetApp()->Run(&Render);
  }
  DestroyShaders();
  DestroyAll();
  return 0;
}
