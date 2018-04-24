#include "../Framework/Common.h"
#include "../Framework/DemoSetup.h"
#include "Application_OGL.h"
#include "ShadowMap_OGL.h"
#include "GLSLShader.h"

extern void Render_DX10(void); // located in main_DX10.cpp
extern void Render_DX9(void); // located in main_DX9.cpp
extern void Render_MP(void); // located in main_MP.cpp

GLSLShader g_Shadows_MP_Shader; // multi-pass shadow rendering
GLSLShader g_Shadows_DX9_Shader; // DX9-level shadow rendering
GLSLShader g_Shadows_DX10_Shader; // DX10-level shadow rendering
GLSLShader g_ShadowMap_DX10_GSC_Shader; // DX10-level (GS cloning) shadow map generation
GLSLShader g_ShadowMap_DX10_Inst_Shader; // DX10-level (instancing) shadow map generation
GLSLShader *g_pActiveShader = NULL;

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
    ShadowMap_OGL *pShadowMap = new ShadowMap_OGL();
    g_ShadowMaps.push_back(pShadowMap);
    if(!pShadowMap->CreateAsTextureArray(g_iShadowMapSize, NUM_SPLITS_IN_SHADER)) return false;

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
      ShadowMap_OGL *pShadowMap = new ShadowMap_OGL();
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
    ShadowMap_OGL *pShadowMap = new ShadowMap_OGL();
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
  return Matrix( 0.5f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.5f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.5f, 0.0f,
                 0.5f, 0.5f, 0.5f, 1.0f);
}


// Renders shadow receivers (common for all methods)
//
//
void RenderReceivers(const std::vector<SceneObject *> &Objects, const Matrix &mViewProj)
{
  g_pActiveShader->SetMatrix("g_mViewProj", mViewProj);
  g_pActiveShader->SetVector("g_vLightDir", g_Light.GetDir());
  g_pActiveShader->SetVector("g_vLightColor", g_Light.m_vLightDiffuse);
  g_pActiveShader->SetVector("g_vAmbient", g_Light.m_vLightAmbient);

  // for each object
  for(unsigned int j = 0; j < Objects.size(); j++)
  {
    SceneObject *pObject = Objects[j];
    g_pActiveShader->SetMatrix("g_mWorld", pObject->m_mWorld);
    pObject->m_pMesh->Draw();
  }
}


// This function is called once per frame
//
//
void Render(void)
{
  // adjust settings
  DoControls();

  // reset triangle counter
  g_iTrisPerFrame = 0;

  // move camera
  g_Camera.DoControls();

  // move light
  g_Light.DoControls();

  // clear the screen
  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
  glClearDepth(1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);

  // render
  if(g_iRenderingMethod == METHOD_MULTIPASS) Render_MP();
  else if(g_iRenderingMethod == METHOD_DX9) Render_DX9();
  else if(g_iRenderingMethod == METHOD_DX10_GSC || g_iRenderingMethod == METHOD_DX10_INST) Render_DX10();

  // render HUD
  RenderHUD();
}


// Load shaders
//
//
bool CreateShaders(void)
{
  if(!g_Shadows_MP_Shader.Load("Shaders\\VS_Shadows_MP.glsl", NULL, "Shaders\\FS_Shadows_MP.glsl")) return false;
  if(!g_Shadows_DX9_Shader.Load("Shaders\\VS_Shadows_DX9.glsl", NULL, "Shaders\\FS_Shadows_DX9.glsl"))
  {
    MessageBox(NULL, TEXT("DX9-level method is not supported - it will be disabled."), TEXT("Error!"), MB_OK);
    g_bMethodSupported[METHOD_DX9] = false;
    return false;
  }

  // geometry shaders and texture arrays supported
  if(GetApp()->m_bGeometryShadersSupported &&
     GetApp()->m_bTextureArraysSupported)
  {
    // DX10 methods supported
    g_bMethodSupported[METHOD_DX10_GSC] = true;
    g_bMethodSupported[METHOD_DX10_INST] = true;

    if(g_Shadows_DX10_Shader.Load("Shaders\\VS_Shadows_DX10.glsl", NULL, "Shaders\\FS_Shadows_DX10.glsl"))
    {
      if(!g_ShadowMap_DX10_GSC_Shader.Load("Shaders\\VS_ShadowMap_DX10_GSC.glsl", "Shaders\\GS_ShadowMap_DX10_GSC.glsl", NULL, NUM_SPLITS_IN_SHADER*3))
      {
        MessageBox(NULL, TEXT("GSC method not supported."), TEXT("Error!"), MB_OK);
        g_bMethodSupported[METHOD_DX10_GSC] = false;
      }
      if(!g_ShadowMap_DX10_Inst_Shader.Load("Shaders\\VS_ShadowMap_DX10_Inst.glsl", "Shaders\\GS_ShadowMap_DX10_Inst.glsl", NULL, 3)
         || !GetApp()->m_bInstancingSupported)
      {
        MessageBox(NULL, TEXT("Instancing method not supported."), TEXT("Error!"), MB_OK);
        g_bMethodSupported[METHOD_DX10_INST] = false;
      }
    } else {
      MessageBox(NULL, TEXT("DX10-level shaders failed to compile."), TEXT("Error!"), MB_OK);
      g_bMethodSupported[METHOD_DX10_GSC] = false;
      g_bMethodSupported[METHOD_DX10_INST] = false;
    }
  }

  return true;
}


// Destroy shaders
//
//
void DestroyShaders(void)
{
  // destroy shaders
  g_Shadows_MP_Shader.Destroy();
  g_Shadows_DX9_Shader.Destroy();
  g_Shadows_DX10_Shader.Destroy();
  g_ShadowMap_DX10_GSC_Shader.Destroy();
  g_ShadowMap_DX10_Inst_Shader.Destroy();
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
