///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : RefractionSimulation.cpp
//  Desc : Generic refraction simulation demo
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "D3dApp.h"
#include "Common.h"
#include "Win32App.h"
#include "Control.h"
#include "Camera.h"
#include "Shader.h"
#include "Mesh.h"
#include "Material.h"
#include "RenderTarget.h"
#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "Timer.h"

class CMyApp:public CD3DApp 
{
public:  
  CMyApp()
  {        
    m_iFrameCounter=0;    
    m_pCGContext=0;
    m_pSkybox=0;
    m_pRefractiveMesh=0;    
    m_pEnvMap=0;
    m_plD3DBackbufferSurf=0;
    m_plD3DDepthStencilSurf=0;
    m_pRTRefraction=0;
  }

  virtual ~CMyApp()
  {
    Release();
  }

  // Initialize/shutdown application
  int InitializeApp();
  int ShutDownApp();

  // Update/render frame
  int Update(float fTimeSpan);
  int Render(float fTimeSpan);

    // Render a screen aligned quad. Assumes textures/shaders are already set.
  void RenderScreenAlignedQuad(float fOffU, float fOffV);
  // Load scene shaders
  int ReloadShaders();

  // Render scene geometry
  void RenderScene();
  // Create refraction map
  void CreateRefractionMap();
  // Render glass
  void RenderGlass();
  
private:
  int m_iFrameCounter;

  CGcontext m_pCGContext;        

  CCamera m_pCamera; 
  CCameraControl m_pCameraControl;    

  CTimer m_pDemoTimer;

  CTexture  *m_pEnvMap;

  CBaseMesh *m_pSkybox, *m_pRefractiveMesh;

  CShader m_pSHSkyBox, m_pSHRefractive, m_pSHRefractiveMask;  

  IDirect3DSurface9 *m_plD3DBackbufferSurf,
                    *m_plD3DDepthStencilSurf;
  
  CRenderTarget     *m_pRTRefraction;

  CMatrix44f  m_pWorldViewProj;
};

int CMyApp::InitializeApp() 
{     
  //_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
  
  // Initialize CG
  cgD3D9SetDevice(m_plD3DDevice);
  
  m_pCGContext=cgCreateContext();
  if(!cgIsContext(m_pCGContext))
  {
    OutputMsg("Error", "Invalid cgContext");
    return APP_ERR_READFAIL;
  }

  // Create camera and camera controler
  int iWidth=0, iHeight=0, iBps=0;
  m_pApp->GetApp()->GetScreenInfo(iWidth, iHeight, iBps);

  m_pCamera.Create(60, iWidth, iHeight, 0.2f, 500.0f);
  m_pCamera.SetPosition(CVector3f(76, 0, 0));
  m_pCamera.SetHeading(90);

  m_pCameraControl.SetCamera(m_pCamera);
  m_pCameraControl.SetInput(m_pApp->GetApp()->m_pInput);
  
  // Load scene shaders
  if(FAILED(ReloadShaders()))
  {
    OutputMsg("Error", "Loading shaders");
    return APP_ERR_INITFAIL;
  }
  
  // Load scene geometry  
  m_pSkybox=new CBaseMesh;
  if(FAILED(m_pSkybox->Create("skybox.tds")))
  {
    return APP_ERR_INITFAIL;
  }

  m_pRefractiveMesh=new CBaseMesh;  
  if(FAILED(m_pRefractiveMesh->Create("refractive.tds")))
  {
    return APP_ERR_INITFAIL;
  }

  m_pEnvMap=new CTexture;
  if(FAILED(m_pEnvMap->Create("envmap.dds", 0)))
  {
    return APP_ERR_INITFAIL;
  }
  
  // Create render targets
  m_pRTRefraction=new CRenderTarget;
  if(FAILED(m_pRTRefraction->Create(iWidth>>1, iHeight>>1, D3DFMT_A8R8G8B8)))
  {
    return APP_ERR_INITFAIL;
  }
  
  // Get backbuffer
  if(FAILED(m_plD3DDevice->GetRenderTarget(0, &m_plD3DBackbufferSurf)))
  {
    OutputMsg("Error", "Getting current backbuffer surface");
    return APP_ERR_INITFAIL;
  }

  // Get depthstencil 
  if(FAILED(m_plD3DDevice->GetDepthStencilSurface(&m_plD3DDepthStencilSurf)))
  {
    OutputMsg("Error", "Getting current depth stencil surface");
    return APP_ERR_INITFAIL;
  }

  m_pDemoTimer.Create();

  return APP_OK;
}

int CMyApp::ShutDownApp()
{  
  cgDestroyContext(m_pCGContext);
  cgD3D9SetDevice(0);

  SAFE_DELETE(m_pSkybox)
  SAFE_DELETE(m_pRefractiveMesh)  
  SAFE_DELETE(m_pEnvMap)
  SAFE_RELEASE(m_plD3DBackbufferSurf)
  SAFE_RELEASE(m_plD3DDepthStencilSurf)
  SAFE_DELETE(m_pRTRefraction)

  return APP_OK;
}

int CMyApp::ReloadShaders()
{
  // Make sure to release all data before reloading  
  m_pSHSkyBox.Release();
  m_pSHRefractive.Release();
  
  // Load shaders
  
  // Create shared shader element declaration (Position, texture coordinates and normal)
  D3DVERTEXELEMENT9 plD3dSharedDecl[] = 
  {    
    { 0,  0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT,  D3DDECLUSAGE_POSITION, 0},
    { 0,  3*sizeof(float), D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},    
    { 0,  3*sizeof(float)+2*sizeof(float), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1},    
    { 0,  3*sizeof(float)+2*sizeof(float)+3*sizeof(float), D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2},
    D3DDECL_END()
  };

  m_pSHSkyBox.CreateVertexShader(m_plD3DDevice, m_pCGContext, "skyboxVP.cg", plD3dSharedDecl, CG_PROFILE_VS_1_1);
  m_pSHSkyBox.CreatePixelShader(m_plD3DDevice, m_pCGContext, "skyboxFP.cg", CG_PROFILE_PS_1_1);

  m_pSHRefractive.CreateVertexShader(m_plD3DDevice, m_pCGContext, "refractiveVP.cg", plD3dSharedDecl, CG_PROFILE_VS_1_1);
  m_pSHRefractive.CreatePixelShader(m_plD3DDevice, m_pCGContext, "refractiveFP.cg", CG_PROFILE_PS_2_0);

  m_pSHRefractiveMask.CreateVertexShader(m_plD3DDevice, m_pCGContext, "refractiveMaskVP.cg", plD3dSharedDecl, CG_PROFILE_VS_1_1);
  m_pSHRefractiveMask.CreatePixelShader(m_plD3DDevice, m_pCGContext, "refractiveMaskFP.cg", CG_PROFILE_PS_2_0);  

  return APP_OK;
}

void CMyApp::RenderScreenAlignedQuad(float fOffU, float fOffV)
{
  float pVertexBuffer[]=
  {
    // pos | uv  
    -fOffU, -fOffV, 0, 0, 0, 
    -fOffU, 1-fOffV, 0, 0, 1,
    1-fOffU, 0-fOffV, 0, 1, 0,
    1-fOffU, 1-fOffV, 0, 1, 1,
  };

  // Setup orthographic projection
  CMatrix44f pOrthoProj;
  pOrthoProj.Identity(); 
  D3DXMatrixOrthoOffCenterRH((D3DXMATRIX *)&pOrthoProj, 0, 1, 1, 0, 0.0f, 1.0f);
  pOrthoProj.Transpose();

  // Set ViewProj Matrix vsh constant (assuming WorldViewProjection is at register 0)
  m_plD3DDevice->SetVertexShaderConstantF(0, &pOrthoProj.m_f11, 4);

  m_plD3DDevice->SetRenderState(D3DRS_ZENABLE, 0);
  m_plD3DDevice->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, pVertexBuffer, 5*sizeof(float));   
  m_plD3DDevice->SetRenderState(D3DRS_ZENABLE, 1);
}

// Update frame logic
int CMyApp::Update(float fTimeSpan)
{
  // Get user input
  if(m_pApp->m_pInput.GetKeyPressed(VK_ESCAPE))
  {
    return APP_ERR_UNKNOWN;
  }

  // Reload shaders  
  if(m_pApp->m_pInput.GetKeyPressed(VK_F5))
  {
    if(FAILED(ReloadShaders()))
    {
      OutputMsg("Error", "Loading shaders");     
    }
  }

  m_pCameraControl.Update(0.1f);    
  
  return APP_OK;
}

void CMyApp::RenderScene()
{
  // Render skybox/background/scene geometry   
  m_pSHSkyBox.SetShader();
  m_pSHSkyBox.SetVertexParam("ModelViewProj", &m_pWorldViewProj.m_f11, 4);
  float pCamPos[]={ m_pCamera.GetPosition().m_fX, m_pCamera.GetPosition().m_fY, m_pCamera.GetPosition().m_fZ, 0.0f};    
  m_pSHSkyBox.SetVertexParam("vCameraPosition", pCamPos, 1);

  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
  m_plD3DDevice->SetRenderState(D3DRS_ZENABLE, 0);

  // Get vertex buffer and material lists
  CVertexBuffer *pVB=m_pSkybox->GetVB();
  const CMaterial *pMaterialList=m_pSkybox->GetMaterialList();

  pVB->Enable();
  CSubMesh *pSubMeshList=m_pSkybox->GetSubMeshes();    
  for(int s=0; s<m_pSkybox->GetSubMeshCount(); s++)
  {
    // Set index buffer      
    CIndexBuffer *pIB=pSubMeshList[s].GetIndexBuffer();
    pIB->Enable();

    // Set decal texture
    const CTexture *pDecal=pMaterialList[pSubMeshList[s].GetMaterialID()].GetDecalTex();
    m_plD3DDevice->SetTexture(0,(IDirect3DTexture9*)pDecal->GetTexture());

    int iVertexCount=pSubMeshList[s].GetFaceCount()*3;
    int iIndicesCount=pSubMeshList[s].GetIndexBuffer()->GetCount();      
    m_plD3DDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, pVB->GetCount(), 0, pSubMeshList[s].GetFaceCount());               
  }

  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
  m_plD3DDevice->SetRenderState(D3DRS_ZENABLE, 1);
}

void CMyApp::CreateRefractionMap()
{  
  m_plD3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE, D3DCOLORWRITEENABLE_ALPHA);
  m_pSHRefractiveMask.SetShader();
  m_pSHRefractiveMask.SetVertexParam("ModelViewProj", &m_pWorldViewProj.m_f11, 4);

  float pTranslation[]={ 0,0,0,0 };      
  m_pSHRefractiveMask.SetVertexParam("vTranslation", pTranslation, 1);

  // Get vertex buffer and material lists
  CVertexBuffer *pVB=m_pRefractiveMesh->GetVB();
  const CMaterial *pMaterialList=m_pRefractiveMesh->GetMaterialList();

  pVB->Enable();
  CSubMesh *pSubMeshList=m_pRefractiveMesh->GetSubMeshes();    
  for(int s=0; s<m_pRefractiveMesh->GetSubMeshCount(); s++)
  {
    // Set index buffer      
    CIndexBuffer *pIB=pSubMeshList[s].GetIndexBuffer();
    pIB->Enable();

    // Set decal texture
    const CTexture *pDecal=pMaterialList[pSubMeshList[s].GetMaterialID()].GetDecalTex();
    m_plD3DDevice->SetTexture(0,(IDirect3DTexture9*)pDecal->GetTexture());

    int iVertexCount=pSubMeshList[s].GetFaceCount()*3;
    int iIndicesCount=pSubMeshList[s].GetIndexBuffer()->GetCount();      
    m_plD3DDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, pVB->GetCount(), 0, pSubMeshList[s].GetFaceCount());               
  }

  m_plD3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE, 0x0000000F);
    
  // Get backbuffer into background texture  
  m_plD3DDevice->StretchRect(m_plD3DBackbufferSurf, 0, m_pRTRefraction->GetSurface(), 0, D3DTEXF_NONE);
}

void CMyApp::RenderGlass()
{  
  // Render refractive geometry   
  m_pSHRefractive.SetShader();
  m_pSHRefractive.SetVertexParam("ModelViewProj", &m_pWorldViewProj.m_f11, 4);

  float pTranslation[]={ 0,0,0,0 };      
  m_pSHRefractive.SetVertexParam("vTranslation", pTranslation, 1);

 // float pParams[]= { 0.5f/(float)iWidth, 0.5f/(float)iHeight, 1, 1 };
  float pParams[]= { 0, 0, 1, 1 };
  m_pSHRefractive.SetFragmentParam("vTexelSize", pParams, 1);

  //float pDiffuse[]= { 1, 0, 0, 1 }; 
  //m_pSHRefractive.SetFragmentParam("Color", pDiffuse, 1);  

  CVector3f pCamPos=m_pCamera.GetPosition();
  float pCameraPos[]= { pCamPos.m_fX, pCamPos.m_fY, pCamPos.m_fZ, 1 };
  m_pSHRefractive.SetVertexParam("vCameraPos", pCameraPos, 1 );

  // Set Background texture
  m_plD3DDevice->SetTexture(2, m_pRTRefraction->GetTexture());
  m_plD3DDevice->SetSamplerState(2, D3DSAMP_MINFILTER, D3DTEXF_POINT);
  m_plD3DDevice->SetSamplerState(2, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
  m_plD3DDevice->SetSamplerState( 2, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
  m_plD3DDevice->SetSamplerState( 2, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
  m_plD3DDevice->SetTexture(3, (IDirect3DTexture9*)m_pEnvMap->GetTexture());

  // Get vertex buffer and material lists
  CVertexBuffer *pVB=m_pRefractiveMesh->GetVB();
  const CMaterial *pMaterialList=m_pRefractiveMesh->GetMaterialList();

  pVB->Enable();
  CSubMesh *pSubMeshList=m_pRefractiveMesh->GetSubMeshes();    
  for(int s=0; s<m_pRefractiveMesh->GetSubMeshCount(); s++)
  {
    // Set index buffer      
    CIndexBuffer *pIB=pSubMeshList[s].GetIndexBuffer();  
    pIB->Enable();

    // Set decal texture
    const CTexture *pDecal=pMaterialList[pSubMeshList[s].GetMaterialID()].GetDecalTex();
    m_plD3DDevice->SetTexture(0,(IDirect3DTexture9*)pDecal->GetTexture());

    const CTexture *pPerturbationMap=pMaterialList[pSubMeshList[s].GetMaterialID()].GetEnvMapTex(); // Using envmap slot to get perturbation map..
    m_plD3DDevice->SetTexture(1,(IDirect3DTexture9*)pPerturbationMap->GetTexture());

    int iVertexCount=pSubMeshList[s].GetFaceCount()*3;
    int iIndicesCount=pSubMeshList[s].GetIndexBuffer()->GetCount();      
    m_plD3DDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, pVB->GetCount(), 0, pSubMeshList[s].GetFaceCount());               
  }

  m_plD3DDevice->SetSamplerState(2, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState(2, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState(2, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
  m_plD3DDevice->SetSamplerState(2, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
}

int CMyApp::Render(float fTimeSpan) 
{    
  int iWidth=0, iHeight=0, iBps=0;
  m_pApp->GetApp()->GetScreenInfo(iWidth, iHeight, iBps);

  m_plD3DDevice->Clear(0, 0, D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(255, 0, 0, 128), 1.0f, 0); 
  if(SUCCEEDED(m_plD3DDevice->BeginScene()))
  {        
    // WorldViewProjection matrix is shared among all vertex shaders
    CMatrix44f pProj=m_pCamera.GetProjectionMatrix(), pView=m_pCamera.GetViewMatrix();
    m_pWorldViewProj=pView*pProj;
    m_pWorldViewProj.Transpose();
    
    // Render the scene
    RenderScene();
    
    // Create the refraction map
    CreateRefractionMap();
    
    // Create refractive/reflective glass
    RenderGlass();

    m_plD3DDevice->EndScene();
    m_plD3DDevice->Present(NULL, NULL, NULL, NULL);
  }    
    
  m_iFrameCounter++;
  // Update each second
  if(m_pTimer->GetCurrTime()>=1.0f) 
  {    
    m_pApp->SetCaption("%s (FPS=%d)" , m_pApp->GetAppName(), m_iFrameCounter);
    m_pTimer->Reset();    
    m_iFrameCounter=0;
  }

  return APP_OK;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) 
{  
  CMyApp pMyApp;
  if(APP_FAILED(pMyApp.Create(hInstance, "Generic Refraction Simulation - Glass rendering demo", 800, 600, 32, 0, 0, 0, 0)))
  {
    return APP_ERR_INITFAIL;
  }

  pMyApp.Run();

  pMyApp.Release();
  return APP_OK;
}
