///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : D3DApp.cpp
//  Desc : Direct3d application handling class
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3DApp.h"
#include "Win32App.h"
#include "Timer.h"

CD3DApp* CD3DApp:: m_pD3DApp;

int OnIdle()
{  
  const CD3DApp *pApp=CD3DApp::GetD3DApp();
  if(pApp)
  {
    return const_cast<CD3DApp*>(pApp)->Run();
  }

  return APP_ERR_UNKNOWN;
}

void OnActivate(WPARAM wParam, LPARAM lParam)
{
  const CD3DApp *pApp=CD3DApp::GetD3DApp();
  if(pApp)
  {
    const_cast<CD3DApp*>(pApp)->OnActivate(wParam, lParam);    
  }
}

int CD3DApp:: Create(HINSTANCE hInstance, const char *pAppName, int iScrx, int iScry, int iSbps, int iAASampleCount, int iAAQuality, bool bFullScreen, bool bVsync)
{ 
  // just in case client calls create more than once
  Release();
  m_pD3DApp=this;

  // macro for forced shutdown 
  // NOTE: CAREFULL USING THIS, ALWAYS USE BRACES OUTSIDE
  #define SHUTDOWN_APP(a)\
    Release();\
    return (a);\
    
  // create aplication
  m_pApp=new CWin32App;
  if(APP_FAILED(m_pApp->Create(hInstance, pAppName, iScrx, iScry, iSbps, bFullScreen)))
  {
    SHUTDOWN_APP(APP_ERR_INITFAIL);
  }

  // set aplication function pointers
  m_pApp->OnIdle=::OnIdle;
  m_pApp->OnActivate=::OnActivate;
  m_pApp->SetActiveFlag(1);

  // create d3d object
  if(!(m_plD3D = Direct3DCreate9( D3D_SDK_VERSION ) ) )
  {
    OutputMsg("CD3DApp", "Directx 9.0 not installed !");
    SHUTDOWN_APP(APP_ERR_NOTSUPPORTED);
  }

  // get current desktop display mode
  D3DDISPLAYMODE pD3Ddm;
  if(APP_FAILED(m_plD3D->GetAdapterDisplayMode( D3DADAPTER_DEFAULT, &pD3Ddm)) )
  {
    OutputMsg("CD3DApp","Error getting display mode");
    SHUTDOWN_APP(APP_ERR_INITFAIL);
  }

  ZeroMemory(&m_plD3Dpp, sizeof(m_plD3Dpp));
  m_plD3Dpp.Windowed = !bFullScreen;
 
  m_plD3Dpp.SwapEffect = (iAASampleCount)? D3DSWAPEFFECT_DISCARD: D3DSWAPEFFECT_FLIP;
  m_plD3Dpp.hDeviceWindow = m_pApp->GetHandle();

  int iSx, iSy, iSBps;
  m_pApp->GetScreenInfo(iSx, iSy, iSBps);
  m_iScrx=iSx; m_iScry=iSy; m_iScrBps=iSBps;

  m_plD3Dpp.BackBufferWidth = iSx;
  m_plD3Dpp.BackBufferHeight = iSy;

  // define back buffer format

  // only 32 bit format supported !
  m_plD3Dpp.BackBufferFormat = D3DFMT_A8R8G8B8; //pD3Ddm.Format;
  m_plD3Dpp.BackBufferCount=1;

  // define depht buffer format
  m_plD3Dpp.EnableAutoDepthStencil = TRUE;
  m_plD3Dpp.AutoDepthStencilFormat = D3DFMT_D24X8;
  m_plD3Dpp.MultiSampleType =(D3DMULTISAMPLE_TYPE)iAASampleCount;
  m_plD3Dpp.MultiSampleQuality=iAAQuality;
  m_plD3Dpp.FullScreen_RefreshRateInHz=0;
  m_plD3Dpp.PresentationInterval=(bVsync)?D3DPRESENT_INTERVAL_ONE:D3DPRESENT_INTERVAL_IMMEDIATE;
  
  m_plD3D->GetDeviceCaps(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, &m_plD3DCaps);

  // check vertex/pixel shaders support
  if(m_plD3DCaps.VertexShaderVersion<D3DVS_VERSION(1,1) || m_plD3DCaps.PixelShaderVersion<D3DPS_VERSION(1,1))
  {
      OutputMsg("CD3DApp", "Vertex/Pixel shaders 1.1 not supported, exiting...");
      SHUTDOWN_APP(APP_ERR_NOTSUPPORTED);
  }

  // create device
  if(APP_FAILED(m_plD3D->CreateDevice(D3DADAPTER_DEFAULT, 
                                      D3DDEVTYPE_HAL,
                                      m_pApp->GetHandle(),
                                      D3DCREATE_HARDWARE_VERTEXPROCESSING,
                                      &m_plD3Dpp, &m_plD3DDevice)))
  {
      OutputMsg("Error...", "Creating d3d9 device");
      SHUTDOWN_APP(APP_ERR_INITFAIL);
  }
  
  // Initialize CG
  cgD3D9SetDevice(m_plD3DDevice);

  m_pCGContext=cgCreateContext();
  if(!cgIsContext(m_pCGContext))
  {
    OutputMsg("Error", "Invalid cgContext");
    return APP_ERR_READFAIL;
  }

  // initialize application
  if(APP_FAILED(InitializeApp()))
  {
    SHUTDOWN_APP(APP_ERR_INITFAIL);
  }

  // set default render state
  m_plD3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
  m_plD3DDevice->SetRenderState(D3DRS_LIGHTING, 0);
  m_plD3DDevice->SetRenderState(D3DRS_DITHERENABLE, 0);  
  m_plD3DDevice->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);

  // set default texture samplers state
  float dwMipLodBias=-1.0;  
  
  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 0, D3DSAMP_MIPMAPLODBIAS, *((LPDWORD) (&dwMipLodBias)));

  m_plD3DDevice->SetSamplerState( 1, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 1, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 1, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 1, D3DSAMP_MIPMAPLODBIAS, *((LPDWORD) (&dwMipLodBias)));  

  m_plD3DDevice->SetSamplerState( 2, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 2, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 2, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 2, D3DSAMP_MIPMAPLODBIAS, *((LPDWORD) (&dwMipLodBias)));  

  m_plD3DDevice->SetSamplerState( 3, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 3, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 3, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
  m_plD3DDevice->SetSamplerState( 3, D3DSAMP_MIPMAPLODBIAS, *((LPDWORD) (&dwMipLodBias)));  

  // default application state is active
  m_bActive=1;

  // initialize application timer
  m_pTimer=new CTimer;
  m_pTimer->Create();
  
  // run win32 application
  m_pApp->m_pInput.ShowMouseCursor(0);
  // set keyboard focus
  SetFocus(m_pApp->GetHandle());
  m_pApp->Run();
  
  return APP_OK;
}

// release resources
void CD3DApp:: Release()
{
  m_pD3DApp=0;
  m_iScrx=m_iScry=m_iScrBps=0;
  m_bActive=0;
  m_fTimeSpan=1.0f;

  if(m_pApp)
  {
    m_pApp->m_pInput.ShowMouseCursor(1);    
  }

  cgDestroyContext(m_pCGContext);
  cgD3D9SetDevice(0);
  ShutDownApp();    
  SAFE_DELETE(m_pTimer)    
  SAFE_RELEASE(m_plD3DDevice)
  SAFE_RELEASE(m_plD3D)  
  SAFE_DELETE(m_pApp)
}

// application main loop
int  CD3DApp:: Run()
{
  if(!m_bActive)
  {
    return APP_OK;
  }

  // update frame
  if(APP_FAILED(Update(1.0f)))
  {
    return APP_ERR_UNKNOWN;
  }

  // render frame
  if(APP_FAILED(Render(1.0f)))
  {
    return APP_ERR_UNKNOWN;      
  }

  return APP_OK;
}

// on activate procedure
void CD3DApp:: OnActivate(WPARAM wParam, LPARAM lParam)
{
  if(LOWORD(wParam)==WA_INACTIVE)
  {    
    if(m_pApp->IsFullScreen())  
    {
      PostQuitMessage(0);
    }
    else 
    {    
      m_pApp->SetActiveFlag(0);
      m_bActive=0;      
    }
  }
  else
  {
    m_pApp->SetActiveFlag(1);
    m_bActive=1;    
  }
}