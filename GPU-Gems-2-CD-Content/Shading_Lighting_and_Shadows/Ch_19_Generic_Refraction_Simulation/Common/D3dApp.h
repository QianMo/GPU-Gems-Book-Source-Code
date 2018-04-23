///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : D3DApp.h
//  Desc : Direct3d application handling class
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "Common.h"

class CWin32App;
class CTimer;

// D3D utils
#define D3DFVF_TEXCOORDSIZE(index, size)  (D3DFVF_TEXCOORDSIZE##size (index))
#define D3D_CREATECOLOR(r, g, b, a) (((uchar)(a) << 24) | ((uchar)(r) << 16) |  ((uchar)(g) << 8) | ((uchar)(b)))

class CD3DApp
{
public:
  CD3DApp(): m_plD3DDevice(0), m_plD3D(0), m_iScrx(0), m_iScry(0), m_iScrBps(0), m_bActive(0), m_fTimeSpan(1)
  {
    m_pTimer=0;
    m_pApp=0;
    m_pD3DApp=0;
  }

  virtual ~CD3DApp()
  {    
    Release();
  }  

  // Create application
  int Create(HINSTANCE hInstance, const char *pAppName, int iScrx, int iScry, int iSbps, int iAASampleCount, int iAAQuality, bool bFullScreen, bool bVsync);
  // Release application resources
  void Release();
  // Application main loop
  int Run();
  // OnActivate procedure
  void OnActivate(WPARAM wParam, LPARAM lParam);
  // OnPaint procedure
  void OnPaint();
  
  // Update/render frame
  virtual int Update(float fTimeSpan) 
  {
    return APP_OK;
  }

  virtual int Render(float fTimeSpan)
  {
    return APP_OK;
  }

  // Create/Release application
  virtual int InitializeApp()
  {
    return APP_OK;
  }

  virtual int ShutDownApp()
  {
    return APP_OK;
  }

  // Get d3d device
  const LPDIRECT3DDEVICE9 GetD3DDevice() const
  {
    return m_plD3DDevice;
  }

  // Get d3d application
  static const CD3DApp *GetD3DApp()
  {
    return m_pD3DApp;
  }

protected:
  static CD3DApp *m_pD3DApp;
  
  // Application data
  bool m_bActive;
  int m_iScrx, m_iScry, m_iScrBps;  
  float m_fTimeSpan;
  CTimer *m_pTimer;
  CWin32App *m_pApp;
    
  // D3d9 data
  LPDIRECT3D9 m_plD3D;
  LPDIRECT3DDEVICE9 m_plD3DDevice;
  D3DCAPS9 m_plD3DCaps;
  D3DPRESENT_PARAMETERS m_plD3Dpp;  

  // NVIDIA CG data
  CGcontext m_pCGContext;        
};
