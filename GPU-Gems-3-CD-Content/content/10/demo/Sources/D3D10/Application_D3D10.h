#pragma once

#pragma comment(lib,"d3d10.lib")
#ifdef _DEBUG
  #define D3D_DEBUG_INFO
  #pragma comment(lib,"d3dx10d.lib")
#else
  #pragma comment(lib,"d3dx10.lib")
#endif

#include <d3d10.h>
#include <d3dx10.h>


#include "../Framework/Application.h"

// Application mostly handles creating a window and D3D
// There can only be one instance, use GetApp()
//
class Application_D3D10 : public Application
{
public:
  // create window and initialize D3D
  bool Create(const CreationParams &cp);

  // run and call framefunc every frame
  typedef void (*FrameFunction) (void);
  void Run(FrameFunction framefunc);

  // destroy window and D3D
  void Destroy(void);

  // D3D
  inline ID3D10Device *GetDevice(void) { return m_pDevice; }
  inline ID3D10RenderTargetView *GetRTV(void) { return m_pRenderTargetView; }
  inline ID3D10DepthStencilView *GetDSV(void) { return m_pDepthStencilView; }
  inline IDXGISwapChain *GetSwapChain(void) { return m_pSwapChain; }

  // reset render target settings
  void SetDefaultRenderTarget(void);

  // render target aspect ratio
  float GetAspectRatio(void);

  Application_D3D10();
  ~Application_D3D10();

private:

  IDXGISwapChain *m_pSwapChain;
  ID3D10RenderTargetView *m_pRenderTargetView;
  ID3D10Device *m_pDevice;
  ID3D10Texture2D *m_pDepthStencil;
  ID3D10DepthStencilView *m_pDepthStencilView;
};

inline Application_D3D10 *GetApp(void) { return (Application_D3D10 *)g_pApplication; }

// Creates an application and returns it
extern Application *CreateApplication(void);