#pragma once

#ifdef _DEBUG
  #define D3D_DEBUG_INFO
#endif

#include <d3d9.h>
#include <d3dx9.h>
#include <dxerr9.h>

#pragma comment(lib,"d3d9.lib")
#ifdef _DEBUG
  #pragma comment(lib,"d3dx9d.lib")
#else
  #pragma comment(lib,"d3dx9.lib")
#endif

#include "../Framework/Application.h"

// Application mostly handles creating a window and D3D
// There can only be one instance, use GetApp()
//
class Application_D3D9 : public Application
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
  inline LPDIRECT3DDEVICE9 GetDevice(void) { return m_pDevice; }
  inline LPDIRECT3D9 GetD3D(void) { return m_pD3D; }

  // capabilities
  D3DCAPS9 GetCaps(void);
  // parameters
  D3DPRESENT_PARAMETERS GetPresentParams(void);
  // render target aspect ratio
  float GetAspectRatio(void);

  Application_D3D9();
  ~Application_D3D9();

private:
  LPDIRECT3DDEVICE9 m_pDevice;
  LPDIRECT3D9 m_pD3D;
};

inline Application_D3D9 *GetApp(void) { return (Application_D3D9 *)g_pApplication; }

// Creates an application and returns it
extern Application *CreateApplication(void);
