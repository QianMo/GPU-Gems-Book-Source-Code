#include "../Framework/Common.h"
#include "../Framework/Resources.h"
#include "Application_D3D9.h"

Application *CreateApplication(void)
{
  return new Application_D3D9();
}

Application_D3D9::Application_D3D9()
{
  m_pDevice = NULL;
  m_pD3D = NULL;
  g_pApplication = this;
}

Application_D3D9::~Application_D3D9()
{
  g_pApplication = NULL;
}

bool Application_D3D9::Create(const CreationParams &cp)
{
  // Create window class
  //
  WNDCLASSEX wc;
  ZeroMemory(&wc, sizeof(WNDCLASSEX));
  wc.cbSize = sizeof(WNDCLASSEX);
  wc.style = CS_CLASSDC;
  wc.lpfnWndProc = MsgProc;
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpszClassName = TEXT("WindowClass");
  wc.hCursor = LoadCursor(NULL, IDC_ARROW);
  wc.lpszMenuName = MAKEINTRESOURCE(ID_FILE);

  if (RegisterClassEx(&wc) == 0)
  {
    MessageBox(NULL, TEXT("Registering window class failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create window
  //
  int iWindowType = (cp.bFullScreen) ? WS_POPUP : WS_OVERLAPPED  | WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU;
  m_hWindow = CreateWindow( TEXT("WindowClass"), cp.strTitle,
                            WS_VISIBLE | iWindowType,
                            (GetSystemMetrics(SM_CXSCREEN) - cp.iWidth) / 2,  // center window
                            (GetSystemMetrics(SM_CYSCREEN) - cp.iHeight) / 2, // center window
                            cp.iWidth, cp.iHeight,
                            NULL, NULL, NULL, NULL );

  if (m_hWindow == NULL)
  {
    MessageBox(NULL, TEXT("Creating window failed!"), TEXT("Error!"), MB_OK);
    return false;
  }


  // Create Direct3D
  //
  m_pD3D = Direct3DCreate9(D3D_SDK_VERSION);
  if (m_pD3D == NULL)
  {
    MessageBox(NULL, TEXT("Could not create Direct3D (DirectX probably not installed right)"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create Device
  //
  D3DPRESENT_PARAMETERS pp;
  ZeroMemory(&pp, sizeof(D3DPRESENT_PARAMETERS));
  pp.Windowed = cp.bFullScreen ? FALSE : TRUE;
  pp.BackBufferWidth = cp.bFullScreen ? cp.iWidth : 0;
  pp.BackBufferHeight = cp.bFullScreen ? cp.iHeight : 0;
  pp.BackBufferFormat = cp.bFullScreen ? D3DFMT_X8R8G8B8 : D3DFMT_UNKNOWN;
  pp.BackBufferCount = 3;
  pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  pp.PresentationInterval = cp.bVSync ? D3DPRESENT_INTERVAL_DEFAULT : D3DPRESENT_INTERVAL_IMMEDIATE;
  pp.EnableAutoDepthStencil = TRUE;
  pp.AutoDepthStencilFormat = D3DFMT_D24X8;

  HRESULT hr;

  DWORD dwVertexProcessing[3] = {D3DCREATE_HARDWARE_VERTEXPROCESSING,
                                 D3DCREATE_MIXED_VERTEXPROCESSING,
                                 D3DCREATE_SOFTWARE_VERTEXPROCESSING};

  // try to create device with different vertex processing modes
  for (int i = 0; i < 3; i++)
  {
    hr = m_pD3D->CreateDevice(D3DADAPTER_DEFAULT, (cp.bReferenceRasterizer) ?  D3DDEVTYPE_REF : D3DDEVTYPE_HAL,
                              m_hWindow, dwVertexProcessing[i], &pp, &m_pDevice);
    if (SUCCEEDED(hr)) break;
  }

  if (FAILED(hr) || m_pDevice == NULL)
  {
    MessageBox(NULL, TEXT("Could not create device!"), TEXT("Error!"), MB_OK);
    return false;
  }

  m_Params = cp;
  return true;
}


void Application_D3D9::Destroy(void)
{
  // Destroy the device
  //
  if (m_pDevice != NULL)
  {
    m_pDevice->Release();
    m_pDevice = NULL;
  }

  // Destroy Direct3D
  //
  if (m_pD3D != NULL)
  {
    m_pD3D->Release();
    m_pD3D = NULL;
  }

  // Destroy the window
  //
  if (IsWindow(m_hWindow) && !DestroyWindow(m_hWindow))
  {
    MessageBox(NULL, TEXT("Destroying window failed!"), TEXT("Error!"), MB_OK);
  }

  // Unregister the window class
  //
  if (!UnregisterClass(TEXT("WindowClass"), NULL))
  {
    MessageBox(NULL, TEXT("Unregistering window class failed!"), TEXT("Error!"), MB_OK);
  }
}


void Application_D3D9::Run(FrameFunction framefunc)
{
  MSG msg;
  ZeroMemory(&msg, sizeof(msg));
  while (msg.message != WM_QUIT)
  {

    // we have messages in the queue
    if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
    {
      // handle them
      TranslateMessage(&msg);
      DispatchMessage(&msg);

    }
    // otherwise we are free to render
    else
    {

      // calculate FPS
      double fTimeNow = GetAccurateTime();
      if (fTimeNow - m_fLastFPSUpdate > 1)
      {
        double fDelta = fTimeNow - m_fLastFPSUpdate;
        m_iFPS = (int)(m_iFramesDrawn / fDelta);
        m_fLastFPSUpdate = fTimeNow;
        m_iFramesDrawn = 0;
      }

      if (SUCCEEDED(m_pDevice->BeginScene()))
      {
        // call frame function
        framefunc();

        m_pDevice->EndScene();
      }

      // present
      HRESULT hr = m_pDevice->Present(NULL, NULL, NULL, NULL);
      if (hr == D3DERR_DEVICENOTRESET || hr == D3DERR_DEVICELOST)
      {
        MessageBox(NULL, TEXT("Device was lost - the demo does not handle this"), TEXT("Error!"), MB_OK);
        break;
      }

      m_iFramesDrawn++;
    }
  }
}

D3DCAPS9 Application_D3D9::GetCaps(void)
{
  D3DCAPS9 HALCaps;
  ZeroMemory(&HALCaps, sizeof(HALCaps));
  m_pD3D->GetDeviceCaps(D3DADAPTER_DEFAULT, (m_Params.bReferenceRasterizer) ?  D3DDEVTYPE_REF : D3DDEVTYPE_HAL, &HALCaps);
  return HALCaps;
}

D3DPRESENT_PARAMETERS Application_D3D9::GetPresentParams(void)
{
  D3DPRESENT_PARAMETERS pp;
  ZeroMemory(&pp, sizeof(D3DPRESENT_PARAMETERS));

  LPDIRECT3DSWAPCHAIN9 pChain;
  HRESULT hr = m_pDevice->GetSwapChain(0, &pChain);
  if (FAILED(hr))
  {
    return pp;
  }

  hr = pChain->GetPresentParameters(&pp);
  if (FAILED(hr))
  {
    return pp;
  }

  pChain->Release();
  return pp;
}

float Application_D3D9::GetAspectRatio(void)
{
  RECT rcClient;
  GetClientRect(m_hWindow, &rcClient);
  unsigned int iClientWidth = rcClient.right - rcClient.left;
  unsigned int iClientHeight = rcClient.bottom - rcClient.top;
  return iClientWidth / (float)iClientHeight;
}
