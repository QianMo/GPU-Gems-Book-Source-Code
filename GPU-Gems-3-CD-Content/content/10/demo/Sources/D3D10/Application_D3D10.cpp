#include "../Framework/Common.h"
#include "../Framework/Resources.h"
#include "Application_D3D10.h"

Application *CreateApplication(void)
{
  return new Application_D3D10();
}

Application_D3D10::Application_D3D10()
{
  m_pSwapChain = NULL;
  m_pRenderTargetView = NULL;
  m_pDevice = NULL;
  m_pDepthStencil = NULL;
  m_pDepthStencilView = NULL;
  g_pApplication = this;
}

Application_D3D10::~Application_D3D10()
{
  g_pApplication = NULL;
}

bool Application_D3D10::Create(const CreationParams &cp)
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

  if (RegisterClassEx(&wc) == NULL)
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

  // Calculate rendering area
  //
  RECT rcClient;
  GetClientRect(m_hWindow, &rcClient);
  unsigned int iClientWidth = rcClient.right - rcClient.left;
  unsigned int iClientHeight = rcClient.bottom - rcClient.top;

  // Setup a swap chain
  //
  DXGI_SWAP_CHAIN_DESC sd;
  ZeroMemory(&sd, sizeof(sd));
  sd.BufferCount = 3;
  sd.BufferDesc.Width = iClientWidth;
  sd.BufferDesc.Height = iClientHeight;
  sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  sd.BufferDesc.RefreshRate.Numerator = 60;
  sd.BufferDesc.RefreshRate.Denominator = 1;
  sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  sd.OutputWindow = m_hWindow;
  sd.SampleDesc.Count = 1;
  sd.SampleDesc.Quality = 0;
  sd.SwapEffect = DXGI_SWAP_EFFECT_SEQUENTIAL;
  sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
  sd.Windowed = cp.bFullScreen ? FALSE : TRUE;

  UINT iCreateDeviceFlags = 0;
#ifdef _DEBUG
  iCreateDeviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

  // Create device and swap chain
  //
  HRESULT hr;

  if (!cp.bReferenceRasterizer)
  {
    // try hardware first
    hr = D3D10CreateDeviceAndSwapChain(NULL, D3D10_DRIVER_TYPE_HARDWARE, NULL, iCreateDeviceFlags,
                                       D3D10_SDK_VERSION, &sd, &m_pSwapChain, &m_pDevice);

  }

  bool bReferenceRasterizer = false;
  if (cp.bReferenceRasterizer || FAILED(hr))
  {
    // try reference rasterizer
    hr = D3D10CreateDeviceAndSwapChain(NULL, D3D10_DRIVER_TYPE_REFERENCE, NULL, iCreateDeviceFlags,
                                       D3D10_SDK_VERSION, &sd, &m_pSwapChain, &m_pDevice);
    bReferenceRasterizer = true;
  }

  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating device failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Setup render target view
  //
  ID3D10Texture2D *pBuffer = NULL;
  hr = m_pSwapChain->GetBuffer(0, __uuidof( ID3D10Texture2D ), (LPVOID*) & pBuffer);
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("GetBuffer failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  hr = m_pDevice->CreateRenderTargetView(pBuffer, NULL, &m_pRenderTargetView);
  pBuffer->Release();
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("CreateRenderTargetView failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create depth stencil texture
  //
  D3D10_TEXTURE2D_DESC descDepth;
  descDepth.Width = iClientWidth;
  descDepth.Height = iClientHeight;
  descDepth.MipLevels = 1;
  descDepth.ArraySize = 1;
  descDepth.Format = DXGI_FORMAT_D32_FLOAT;
  descDepth.SampleDesc.Count = 1;
  descDepth.SampleDesc.Quality = 0;
  descDepth.Usage = D3D10_USAGE_DEFAULT;
  descDepth.BindFlags = D3D10_BIND_DEPTH_STENCIL;
  descDepth.CPUAccessFlags = 0;
  descDepth.MiscFlags = 0;
  hr = m_pDevice->CreateTexture2D(&descDepth, NULL, &m_pDepthStencil);
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating depth stencil texture failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create the depth stencil view
  //
  D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
  descDSV.Format = descDepth.Format;
  descDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
  descDSV.Texture2D.MipSlice = 0;
  hr = m_pDevice->CreateDepthStencilView(m_pDepthStencil, &descDSV, &m_pDepthStencilView);
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("CreateDepthStencilView failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  SetDefaultRenderTarget();

  m_Params = cp;
  m_Params.bReferenceRasterizer = bReferenceRasterizer;
  return true;
}

void Application_D3D10::Destroy(void)
{
  if (GetParams().bFullScreen) m_pSwapChain->SetFullscreenState(FALSE, NULL);
  if (m_pDevice != NULL) m_pDevice->ClearState();
  if (m_pRenderTargetView != NULL) m_pRenderTargetView->Release();
  if (m_pSwapChain != NULL) m_pSwapChain->Release();
  if (m_pDevice != NULL) m_pDevice->Release();
}

void Application_D3D10::Run(FrameFunction framefunc)
{
  MSG msg = {0};
  while (WM_QUIT != msg.message)
  {
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
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

      framefunc();

      m_iFramesDrawn++;
    }
  }
}

void Application_D3D10::SetDefaultRenderTarget(void)
{
  RECT rcClient;
  GetClientRect(m_hWindow, &rcClient);
  unsigned int iClientWidth = rcClient.right - rcClient.left;
  unsigned int iClientHeight = rcClient.bottom - rcClient.top;

  // Set render targets
  //
  m_pDevice->OMSetRenderTargets(1, &m_pRenderTargetView, m_pDepthStencilView);

  // Setup the viewport
  //
  D3D10_VIEWPORT vp;
  vp.Width = iClientWidth;
  vp.Height = iClientHeight;
  vp.MinDepth = 0.0f;
  vp.MaxDepth = 1.0f;
  vp.TopLeftX = 0;
  vp.TopLeftY = 0;
  m_pDevice->RSSetViewports( 1, &vp );
}

float Application_D3D10::GetAspectRatio(void)
{
  RECT rcClient;
  GetClientRect(m_hWindow, &rcClient);
  unsigned int iClientWidth = rcClient.right - rcClient.left;
  unsigned int iClientHeight = rcClient.bottom - rcClient.top;
  return iClientWidth / (float)iClientHeight;
}
