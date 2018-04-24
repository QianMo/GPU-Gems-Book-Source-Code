#include "../Framework/Common.h"
#include "../Framework/Resources.h"
#include "Application_OGL.h"

Application *CreateApplication(void)
{
  return new Application_OGL();
}

Application_OGL::Application_OGL()
{
  m_hRC = NULL;
  m_hDC = NULL;
  m_bTextureArraysSupported = false;
  m_bGeometryShadersSupported = false;
  m_bInstancingSupported = false;
  g_pApplication = this;
}

Application_OGL::~Application_OGL()
{
  g_pApplication = NULL;
}

bool Application_OGL::Create(const CreationParams &cp)
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

  // change display mode
  if(cp.bFullScreen)
  {
		DEVMODE dmScreenSettings;
		ZeroMemory(&dmScreenSettings, sizeof(DEVMODE));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth	= cp.iWidth;
		dmScreenSettings.dmPelsHeight	= cp.iHeight;
		dmScreenSettings.dmBitsPerPel	= 32;
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		if(ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
		{
      MessageBox(NULL, TEXT("Changing to fullscreen mode failed"), TEXT("Error!"), MB_OK);
      return false;
		}
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


  m_hDC = GetDC(m_hWindow);

  PIXELFORMATDESCRIPTOR pfd;
  ZeroMemory(&pfd, sizeof(pfd));
  pfd.nSize = sizeof(pfd);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 24;
  pfd.cDepthBits = 16;
  pfd.iLayerType = PFD_MAIN_PLANE;
  int iFormat = ChoosePixelFormat(m_hDC, &pfd);
  if(!SetPixelFormat(m_hDC, iFormat, &pfd))
  {
    MessageBox(NULL, TEXT("Setting pixel format failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  m_hRC = wglCreateContext(m_hDC);
  if(m_hRC == NULL)
  {
    MessageBox(NULL, TEXT("Creating OGL context failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  wglMakeCurrent(m_hDC, m_hRC);

  if(cp.bVSync == false)
  {
    // vsync toggle is supported
    if(strstr((char *)glGetString(GL_EXTENSIONS), "WGL_EXT_swap_control") != NULL)
    {
      typedef BOOL (APIENTRY *PFNWGLSWAPINTERVALFARPROC)( int );
      PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;
      wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress( "wglSwapIntervalEXT" );
      if(wglSwapIntervalEXT != NULL) wglSwapIntervalEXT(0);
    }
  }

  // load extensions
  extern bool LoadExtensions(void);
  if(!LoadExtensions()) return false;

  // create font
  SelectObject(m_hDC, GetStockObject (SYSTEM_FONT)); 
  wglUseFontBitmaps(m_hDC, 0, 255, 1000); 

  m_Params = cp;
  return true;
}


void Application_OGL::Destroy(void)
{
  // destroy font
  glDeleteLists(1000, 255);

  wglMakeCurrent(NULL, NULL);
  wglDeleteContext(m_hRC);
  ReleaseDC(m_hWindow, m_hDC);

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


void Application_OGL::Run(FrameFunction framefunc)
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

      // call frame function
      framefunc();

      SwapBuffers(m_hDC);
      m_iFramesDrawn++;
    }
  }
}

float Application_OGL::GetAspectRatio(void)
{
  RECT rcClient;
  GetClientRect(m_hWindow, &rcClient);
  unsigned int iClientWidth = rcClient.right - rcClient.left;
  unsigned int iClientHeight = rcClient.bottom - rcClient.top;
  return iClientWidth / (float)iClientHeight;
}
