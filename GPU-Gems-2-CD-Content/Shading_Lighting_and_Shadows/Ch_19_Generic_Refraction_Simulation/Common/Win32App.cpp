///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Win32.cpp
//  Desc : Win32 application handling class
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Win32App.h"
#include <stdio.h>

CWin32App* CWin32App::m_pWin32App;

// update input 
void CWin32Input:: UpdateMouseInput(bool bRelativeCoords)
{
  int hsizex=GetSystemMetrics(SM_CXFULLSCREEN)>>1,
      hsizey=GetSystemMetrics(SM_CYFULLSCREEN)>>1;

  POINT hPos;
  GetCursorPos(&hPos);
  
  // get relative coordinates
  if(bRelativeCoords)
  {
    SetCursorPos(hsizex,hsizey);
    m_pAbsolute.Set(0,0,0);
    m_pRelative.Set((float)hsizex-hPos.x,(float)hsizey-hPos.y,0);
  }
  else
  {
    // get absolute coordinates
    m_pRelative.Set(0,0,0);
    m_pAbsolute.Set((float) hPos.x, (float) hPos.y, 0);
  }
}

// msg handler which passes messages to the application class
LRESULT CALLBACK WinProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
  if(CWin32App::GetApp())
  {
    return CWin32App::GetApp()->MsgProc(hWnd, uMsg, wParam, lParam);
  }
  return 0;
}

// class constructor
CWin32App:: CWin32App(): m_iWidth(0),m_iHeight(0),m_iBps(0),m_bFullScreen(0),
  m_bActive(0), m_hWndHandle(0),m_hInstance(0), OnIdle(0), OnActivate(0), OnPaint(0) 
{
  ZeroMemory(&m_pMsg, sizeof(m_pMsg));
  ZeroMemory(&m_pClassName,sizeof(m_pClassName));
  m_pWin32App=this;
}

// create a window 
int CWin32App:: Create(HINSTANCE hInstance, const char *pName, int iWidth, int iHeight, int iBps, bool bFullScr)
{
  DWORD     dwStyle;    // style
  DWORD     dwExStyle;  // extended Style

  // save screen values
  m_bFullScreen=bFullScr;
  strcpy(m_pClassName, pName);
  
  // save the current display state 
  EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &m_pPreviousMode);

  if(APP_FAILED(SetResolution(iWidth, iHeight, iBps)))
  {
    OutputMsg("Error","Error creating window");
    return APP_ERR_INITFAIL;
  }

  // get module handle
  m_hInstance = hInstance;

  // setup window
  WNDCLASS pWc= 
  {
    CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
    (WNDPROC) WinProc ,
    0, 0,
    m_hInstance,
    LoadIcon(m_hInstance, MAKEINTRESOURCE(IDI_WINLOGO)),
    LoadCursor(NULL, IDC_CROSS),
    (HBRUSH)GetStockObject(BLACK_BRUSH),
    NULL,
    pName
  };

  // register window class
  if(!RegisterClass(&pWc))
  {
    OutputMsg("Error...","Error registering window class");
    return APP_ERR_INITFAIL;
  }

  // set window extended style
  if(!m_bFullScreen)
  {
    dwExStyle=WS_EX_APPWINDOW; // window extended style
    dwStyle= WS_MINIMIZEBOX|WS_SYSMENU| WS_OVERLAPPED;//WS_POPUP|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX|WS_VISIBLE;
  }
  else
  {
    dwExStyle=WS_EX_APPWINDOW; //|WS_EX_TOPMOST;      // window extended style
    dwStyle=WS_VISIBLE| WS_POPUP;   // window style
  }

  // get desktop work area
  RECT DesktopRect;	
  DesktopRect.right=GetSystemMetrics(SM_CXFULLSCREEN);
  DesktopRect.bottom=GetSystemMetrics(SM_CYFULLSCREEN);
  int m_dwPosX=0, m_dwPosY=0;

  // Set the window's initial style
  RECT rc;
  SetRect( &rc, 0, 0, m_iWidth, m_iHeight);
  // Set the window's initial width
  AdjustWindowRectEx( &rc, dwStyle, FALSE, dwExStyle);

  // centralized position
  if(!bFullScr)
  {
    m_dwPosX=(DesktopRect.right  - (rc.right-rc.left))>>1;
    m_dwPosY=(DesktopRect.bottom - (rc.bottom-rc.top))>>1;
  }

  // create the window
  m_hWndHandle=CreateWindowEx(dwExStyle,                                  // extended style
                              pName,                                      // class name
                              pName,                                      // title
                              dwStyle,                                    // style
                              m_dwPosX, m_dwPosY,                         // position
                              (rc.right-rc.left), (rc.bottom-rc.top),     // size
                              NULL,                                       // no parent window
                              NULL,                                       // no menu
                              m_hInstance,                                //instance
                              NULL);

  if(!m_hWndHandle) 
  {
    Release();
    OutputMsg("Error..","Error creating window");
    return APP_ERR_INITFAIL;
  }

  ShowWindow(m_hWndHandle, SW_SHOW);
  SetForegroundWindow(m_hWndHandle);
  UpdateWindow(m_hWndHandle);
  // adjust window rectangle
  GetClientRect(m_hWndHandle,&m_rcWndRect);
  // set keyboard focus
  SetFocus(m_hWndHandle);
  // center cursor
  SetCursorPos(GetSystemMetrics(SM_CXFULLSCREEN)>>1,GetSystemMetrics(SM_CYFULLSCREEN)>>1);

  // save current display state 
  EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &m_pCurrentMode);

  return APP_OK;
}

// release a window 
int CWin32App:: Release()
{
  // destroy window
  if(m_hWndHandle && !DestroyWindow(m_hWndHandle))
  {
    OutputMsg("Error...","Error releasing window");
  }

  // unregister window class
  if(m_hInstance && !UnregisterClass(m_pClassName, m_hInstance))
  {
    OutputMsg("Error...", "Could not unregister window class");
  }

  // restore previous screen settings
  if(m_bFullScreen) ChangeDisplaySettings(&m_pPreviousMode,CDS_RESET);

  m_hWndHandle=0;
  m_hInstance=0;

 // m_pClassName(0)
  m_iWidth=0;
  m_iHeight=0;
  m_iBps=0;
  m_bFullScreen=0;

  m_bActive=0;
  OnIdle=0;
  OnActivate=0;
  m_pWin32App = 0;
  m_pInput.Release();
  
  ZeroMemory(&m_pClassName,sizeof(m_pClassName));
  ZeroMemory(&m_pMsg, sizeof(m_pMsg));
  
  return APP_OK;
}

// run application
int CWin32App:: Run()
{  
  // initialize msg struct
  m_pMsg.message=NULL;
  PeekMessage(&m_pMsg, NULL, 0, 0, PM_NOREMOVE );

  while(m_pMsg.message!=WM_QUIT)
  {
    if(PeekMessage( &m_pMsg, NULL, 0, 0, PM_NOREMOVE ))
    {
      if(!GetMessage( &m_pMsg, NULL, 0, 0))
      {
        continue;
      }
      TranslateMessage(&m_pMsg);
      DispatchMessage(&m_pMsg);
    }
    // run application specific calls
    if(OnIdle && m_bActive)
    {
      if(APP_FAILED(OnIdle()))     
      {
        SendMessage( m_hWndHandle, WM_CLOSE, 0, 0 );
      }
    }
  }

  return APP_OK;
}

// change window name
void CWin32App:: SetCaption(const char *pStr, ...)
{
  char pLastStr[255];
  va_list pVl;
  char bStr[255];
  // get variable-argument list
  va_start(pVl, pStr);
  // write formatted output 
  vsprintf(bStr, pStr, pVl);

  // check if strings are equal
  if(!strcmp(bStr, pLastStr)) return;

  SetWindowText(m_hWndHandle, bStr);
  va_end(pVl);

  strcpy(pLastStr, bStr);
}

// change window resolution
int CWin32App:: SetResolution(int iWidth, int iHeight, int iBps)
{
  // device settings
  DEVMODE pNewScr;
  // clear it
  ZeroMemory(&pNewScr,sizeof(pNewScr));
  // setsize
  pNewScr.dmSize=sizeof(pNewScr);

  // save screen values
  m_iWidth=iWidth; 
  m_iHeight=iHeight; 
  m_iBps=iBps;

  // define resolution
  pNewScr.dmPelsWidth= iWidth;
  pNewScr.dmPelsHeight = iHeight;
  pNewScr.dmBitsPerPel = iBps;
  pNewScr.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

  if(m_bFullScreen)
  {
    // try change resolution
    if(ChangeDisplaySettings(&pNewScr,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
    {
      // if fails, quit or continue in windowed mode
      if(MessageBox(NULL,"Requested fullscreen mode not supported.\nUse windowed mode ?","Error",MB_YESNO|MB_ICONEXCLAMATION)==IDYES)
      {
        m_bFullScreen=0;
      }
      else 
      {
        // close app
        OutputMsg("Fatal Error","Changing screen resolution...");
        return APP_ERR_NOTSUPPORTED;
      }
    }

    return APP_OK;
  }

  //if(!m_bFullScreen) 
  //  MoveWindow(m_hWndHandle, 0, 0, m_iWidth, m_iHeight, TRUE);

  return APP_OK;
}

// window aplication procedure
LRESULT CWin32App:: MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) 
{
  switch (uMsg)
  {
  case WM_SYSCOMMAND:
    // stop interruptions
    switch (wParam)
    {
      case SC_SCREENSAVE:
      case SC_MONITORPOWER:
        return 0;
    }
  break;    
  case WM_KEYDOWN:
    m_pInput.m_bKeys[wParam] = 1;
    return 0;
  case WM_KEYUP:
    m_pInput.m_bKeys[wParam] = 0;
    return 0;    
  case WM_LBUTTONUP:
    m_pInput.m_bMousebutton[0]=0;
    return 0;
  case WM_LBUTTONDOWN:
    m_pInput.m_bMousebutton[0]=1;
    return 0;
  case WM_RBUTTONUP:
    m_pInput.m_bMousebutton[1]=0;
    return 0;
  case WM_RBUTTONDOWN:
    m_pInput.m_bMousebutton[1]=1;
    return 0;
  case WM_MBUTTONUP:
    m_pInput.m_bMousebutton[2]=0;
    return 0;
  case WM_MBUTTONDOWN:
    m_pInput.m_bMousebutton[2]=1;
    return 0;  
  case WM_ACTIVATE:
    if(OnActivate)
    {
      OnActivate(wParam, lParam);
    }
  return 0;

    // close window
  case WM_CLOSE: PostQuitMessage(0); return 0;
  }

  // non handled messages go to DefWindowProc
  return DefWindowProc(hWnd,uMsg,wParam,lParam);
}
