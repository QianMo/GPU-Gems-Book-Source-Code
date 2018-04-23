///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Win32.h
//  Desc : Win32 application handling class
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Common.h"

typedef int WindowIdleProc();
typedef void WindowNotifyProc(WPARAM wParam, LPARAM lParam);
typedef void WindowPaintProc();

class CWin32App;
class CWin32Input;

class CWin32Input
{
  friend CWin32App;
public:

  enum MouseButtons
  {
    MOUSE_LEFT,
    MOUSE_RIGHT,
    MOUSE_MIDLE
  };

  // constructor
  CWin32Input()
  {
    // reset data
    ZeroMemory(m_bKeys, sizeof(m_bKeys));
    ZeroMemory(m_bMousebutton, sizeof(m_bMousebutton));
    m_pAbsolute.Set(0,0,0);
    m_pRelative.Set(0,0,0);
  };

  // destructor
  ~CWin32Input()
  {
    Release();
  }

  void Release()
  {
    // reset data
    ZeroMemory(m_bKeys, sizeof(m_bKeys));
    ZeroMemory(m_bMousebutton, sizeof(m_bMousebutton));
    m_pAbsolute.Set(0,0,0);
    m_pRelative.Set(0,0,0);
  }

  // show/hide cursor
  void ShowMouseCursor(bool bShow) const
  {
    ShowCursor(bShow);
  }

  // check if key was pressed
  bool GetKeyPressed(char bKey)
  {    
    return (bool) (GetAsyncKeyState(bKey)& 0x8000);
  }

  // check if mouse button was pressed
  bool GetMouseButtonPressed(MouseButtons bButton)
  {
    return m_bMousebutton[bButton]; 
  }

  // get screen coordinates
  CVector3f *GetMouseScreenCoordinates()
  {
    return &m_pAbsolute;
  }

  // get relative coordinates
  CVector3f *GetMouseRelativeCoordinates()
  {
    return &m_pRelative;
  }

  // update input
  void UpdateMouseInput(bool bRelativeCoords);

private:

  // keyboard map
  bool m_bKeys[256];
  // mouse input handling
  bool m_bMousebutton[3]; // 0=left, 1=right, 2=midle

  // mouse screen coordinates
  CVector3f m_pAbsolute;
  // mouse relative coordinates
  CVector3f m_pRelative;
};

class CWin32App
{
public:
  // input handling
  CWin32Input m_pInput;

  // class methods
  CWin32App();

  ~CWin32App()
  {
    Release();
  };

  WindowIdleProc    *OnIdle;
  WindowNotifyProc  *OnActivate;
  WindowPaintProc   *OnPaint;

  // create a window 
  int Create(HINSTANCE hInstance, const char *pName, int iWidth, int iHeight, int iBps, bool bFullScr);
  // release a window 
  int Release();

  // run application
  int Run();

  // set methods

  // change window name
  void SetCaption(const char *pStr, ...);
  
  // set activation flag
  void SetActiveFlag(const bool bActive) { m_bActive=bActive; };
  // change window resolution
  int SetResolution(int iWidth, int iHeight, int iBps);

  // get member data
  HWND  GetHandle()
  {
    return m_hWndHandle;
  };

  HINSTANCE   GetInstance()
  {
    return m_hInstance;
  };

  void GetScreenInfo(int &iWidth, int &iHeight, int &iBps)
  {
    iWidth=m_rcWndRect.right-m_rcWndRect.left;
    iHeight=m_rcWndRect.bottom-m_rcWndRect.top;
    iBps=m_iBps;
  };

  // Get application name
  const char *GetAppName() const
  {
    return m_pClassName;
  }

  // Window aplication procedure
  virtual LRESULT MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  bool IsFullScreen()
  {
    return m_bFullScreen;
  };

  
  static CWin32App *GetApp()
  {
    return m_pWin32App;
  }

private:
  static CWin32App  *m_pWin32App;

  // windows class name
  char m_pClassName[256];
  // screen options
  int  m_iWidth, m_iHeight, m_iBps;

  // window flags
  bool m_bFullScreen, m_bActive;

  // screen settings
  DEVMODE   m_pPreviousMode, m_pCurrentMode;

  // window handle
  HWND m_hWndHandle;
  // aplication instance
  HINSTANCE m_hInstance;

  // window "rectangle"
  RECT m_rcWndRect;
  // message handling
  MSG  m_pMsg;
};