#pragma once

// Base class for applications
//
// Application_D3D9, Application_D3D10 and Application_OGL will derive from this
class Application
{
public:
  struct CreationParams
  {
    int iWidth;
    int iHeight;
    bool bVSync;
    bool bFullScreen;
    bool bReferenceRasterizer;
    TCHAR *strTitle;
  };

  Application();
  virtual ~Application() {}

  virtual bool Create(const CreationParams &cp) = 0;
  virtual void Destroy(void) = 0;

  // frames per second
  inline int GetFPS(void) { return m_iFPS; }

  // aspect ratio of back buffer
  virtual float GetAspectRatio(void) = 0;

  // handle to window
  inline HWND GetHWND(void) { return m_hWindow; }

  // set menu callback
  typedef void (*MenuFunction) (int iID);
  inline void SetMenuFunction(MenuFunction menufunc) { m_pMenuFunc = menufunc; }

  // message handling function
  static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

  // returns creation parameters
  inline const CreationParams &GetParams(void) { return m_Params; }

protected:
  HWND m_hWindow;
  MenuFunction m_pMenuFunc;

  CreationParams m_Params;

  double m_fLastFPSUpdate;
  int m_iFramesDrawn;
  int m_iFPS;
};

extern Application *g_pApplication;
inline Application *GetAppBase(void) { return g_pApplication; }

// returns true if virtual key is down
inline bool GetKeyDown(int iVirtualKey)
{
  if(GetFocus()!=GetAppBase()->GetHWND()) return false;
  return (GetKeyState(iVirtualKey) & 0xfe) ? true : false;
}

// returns true if mouse button VK_LBUTTON / VK_RBUTTON / VK_MBUTTON is down
inline bool GetMouseDown(int iVirtualKey)
{
  if(GetFocus()!=GetAppBase()->GetHWND()) return false;
  return (GetAsyncKeyState(iVirtualKey) & 0x8000) ? true : false;
}
