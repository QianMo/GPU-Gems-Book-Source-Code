#include "Common.h"
#include "Application.h"
#include "Resources.h"

Application *g_pApplication = NULL;

extern void MenuFunc(int iID);

Application::Application()
{
  m_hWindow = NULL;
  m_pMenuFunc = &MenuFunc;
  ZeroMemory(&m_Params, sizeof(CreationParams));
  m_fLastFPSUpdate = 0;
  m_iFramesDrawn = 0;
  m_iFPS = 0;
}

// message handler for about box
INT_PTR CALLBACK AboutDlgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
   case WM_INITDIALOG:
      return TRUE;
   case WM_COMMAND:
      switch( wParam ) 
      {
         case IDOK:
            EndDialog( hWnd, TRUE );
            return TRUE;
      }
   break;
	}
  return FALSE;
}

LRESULT WINAPI Application::MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
  if (msg == WM_DESTROY)
  {
    PostQuitMessage(0);
    return 0;
  }
  else if(msg == WM_COMMAND && HIWORD(wParam)==0)
  {
    // File / exit
    if(LOWORD(wParam) == ID_FILE_EXIT)
    {
      PostMessage(hWnd, WM_CLOSE, 0, 0);
    }
    // Help / about
    else if(LOWORD(wParam) == ID_HELP_ABOUT)
    {
      DialogBox(GetModuleHandle(NULL), MAKEINTRESOURCE(IDD_ABOUT), hWnd, AboutDlgProc);
    }

    if(GetAppBase()!=NULL && GetAppBase()->m_pMenuFunc != NULL)
    {
      GetAppBase()->m_pMenuFunc(LOWORD(wParam));
    }
  }

  return DefWindowProc(hWnd, msg, wParam, lParam);
}
