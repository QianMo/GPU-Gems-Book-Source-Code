#include <windows.h>
#include <tchar.h>

#include "Resources.h" 

inline void AddToComboBox(HWND hwnd, int iComponentID, const TCHAR *strText)
{
  int iIndex = (int)SendDlgItemMessage(hwnd, iComponentID, CB_ADDSTRING, 0, (LPARAM)strText);
}

inline void SetComboBoxItem(HWND hwnd, int iComponentID, int iItem)
{
  int iIndex = (int)SendDlgItemMessage(hwnd, iComponentID, CB_SETCURSEL, (WPARAM)iItem, 0);
}

inline int GetComboBoxSelection(HWND hwnd, int iComponentID, TCHAR *str)
{
  int iIndex = (int)SendDlgItemMessage(hwnd, iComponentID, CB_GETCURSEL, 0, 0);
  return (int)SendDlgItemMessage(hwnd, iComponentID, CB_GETLBTEXT, iIndex, (LPARAM)str);
}

inline bool StringMatch(const TCHAR *str, const TCHAR *str2)
{
  return _tcsstr(str, str2) != NULL;
}
void LaunchApp(HWND hwnd)
{
  TCHAR str[1024];
  str[0] = 0;
  TCHAR strParams[1024];
  strParams[0] = 0;

  // get executable path
  TCHAR szName[_MAX_PATH]; szName[0]=0;
  TCHAR szDrive[_MAX_DRIVE]; szDrive[0]=0;
  TCHAR szDir[_MAX_DIR]; szDir[0]=0;
  TCHAR szFilename[_MAX_DIR]; szFilename[0]=0;
  TCHAR szExt[_MAX_DIR]; szExt[0]=0;
  GetModuleFileName(0, szName, _MAX_PATH);
  _tsplitpath_s(szName, szDrive, szDir, szFilename, szExt);
  szName[0]=0;
  _tcscat_s(szName, _MAX_PATH, szDrive);
  _tcscat_s(szName, _MAX_PATH, szDir);

  // get selected res
  int iWidth = 640, iHeight = 480;
  GetComboBoxSelection(hwnd, ID_WINDOW, str);
  _stscanf_s(str, TEXT("%ix%i"), &iWidth, &iHeight);

  // get other params
  int iFullScreen = (IsDlgButtonChecked(hwnd, ID_WINDOWED) == BST_CHECKED) ? 0 : 1;
  int iVSync = (IsDlgButtonChecked(hwnd, ID_VSYNC) == BST_CHECKED) ? 1 : 0;
  int iRefRasterizer = (IsDlgButtonChecked(hwnd, ID_REFR) == BST_CHECKED) ? 1 : 0;

  // print command line arguments
  _stprintf_s(strParams, TEXT("%i %i %i %i %i"), iWidth, iHeight, iFullScreen, iVSync, iRefRasterizer);


  // get selected api and launch
  GetComboBoxSelection(hwnd, ID_API, str);
  if(StringMatch(str, TEXT("OpenGL")))
  {
    ShellExecute(NULL, TEXT("open"), TEXT("PSSM_OGL.exe"), strParams, szDir, SW_SHOW);
  }
  else if(StringMatch(str, TEXT("Direct3D 9")))
  {
    ShellExecute(NULL, TEXT("open"), TEXT("PSSM_D3D9.exe"), strParams, szDir, SW_SHOW);
  }
  else if(StringMatch(str, TEXT("Direct3D 10")))
  {
    ShellExecute(NULL, TEXT("open"), TEXT("PSSM_D3D10.exe"), strParams, szDir, SW_SHOW);
  }
}

BOOL CALLBACK DlgProc(HWND hwnd, UINT Message, WPARAM wParam, LPARAM lParam)
{
	switch(Message)
	{
		case WM_INITDIALOG:
      AddToComboBox(hwnd, ID_API, TEXT("OpenGL"));
      AddToComboBox(hwnd, ID_API, TEXT("Direct3D 9"));
      AddToComboBox(hwnd, ID_API, TEXT("Direct3D 10"));
      SetComboBoxItem(hwnd, ID_API, 1);
      AddToComboBox(hwnd, ID_WINDOW, TEXT("640x480"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("800x600"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("1024x768"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("1152x864"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("1280x800"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("1280x1024"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("1680x1020"));
      AddToComboBox(hwnd, ID_WINDOW, TEXT("1600x1200"));
      SetComboBoxItem(hwnd, ID_WINDOW, 2);
      CheckDlgButton(hwnd, ID_WINDOWED, BST_CHECKED);
		break;
		case WM_COMMAND:
      if(LOWORD(wParam) == ID_START)
      {
			  PostMessage(hwnd, WM_CLOSE, 0, 0);
        LaunchApp(hwnd);
      }
      if(LOWORD(wParam) == ID_CANCEL)
      {
			  PostMessage(hwnd, WM_CLOSE, 0, 0);
      }
		break;
		case WM_CLOSE:
			EndDialog(hwnd, 0);
		break;
		default:
			return FALSE;
	}
	return TRUE;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
	LPSTR lpCmdLine, int nCmdShow)
{
	return (int)DialogBox(hInstance, MAKEINTRESOURCE(IDD_MAIN), NULL, DlgProc);
}
