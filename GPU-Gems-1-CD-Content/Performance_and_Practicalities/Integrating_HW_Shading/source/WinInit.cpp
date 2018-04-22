// from C4Dfx by Jörn Loviscach, www.l7h.cn
// functions to build a dummy window and to initialize OpenGL extensions
// and to call an external editor software

#include <windows.h>
#define GLH_EXT_SINGLE_FILE
#include <glh/glh_extensions.h>
#include "WinInit.h"
#include "WinHack.h"
#include "PrefDialog.h"
#include "C4DWrapper.h"
#include <assert.h>

HWND hWnd = NULL;

static HDC hDC = NULL;
static HGLRC hRC = NULL;

static bool ok = false;
static bool multisampleFilterHintNV = false;

static DWORD WINAPI WinThreadFunc(LPVOID hIns)
{
	HINSTANCE hInstance = (HINSTANCE)hIns;

	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style			= CS_OWNDC;
	wcex.lpfnWndProc	= (WNDPROC)DefWindowProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= NULL;
	wcex.hCursor		= NULL;
	wcex.hbrBackground	= (HBRUSH)NULL;
	wcex.lpszMenuName	= (LPCSTR)NULL;
	wcex.lpszClassName	= "C4Dfx";
	wcex.hIconSm		= NULL;

	if(RegisterClassEx(&wcex) == NULL)
	{
		C4DWrapper::MsgBox("Couldn't register window class.");
	}
	else
	{
		hWnd = CreateWindowEx(WS_EX_TOPMOST, "C4Dfx", "C4Dfx",
			WS_OVERLAPPEDWINDOW, 0, 0, 200, 100, NULL, NULL, hInstance, NULL);

		if(hWnd == NULL)
		{
			C4DWrapper::MsgBox("Couldn't create initial window.");
		}
		else
		{
			hDC = GetDC(hWnd);
			if(hDC == NULL)
			{
				C4DWrapper::MsgBox("Couldn't get device context of initial window.");
			}
			else
			{
				static PIXELFORMATDESCRIPTOR pfd = {
					sizeof(PIXELFORMATDESCRIPTOR),
					1,
					PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL,
					PFD_TYPE_RGBA,
					32,
					0,0,0,0,0,0,0,0,0,0,0,0,0,
					32,
					0,0,0,0,0,0,0 };
				int nPixelFormat = ChoosePixelFormat(hDC, &pfd);
				if(nPixelFormat == 0)
				{
					C4DWrapper::MsgBox("No suitable OpenGL pixel format for initial window.");
				}
				else
				{				
					if(! SetPixelFormat(hDC, nPixelFormat, &pfd))
					{
						C4DWrapper::MsgBox("Couldn't set OpenGL pixel format for initial window.");
					}
					else
					{
						hRC = wglCreateContext(hDC);
						if(hRC == NULL)
						{
							C4DWrapper::MsgBox("Couldn't create rendering context for initial window.");
						}
						else
						{
							if(! wglMakeCurrent(hDC, hRC))
							{
								C4DWrapper::MsgBox("Couldn't switch to rendering context for initial window.");
							}
							else
							{
								if(wglGetProcAddress==NULL)
								{
									C4DWrapper::MsgBox("Not support of wglGetProcAddress. Use vendor specific driver?");
								}
								else
								{
									ok = true;

									// mission-critical OpenGL extensions
									const int numExt = 8;
									char* ext[numExt] = {
										"WGL_ARB_extensions_string",
										"GL_ARB_multitexture",
										"GL_ARB_multisample",
										"WGL_ARB_pixel_format",
										"WGL_ARB_pbuffer",
										"GL_ARB_depth_texture",
										"WGL_ARB_render_texture",
										"WGL_NV_render_depth_texture"
									};

									int i;
									for(i = 0; i < numExt; ++i)
									{
										if(!glh_init_extensions(ext[i]))
										{
											C4DWrapper::MsgBox("Driver does not support", ext[i]);
											ok = false;
										}
									}
	
									// uncritical OpenGL extensions
									multisampleFilterHintNV = ( 0 != glh_init_extensions("GL_NV_multisample_filter_hint") );
								}
								BOOL result = wglMakeCurrent(hDC, NULL);
								assert(result);
							}
							BOOL result = wglDeleteContext(hRC);
							assert(result);
						}
					}
				}
			}
			BOOL result = DestroyWindow(hWnd);
			assert(result);
		}
	}
	return 0;
}

static DWORD dwWinThreadId = 0;
static HANDLE hWinThread = NULL;
static bool calledBefore = false;

// get a window to build our first OpenGL context
void WinInit::StartWin(void)
{
	assert(!calledBefore);
	calledBefore = true;

	hWinThread = CreateThread(NULL, 0, WinThreadFunc, GetInstance(), 0, &dwWinThreadId);
	CloseHandle(hWinThread);
}

bool WinInit::IsOpenGLOK(void)
{
	return ok;
}

bool WinInit::HasMultisampleFilterHintNV(void)
{
	return multisampleFilterHintNV;
}

void WinInit::StartEditor(const char* file, int line)
{

	char s[256], t[256]; // These should better be of dynamic size.
	if(! GetEditorPath(s))
		return;

	if(! GetCommandL(t, line, file))
		return;

	STARTUPINFO si;
	si.cb = sizeof(STARTUPINFO);
	si.lpReserved = NULL;
	si.lpDesktop = NULL;
	si.lpTitle = NULL;
	si.dwFlags = 0L;
	si.cbReserved2 = 0L;
	si.lpReserved2 = NULL;
	PROCESS_INFORMATION pi;
	pi.hThread = NULL;
	pi.hProcess = NULL;
	
	if (!CreateProcess(s, t, NULL, NULL, FALSE, 0L, NULL, NULL, &si, &pi))
	{
		DWORD dw = GetLastError();
		FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, dw, 0L, s, 255, NULL);
		C4DWrapper::MsgBox("Cannot open editor:\n", s);
	}
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
}