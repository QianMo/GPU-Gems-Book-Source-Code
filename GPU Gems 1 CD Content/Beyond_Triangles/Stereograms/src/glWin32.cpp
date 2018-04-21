#include <windows.h>
#include "..\lib\paralelo3d.h"
#include "resource.h"

extern char *appname;
char app_path[256];

pRender *render=0;
int active=0;

int mouse=0,mousex,mousey;

void draw_scene();
void draw_info();
void key_pressed(int key,int flags);
void animate();
void init();
void destroy();
void update_menu();
void command(UINT msg, WPARAM wParam, LPARAM lParam);

HWND hWndMain;
HINSTANCE hInstance;

LRESULT CALLBACK WinFunc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lp, int nCmd)
{
	WNDCLASS wcl;
	MSG  msg;

	hInstance=hInst;
	HACCEL hAccelTable = LoadAccelerators(hInst, MAKEINTRESOURCE(IDR_ACCELERATOR));

	char *c,*cmdline=GetCommandLine();
	if (cmdline[0]=='"') 
	{
		cmdline++;
		c=strchr(cmdline,'"');
		if (c) *c=0;
	}
	else
	{
		c=strchr(cmdline,' ');
		if (c) *c=0;
	}
	strcpy(app_path,cmdline);
	c=strrchr(app_path,'\\');
	if (c) *(c+1)=0;

	wcl.style			= CS_HREDRAW | CS_VREDRAW;
	wcl.lpfnWndProc		= (WNDPROC)WinFunc;
	wcl.cbClsExtra		= 0;
	wcl.cbWndExtra		= 0;
	wcl.hInstance		= hInst;
	wcl.hIcon			= LoadIcon(hInst, MAKEINTRESOURCE(IDR_MAINFRAME));
	wcl.hCursor			= LoadCursor(NULL, IDC_ARROW);
	wcl.hbrBackground	= 0;
	wcl.lpszMenuName	= MAKEINTRESOURCE(IDR_MENU);
	wcl.lpszClassName	= "glWin32";

	if (!RegisterClass (&wcl))
	{
	  MessageBox (0, "Can't register Window", "ERROR", MB_OK);
	  return 0;
	}

	CreateWindowEx(	0,"glWin32",appname,
					WS_OVERLAPPEDWINDOW,
					CW_USEDEFAULT,CW_USEDEFAULT,800,600,
					NULL,NULL,hInst,NULL);

	ShowWindow (hWndMain, SW_NORMAL);

	while (1) 
		{
		while (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE) == TRUE) 
			{
			if (GetMessage(&msg, NULL, 0, 0)) 
				{
				if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) 
					{
					TranslateMessage(&msg);
					DispatchMessage(&msg);
					}
				}
			else 
				return TRUE;
			}
		if (render)
			{
				render->update();
				if (active) 
					render->check_input();
				animate();
				render->begin_draw();
				draw_scene();
				render->set_draw2d();
				draw_info();
				render->end_draw();
			}
		}
}

LRESULT CALLBACK WinFunc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
	case WM_CREATE:
		{
			hWndMain=hWnd;
			render=new pRender(app_path);
			render->pre_create(hWnd);
			render->create(0);
/*
			HMENU menu;
			MENUITEMINFO mi;
			menu=GetSubMenu(GetMenu(hWnd),2);
			memset(&mi,0,sizeof(mi));
			mi.cbSize=sizeof(mi);
			mi.fMask=MIIM_ID|MIIM_TYPE;
			mi.fType=MFT_STRING;
			pString str;
			for( int i=0;i<render->profile.num;i++ )
			{
				mi.wID=32900+i;
				str.format("%s\tCtrl+%i",(const char *)render->profile[i]->name,i+1);
				mi.dwTypeData=(char *)(const char *)str;
				mi.cch=str.length();
				InsertMenuItem(menu,2+i,1,&mi);
			}
*/
			init();
		}
		break;

	case WM_SIZE:
		if (render)
			render->resize(LOWORD(lParam),HIWORD(lParam));
		break;

	case WM_DESTROY:
		destroy();
		delete render;
		PostQuitMessage(0);
		break;

	case WM_ERASEBKGND:
		if (render && render->hrc)
			return 1;
		else
			return 0;

	case WM_KEYDOWN:
		if (wParam>='1' && wParam<='1'+(unsigned)render->profile.num)
		if (GetAsyncKeyState(VK_CONTROL)&(1<<15))
			render->profile_select(wParam-'1');
		break;

	case WM_CHAR:
		if (render)
		{
			render->key_pressed(wParam);
			key_pressed(wParam,GetAsyncKeyState(VK_SHIFT)&0x80000000);
			InvalidateRect(hWnd,0,0);
		}
		return 0;

	case WM_PAINT:
		if (render)
		{
			render->begin_draw();
			draw_scene();
			render->set_draw2d();
			draw_info();
			render->end_draw();
		}
		break;

	case WM_COMMAND:
		command(msg,wParam,lParam);
		break;

	case WM_ACTIVATE:
		active=LOWORD(wParam)==WA_INACTIVE?0:1;
		update_menu();
		break;

	case WM_RBUTTONDOWN:
		if (mouse==0)
		{
			mouse=2;
			SetCapture(hWnd);
			mousex=LOWORD(lParam);
			mousey=HIWORD(lParam);  
		}
		break;
	case WM_LBUTTONDOWN:
		if (mouse==0)
		{
			mouse=1;
			SetCapture(hWnd);
			mousex=LOWORD(lParam);
			mousey=HIWORD(lParam);  
		}
		break;
	case 0x20a:
		{
			float wheelspeed=0.2f;
			short dz=(short)HIWORD(wParam);
 			render->camera.pos-=render->camera.Z*(dz*wheelspeed);
		}
		break;
	case WM_RBUTTONUP:
		mouse=0;
		ReleaseCapture();
		break;
	case WM_LBUTTONUP:
		mouse=0;
		ReleaseCapture();
		break;
	case WM_MOUSEMOVE:
		if (mouse)
		{
			float rotspeed=0.05f;
			float panspeed=0.2f;

			int dx=(short)LOWORD(lParam)-mousex;
			int dy=(short)HIWORD(lParam)-mousey;

			if (mouse==1)
			{
				render->camera.rotate(-dx*rotspeed,render->camera.Y);
				render->camera.rotate(-dy*rotspeed,render->camera.X);
			}
			else
				render->camera.pos+=
					render->camera.X*(-dx*panspeed)+
					render->camera.Y*(dy*panspeed);

			mousex=(short)LOWORD(lParam);
			mousey=(short)HIWORD(lParam);
		}
		break;
	}

	return DefWindowProc (hWnd, msg, wParam, lParam);
}
