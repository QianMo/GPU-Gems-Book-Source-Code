//--------------------------------------------------------------------------------------
// File: DXUT.h
//
// DirectX SDK Direct3D sample framework
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma once
#ifndef DXUT_H
#define DXUT_H

#ifndef UNICODE
#error "The sample framework requires a Unicode build. If you are using Microsoft Visual C++ .NET, under the General tab of the project properties change the Character Set to 'Use Unicode Character Set'."
#endif


//--------------------------------------------------------------------------------------
// Structs
//--------------------------------------------------------------------------------------
class CD3DEnumeration;
class CD3DSettingsDlg;

struct DXUTDeviceSettings
{
    UINT AdapterOrdinal;
    D3DDEVTYPE DeviceType;
    D3DFORMAT AdapterFormat;
    DWORD BehaviorFlags;
    D3DPRESENT_PARAMETERS pp;
};


//--------------------------------------------------------------------------------------
// Error codes
//--------------------------------------------------------------------------------------
#define DXUTERR_NODIRECT3D              MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0901)
#define DXUTERR_NOCOMPATIBLEDEVICES     MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0902)
#define DXUTERR_MEDIANOTFOUND           MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0903)
#define DXUTERR_NONZEROREFCOUNT         MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0904)
#define DXUTERR_CREATINGDEVICE          MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0905)
#define DXUTERR_RESETTINGDEVICE         MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0906)
#define DXUTERR_CREATINGDEVICEOBJECTS   MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0907)
#define DXUTERR_RESETTINGDEVICEOBJECTS  MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0908)
#define DXUTERR_INCORRECTVERSION        MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0909)


//--------------------------------------------------------------------------------------
// Callback registration 
//--------------------------------------------------------------------------------------
typedef bool    (CALLBACK *LPDXUTCALLBACKISDEVICEACCEPTABLE)( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat, bool bWindowed );
typedef void    (CALLBACK *LPDXUTCALLBACKMODIFYDEVICESETTINGS)( DXUTDeviceSettings* pDeviceSettings, const D3DCAPS9* pCaps );
typedef HRESULT (CALLBACK *LPDXUTCALLBACKDEVICECREATED)( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc );
typedef HRESULT (CALLBACK *LPDXUTCALLBACKDEVICERESET)( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc );
typedef void    (CALLBACK *LPDXUTCALLBACKDEVICEDESTROYED)();
typedef void    (CALLBACK *LPDXUTCALLBACKDEVICELOST)();
typedef void    (CALLBACK *LPDXUTCALLBACKFRAMEMOVE)( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime );
typedef void    (CALLBACK *LPDXUTCALLBACKFRAMERENDER)( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime );
typedef void    (CALLBACK *LPDXUTCALLBACKKEYBOARD)( UINT nChar, bool bKeyDown, bool bAltDown );
typedef void    (CALLBACK *LPDXUTCALLBACKMOUSE)( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos );
typedef LRESULT (CALLBACK *LPDXUTCALLBACKMSGPROC)( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing );
typedef void    (CALLBACK *LPDXUTCALLBACKTIMER)( UINT idEvent );

// Device callbacks
void DXUTSetCallbackDeviceCreated( LPDXUTCALLBACKDEVICECREATED pCallbackDeviceCreated );
void DXUTSetCallbackDeviceReset( LPDXUTCALLBACKDEVICERESET pCallbackDeviceReset );
void DXUTSetCallbackDeviceLost( LPDXUTCALLBACKDEVICELOST pCallbackDeviceLost );
void DXUTSetCallbackDeviceDestroyed( LPDXUTCALLBACKDEVICEDESTROYED pCallbackDeviceDestroyed );

// Frame callbacks
void DXUTSetCallbackFrameMove( LPDXUTCALLBACKFRAMEMOVE pCallbackFrameMove );
void DXUTSetCallbackFrameRender( LPDXUTCALLBACKFRAMERENDER pCallbackFrameRender );

// Message callbacks
void DXUTSetCallbackKeyboard( LPDXUTCALLBACKKEYBOARD pCallbackKeyboard );
void DXUTSetCallbackMouse( LPDXUTCALLBACKMOUSE pCallbackMouse, bool bIncludeMouseMove = false );
void DXUTSetCallbackMsgProc( LPDXUTCALLBACKMSGPROC pCallbackMsgProc );


//--------------------------------------------------------------------------------------
// Initialization
//--------------------------------------------------------------------------------------
HRESULT DXUTInit( bool bParseCommandLine = true, bool bHandleDefaultHotkeys = true, bool bShowMsgBoxOnError = true );

// Choose either DXUTCreateWindow or DXUTSetWindow.  If using DXUTSetWindow, consider using DXUTStaticWndProc
HRESULT DXUTCreateWindow( const WCHAR* strWindowTitle = L"Direct3D Window", 
                          HINSTANCE hInstance = NULL, HICON hIcon = NULL, HMENU hMenu = NULL,
                          int x = CW_USEDEFAULT, int y = CW_USEDEFAULT );
HRESULT DXUTSetWindow( HWND hWndFocus, HWND hWndDeviceFullScreen, HWND hWndDeviceWindowed, bool bHandleMessages = true );
LRESULT CALLBACK DXUTStaticWndProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam );

// Choose either DXUTCreateDevice or DXUTSetDevice or DXUTCreateDeviceFromSettings
HRESULT DXUTCreateDevice( UINT AdapterOrdinal = D3DADAPTER_DEFAULT, bool bWindowed = true, 
                          int nSuggestedWidth = 640, int nSuggestedHeight = 480,
                          LPDXUTCALLBACKISDEVICEACCEPTABLE pCallbackIsDeviceAcceptable = NULL,
                          LPDXUTCALLBACKMODIFYDEVICESETTINGS pCallbackModifyDeviceSettings = NULL );
HRESULT DXUTCreateDeviceFromSettings( DXUTDeviceSettings* pDeviceSettings, bool bPreserveInput = false );
HRESULT DXUTSetDevice( IDirect3DDevice9* pd3dDevice );

// Choose either DXUTMainLoop or implement your own main loop 
HRESULT DXUTMainLoop( HACCEL hAccel = NULL );

// If not using DXUTMainLoop consider using DXUTRender3DEnvironment
void DXUTRender3DEnvironment(); 


//--------------------------------------------------------------------------------------
// Finding valid device settings
//--------------------------------------------------------------------------------------
enum DXUT_MATCH_TYPE
{
    DXUTMT_IGNORE_INPUT = 0,  // Use the closest valid value to a default 
    DXUTMT_PRESERVE_INPUT,    // Use input without change, but may cause no valid device to be found
    DXUTMT_CLOSEST_TO_INPUT   // Use the closest valid value to the input 
};

struct DXUTMatchOptions
{
    DXUT_MATCH_TYPE eAdapterOrdinal;
    DXUT_MATCH_TYPE eDeviceType;
    DXUT_MATCH_TYPE eWindowed;
    DXUT_MATCH_TYPE eAdapterFormat;
    DXUT_MATCH_TYPE eVertexProcessing;
    DXUT_MATCH_TYPE eResolution;
    DXUT_MATCH_TYPE eBackBufferFormat;
    DXUT_MATCH_TYPE eBackBufferCount;
    DXUT_MATCH_TYPE eMultiSample;
    DXUT_MATCH_TYPE eSwapEffect;
    DXUT_MATCH_TYPE eDepthFormat;
    DXUT_MATCH_TYPE eStencilFormat;
    DXUT_MATCH_TYPE ePresentFlags;
    DXUT_MATCH_TYPE eRefreshRate;
    DXUT_MATCH_TYPE ePresentInterval;
};

HRESULT DXUTFindValidDeviceSettings( DXUTDeviceSettings* pOut, DXUTDeviceSettings* pIn = NULL, DXUTMatchOptions* pMatchOptions = NULL );


//--------------------------------------------------------------------------------------
// Common Tasks 
//--------------------------------------------------------------------------------------
void    DXUTSetCursorSettings( bool bShowCursorWhenFullScreen, bool bClipCursorWhenFullScreen );
void    DXUTSetMultimonSettings( bool bAutoChangeAdapter );
void    DXUTSetConstantFrameTime( bool bConstantFrameTime, float fTimePerFrame = 0.0333f );
void    DXUTSetShowSettingsDialog( bool bShow ); 
HRESULT DXUTSetTimer( LPDXUTCALLBACKTIMER pCallbackTimer, float fTimeoutInSecs = 1.0f, UINT* pnIDEvent = NULL );
HRESULT DXUTKillTimer( UINT nIDEvent );
HRESULT DXUTToggleFullScreen();
HRESULT DXUTToggleREF();
void    DXUTPause( bool bPauseTime, bool bPauseRendering );
void    DXUTResetFrameworkState();
void    DXUTShutdown();


//--------------------------------------------------------------------------------------
// State Retrieval  
//--------------------------------------------------------------------------------------
IDirect3D9*             DXUTGetD3DObject(); // Does not addref unlike typical Get* APIs
IDirect3DDevice9*       DXUTGetD3DDevice(); // Does not addref unlike typical Get* APIs
DXUTDeviceSettings      DXUTGetDeviceSettings(); 
D3DPRESENT_PARAMETERS   DXUTGetPresentParameters();
const D3DSURFACE_DESC*  DXUTGetBackBufferSurfaceDesc();
const D3DCAPS9*         DXUTGetDeviceCaps();
HWND                    DXUTGetHWND();
HWND                    DXUTGetHWNDFocus();
HWND                    DXUTGetHWNDDeviceFullScreen();
HWND                    DXUTGetHWNDDeviceWindowed();
const RECT&             DXUTGetWindowClientRect();
double                  DXUTGetTime();
float                   DXUTGetElapsedTime();
bool                    DXUTIsWindowed();
float                   DXUTGetFPS();
LPCWSTR                 DXUTGetWindowTitle();
LPCWSTR                 DXUTGetFrameStats();
LPCWSTR                 DXUTGetDeviceStats();
bool                    DXUTGetShowSettingsDialog(); 
bool                    DXUTIsRenderingPaused();
bool                    DXUTIsTimePaused();
int                     DXUTGetExitCode();
bool                    DXUTIsKeyDown( BYTE vKey ); // Pass a virtual-key code, ex. VK_F1, 'A', VK_RETURN, VK_LSHIFT, etc
bool                    DXUTIsMouseButtonDown( BYTE vButton ); // Pass a virtual-key code: VK_LBUTTON, VK_RBUTTON, VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2

#endif




