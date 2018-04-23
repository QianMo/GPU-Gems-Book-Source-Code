//--------------------------------------------------------------------------------------
// File: Water.h
//
// Simple Shader 2.0 clFramework demo
//
// Copyright (c) Jens Krüger All rights reserved.
//--------------------------------------------------------------------------------------


#include "common/dxstdafx.h"
#include "resource.h"
#include "direct.h"

#include "tchar.h"
#include <tools/directXUtils.h>
#include "../clFramework/clCGSolver.h"
#include "../clFramework/clCrNiVector.h"

#ifdef useUpacked
#include "../clFramework/clCrNiMatrix.h"
#else
#include "../clFramework/clPackedCrNiMatrix.h"
#endif

//--------------------------------------------------------------------------------------
// Global Structures
//--------------------------------------------------------------------------------------
struct GRIDVERTEX {
    FLOAT      x,y,z;	// position
    D3DCOLOR   diffuse;
    static const DWORD FVF;
};
const DWORD GRIDVERTEX::FVF = D3DFVF_XYZ | D3DFVF_DIFFUSE;

struct TEXVERTEX2D {
    FLOAT      x,y,z;	// position
    FLOAT      tu, tv;		// tex-coords
    static const DWORD FVF;
};
const DWORD TEXVERTEX2D::FVF = D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE2(0);

//--------------------------------------------------------------------------------------
// Global "Constants"
//--------------------------------------------------------------------------------------
TEXVERTEX2D	g_hMainQuad[]= {	{ -1, -1, 1, 0, 1},
								{  1, -1, 1, 1, 1},
								{ -1,  1, 1, 0, 0},
								{  1,  1, 1, 1, 0}};
bool		g_bStartWindowed=true;
int			g_size  = 512;
TCHAR		g_strTitle[] = _T("Shader 2.0 Water Surface V. 1.1 (Build:")_T(__TIMESTAMP__)_T(")");
bool		g_bScreesaverMode;

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
ID3DXFont*              g_pFont = NULL;         // Font for drawing text
ID3DXSprite*            g_pTextSprite = NULL;   // Sprite for batching draw text calls
CModelViewerCamera      g_Camera;               // A model viewing camera
bool                    g_bShowHelp = false;    // If true, it renders the UI control text
CDXUTDialog             g_HUD;                  // dialog for standard controls
CDXUTDialog             g_WaterUI;              // dialog for sample specific controls

IDirect3DDevice9*		g_pd3dDevice;
const D3DSURFACE_DESC*	g_pBackBufferSurfaceDesc;

LPDIRECT3DVERTEXBUFFER9 g_pVB;                  // Single Quad Vextex buffer 
LPDIRECT3DVERTEXBUFFER9 g_pVBGrid;              // Grid Vextex buffer 
LPDIRECT3DINDEXBUFFER9  g_pIBGrid;				// Grid Index buffer

LPD3DXEFFECT			g_pMainShader;
D3DXMATRIX				g_mModelViewProj, g_matProj, g_matView;

// cg-solver variables
FLOAT*					g_pfRHSData;
int						g_iGridSizeX, g_iGridSizeY;
clCGSolver*				g_pCGSolver;
clUnpackedVector*		g_cluUCurrent;
clUnpackedVector*		g_cluULast;
clCrNiVector*			g_cluRHS;

#ifdef useUpacked
	clUnpackedVector*		g_cluUNext;
	#define timeVec1 g_cluRHS
	#define timeVec2 g_cluUNext
#else
	clPackedVector*			g_clUNext;
	clPackedVector*			g_clRHS;
	#define timeVec1 g_clRHS
	#define timeVec2 g_clUNext
#endif

float					g_fDt, g_fC, g_fDX, g_fDY;
int						g_iSteps;

float					g_fMouseX;
float					g_fMouseY;
bool					g_bMouseLDown;

bool					g_bRainMode = false;
int						g_iRainCounter;
int						g_iRainDelay;

bool					g_bTimeclPerformance;

LPDIRECT3DCUBETEXTURE9  g_pCubePoolView;

LPD3DXRENDERTOSURFACE	g_pBufferRenderSurface;
PDIRECT3DTEXTURE9		g_pBufferTexture;
LPDIRECT3DSURFACE9		g_pBufferTextureSurface;
D3DXMATRIXA16			g_mBufferModelViewProj;

bool					g_bShowText=false;
bool					g_bShowUI=true;
bool					g_bWireFrame;

int						g_iViewHeight;
int						g_iViewWidth;

bool					g_bIsGUIEvent;


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN    1
#define IDC_TOGGLEREF           2
#define IDC_CHANGEDEVICE        3
#define IDC_STATIC_VIS		    4
#define IDC_SLIDER_VIS		    5
#define IDC_SLIDER_RAIN		    6
#define IDC_CBOX_RAIN		    7
#define IDC_TOGGLEUI			8

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool    CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat, bool bWindowed );
void    CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, const D3DCAPS9* pCaps );
HRESULT CALLBACK OnCreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc );
HRESULT CALLBACK OnResetDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc );
void    CALLBACK OnFrameMove( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime );
void    CALLBACK OnFrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing );
void    CALLBACK KeyboardProc( UINT nChar, bool bKeyDown, bool bAltDown  );
void	CALLBACK OnMouseEvent( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos );
void    CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl );
void    CALLBACK OnLostDevice();
void    CALLBACK OnDestroyDevice();

void    InitApp();
void    RenderText(float fElapsedTime);
HRESULT	RenderUI(float fElapsedTime);

void	ToggleRain(bool bIsRain);
HRESULT CreateGridBuffer(IDirect3DDevice9* pd3dDevice);
HRESULT createCubeMap(IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, _TCHAR *face);
void setDrop(IDirect3DDevice9* pd3dDevice, float x, float y);
void ChangeViscosity(int iValue);
void ChangeRain(int iValue);
void ResizeGrid();