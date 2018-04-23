#pragma once
//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
enum
{
IDC_TOGGLEFULLSCREEN,
IDC_TOGGLEREF,
IDC_CHANGEDEVICE,
IDC_CYCLECUBEMAPS,
IDC_CYCLELIGHTING,
//IDC_DEBUGIMAGE,
IDC_LAST,
};



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
void    CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl );
void    CALLBACK OnLostDevice();
void    CALLBACK OnDestroyDevice();

HRESULT RenderParaboloidEnvMap( IDirect3DDevice9* pd3dDevice );
HRESULT ProjectParaboloidToSH( IDirect3DDevice9* pd3dDevice );
HRESULT EvaluateConvolvedSH( LPDIRECT3DDEVICE9 pd3dDevice, LPDIRECT3DCUBETEXTURE9 pResultCube, const LPDIRECT3DTEXTURE9* pEvalSHFunction );
HRESULT DisplayDebugImage( LPDIRECT3DDEVICE9 pd3dDevice, LPDIRECT3DTEXTURE9 pTexture );
void    InitApp();
void    RenderText();

//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
CModelViewerCamera      g_Camera;                 // A model viewing camera
CDXUTDialog             g_HUD;                    // dialog for standard controls
ID3DXFont*              g_pFont = NULL;           // Font for drawing text
ID3DXSprite*            g_pTextSprite = NULL;     // Sprite for batching draw text calls
CubeMesh*               g_pCubeMesh = NULL;       // Cubemap background mesh
FSQuadMesh*             g_pFSQuadMesh = NULL;     // fullscreen quad mesh
LPDIRECT3DCUBETEXTURE9  g_pBaseCubeMap = NULL;    // non-convolved cube map
LPDIRECT3DCUBETEXTURE9  g_pDiffuseCubeMap = NULL; // diffuse convolved cube map
LPDIRECT3DCUBETEXTURE9  g_pSpecularCubeMap = NULL; // specular convolved cube map
LPDIRECT3DSURFACE9      g_pBackBuffer = NULL;     // primary surf back buffer
LPDIRECT3DSURFACE9      g_pZBuffer = NULL;        // primary surf Z buffer
LPDIRECT3DTEXTURE9      g_pParaboloidMap[2] = { NULL, NULL };  // cubemap converted into paraboloid map
LPDIRECT3DTEXTURE9      g_pParaboloidSHWeights[2] = { NULL, NULL };  // SH basis functions for paraboloid map
LPDIRECT3DTEXTURE9      g_pLambertSHEval[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
LPDIRECT3DTEXTURE9      g_pPhongSHEval[6] = { NULL, NULL, NULL, NULL, NULL, NULL };
LPDIRECT3DTEXTURE9      g_pIrradianceSHCoefficients;  // texture storing num_order^2 SH coefficients for the cube map
LPD3DXEFFECT            g_pConvolveEffect = NULL; // shaders which perform SH projection & eval
LPD3DXEFFECT            g_pDisplayEffect = NULL;  // simple decal texture display
LPD3DXEFFECT            g_pParaboloidEffect = NULL; // project cubemap to paraboloid map
LPD3DXEFFECT            g_pDragonEffect = NULL;
LPDIRECT3DVERTEXDECLARATION9 g_pDragonVBDecl = NULL;
NVBScene*               g_pDragon = NULL;           // Dragon scene
bool					g_bShowHelp = false;
bool					g_bShowUI = true;
bool                    g_bShowDebug = false;
UINT                    g_uDisplayCubeMap = 0;    // cycle through standard, diffuse convolved, specular convolved
UINT                    g_uTechnique = 0;         // cycle through cubemap & directional lighting
