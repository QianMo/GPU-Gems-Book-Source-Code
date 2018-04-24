//--------------------------------------------------------------------------------------
// File: MultipleReflections.cpp
//
// Empty starting point for new Direct3D applications
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "dxstdafx.h"
#include "resource.h"
#include "L.h"
#include "Mesh.h"
#include "utilFunctions.h"

//-------------------------------------------------------------------------
// D3dDevice and UI elements
IDirect3DDevice9*			g_pd3dDevice;				///< THE D3D DEVICE
CDXUTDialogResourceManager g_DialogResourceManager;			// manager for shared resources of dialogs
CDXUTDialog             g_HUD;								// dialog for standard controls
CDXUTComboBox*			g_objectMaterialComboBox;			
CDXUTComboBox*			g_objectModelComboBox;
CDXUTComboBox*			g_environmentModelComboBox;
ID3DXFont*					g_pFont;					///< Font for drawing text
ID3DXSprite*				g_pTextSprite;				///< Sprite for batching draw text calls
bool g_showWelcomeScreen = true;
bool doScreenShoot = false;
	//slider ids
#define FIRST_SLIDER 10
#define LINEAR_MIN_ITERATION_SLIDER 10
#define LINEAR_MAX_ITERATION_SLIDER 11
#define SECANT_ITERATION_SLIDER 12
#define MAX_RAY_DEPTH_SLIDER 13
#define CUBEMAP_UPDATE_INTERVAL 14
#define OBJECT_SIZE 15
#define REFRACT_FACTOR_SLIDER 16
#define FRESHEL_RED_SLIDER 17
#define FRESHEL_GREEN_SLIDER 18
#define FRESHEL_BLUE_SLIDER 19
#define LAST_SLIDER 19

    //combo box ids
#define OBJECT_MATERIAL_COMBO 20
#define OBJECT_MODEL_COMBO 21
#define ENVIRONMENT_MODEL_COMBO 22
	//slider values
int g_linearMinIter = 10;
int g_linearMaxIter = 30;
int g_secantIter = 1;
int g_rayDepth = 2;
int	g_iSelectedMenuItem = 0;
int g_iObjectSize = 30;
float g_objectSize = 0.3;
bool g_showHelp = false;
bool g_showGUI = true;
unsigned int g_updateInterval = 1;
unsigned int g_currentFrame = 0;
int g_currentMaterial = 0;
int g_iRefractFactor = 0;
int g_iFreshnelR = 80;
int g_iFreshnelG = 80;
int g_iFreshnelB = 80;
float g_refractFactor = -0.01f;
float g_freshnelR = 0.8f;
float g_freshnelG = 0.8f;
float g_freshnelB = 0.8f;

D3DXVECTOR4 g_eyePos;
D3DXVECTOR4 g_refPos;
D3DXMATRIXA16 g_viewMatrix;
D3DXMATRIXA16 g_projMatrix;
D3DXMATRIXA16 g_worldMatrix;

//---------------------------------------------------------------
// Mesh data
LPCWSTR MESHNAMES[] = { L"Media\\Objects\\sphere.x",
						L"Media\\Objects\\teapot.x",
						L"Media\\Objects\\trollhead.x"};
LPCWSTR EnvMESHNAMES[] = {L"Media\\Objects\\colorCube.x",
						  L"Media\\Objects\\labour.x"};
Mesh* g_environmentMesh;
Mesh* g_objectMesh;


//----------------------------------------------------------------
// shader program data
ID3DXEffect*				g_basicShaders;
ID3DXEffect*				g_reflectionShaders;

//---------------------------------------------------------------
//Camera settings
CModelViewerCamera camera;								///< the camera
D3DXVECTOR3 cameraPosition(0,0,-2);
D3DXVECTOR3 cameraLookat(0,0,0);
unsigned int g_width = 800;
unsigned int g_height = 600;

//-----------------------------------------------------------------
//RenderTargets
unsigned int g_cubeMapSize = 512;
  //color and refraction index information of the first layer
IDirect3DCubeTexture9*	g_CubeTexture1 = NULL;
  //color and refraction index information of the second layer
IDirect3DCubeTexture9*	g_CubeTexture2 = NULL;
  //color and refraction index information of the third layer
IDirect3DCubeTexture9*	g_CubeTexture3 = NULL;

  //normal and distance information of the first layer
IDirect3DCubeTexture9*	g_CubeTexture4 = NULL;
  //normal and distance information of the second layer
IDirect3DCubeTexture9*	g_CubeTexture5 = NULL;
  //normal and distance information of the third layer
IDirect3DCubeTexture9*	g_CubeTexture6 = NULL;

IDirect3DTexture9*	g_ScreenShootTexture = NULL;

//-----------------------------------------------------------------
//function forward declarations
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void UpdateParamFromSlider();

//--------------------------------------------------------------------------------------
// Rejects any devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, 
                                  D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    // Typically want to skip backbuffer formats that don't support alpha blending
    IDirect3D9* pD3D = DXUTGetD3DObject(); 
    if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType,
                    AdapterFormat, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, 
                    D3DRTYPE_TEXTURE, BackBufferFormat ) ) )
        return false;

    return true;
}


//--------------------------------------------------------------------------------------
// Before a device is created, modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, const D3DCAPS9* pCaps, void* pUserContext )
{
	//VSync Off
	pDeviceSettings->pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
	return true;
}

void compileShaders()
{
	HRESULT hr;
	//load effect files
	DWORD dwShaderFlags = 0;
	ID3DXBuffer* errBuff;	// buffer for error message
	if (FAILED(hr = D3DXCreateEffectFromFile( g_pd3dDevice, L"Media\\Shaders\\basicShaders.fx", NULL, NULL, dwShaderFlags, 
                                        NULL, &g_basicShaders, &errBuff )))	// if compilation error occurs
	{
		int BufSize = errBuff->GetBufferSize();

		wchar_t* wbuf = new wchar_t[BufSize];
		mbstowcs( wbuf, (const char*)errBuff->GetBufferPointer(), BufSize );
		MessageBox(NULL, wbuf, L".fx Compilation Error", MB_ICONERROR);		// error message

		delete wbuf;
		exit(-1);
	}
	dwShaderFlags = D3DXSHADER_PREFER_FLOW_CONTROL;
	//dwShaderFlags = D3DXSHADER_SKIPOPTIMIZATION|D3DXSHADER_PREFER_FLOW_CONTROL;
	if (FAILED(hr = D3DXCreateEffectFromFile( g_pd3dDevice, L"Media\\Shaders\\MultipleReflection.fx", NULL, NULL, dwShaderFlags, 
                                        NULL, &g_reflectionShaders, &errBuff )))	// if compilation error occurs
	{
		int BufSize = errBuff->GetBufferSize();

		wchar_t* wbuf = new wchar_t[BufSize];
		mbstowcs( wbuf, (const char*)errBuff->GetBufferPointer(), BufSize );
		MessageBox(NULL, wbuf, L".fx Compilation Error", MB_ICONERROR);		// error message

		delete wbuf;
		exit(-1);
	}
}
//--------------------------------------------------------------------------------------
// Create any D3DPOOL_MANAGED resources here 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnCreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr;

	g_pd3dDevice = pd3dDevice;
	g_DialogResourceManager.OnCreateDevice( pd3dDevice );
    
	//load meshes
	g_environmentMesh = new Mesh(EnvMESHNAMES[0], 1, D3DXVECTOR3(0,0,0));
	D3DXVECTOR3 environmentSize = g_environmentMesh->GetMeshSize();
	g_objectMesh = new Mesh(MESHNAMES[0], g_objectSize, D3DXVECTOR3(0,0,0));
	g_objectMesh->SetContainerSize(environmentSize);
    
	 // Initialize the font
    V_RETURN( D3DXCreateFont( g_pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                         OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                         L"Arial", &g_pFont ) );
	return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3DPOOL_DEFAULT resources here 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnResetDevice( IDirect3DDevice9* pd3dDevice, 
                                const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr;
	//create cubemap rendertargets
	g_CubeTexture1 = CreateCubeTexture( g_cubeMapSize, D3DFMT_A16B16G16R16F, g_pd3dDevice);
	g_CubeTexture2 = CreateCubeTexture( g_cubeMapSize, D3DFMT_A16B16G16R16F, g_pd3dDevice);
	g_CubeTexture3 = CreateCubeTexture( g_cubeMapSize, D3DFMT_A16B16G16R16F, g_pd3dDevice);
	g_CubeTexture4 = CreateCubeTexture( g_cubeMapSize, D3DFMT_A16B16G16R16F, g_pd3dDevice);
	g_CubeTexture5 = CreateCubeTexture( g_cubeMapSize, D3DFMT_A16B16G16R16F, g_pd3dDevice);
	g_CubeTexture6 = CreateCubeTexture( g_cubeMapSize, D3DFMT_A16B16G16R16F, g_pd3dDevice);

	g_ScreenShootTexture = CreateTexture(1600, 1200, D3DFMT_R8G8B8, g_pd3dDevice);
	
	g_DialogResourceManager.OnResetDevice();
	if( g_basicShaders ) V_RETURN( g_basicShaders->OnResetDevice() );
	if( g_reflectionShaders ) V_RETURN( g_reflectionShaders->OnResetDevice() );
	if( g_pFont )	g_pFont->OnResetDevice();
    camera.SetProjParams(3.14 / 4.0, (float)g_width / (float)g_height, 0.01, 20);
	g_currentFrame = 0;
    return S_OK;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
	camera.FrameMove( fElapsedTime );
}

//--------------------------------------------------------------------------------------
// Renders a mesh to all 6 faces of a cubemap
//--------------------------------------------------------------------------------------
void renderEnvMap(Mesh* mesh, IDirect3DCubeTexture9* target, enum RenderMode mode)
{
	IDirect3DSurface9*	oldRenderTarget; 
	IDirect3DSurface9*	oldDepthStencil;
	IDirect3DSurface9*  newDepthStencil;
	// create a CUBEMAP_SIZE x CUBEMAP_SIZE size depth buffer
	g_pd3dDevice->CreateDepthStencilSurface(g_cubeMapSize, g_cubeMapSize, D3DFMT_D16,
		D3DMULTISAMPLE_NONE, 0, true, &newDepthStencil, NULL);
		// replace old depth buffer
	g_pd3dDevice->GetDepthStencilSurface(&oldDepthStencil);
	g_pd3dDevice->SetDepthStencilSurface(newDepthStencil);
	g_pd3dDevice->GetRenderTarget(0,&oldRenderTarget);

	HRESULT hr;
		
	for(int i = 0; i < 6; i++)
	{
		IDirect3DSurface9* pFace;
		V( target->GetCubeMapSurface((D3DCUBEMAP_FACES)i, 0, &pFace) );
		V( g_pd3dDevice->SetRenderTarget(0, pFace) );
        V( g_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 
			D3DCOLOR_ARGB(0,0,0,0), 1.0f, 0) ); 

		getViewProjForCubeFace(&g_viewMatrix, &g_projMatrix, i, g_objectMesh->GetMeshPosition());
		D3DXMATRIXA16 rotationMatrix = mesh->getRotation();
		D3DXMATRIXA16 scaleOffsetMatrix = ScaleAndOffset(mesh->GetMeshScale(), mesh->GetMeshPosition());
		D3DXMatrixMultiply(&g_worldMatrix, &rotationMatrix, &scaleOffsetMatrix);
		mesh->Draw(mode);

		SAFE_RELEASE( pFace );
	}

	// restore old rendertarget & depth buffer
	g_pd3dDevice->SetRenderTarget(0, oldRenderTarget);
	g_pd3dDevice->SetDepthStencilSurface(oldDepthStencil);
	SAFE_RELEASE( oldDepthStencil );
	SAFE_RELEASE( oldRenderTarget );
	SAFE_RELEASE( newDepthStencil );
}

//--------------------------------------------------------------------------------------
// Cubemap updates
//--------------------------------------------------------------------------------------
void renderEnvMaps()
{
	HRESULT hr;
	//check if update is needed
	static bool first = true;
	if(first)//skip first frame (why?)
	{
			first = false;
			return;
	}
    if(g_currentFrame == g_updateInterval)
	{
		g_currentFrame = 0;
	    //update color and distance envmap for layer 1 (close objects back facing polygons)
		g_pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
		renderEnvMap(g_objectMesh, g_CubeTexture1, RENDERMODE_COLOR);
		renderEnvMap(g_objectMesh, g_CubeTexture4, RENDERMODE_NORMAL);
		g_pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
		//update color and distance envmap for layer 2 (close objects front facing polygons)
		renderEnvMap(g_objectMesh,g_CubeTexture2, RENDERMODE_COLOR);
		renderEnvMap(g_objectMesh, g_CubeTexture5, RENDERMODE_NORMAL);
		//update color and distance envmap for layer 3 (far objects front facing polygons)
		renderEnvMap(g_environmentMesh, g_CubeTexture3, RENDERMODE_COLOR);			
		renderEnvMap(g_environmentMesh, g_CubeTexture6, RENDERMODE_NORMAL);
	}
	if(g_updateInterval == 0)
		g_currentFrame = 1;
	else
		g_currentFrame++;
}

//--------------------------------------------------------------------------------------
// Final rendering to the screen
//--------------------------------------------------------------------------------------
void renderToScreen()
{
	HRESULT hr;
	// Clear the render target and the zbuffer 
    V( g_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 45, 50, 170), 1.0f, 0) );

	g_viewMatrix = *camera.GetViewMatrix();
	g_projMatrix = *camera.GetProjMatrix();
	UINT uPasses;
	
	D3DXMATRIXA16 rotationMatrix = g_environmentMesh->getRotation();
	D3DXMATRIXA16 scaleOffsetMatrix = ScaleAndOffset(g_environmentMesh->GetMeshScale(),
													g_environmentMesh->GetMeshPosition());
	D3DXMatrixMultiply(&g_worldMatrix, &rotationMatrix, &scaleOffsetMatrix);
	g_environmentMesh->Draw(RENDERMODE_FINAL);

	rotationMatrix = g_objectMesh->getRotation();
	scaleOffsetMatrix = ScaleAndOffset(g_objectMesh->GetMeshScale(),
									   g_objectMesh->GetMeshPosition());
	D3DXMatrixMultiply(&g_worldMatrix, &rotationMatrix, &scaleOffsetMatrix);
	g_objectMesh->Draw(RENDERMODE_FINAL);
	
}	


//----------------------------------------------------------------------------------
// Render text to screen
//----------------------------------------------------------------------------------
void renderText()
{
	const D3DSURFACE_DESC* backBufferDesc = DXUTGetBackBufferSurfaceDesc();

    CDXUTTextHelper txtHelper( g_pFont, g_pTextSprite, 15 );
    txtHelper.Begin();

	txtHelper.SetInsertionPos( 5, 5 );
	txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );

	txtHelper.DrawFormattedTextLine( L"%.2f fps @ %i x %i", 
		DXUTGetFPS(), backBufferDesc->Width, backBufferDesc->Height );
	
	if(g_showGUI)
	{
		int xpos =  135;
		int ypos =  30;
		txtHelper.SetInsertionPos( xpos, ypos );
		txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
		
		txtHelper.DrawFormattedTextLine( L"linear iteration (min) : %d", g_linearMinIter);
		txtHelper.DrawFormattedTextLine( L"linear iteration (max) : %d", g_linearMaxIter);
		txtHelper.DrawFormattedTextLine( L"secant iteration : %d", g_secantIter);
		txtHelper.DrawFormattedTextLine( L"max ray depth : %d", g_rayDepth);
		txtHelper.DrawFormattedTextLine( L"cubemap update frequency : %d", g_updateInterval);
		txtHelper.DrawFormattedTextLine( L"object scale : %0.2f", g_objectSize);
		txtHelper.DrawFormattedTextLine( L"index of refraction : %0.2f", g_refractFactor);
		txtHelper.DrawFormattedTextLine( L"Fresnel at 90 deg(RED) : %0.2f", g_freshnelR);
		txtHelper.DrawFormattedTextLine( L"Fresnel at 90 deg(GREEN) : %0.2f", g_freshnelG);
		txtHelper.DrawFormattedTextLine( L"Fresnel at 90 deg(BLUE) : %0.2f", g_freshnelB);
	}
	
	if ( !g_showHelp )
	{
		txtHelper.SetInsertionPos( backBufferDesc->Width / 2 -50, 5 );
		txtHelper.DrawTextLine( L"Press F1 for help" );
	}
	
	if(g_showHelp)
	{
		txtHelper.SetInsertionPos( backBufferDesc->Width - 260, backBufferDesc->Height-24*22 );

		txtHelper.DrawTextLine( 
				L"Controls (F1 to hide):\n"
				L"___________________________________\n"
				L"      APPLICATION CONTROLS\n"
				L"\n"
				L"Right click+drag: Rotate mesh\n"
				L"Left click+drag: Move camera\n"
				L"Mouse wheel: Zoom\n"
				L"Arrow keys: Move object\n"
				L"F1: Show/Hide Help\n"
				L"TAB: Show/Hide User Interface\n"
				L"F12: Save Screenshot"
				L"\n"
				L"___________________________________\n"
				L"            Quit: ESC");
	}

	txtHelper.End();
}

void renderWelcomeText()
{
	HRESULT hr;
	V( g_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 45, 50, 170), 1.0f, 0) );

	const D3DSURFACE_DESC* backBufferDesc = DXUTGetBackBufferSurfaceDesc();

    CDXUTTextHelper txtHelper( g_pFont, g_pTextSprite, 20 );
    txtHelper.Begin();

	txtHelper.SetInsertionPos( backBufferDesc->Width / 2 - 100, backBufferDesc->Height /2 );
	txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );

	txtHelper.DrawFormattedTextLine( L"Compiling shaders. Please wait..." );
}
//--------------------------------------------------------------------------------------
// Render the scene 
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
    HRESULT hr;
    
    // Render the scene
    if( SUCCEEDED( pd3dDevice->BeginScene() ) )
    {
		if(g_showWelcomeScreen)
		{
			renderWelcomeText();
			g_showWelcomeScreen = false;
		}
		else
		{
			//if shaders are not complied compile them
            if(g_basicShaders == 0)
				compileShaders();
			//set some parameters
			D3DXVECTOR3 eye = *camera.GetEyePt();
			g_eyePos = D3DXVECTOR4(eye.x, eye.y, eye.z, 1);			
			D3DXVECTOR3 refpos = g_objectMesh->GetMeshPosition();
			g_refPos = D3DXVECTOR4(refpos.x, refpos.y, refpos.z, 1);
			g_objectMesh->getMaterial(0).F0 = D3DXVECTOR4(g_freshnelR, g_freshnelG, g_freshnelB, 1);
			g_objectMesh->getMaterial(0).N0 = g_refractFactor;
			switch(g_currentMaterial)
			{
			case 0:
				strcpy(g_objectMesh->getMaterial(0).finalRenderTechnique, "EnvMapped");
				g_objectMesh->getMaterial(0).basicFinalRender = true;
				break;
			case 1:
				strcpy(g_objectMesh->getMaterial(0).finalRenderTechnique, "SingleReflection");
				g_objectMesh->getMaterial(0).basicFinalRender = false;
				break;
			case 2:
				strcpy(g_objectMesh->getMaterial(0).finalRenderTechnique, "MultipleReflection");
				g_objectMesh->getMaterial(0).basicFinalRender = false;
				break;
			}
		//rendering
			//render the envmaps
			renderEnvMaps();
			//final render to screen
			if(doScreenShoot)
			{
				IDirect3DSurface9*	oldRenderTarget; 
				g_pd3dDevice->GetRenderTarget(0,&oldRenderTarget);
				IDirect3DSurface9*	oldDepthStencil;
				IDirect3DSurface9*  newDepthStencil;
				g_pd3dDevice->CreateDepthStencilSurface(1600, 1200, D3DFMT_D16,
					D3DMULTISAMPLE_NONE, 0, true, &newDepthStencil, NULL);
		    	g_pd3dDevice->GetDepthStencilSurface(&oldDepthStencil);
				g_pd3dDevice->SetDepthStencilSurface(newDepthStencil);
				IDirect3DSurface9* pSurface;
				V( g_ScreenShootTexture->GetSurfaceLevel(0, &pSurface) );
				V( g_pd3dDevice->SetRenderTarget(0, pSurface) ); 
				renderToScreen();     			
				SAFE_RELEASE( pSurface );
				g_pd3dDevice->SetRenderTarget(0, oldRenderTarget);
				g_pd3dDevice->SetDepthStencilSurface(oldDepthStencil);
				SAFE_RELEASE( oldDepthStencil );
				SAFE_RELEASE( oldRenderTarget );
				SAFE_RELEASE( newDepthStencil );
				doScreenShoot = false;
				static int nr = 0;
				char filename[255];
				sprintf(filename, "screen_%i.png", nr);
				V(D3DXSaveTextureToFile(L::l+filename, D3DXIFF_PNG, g_ScreenShootTexture, 0));
				nr++;
			}
			else
				renderToScreen(); 
			//render all text
			renderText();
			//rendre GUI
			if(g_showGUI)
				g_HUD.OnRender( fElapsedTime );
		}
        V( pd3dDevice->EndScene() );
    }	
}

//--------------------------------------------------------------------------------------
// Handle messages to the application 
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, 
                          bool* pbNoFurtherProcessing, void* pUserContext )
{
	*pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
	
	*pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
	
	camera.HandleMessages( hWnd, uMsg, wParam, lParam );
	D3DXMATRIXA16 rotation = *camera.GetWorldMatrix();
	if(g_objectMesh)
		g_objectMesh->setRotation(rotation);

    return 0;
}

//--------------------------------------------------------------------------------------
// Keyboard message handler
//--------------------------------------------------------------------------------------
void CALLBACK KeyboardProc( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	if( bKeyDown )
    {	
		float step = 0.01f;

		switch( nChar )
        {
			case VK_RIGHT: g_objectMesh->Move( D3DXVECTOR3( step,0,0), true ); break;
			case VK_LEFT:  g_objectMesh->Move( D3DXVECTOR3(-step,0,0), true ); break;
			case VK_UP:	   g_objectMesh->Move( D3DXVECTOR3(0, step,0), true ); break;
			case VK_DOWN:  g_objectMesh->Move( D3DXVECTOR3(0,-step,0), true ); break;
			case VK_PRIOR: g_objectMesh->Move( D3DXVECTOR3(0,0, step), true ); break;
			case VK_NEXT:  g_objectMesh->Move( D3DXVECTOR3(0,0,-step), true ); break;
			case VK_TAB:   g_showGUI = !g_showGUI; break;
			case VK_F1:    g_showHelp = !g_showHelp; break;
			case VK_F12: doScreenShoot = true; break;
       }
    }
}


//--------------------------------------------------------------------------------------
// Release resources created in the OnResetDevice callback here 
//--------------------------------------------------------------------------------------
void CALLBACK OnLostDevice( void* pUserContext )
{
	g_DialogResourceManager.OnLostDevice(); 
	if( g_basicShaders ) g_basicShaders->OnLostDevice();
	if( g_reflectionShaders ) g_reflectionShaders->OnLostDevice();	
	if( g_pFont )	g_pFont->OnLostDevice();
	SAFE_RELEASE( g_CubeTexture1 );
	SAFE_RELEASE( g_CubeTexture2 );
	SAFE_RELEASE( g_CubeTexture3 );
	SAFE_RELEASE( g_CubeTexture4 );
	SAFE_RELEASE( g_CubeTexture5 );
	SAFE_RELEASE( g_CubeTexture6 );
	SAFE_RELEASE( g_ScreenShootTexture );
}

//--------------------------------------------------------------------------------------
// Release resources created in the OnCreateDevice callback here
//--------------------------------------------------------------------------------------
void CALLBACK OnDestroyDevice( void* pUserContext )
{
	 g_DialogResourceManager.OnDestroyDevice(); 
	 delete g_environmentMesh;
	 delete g_objectMesh;
	 SAFE_RELEASE(g_basicShaders);
	 SAFE_RELEASE(g_reflectionShaders);
	 SAFE_RELEASE(g_pFont);	 
}

//--------------------------------------------------------------------------------------
// GUI initialization
//--------------------------------------------------------------------------------------
void InitializeDialogs()
{
    g_HUD.Init( &g_DialogResourceManager );
    g_HUD.SetCallback( OnGUIEvent ); 
	
	// Initialize dialog
	int posX = 10; 
	int posY = 30; 
    int sliderheight = 14;
	int sliderYoffset = 15;
	// Setup for sliders
	g_HUD.AddSlider( LINEAR_MIN_ITERATION_SLIDER, posX, posY, 120, sliderheight, 10, 100, g_linearMinIter );
	g_HUD.AddSlider( LINEAR_MAX_ITERATION_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 10, 100, g_linearMaxIter);
	g_HUD.AddSlider( SECANT_ITERATION_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 0, 20, g_secantIter );
	g_HUD.AddSlider( MAX_RAY_DEPTH_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 1, 10, g_rayDepth );
	g_HUD.AddSlider( CUBEMAP_UPDATE_INTERVAL, posX, posY += sliderYoffset, 120, sliderheight, 0, 10, g_updateInterval );
	g_HUD.AddSlider( OBJECT_SIZE, posX, posY += sliderYoffset, 120, sliderheight, 1, 100, g_iObjectSize );
	g_HUD.AddSlider( REFRACT_FACTOR_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 0, 101, g_iRefractFactor );
	g_HUD.AddSlider( FRESHEL_RED_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 0, 100, g_iFreshnelR );
	g_HUD.AddSlider( FRESHEL_GREEN_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 0, 100, g_iFreshnelG );
	g_HUD.AddSlider( FRESHEL_BLUE_SLIDER, posX, posY += sliderYoffset, 120, sliderheight, 0, 100, g_iFreshnelB );
	// Setup fo combo boxes
	g_HUD.AddComboBox( OBJECT_MATERIAL_COMBO, 10, posY += 20, 160, 20, 0, false, &g_objectMaterialComboBox );
	g_HUD.AddComboBox( OBJECT_MODEL_COMBO, 10, posY += 20, 160, 20, 0, false, &g_objectModelComboBox );	
	g_HUD.AddComboBox( ENVIRONMENT_MODEL_COMBO, 10, posY += 20, 160, 20, 0, false, &g_environmentModelComboBox );	

	g_objectMaterialComboBox->AddItem( L"classical envmapping", NULL );
	g_objectMaterialComboBox->AddItem( L"single reflection/refraction", NULL );
	g_objectMaterialComboBox->AddItem( L"multiple reflection/refraction", NULL );
    
	g_objectModelComboBox->AddItem( L"sphere", NULL );
	g_objectModelComboBox->AddItem( L"teapot", NULL );    
	g_objectModelComboBox->AddItem( L"troll head", NULL );    
	//g_objectModelComboBox->AddItem( L"skull", NULL ); 

	g_environmentModelComboBox->AddItem( L"colored box", NULL ); 
	g_environmentModelComboBox->AddItem( L"labour", NULL ); 
}

//--------------------------------------------------------------------------------------
// GUI message handler
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	if( nControlID >= FIRST_SLIDER && nControlID <= LAST_SLIDER )
	{
		g_iSelectedMenuItem = nControlID;
		UpdateParamFromSlider();
	}
	else
	{
		if(nControlID == OBJECT_MODEL_COMBO)
		{
			D3DXVECTOR3 lastposition = g_objectMesh->GetMeshPosition();
			int currentmodel = g_objectModelComboBox->FindItem(g_objectModelComboBox->GetSelectedItem()->strText);
			delete g_objectMesh;
			g_objectMesh = new Mesh(MESHNAMES[currentmodel], g_objectSize, lastposition);
			D3DXVECTOR3 environmentSize = g_environmentMesh->GetMeshSize();
			g_objectMesh->SetContainerSize(environmentSize);
			g_currentFrame = 0;
		}

		if(nControlID == ENVIRONMENT_MODEL_COMBO)
		{
			int currentmodel = g_environmentModelComboBox->FindItem(g_environmentModelComboBox->GetSelectedItem()->strText);
			delete g_environmentMesh;
			g_environmentMesh = new Mesh(EnvMESHNAMES[currentmodel], 1, D3DXVECTOR3(0,0,0));
			D3DXVECTOR3 environmentSize = g_environmentMesh->GetMeshSize();
			g_objectMesh->SetContainerSize(environmentSize);
			g_currentFrame = 0;
		}
		
		if(nControlID == OBJECT_MATERIAL_COMBO)
		{
			g_currentMaterial = g_objectMaterialComboBox->FindItem(g_objectMaterialComboBox->GetSelectedItem()->strText);
		}
	}
}

//--------------------------------------------------------------------------------------
// Update slider parameters
//--------------------------------------------------------------------------------------
void UpdateParamFromSlider()
{
	int iSelectedValue = g_HUD.GetSlider( g_iSelectedMenuItem )->GetValue();

	switch (g_iSelectedMenuItem)
	{
	  case LINEAR_MIN_ITERATION_SLIDER: g_linearMinIter = min(iSelectedValue, g_linearMaxIter);	break;
	  case LINEAR_MAX_ITERATION_SLIDER: g_linearMaxIter = max(iSelectedValue, g_linearMinIter);	break;
	  case SECANT_ITERATION_SLIDER: g_secantIter = iSelectedValue;	break;
	  case CUBEMAP_UPDATE_INTERVAL: g_updateInterval = iSelectedValue;	g_currentFrame = 0; break;
	  case MAX_RAY_DEPTH_SLIDER: g_rayDepth = iSelectedValue;	break;
	  case OBJECT_SIZE: g_iObjectSize = iSelectedValue; 
						g_objectSize = (float) g_iObjectSize / 100.0f;
						g_objectMesh->SetPreferredDiameter(g_objectSize);
						g_currentFrame = 0;
						break;
	  case REFRACT_FACTOR_SLIDER: g_iRefractFactor = iSelectedValue; g_refractFactor = (g_iRefractFactor - 1) / 100.0;	break;
	  case FRESHEL_RED_SLIDER: g_iFreshnelR = iSelectedValue; g_freshnelR = g_iFreshnelR / 100.0;	break;
	  case FRESHEL_GREEN_SLIDER: g_iFreshnelG = iSelectedValue; g_freshnelG = g_iFreshnelG / 100.0; break;
	  case FRESHEL_BLUE_SLIDER: g_iFreshnelB = iSelectedValue; g_freshnelB = g_iFreshnelB / 100.0;	break;
	}
}

//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
INT WINAPI WinMain( HINSTANCE, HINSTANCE, LPSTR, int )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	// Set the callback functions
    DXUTSetCallbackDeviceCreated( OnCreateDevice );
    DXUTSetCallbackDeviceReset( OnResetDevice );
    DXUTSetCallbackDeviceLost( OnLostDevice );
    DXUTSetCallbackDeviceDestroyed( OnDestroyDevice );
    DXUTSetCallbackMsgProc( MsgProc );
	DXUTSetCallbackKeyboard( KeyboardProc );
    DXUTSetCallbackFrameRender( OnFrameRender );
    DXUTSetCallbackFrameMove( OnFrameMove );
   
    // TODO: Perform any application-level initialization here
	InitializeDialogs();
	camera.SetViewParams(&cameraPosition, &cameraLookat);
	camera.SetButtonMasks( MOUSE_RIGHT_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );
	// Initialize DXUT and create the desired Win32 window and Direct3D device for the application
    DXUTInit( true, true, true ); // Parse the command line, handle the default hotkeys, and show msgboxes
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"MultipleReflections" );
    DXUTCreateDevice( D3DADAPTER_DEFAULT, true, g_width, g_height, IsDeviceAcceptable, ModifyDeviceSettings );

    // Start the render loop
    DXUTMainLoop();
    // TODO: Perform any application-level cleanup here
    return DXUTGetExitCode();
}


