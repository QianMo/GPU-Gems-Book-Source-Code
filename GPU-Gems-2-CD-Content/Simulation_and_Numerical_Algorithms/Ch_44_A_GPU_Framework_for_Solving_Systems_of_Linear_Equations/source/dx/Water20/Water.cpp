//--------------------------------------------------------------------------------------
// File: Water.cpp
//
// Simple Shader 2.0 clFramework demo
//
// Copyright (c) Jens Krüger All rights reserved.
//--------------------------------------------------------------------------------------
#include "water.h"

void SetCurrentDir(LPSTR cmdLine) {
	LPWSTR wcLine;
	wcLine = GetCommandLine();

	char *cline = new char[int(_tcslen(wcLine))+1];

	BOOL bConvProblem;
	WideCharToMultiByte( CP_ACP, 0, wcLine, -1, cline, int(_tcslen(wcLine))+1,"_",&bConvProblem );

	char strAppDir[_MAX_PATH];
	strncpy(strAppDir,cline,strlen(cline)-strlen(cmdLine));
	strAppDir[strlen(cline)-strlen(cmdLine)] = 0;
	
	unsigned int end=0;
	for (unsigned int i = 0;i<strlen(strAppDir);i++) if (strAppDir[i] == '\\') {
		end = i;
	}

	strncpy(strAppDir,strAppDir,end);
	strAppDir[end] = 0;	strupr(strAppDir);
	if( !_chdrive( strAppDir[0]-'A'+1 ) ) _chdir(strAppDir+2);

	delete [] cline;
}

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
INT WINAPI WinMain( HINSTANCE, HINSTANCE, LPSTR cmdLine, int )
{
	// This is for "cheap screensaver" support
	if (strstr(cmdLine,"/p") != 0) {
		exit(0);
	}
	if (strstr(cmdLine,"/c") != 0) {
		MessageBoxA(NULL,"Water Screensaver\n(c) 2004 by Jens Krüger\n http://wwwcg.in.tum.de/\nHit ESC to exit the Screensaver\nHit F1 for help ","Implict Water Screensaver",MB_OK);
		exit(0);
	}
	if (strstr(cmdLine,"/s") != 0) {
		SetCurrentDir(cmdLine);
		g_bStartWindowed = false;
		g_bShowUI = false;
		g_bRainMode = true;
		g_bScreesaverMode = true;
	} else g_bScreesaverMode = false;
	// end screensaver


	if (strlen(cmdLine) != 0 && atoi(cmdLine) != 0 ) g_size = atoi(cmdLine);
	
	// Set the callback functions. These functions allow the sample framework to notify
    // the application about device changes, user input, and windows messages.  The 
    // callbacks are optional so you need only set callbacks for events you're interested 
    // in. However, if you don't handle the device reset/lost callbacks then the sample 
    // framework won't be able to reset your device since the application must first 
    // release all device resources before resetting.  Likewise, if you don't handle the 
    // device created/destroyed callbacks then the sample framework won't be able to 
    // recreate your device resources.
    DXUTSetCallbackDeviceCreated( OnCreateDevice );
    DXUTSetCallbackDeviceReset( OnResetDevice );
    DXUTSetCallbackDeviceLost( OnLostDevice );
    DXUTSetCallbackDeviceDestroyed( OnDestroyDevice );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( KeyboardProc );
    DXUTSetCallbackFrameRender( OnFrameRender );
    DXUTSetCallbackFrameMove( OnFrameMove );
	DXUTSetCallbackMouse( OnMouseEvent, true);

    // Show the cursor and clip it when in full screen
    DXUTSetCursorSettings( g_bScreesaverMode, true );

    InitApp();

    // Initialize the sample framework and create the desired Win32 window and Direct3D 
    // device for the application. Calling each of these functions is optional, but they
    // allow you to set several options which control the behavior of the framework.
    DXUTInit( true, true, true ); // Parse the command line, handle the default hotkeys, and show msgboxes

    DXUTCreateWindow( g_strTitle );
    DXUTCreateDevice( D3DADAPTER_DEFAULT, g_bStartWindowed, MIN(600,(g_iGridSizeX > 64) ? g_iGridSizeX*2 : 128),
		                                        MIN(600,(g_iGridSizeY > 64) ? g_iGridSizeY*2 : 128),
												IsDeviceAcceptable, ModifyDeviceSettings );

    // Pass control to the sample framework for handling the message pump and 
    // dispatching render calls. The sample framework will call your FrameMove 
    // and FrameRender callback when there is idle time between handling window messages.
    DXUTMainLoop();

    // Perform any application-level cleanup here. Direct3D device resources are released within the
    // appropriate callback functions and therefore don't require any cleanup code here.

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
	// Initialize members
	g_iGridSizeX	= g_size;
	g_iGridSizeY	= g_size;

	g_bWireFrame				= false;
	g_pMainShader				= NULL;
	g_pVB						= NULL;
	g_pVBGrid					= NULL;
	g_pIBGrid					= NULL;

	g_cluUCurrent				= NULL;
	g_cluULast					= NULL;
	g_cluRHS					= NULL;
#ifdef useUpacked 
	g_cluUNext					= NULL;
#else
	g_clRHS						= NULL;
	g_clUNext					= NULL;
#endif
	g_fDt						= 0;
	g_fDX						= 0;
	g_fDY						= 0;

	g_fMouseX					= 0;
	g_fMouseY					= 0;
	g_bMouseLDown				= false;
	g_bIsGUIEvent				= false;

    g_pCubePoolView             = NULL;

	g_fC						= 0.3f;

	g_iRainCounter				= 0;
	g_iRainDelay				= 10;

	g_bTimeclPerformance		= false;

	// Initialize UI
    g_HUD.SetCallback( OnGUIEvent ); int iY = 10; 
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, 125, 22 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 35, iY += 24, 125, 22 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 35, iY += 24, 125, 22 );
    g_HUD.AddButton( IDC_TOGGLEUI, L"Hide UI (u)", 35, iY += 24, 125, 22 );

	g_WaterUI.SetCallback( OnGUIEvent ); iY = 2;
	g_WaterUI.AddStatic( IDC_STATIC_VIS, _T("Velocity"),50, iY += 24, 100, 22);
	g_WaterUI.GetStatic( IDC_STATIC_VIS )->GetElement(0)->dwTextFormat = DT_LEFT|DT_TOP|DT_WORDBREAK;
    g_WaterUI.AddSlider( IDC_SLIDER_VIS, 50, iY += 24, 100, 22 );
    g_WaterUI.GetSlider( IDC_SLIDER_VIS )->SetRange( 10, 100 );
    g_WaterUI.GetSlider( IDC_SLIDER_VIS )->SetValue( int(g_fC*100) );
	g_WaterUI.AddCheckBox( IDC_CBOX_RAIN, _T("Rain"), 50, iY += 50, 125, 22,g_bRainMode );
    g_WaterUI.AddSlider( IDC_SLIDER_RAIN, 50, iY += 24, 100, 22 );
    g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->SetRange( 1, 100 );
	g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->SetEnabled(g_bRainMode);
    g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->SetValue( 100/g_iRainDelay );
}

HRESULT createCubeMap(IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, _TCHAR *face) {
	if (FAILED(DirectXUtils::checkLoadCubemap(_T("textures"), &g_pCubePoolView,pBackBufferSurfaceDesc->Format,pd3dDevice,512,face))) {
		MessageBox(NULL, _T("Unable to load pool cubemap"), _T("Error"),MB_OK);
		return E_FAIL;
	}
	return S_OK;
}


//--------------------------------------------------------------------------------------
// Called during device initialization, this code checks the device for some 
// minimum set of capabilities, and rejects those that don't pass by returning false.
//--------------------------------------------------------------------------------------
bool CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, 
                                  D3DFORMAT BackBufferFormat, bool bWindowed )
{
    IDirect3D9* pD3D = DXUTGetD3DObject(); 

   if(	pCaps->PixelShaderVersion >= D3DPS_VERSION(2,0) &&
		pCaps->VertexShaderVersion >= D3DVS_VERSION(2,0) &&
		DirectXUtils::isTextureFormatOk( D3DFMT_R32F, AdapterFormat, pD3D) &&
		DirectXUtils::isTextureFormatOk( D3DFMT_A32B32G32R32F, AdapterFormat, pD3D))
        return true;
    else
        return false;
	
}

void ClearSimulation() {
	g_cluULast->clear();
#ifdef useUpacked
	g_cluUNext->clear();
#else
	g_clUNext->clear();
#endif
	g_cluUCurrent->clear();
}

void CALLBACK OnMouseEvent( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, int xPos, int yPos ) {
	g_fMouseX = float(xPos)/g_iViewWidth;
	g_fMouseY = float(yPos)/g_iViewHeight;

	if (bRightButtonDown) ClearSimulation();

	g_bMouseLDown = bLeftButtonDown;
}

//--------------------------------------------------------------------------------------
// This callback function is called immediately before a device is created to allow the 
// application to modify the device settings. The supplied pDeviceSettings parameter 
// contains the settings that the framework has selected for the new device, and the 
// application can make any desired changes directly to this structure.  Note however that 
// the sample framework will not correct invalid device settings so care must be taken 
// to return valid device settings, otherwise IDirect3D9::CreateDevice() will fail.  
//--------------------------------------------------------------------------------------
void CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, const D3DCAPS9* pCaps )
{
    // If device doesn't support HW T&L or doesn't support 1.1 vertex shaders in HW 
    // then switch to SWVP.
    if( (pCaps->DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT) == 0 ||
         pCaps->VertexShaderVersion < D3DVS_VERSION(1,1) )
    {
        pDeviceSettings->BehaviorFlags = D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }
    else
    {
        pDeviceSettings->BehaviorFlags = D3DCREATE_HARDWARE_VERTEXPROCESSING;
    }

    // This application is designed to work on a pure device by not using 
    // IDirect3D9::Get*() methods, so create a pure device if supported and using HWVP.
    if ((pCaps->DevCaps & D3DDEVCAPS_PUREDEVICE) != 0 && 
        (pDeviceSettings->BehaviorFlags & D3DCREATE_HARDWARE_VERTEXPROCESSING) != 0 )
        pDeviceSettings->BehaviorFlags |= D3DCREATE_PUREDEVICE;

    // Debugging vertex shaders requires either REF or software vertex processing 
    // and debugging pixel shaders requires REF.  
#ifdef DEBUG_VS
    if( pDeviceSettings->DeviceType != D3DDEVTYPE_REF )
    {
        pDeviceSettings->BehaviorFlags &= ~D3DCREATE_HARDWARE_VERTEXPROCESSING;
        pDeviceSettings->BehaviorFlags &= ~D3DCREATE_PUREDEVICE;
        pDeviceSettings->BehaviorFlags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }
#endif
#ifdef DEBUG_PS
    pDeviceSettings->DeviceType = D3DDEVTYPE_REF;
#endif
}


//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has been 
// created, which will happen during application initialization and windowed/full screen 
// toggles. This is the best location to create D3DPOOL_MANAGED resources since these 
// resources need to be reloaded whenever the device is destroyed. Resources created  
// here should be released in the OnDestroyDevice callback. 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnCreateDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc )
{
    HRESULT hr;

	g_pd3dDevice = pd3dDevice;
	g_pBackBufferSurfaceDesc = pBackBufferSurfaceDesc;

    // Initialize the font
    V_RETURN( D3DXCreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                         OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                         L"Arial", &g_pFont ) );

    // Define DEBUG_VS and/or DEBUG_PS to debug vertex and/or pixel shaders with the 
    // shader debugger. Debugging vertex shaders requires either REF or software vertex 
    // processing, and debugging pixel shaders requires REF.  The 
    // D3DXSHADER_FORCE_*_SOFTWARE_NOOPT flag improves the debug experience in the 
    // shader debugger.  It enables source level debugging, prevents instruction 
    // reordering, prevents dead code elimination, and forces the compiler to compile 
    // against the next higher available software target, which ensures that the 
    // unoptimized shaders do not exceed the shader model limitations.  Setting these 
    // flags will cause slower rendering since the shaders will be unoptimized and 
    // forced into software.  See the DirectX documentation for more information about 
    // using the shader debugger.
    DWORD dwShaderFlags = 0;
    #ifdef DEBUG_VS
        dwShaderFlags |= D3DXSHADER_FORCE_VS_SOFTWARE_NOOPT;
    #endif
    #ifdef DEBUG_PS
        dwShaderFlags |= D3DXSHADER_FORCE_PS_SOFTWARE_NOOPT;
    #endif

    // Setup the camera's view parameters
    D3DXVECTOR3 vecEye(0.0f, 0.0f, -5.0f);
    D3DXVECTOR3 vecAt (0.0f, 0.0f, -0.0f);
    g_Camera.SetViewParams( &vecEye, &vecAt );

    return S_OK;
}



//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has been 
// reset, which will happen after a lost device scenario. This is the best location to 
// create D3DPOOL_DEFAULT resources since these resources need to be reloaded whenever 
// the device is lost. Resources created here should be released in the OnLostDevice 
// callback. 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnResetDevice( IDirect3DDevice9* pd3dDevice, 
                                const D3DSURFACE_DESC* pBackBufferSurfaceDesc )
{
    HRESULT hr;

	// kick out managed resources to make sure they are uploaded after the default data is loaded to the GPU
	pd3dDevice->EvictManagedResources();

	g_fDt = 1.0f;	g_fDX = 1.0f;	g_fDY = 1.0f;

	/*
	   Initialization of conjugate gradients objects
	 */	
	clClass::StartupCLFrameWork(pd3dDevice);

	// The pre-order code reserves memory on the GPU thus making texture 
	// storage more efficiente on some GPUs This code is optional,
	// if omitted memory is allocated when needed
	#ifdef optNVIDIA
		clUnpackedVector::preOrder(g_iGridSizeX,g_iGridSizeY,4+2);  // reserve code for 4 unpacked vectors of size g_iGridSizeX*g_iGridSizeY + two for temp purposes
		clCrNiMatrix::preOrder(g_iGridSizeX,g_iGridSizeY,1);
		clCGSolver::preOrder(g_iGridSizeX,g_iGridSizeY,CL_UNPACKED,1);
		clFloat::preOrder(1);
		clClass::ms_memoryMananger->createOrders(false);
	#else
		clUnpackedVector::preOrder(g_iGridSizeX,g_iGridSizeY,3+1);
		clPackedVector::preOrder(g_iGridSizeX,g_iGridSizeY,3+2);
		clPackedCrNiMatrix::preOrder(g_iGridSizeX,g_iGridSizeY,1);
		clCGSolver::preOrder(g_iGridSizeX,g_iGridSizeY,CL_PACKED,1);
		clFloat::preOrder(1);
		clClass::ms_memoryMananger->createOrders(true);
	#endif
	// END Pre-Order

	g_cluRHS      = new clCrNiVector(pd3dDevice,g_iGridSizeX,g_iGridSizeY);
	g_cluRHS->setSimulParam(g_fDt,g_fC,g_fDX,g_fDY);

	#ifdef useUpacked
		clCrNiMatrix *cnMatrix = new clCrNiMatrix(pd3dDevice,g_iGridSizeX,g_iGridSizeY,g_fDt,g_fC,g_fDX,g_fDY);
		g_cluUNext  = new clUnpackedVector(pd3dDevice,g_iGridSizeX,g_iGridSizeY); g_cluUNext->clear();
		g_pCGSolver = new clCGSolver(cnMatrix,g_cluUNext,g_cluRHS,CL_UNPACKED);
	#else
		clPackedCrNiMatrix *cnMatrix = new clPackedCrNiMatrix(pd3dDevice,g_iGridSizeX,g_iGridSizeY,g_fDt,g_fC,g_fDX,g_fDY);
		g_clUNext   = new clPackedVector(pd3dDevice,g_iGridSizeX,g_iGridSizeY); g_clUNext->clear();
		g_clRHS     = new clPackedVector(pd3dDevice,g_iGridSizeX,g_iGridSizeY);
		g_pCGSolver = new clCGSolver(cnMatrix,g_clUNext,g_clRHS,CL_PACKED);
	#endif

	g_cluULast    = new clUnpackedVector(pd3dDevice,g_iGridSizeX,g_iGridSizeY); g_cluULast->clear();
	g_cluUCurrent = new clUnpackedVector(pd3dDevice,g_iGridSizeX,g_iGridSizeY); g_cluUCurrent->clear();


	g_iViewWidth = pBackBufferSurfaceDesc->Width;
	g_iViewHeight = pBackBufferSurfaceDesc->Height;

    if( g_pFont ) V_RETURN( g_pFont->OnResetDevice() );

    // Create a sprite to help batch calls when drawing many lines of text
    V_RETURN( D3DXCreateSprite( pd3dDevice, &g_pTextSprite ) );

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( D3DX_PI/4, fAspectRatio, 0.1f, 1000.0f );
    g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width-170, 0 );
    g_HUD.SetSize( 170, 170 );
    g_WaterUI.SetLocation( pBackBufferSurfaceDesc->Width-170, pBackBufferSurfaceDesc->Height-150 );
    g_WaterUI.SetSize( 170, 150 );

	if FAILED(createCubeMap(pd3dDevice, pBackBufferSurfaceDesc, _T("pool.png"))) exit(-1);

    // Set up the textures
    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_MODULATE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP,   D3DTOP_MODULATE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE );
    pd3dDevice->SetSamplerState( 0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR );
    pd3dDevice->SetSamplerState( 0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR );
	pd3dDevice->SetRenderState( D3DRS_CULLMODE,  D3DCULL_NONE );

    // Set miscellaneous render states
    pd3dDevice->SetRenderState( D3DRS_DITHERENABLE,   FALSE );
    pd3dDevice->SetRenderState( D3DRS_SPECULARENABLE, FALSE );
    pd3dDevice->SetRenderState( D3DRS_AMBIENT,        0x000F0F0F );

    // Set the world matrix
    D3DXMATRIX matIdentity;
    D3DXMatrixIdentity( &matIdentity );
    pd3dDevice->SetTransform( D3DTS_WORLD,  &matIdentity );

    // Set up our view matrix. A view matrix can be defined given an eye point,
    // a point to lookat, and a direction for which way is up. Here, we set the
    // eye five units back along the z-axis and up three units, look at the
    // origin, and define "up" to be in the y-direction.
    D3DXVECTOR3 vFromPt   = D3DXVECTOR3( 0.0f, 0.0f, -5.0f );
    D3DXVECTOR3 vLookatPt = D3DXVECTOR3( 0.0f, 0.0f, 0.0f );
    D3DXVECTOR3 vUpVec    = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
    D3DXMatrixLookAtLH( &g_matView, &vFromPt, &vLookatPt, &vUpVec );
    pd3dDevice->SetTransform( D3DTS_VIEW, &g_matView );

    // Set the projection matrix
    D3DXMatrixOrthoLH( &g_matProj, 2, 2, 1.0f, 100.0f );
    pd3dDevice->SetTransform( D3DTS_PROJECTION, &g_matProj );

	g_mModelViewProj    = g_matView * g_matProj;

    // Create the single quad vertex buffer.
	V_RETURN(DirectXUtils::createFilledVertexBuffer(pd3dDevice,g_hMainQuad,sizeof(g_hMainQuad),TEXVERTEX2D::FVF,D3DPOOL_DEFAULT,g_pVB));

	// create shader
#ifdef optNVIDIA
    #define PS_PROFILE "ps_2_a"
#else
    #define PS_PROFILE "ps_2_0"
#endif
    LPCSTR profile[4] = {"PS_PROFILE", PS_PROFILE, 0, 0};
	V_RETURN(DirectXUtils::checkLoadShader(_T("main.fx"), g_pMainShader, pd3dDevice,_T("shader/"), 0, (D3DXMACRO*)profile));

	// do another check on the assembly technique
	D3DXHANDLE hDisplaceTechnique = g_pMainShader->GetTechniqueByName("tDisplacePass");
	V_RETURN( g_pMainShader->ValidateTechnique(hDisplaceTechnique));

	// create render surface for indirect result display with linear filtering
	V_RETURN(DirectXUtils::createRenderTexture( g_pBufferRenderSurface, g_pBufferTexture, g_pBufferTextureSurface, pd3dDevice, g_mBufferModelViewProj, g_iGridSizeX, g_iGridSizeY, D3DFMT_A8R8G8B8));

    return S_OK;
}

void setDrop(IDirect3DDevice9* pd3dDevice, float x, float y) {
	g_cluUCurrent->BeginScene();
		D3DXVECTOR4 vPosition = D3DXVECTOR4(x,1-y,0,0);
		g_pMainShader->SetTechnique("tInsertDrop");
		g_pMainShader->SetVector("f4Position",&vPosition);
		g_pMainShader->SetTexture("tHeightMap",	 g_cluUCurrent->m_pVectorTexture);
		pd3dDevice->SetStreamSource( 0, g_pVB, 0, sizeof(TEXVERTEX2D) );
		pd3dDevice->SetFVF( TEXVERTEX2D::FVF );

		UINT cPasses;
		g_pMainShader->Begin(&cPasses, 0);
			g_pMainShader->BeginPass(0);
				pd3dDevice->DrawPrimitive( D3DPT_TRIANGLESTRIP, 0, 2 );	
			g_pMainShader->EndPass();
		g_pMainShader->End();
	g_cluUCurrent->EndScene();
}

//--------------------------------------------------------------------------------------
// This callback function will be called once at the beginning of every frame. This is the
// best location for your application to handle updates to the scene, but is not 
// intended to contain actual rendering calls, which should instead be placed in the 
// OnFrameRender callback.  
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime )
{
    // Update the camera's position based on user input 
    g_Camera.FrameMove( fElapsedTime );

	pd3dDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID );

	pd3dDevice->BeginScene();

	if (g_bMouseLDown && !	g_bIsGUIEvent) setDrop(pd3dDevice,g_fMouseX,g_fMouseY);

	if (g_bRainMode) {
		if (g_iRainCounter) {
			g_iRainCounter--;
		} else {
			g_iRainCounter = g_iRainDelay;
			setDrop(pd3dDevice,(float)rand()/RAND_MAX,(float)rand()/RAND_MAX);
		}
	}

#ifdef useUpacked
	g_cluRHS->computeRHS(g_cluULast, g_cluUCurrent);
	g_iSteps = g_pCGSolver->solveNT(2);
	g_cluULast->copyVector(g_cluUCurrent);
	g_cluUCurrent->copyVector(g_cluUNext);
#else
	g_cluRHS->computeRHS(g_cluULast, g_cluUCurrent);
	g_clRHS->repack(g_cluRHS);
	g_iSteps = g_pCGSolver->solveNT(2);
	g_cluULast->copyVector(g_cluUCurrent);
	g_clUNext->unpack(g_cluUCurrent);
#endif

	pd3dDevice->EndScene();
}


//--------------------------------------------------------------------------------------
// This callback function will be called at the end of every frame to perform all the 
// rendering calls for the scene, and it will also be called if the window needs to be 
// repainted. After this function has returned, the sample framework will call 
// IDirect3DDevice9::Present to display the contents of the next buffer in the swap chain
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime )
{
	HRESULT hr;

	if (g_bTimeclPerformance) {
		pd3dDevice->BeginScene();

		clFloat timeFloat(pd3dDevice);

		double dStartTime = DXUTGetGlobalTimer()->GetTime();
		for (int i = 0; i<1000;i++)	timeVec1->reduceAdd(&timeFloat);
		double dReduceTime = DXUTGetGlobalTimer()->GetTime()-dStartTime;

		dStartTime = DXUTGetGlobalTimer()->GetTime();
		for (int i = 0; i<1000;i++) timeVec1->addVector(timeVec2,timeVec2);
		double dVectorOpTime = DXUTGetGlobalTimer()->GetTime()-dStartTime;

		TCHAR strTimings[1000];
		_stprintf(strTimings,_T("vector size %i %i\na vector reduce takes %gms\na vector-vector add takes %gms"),g_iGridSizeX,g_iGridSizeY,dReduceTime,dVectorOpTime);

		MessageBox(NULL,strTimings,_T("Timings"), MB_OK);

		g_bTimeclPerformance = false;

		ClearSimulation();

		pd3dDevice->EndScene();
		return;
	}

	// render the water surface to a temp buffer
	g_pBufferRenderSurface->BeginScene( g_pBufferTextureSurface, NULL);
		pd3dDevice->Clear( 0L, NULL, D3DCLEAR_TARGET, D3DCOLOR_RGBA(0,0,0,0), 1.0f, 0L );

		g_pMainShader->SetTechnique("tMainPass");

		D3DXVECTOR4 f4StepSize = D3DXVECTOR4(1.0f/(float)g_iGridSizeX, 1.0f/(float)g_iGridSizeY,0,0);

		UINT cPasses;
		g_pMainShader->Begin(&cPasses, 0);
			g_pMainShader->SetTexture("tHeightMap",g_cluUCurrent->m_pVectorTexture);
			g_pMainShader->SetTexture("tRefractMap", g_pCubePoolView);
			g_pMainShader->SetVector("f4StepSize", &f4StepSize);
			pd3dDevice->SetStreamSource( 0, g_pVB, 0, sizeof(TEXVERTEX2D) );
			pd3dDevice->SetFVF( TEXVERTEX2D::FVF );

			g_pMainShader->BeginPass(0);
				pd3dDevice->DrawPrimitive( D3DPT_TRIANGLESTRIP, 0, 2 );
			g_pMainShader->EndPass();
		g_pMainShader->End();
	g_pBufferRenderSurface->EndScene( 0 );

	if (g_bWireFrame) pd3dDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME );

	// Begin the scene
    if( SUCCEEDED( pd3dDevice->BeginScene() ) ) {

		// Clear the viewport
		pd3dDevice->Clear( 0L, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, D3DCOLOR_RGBA(0,0,0,0), 1.0f, 0L );

        pd3dDevice->SetTexture( 0, g_pBufferTexture );
        pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_SELECTARG1 );
        pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
        pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP,   D3DTOP_DISABLE );
		pd3dDevice->SetSamplerState( 0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR );
		pd3dDevice->SetSamplerState( 0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR );
		pd3dDevice->SetSamplerState( 0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP );
		pd3dDevice->SetSamplerState( 0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP );

		pd3dDevice->SetStreamSource( 0, g_pVB, 0, sizeof(TEXVERTEX2D) );
		pd3dDevice->SetFVF( TEXVERTEX2D::FVF );
		pd3dDevice->DrawPrimitive( D3DPT_TRIANGLESTRIP, 0, 2 );

		pd3dDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID );

        RenderText(fElapsedTime);
		RenderUI(fElapsedTime);

        V( pd3dDevice->EndScene() );
    }
}

//--------------------------------------------------------------------------------------
// Render the UI elements such as sliders buttons etc.
//--------------------------------------------------------------------------------------
HRESULT RenderUI(float fElapsedTime)
{
	HRESULT hr;

	if (!g_bShowUI) return S_OK;

    V( g_HUD.OnRender( fElapsedTime ) );
    V( g_WaterUI.OnRender( fElapsedTime ) );

	return S_OK;
}

//--------------------------------------------------------------------------------------
// Render the help and statistics text. This function uses the ID3DXFont interface for 
// efficient text rendering.
//--------------------------------------------------------------------------------------
void RenderText(float fElapsedTime)
{
	if (!g_bShowText) return;

    // The helper object simply helps keep track of text position, and color
    // and then it calls pFont->DrawText( g_pSprite, strMsg, -1, &rc, DT_NOCLIP, g_clr );
    // If NULL is passed in as the sprite object, then it will work however the 
    // pFont->DrawText() will not be batched together.  Batching calls will improves performance.
    const D3DSURFACE_DESC* pd3dsdBackBuffer = DXUTGetBackBufferSurfaceDesc();
    CDXUTTextHelper txtHelper( g_pFont, g_pTextSprite, 15 );

    // Output statistics
    txtHelper.Begin();
    txtHelper.SetInsertionPos( 5, 5 );
    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    txtHelper.DrawTextLine( DXUTGetFrameStats() );
    txtHelper.DrawTextLine( DXUTGetDeviceStats() );

    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
	TCHAR test[100];

	_sntprintf(test,99,_T("Gridsize %i x %i"),g_iGridSizeX,g_iGridSizeY);
    txtHelper.DrawTextLine(test);

    
    // Draw help
    if( g_bShowHelp )
    {
        txtHelper.SetInsertionPos( 10, pd3dsdBackBuffer->Height-15*6 );
        txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 0.75f, 0.0f, 1.0f ) );
        txtHelper.DrawTextLine( L"Controls (F1 to hide):" );

        txtHelper.SetInsertionPos( 40, pd3dsdBackBuffer->Height-15*5 );
        txtHelper.DrawTextLine( L"Toggle UI Display: u\n"
								L"Toggle Text Display: t\n"
								L"Toggle Rain: SPACE\n"
                                L"Quit: ESC" );
    }
    else
    {
        txtHelper.SetInsertionPos( 10, pd3dsdBackBuffer->Height-15*2 );
        txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
        txtHelper.DrawTextLine( L"Press F1 for help" );
    }
    txtHelper.End();
}


//--------------------------------------------------------------------------------------
// Before handling window messages, the sample framework passes incoming windows 
// messages to the application through this callback function. If the application sets 
// *pbNoFurtherProcessing to TRUE, then the sample framework will not process this message.
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing )
{
	g_bIsGUIEvent = true;

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing ) return 0;

    *pbNoFurtherProcessing = g_WaterUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing ) return 0;

	g_bIsGUIEvent = false;

    // Pass all remaining windows messages to camera so it can respond to user input
    g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

    return 0;
}


//--------------------------------------------------------------------------------------
// As a convenience, the sample framework inspects the incoming windows messages for
// keystroke messages and decodes the message parameters to pass relevant keyboard
// messages to the application.  The framework does not remove the underlying keystroke 
// messages, which are still passed to the application's MsgProc callback.
//--------------------------------------------------------------------------------------
void CALLBACK KeyboardProc( UINT nChar, bool bKeyDown, bool bAltDown )
{
    if( bKeyDown )
    {
        switch( nChar )
        {
            case VK_F1	: g_bShowHelp = !g_bShowHelp; break;

			case 'W'    : g_bWireFrame = !g_bWireFrame; break;

			case ' '	: g_WaterUI.EnableNonUserEvents(true);
						  g_WaterUI.GetCheckBox(IDC_CBOX_RAIN)->SetChecked(!g_bRainMode);
						  g_WaterUI.EnableNonUserEvents(false);
				          break;

			case 'U'	: g_bShowUI = !g_bShowUI; break;

			case 'T'	: g_bShowText = !g_bShowText; break;

			case 'X'	: g_iGridSizeX = (bAltDown) ? g_iGridSizeX*2 : g_iGridSizeX/2; ResizeGrid();break;
			case 'Y'	: g_iGridSizeY = (bAltDown) ? g_iGridSizeY*2 : g_iGridSizeY/2; ResizeGrid();break;

			case VK_DOWN : g_iRainDelay = MIN(g_iRainDelay+1,100); g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->SetValue(100/g_iRainDelay); break;
			case VK_UP   : g_iRainDelay = MAX(g_iRainDelay-1,1);   g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->SetValue(100/g_iRainDelay); break;

			case VK_RIGHT : g_fC = MIN(g_fC*1.1f,1); ChangeViscosity(int(g_fC*100)); g_WaterUI.GetSlider( IDC_SLIDER_VIS )->SetValue(int(g_fC*100)); break;
			case VK_LEFT  : g_fC = MAX(g_fC/1.1f,0.1f); ChangeViscosity(int(g_fC*100)); g_WaterUI.GetSlider( IDC_SLIDER_VIS )->SetValue(int(g_fC*100)); break;

			case 'P'	 : g_bTimeclPerformance = true;
		}
    }
}

void ResizeGrid() {
	OnLostDevice();
	OnDestroyDevice();
	OnCreateDevice(g_pd3dDevice, g_pBackBufferSurfaceDesc);
	OnResetDevice(g_pd3dDevice, g_pBackBufferSurfaceDesc);
}

void ChangeViscosity(int iValue) {
	g_fC = iValue / 100.0f;

#ifdef useUpacked
	clCrNiMatrix *cnMatrix = static_cast<clCrNiMatrix*>(g_pCGSolver->getMatrix());
#else
	clPackedCrNiMatrix *cnMatrix = static_cast<clPackedCrNiMatrix*>(g_pCGSolver->getMatrix());
#endif

	cnMatrix->setC(g_fC);
	g_cluRHS->setC(g_fC);
}

void ChangeRain(int iValue) {
	g_iRainDelay = int(100/iValue);
}

void ToggleRain(bool bIsRain) {
	g_bRainMode = bIsRain;
	g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->SetEnabled(bIsRain);
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl )
{
    switch( nControlID )
    {
        case IDC_TOGGLEFULLSCREEN:	DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:			DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:		DXUTSetShowSettingsDialog( !DXUTGetShowSettingsDialog() ); break;
		case IDC_TOGGLEUI:			g_bShowUI = !g_bShowUI;break;
		case IDC_CBOX_RAIN:			ToggleRain(g_WaterUI.GetCheckBox(IDC_CBOX_RAIN)->GetChecked() );break;
		case IDC_SLIDER_VIS:		ChangeViscosity(g_WaterUI.GetSlider( IDC_SLIDER_VIS )->GetValue());
		case IDC_SLIDER_RAIN:		ChangeRain(g_WaterUI.GetSlider( IDC_SLIDER_RAIN )->GetValue());
    }

	
}


//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has 
// entered a lost state and before IDirect3DDevice9::Reset is called. Resources created
// in the OnResetDevice callback should be released here, which generally includes all 
// D3DPOOL_DEFAULT resources. See the "Lost Devices" section of the documentation for 
// information about lost devices.
//--------------------------------------------------------------------------------------
void CALLBACK OnLostDevice()
{
    if( g_pFont ) g_pFont->OnLostDevice();
    SAFE_RELEASE( g_pTextSprite );

	SAFE_RELEASE( g_pVB );
	SAFE_RELEASE( g_pVBGrid );
	SAFE_RELEASE( g_pIBGrid );

	SAFE_RELEASE( g_pMainShader );

	SAFE_RELEASE( g_pBufferRenderSurface );
	SAFE_RELEASE( g_pBufferTexture );
	SAFE_RELEASE( g_pBufferTextureSurface );

 	SAFE_DELETE( g_pCGSolver ); 	// deleting the solver implitly deletes "g_clUNext" and "g_clRHS"
#ifndef useUpacked
	SAFE_DELETE( g_cluRHS );
#endif
	SAFE_DELETE( g_cluUCurrent );
	SAFE_DELETE( g_cluULast );

	SAFE_RELEASE( g_pCubePoolView );

	clClass::ShutdownCLFrameWork();
}

//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has 
// been destroyed, which generally happens as a result of application termination or 
// windowed/full screen toggles. Resources created in the OnCreateDevice callback 
// should be released here, which generally includes all D3DPOOL_MANAGED resources. 
//--------------------------------------------------------------------------------------
void CALLBACK OnDestroyDevice()
{
    SAFE_RELEASE( g_pFont );
}
