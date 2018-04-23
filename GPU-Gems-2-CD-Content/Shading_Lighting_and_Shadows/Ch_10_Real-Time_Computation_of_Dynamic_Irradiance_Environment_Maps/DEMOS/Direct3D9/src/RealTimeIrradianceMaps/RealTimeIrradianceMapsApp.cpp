#define STRICT
#include "nvafx.h"
#include "RealTimeIrradianceMapsApp.h"
#include <shared/GetFilePath.h>

const struct { LPDIRECT3DCUBETEXTURE9 *cubeTex; LPCWSTR name; } g_DisplayCubemaps[3] =
{
    { &g_pBaseCubeMap, L"Original Cubemap" },
    { &g_pDiffuseCubeMap, L"Diffuse-convolved Cubemap" },
    { &g_pSpecularCubeMap, L"Specular-convolved (s=9) Cubemap" }
};

const struct { D3DXHANDLE technique; LPCWSTR name; } g_LightingMethods[2] =
{
    { "RenderCubemap", L"Cubemap Diffuse+Specular Lighting" },
    { "RenderDirectional", L"Directional Diffuse+Specular Lighting" }
};

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
INT WINAPI WinMain( HINSTANCE, HINSTANCE, LPSTR, int )
{
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

    // Show the cursor and clip it when in full screen
    DXUTSetCursorSettings( true, true );

    InitApp();

    // Initialize the sample framework and create the desired Win32 window and Direct3D 
    // device for the application. Calling each of these functions is optional, but they
    // allow you to set several options which control the behavior of the framework.
    DXUTInit( true, true, true ); // Parse the command line, handle the default hotkeys, and show msgboxes
    DXUTCreateWindow( L"RealTimeIrradianceMaps" );
    DXUTCreateDevice( D3DADAPTER_DEFAULT, true, 512, 512, IsDeviceAcceptable, ModifyDeviceSettings );

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
    // Initialize dialogs
    g_HUD.SetCallback( OnGUIEvent );
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, 34, 125, 22 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 35, 58, 125, 22 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 35, 82, 125, 22 );
    g_HUD.AddButton( IDC_CYCLECUBEMAPS, L"Cycle Cubemaps", 35, 106, 125, 22 );
    g_HUD.AddButton( IDC_CYCLELIGHTING, L"Cycle Lighting", 35, 130, 125, 22 );
//    g_HUD.AddButton( IDC_DEBUGIMAGE, L"Toggle Debug Image", 35, 154, 125, 22 );
}


//--------------------------------------------------------------------------------------
// Called during device initialization, this code checks the device for some 
// minimum set of capabilities, and rejects those that don't pass by returning false.
//--------------------------------------------------------------------------------------
bool CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, 
                                  D3DFORMAT BackBufferFormat, bool bWindowed )
{
    bool bCapsAcceptable = true;

    bool bAcceptableSM30 = true;
    //  code sample needs shader model 3.0 support
    bAcceptableSM30 &= (pCaps->PixelShaderVersion >= D3DPS_VERSION(3,0));
    bAcceptableSM30 &= (pCaps->VertexShaderVersion >= D3DVS_VERSION(3,0));
    bAcceptableSM30 &= (pCaps->MaxPShaderInstructionsExecuted >= 16384);
    bCapsAcceptable &= bAcceptableSM30;

    if (!bAcceptableSM30)
        DO_ONCE(ALERT("Demo requires a graphics device with shader model 3.0\nsupport and a executed pixel shader length of at least 16384"));

    LPDIRECT3D9 pD3D = DXUTGetD3DObject();

    bool bAcceptableFP16 = true;
    //  code sample needs A16B16G16R16F filtering & render target support for cubemaps and 2D textures
    bAcceptableFP16 &= SUCCEEDED( pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, pCaps->DeviceType, AdapterFormat, D3DUSAGE_RENDERTARGET, D3DRTYPE_CUBETEXTURE, D3DFMT_A16B16G16R16F) );
    bAcceptableFP16 &= SUCCEEDED( pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, pCaps->DeviceType, AdapterFormat, D3DUSAGE_QUERY_FILTER, D3DRTYPE_CUBETEXTURE, D3DFMT_A16B16G16R16F) );
    bAcceptableFP16 &= SUCCEEDED( pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, pCaps->DeviceType, AdapterFormat, D3DUSAGE_RENDERTARGET, D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16F) );
    bAcceptableFP16 &= SUCCEEDED( pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, pCaps->DeviceType, AdapterFormat, D3DUSAGE_QUERY_FILTER, D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16F) );
    bCapsAcceptable &= bAcceptableFP16;

    if (!bAcceptableFP16)
        DO_ONCE(ALERT("Demo requires a graphics device with 16-bit floating point\n(A16B16G16R16F) filtering and rendering for 2D and cubemap textures"));

    bool bAcceptableFP32 = true;
    //  code sample needs point-sampled R32F texture support
    bAcceptableFP32 &= SUCCEEDED( pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, pCaps->DeviceType, AdapterFormat, 0, D3DRTYPE_TEXTURE, D3DFMT_R32F) );
    bCapsAcceptable &= bAcceptableFP32;

    if (!bAcceptableFP32)
        DO_ONCE(ALERT("Demo requires a graphics device with 32-bit floating point (R32F) texturing support"));

    return bCapsAcceptable;
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

    // Initialize the font
    V_RETURN( D3DXCreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                         OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                         L"Arial", &g_pFont ) );

    // Set up our view matrix. A view matrix can be defined given an eye point and
    // a point to lookat. Here, we set the eye five units back along the z-axis and 
	// up three units and look at the origin.
    D3DXVECTOR3 vFromPt   = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 vLookatPt = D3DXVECTOR3(0.0f, 0.0f,-1.0f);
	g_Camera.SetViewParams( &vFromPt, &vLookatPt);
    g_Camera.SetProjParams( 60.f, pBackBufferSurfaceDesc->Width/pBackBufferSurfaceDesc->Height, 1.f, 10000.f );

    g_pCubeMesh = new CubeMesh();
    g_pFSQuadMesh = new FSQuadMesh();

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
    // TODO: setup render states

	HRESULT hr;

    V_RETURN( pd3dDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &g_pBackBuffer) );
    V_RETURN( pd3dDevice->GetDepthStencilSurface(&g_pZBuffer) );

    V_RETURN( pd3dDevice->CreateTexture(PARABOLOID_SAMPLES, PARABOLOID_SAMPLES, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &g_pParaboloidMap[0], NULL) );
    V_RETURN( pd3dDevice->CreateTexture(PARABOLOID_SAMPLES, PARABOLOID_SAMPLES, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &g_pParaboloidMap[1], NULL) );
    V_RETURN( pd3dDevice->CreateCubeTexture(NUM_RESULT_SAMPLES, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &g_pDiffuseCubeMap, NULL) );
    V_RETURN( pd3dDevice->CreateCubeTexture(NUM_RESULT_SAMPLES, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &g_pSpecularCubeMap, NULL) );
    V_RETURN( pd3dDevice->CreateTexture(NUM_ORDER_P2, NUM_ORDER_P2, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &g_pIrradianceSHCoefficients, NULL) );

    D3DVERTEXELEMENT9 nvbDecl[] = {
        { 0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
        { 0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
        { 0, 24, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
        { 0, 28, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
        { 0, 36, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
        { 0, 48, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 }, 
        { 0, 60, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 }, 
        D3DDECL_END()
    };

    V_RETURN( pd3dDevice->CreateVertexDeclaration(nvbDecl, &g_pDragonVBDecl) );

    
    if( g_pFont )
        V_RETURN( g_pFont->OnResetDevice() );

    if ( g_pCubeMesh )
        V_RETURN( g_pCubeMesh->RestoreDeviceObjects(pd3dDevice) );

    if ( g_pFSQuadMesh )
        V_RETURN( g_pFSQuadMesh->RestoreDeviceObjects(pd3dDevice) );

    // Create a sprite to help batch calls when drawing many lines of text
    V_RETURN( D3DXCreateSprite( pd3dDevice, &g_pTextSprite ) );

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( D3DX_PI/4, fAspectRatio, 0.1f, 1000.0f );
    g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

    g_HUD.SetLocation( 0, 0 );
    g_HUD.SetSize( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );

	int iY = 15;
    g_HUD.GetControl( IDC_TOGGLEFULLSCREEN )->SetLocation( pBackBufferSurfaceDesc->Width - 135, iY);
    g_HUD.GetControl( IDC_TOGGLEREF )->SetLocation( pBackBufferSurfaceDesc->Width - 135, iY += 24 );
    g_HUD.GetControl( IDC_CHANGEDEVICE )->SetLocation( pBackBufferSurfaceDesc->Width - 135, iY += 24 );
    g_HUD.GetControl( IDC_CYCLECUBEMAPS )->SetLocation( pBackBufferSurfaceDesc->Width - 135, iY += 24 );
    g_HUD.GetControl( IDC_CYCLELIGHTING )->SetLocation( pBackBufferSurfaceDesc->Width - 135, iY += 24 );
//    g_HUD.GetControl( IDC_DEBUGIMAGE )->SetLocation( pBackBufferSurfaceDesc->Width - 135, iY += 24 );

    pd3dDevice->SetRenderState(D3DRS_ZENABLE, TRUE);

    //V_RETURN(D3DXCreateCubeTextureFromFile( pd3dDevice, GetFilePath::GetFilePath(_T("textures\\cubemaps\\CloudyHillsCubeMap.dds")).c_str(), &g_pBaseCubeMap ));
V_RETURN(D3DXCreateCubeTextureFromFile( pd3dDevice, GetFilePath::GetFilePath(_T("textures\\cubemaps\\nvlobby_new_cube_mipmap.dds")).c_str(), &g_pBaseCubeMap ));

    V_RETURN(D3DXCreateEffectFromFile( pd3dDevice, GetFilePath::GetFilePath(_T("programs\\RealTimeIrradianceMaps\\ConvolveIrradiance.fx")).c_str(), NULL, NULL, D3DXSHADER_SKIPOPTIMIZATION, NULL, &g_pConvolveEffect, NULL ));

    V_RETURN(D3DXCreateEffectFromFile( pd3dDevice, GetFilePath::GetFilePath(_T("programs\\RealTimeIrradianceMaps\\Simple.fx")).c_str(), NULL, NULL, 0, NULL, &g_pDisplayEffect, NULL ));

    V_RETURN(D3DXCreateEffectFromFile( pd3dDevice, GetFilePath::GetFilePath(_T("programs\\RealTimeIrradianceMaps\\RenderParaboloid.fx")).c_str(), NULL, NULL, D3DXSHADER_SKIPOPTIMIZATION, NULL, &g_pParaboloidEffect, NULL ));

    V_RETURN(D3DXCreateEffectFromFile( pd3dDevice, GetFilePath::GetFilePath(_T("programs\\RealTimeIrradianceMaps\\Dragon.fx")).c_str(), NULL, NULL, 0, NULL, &g_pDragonEffect, NULL ));

    V_RETURN(BuildDualParaboloidWeightTextures(g_pParaboloidSHWeights, pd3dDevice, NUM_ORDER, NUM_RADIANCE_SAMPLES));

    V_RETURN(BuildLambertIrradianceTextures(g_pLambertSHEval, pd3dDevice, NUM_ORDER, NUM_RESULT_SAMPLES));

    g_pDragon = new NVBScene();
    V_RETURN(g_pDragon->Load(_T("RocketCar.nvb"), pd3dDevice, GetFilePath::GetFilePath));

    const FLOAT specExponent = 9.0f;
    V_RETURN(BuildPhongIrradianceTextures(g_pPhongSHEval, pd3dDevice, NUM_ORDER, NUM_RESULT_SAMPLES, specExponent));

    return S_OK;
}


//--------------------------------------------------------------------------------------
// This callback function will be called once at the beginning of every frame. This is the
// best location for your application to handle updates to the scene, but is not 
// intended to contain actual rendering calls, which should instead be placed in the 
// OnFrameRender callback.  
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime )
{
    // TODO: update world
    g_Camera.FrameMove( fElapsedTime );
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

    const D3DXVECTOR3 dragonXlate(0.f, 0.f, 350.f);

    // Set the world matrix
    pd3dDevice->SetTransform(D3DTS_WORLD, g_Camera.GetWorldMatrix());

    static float time = 0.f;
    static bool  forward = true;

    
    time += (forward) ? fElapsedTime/0.0333333f : -fElapsedTime/0.0333333f;
    if ( time > g_pDragon->m_NumMeshKeys - 1 )
    {
        forward = false;
        time = g_pDragon->m_NumMeshKeys - 2.f;
    }
    else if ( time < 1 )
        forward = true;

    g_pDragon->Update( time, NULL );

    D3DXMATRIX dragonViewMatrix;  D3DXMatrixTranslation(&dragonViewMatrix, dragonXlate.x, dragonXlate.y, dragonXlate.z);
    D3DXMATRIX viewProjectionMatrix;
    D3DXMatrixMultiply( &viewProjectionMatrix, g_Camera.GetWorldMatrix(), g_Camera.GetViewMatrix() );
    D3DXMatrixMultiply( &viewProjectionMatrix, &viewProjectionMatrix, &dragonViewMatrix );

    D3DXMATRIX viewInverse; D3DXMatrixInverse( &viewInverse, NULL, &viewProjectionMatrix );
    
    D3DXVECTOR4 eyePosition( -viewInverse._41, -viewInverse._42, -viewInverse._43, 1.f );
    D3DXMatrixMultiply( &viewProjectionMatrix, &viewProjectionMatrix, g_Camera.GetProjMatrix() );

    // Set the projection matrix
    pd3dDevice->SetTransform(D3DTS_PROJECTION, g_Camera.GetProjMatrix());

	// Set the view matrix
	pd3dDevice->SetTransform(D3DTS_VIEW, g_Camera.GetViewMatrix());

	// Begin the scene
    if (SUCCEEDED(pd3dDevice->BeginScene()))
    {
        //  Project cubemap into dual-paraboloid map
        UINT uPasses;

        ALERT_RETURN( RenderParaboloidEnvMap(pd3dDevice), "Failed to render dual-paraboloid environment map" );
        ALERT_RETURN( ProjectParaboloidToSH(pd3dDevice), "Failed to project dual-paraboloid map to spherical harmonics" );
        ALERT_RETURN( EvaluateConvolvedSH(pd3dDevice, g_pDiffuseCubeMap, g_pLambertSHEval), "Failed to evaluate diffuse cube map" );
        ALERT_RETURN( EvaluateConvolvedSH(pd3dDevice, g_pSpecularCubeMap, g_pPhongSHEval), "Failed to evaluate specular cube map" );

        ALERT_RETURN(pd3dDevice->SetRenderTarget(0, g_pBackBuffer), "Failed to set render target to primary display");
        ALERT_RETURN(pd3dDevice->SetDepthStencilSurface(g_pZBuffer), "Failed to set zbuffer to primary back buffer");
        // Clear the viewport
        pd3dDevice->Clear(0L, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0x000000ff, 1.0f, 0L);

        ALERT_RETURN(g_pDisplayEffect->SetTechnique("SimpleCubeMapRender"), "Failed to set technique: SimpleCubeLookup");
        ALERT_RETURN(g_pDisplayEffect->SetTexture( "CubeTexture", *(g_DisplayCubemaps[g_uDisplayCubeMap].cubeTex) ), "Failed to set texture: CubeTexture");

        if ( SUCCEEDED(g_pDisplayEffect->Begin(&uPasses, 0)) )
        {
            for ( UINT i=0; i<uPasses; i++ )
            {
                g_pDisplayEffect->BeginPass(i);
                ALERT_RETURN(g_pCubeMesh->Draw(pd3dDevice), "Failed to draw cube mesh");
                g_pDisplayEffect->EndPass();
            }
            g_pDisplayEffect->End();
        }

        ALERT_RETURN(g_pDragonEffect->SetTechnique(g_LightingMethods[g_uTechnique].technique), "Failed to set rendering technique");
        ALERT_RETURN(g_pDragonEffect->SetTexture("DiffuseIrradiance", g_pDiffuseCubeMap), "Failed to set texture: DiffuseIrradiance");
        ALERT_RETURN(g_pDragonEffect->SetTexture("SpecularIrradiance", g_pSpecularCubeMap), "Failed to set texture: SpecularIrradiance");
        ALERT_RETURN(g_pDragonEffect->SetTexture("EnvironmentReflection", g_pBaseCubeMap), "Failed to set texture: EnvironmentReflection");
        ALERT_RETURN(g_pDragonEffect->SetVector("cEyePosition", &eyePosition), "Failed to set eye position");
        ALERT_RETURN(g_pDragonEffect->SetMatrix("cViewProjection", &viewProjectionMatrix), "Failed to set view projection matrix");
        ALERT_RETURN(pd3dDevice->SetVertexDeclaration(g_pDragonVBDecl), "Failed to set vertex declaration");

        for ( UINT i=0; i<g_pDragon->m_NumMeshes; i++ )
        {
            const NVBScene::Mesh& mesh = g_pDragon->m_Meshes[i];
            D3DXMATRIX dragonXFormIT; D3DXMatrixInverse( &dragonXFormIT, NULL, &mesh.m_Transform ); D3DXMatrixTranspose(&dragonXFormIT, &dragonXFormIT);
            g_pDragonEffect->SetMatrix("cWorld", &mesh.m_Transform);
            g_pDragonEffect->SetMatrix("cWorldIT", &dragonXFormIT);
            g_pDragonEffect->SetTexture("DiffuseTexture", mesh.m_DiffuseMap);

            UINT uPasses;

            if ( SUCCEEDED(g_pDragonEffect->Begin(&uPasses, 0)) )
            {
                g_pDragonEffect->BeginPass(0);
                ALERT_RETURN(mesh.Draw(), "Failed to draw dragon sub-mesh");
                g_pDragonEffect->EndPass();
                g_pDragonEffect->End();
            }
        }

        if ( g_bShowDebug )
            ALERT_RETURN(DisplayDebugImage(pd3dDevice, g_pParaboloidMap[0]), "Failed to draw debug image");

        pd3dDevice->SetTexture( 0, NULL );

        // Render stats and help text  
        RenderText();

		V( g_HUD.OnRender( fElapsedTime ) );

        // End the scene.
        V( pd3dDevice->EndScene() );
    }
}

HRESULT RenderParaboloidEnvMap( LPDIRECT3DDEVICE9 pd3dDevice )
{
    UINT uPasses;
    HRESULT hr = S_OK;

    for ( UINT face=0; face<2; face++ )
    {
        LPDIRECT3DSURFACE9 pRTSurf = NULL;
        V_RETURN(g_pParaboloidMap[face]->GetSurfaceLevel( 0, &pRTSurf ));
        V_RETURN(pd3dDevice->SetRenderTarget(0, pRTSurf));
        V_RETURN(pd3dDevice->SetDepthStencilSurface(NULL));
        
        pd3dDevice->Clear(0L, NULL, D3DCLEAR_TARGET, D3DCOLOR_ARGB(0xff,0x7f,0x7f,0xff), 1.0f, 0L);
        V_RETURN(g_pParaboloidEffect->SetTechnique("ConvertHemisphere"));
        V_RETURN(g_pParaboloidEffect->SetTexture("CubeMap", g_pBaseCubeMap));

        D3DXVECTOR4 vDirectionVec(0.f, 0.f, (face)?1.f : -1.f, 1.f);
        V_RETURN(g_pParaboloidEffect->SetVector("DirectionVec", &vDirectionVec));

        if ( SUCCEEDED(g_pParaboloidEffect->Begin(&uPasses, 0)) )
        {
            for ( UINT i=0; i<uPasses; i++ )
            {
                V_RETURN(g_pParaboloidEffect->BeginPass(i));
                V_RETURN(g_pFSQuadMesh->Draw(pd3dDevice, FALSE));
                V_RETURN(g_pParaboloidEffect->EndPass());
            }
            g_pParaboloidEffect->End();
        }
        pRTSurf->Release();
    }
    return S_OK;
}

HRESULT ProjectParaboloidToSH( LPDIRECT3DDEVICE9 pd3dDevice )
{
    UINT uPasses;
    HRESULT hr = S_OK;

    LPDIRECT3DSURFACE9 pRTSurf = NULL;
    V_RETURN(g_pIrradianceSHCoefficients->GetSurfaceLevel(0, &pRTSurf));
    V_RETURN(pd3dDevice->SetRenderTarget(0, pRTSurf));
    V_RETURN(pd3dDevice->SetDepthStencilSurface(NULL));
    pd3dDevice->Clear(0L, NULL, D3DCLEAR_TARGET, 0x00000000, 1.f, 0L);
    
    V_RETURN(g_pConvolveEffect->SetTechnique("ProjectDualParaboloidToSH"));
    V_RETURN(g_pConvolveEffect->SetTexture("SH_Convolve_dE_0", g_pParaboloidMap[0]));
    V_RETURN(g_pConvolveEffect->SetTexture("SH_Convolve_dE_1", g_pParaboloidMap[1]));
    V_RETURN(g_pConvolveEffect->SetTexture("SH_Convolve_Ylm_dW_0", g_pParaboloidSHWeights[0]));
    V_RETURN(g_pConvolveEffect->SetTexture("SH_Convolve_Ylm_dW_1", g_pParaboloidSHWeights[1]));

    if ( SUCCEEDED(g_pConvolveEffect->Begin(&uPasses, 0)) )
    {
        for ( UINT i=0; i<uPasses; i++ )
        {
            V_RETURN(g_pConvolveEffect->BeginPass(i));
            V_RETURN(g_pFSQuadMesh->Draw(pd3dDevice, FALSE));
            V_RETURN(g_pConvolveEffect->EndPass());
        }
        g_pConvolveEffect->End();
    }
    pRTSurf->Release();
    return S_OK;
}

HRESULT EvaluateConvolvedSH( LPDIRECT3DDEVICE9 pd3dDevice, LPDIRECT3DCUBETEXTURE9 pResultCube, const LPDIRECT3DTEXTURE9* pEvalSHFunction )
{
    UINT uPasses;
    HRESULT hr = S_OK;

    for ( UINT face=0; face<6; face++ )
    {
        LPDIRECT3DSURFACE9 pRTSurf = NULL;
        V_RETURN(pResultCube->GetCubeMapSurface((D3DCUBEMAP_FACES)face, 0, &pRTSurf));
        V_RETURN(pd3dDevice->SetRenderTarget(0, pRTSurf));
        V_RETURN(pd3dDevice->SetDepthStencilSurface(NULL));
        pd3dDevice->Clear(0L, NULL, D3DCLEAR_TARGET, 0x00000000, 1.f, 0L);

        const D3DXHANDLE EvalTextureNames[6] = { "SH_Integrate_Ylm_Al_xpos",
                                                 "SH_Integrate_Ylm_Al_xneg",
                                                 "SH_Integrate_Ylm_Al_ypos",
                                                 "SH_Integrate_Ylm_Al_yneg",
                                                 "SH_Integrate_Ylm_Al_zpos",
                                                 "SH_Integrate_Ylm_Al_zneg" };
        
        V_RETURN(g_pConvolveEffect->SetTechnique("EvaluateSHCubemap"));
        V_RETURN(g_pConvolveEffect->SetTexture(EvalTextureNames[face], pEvalSHFunction[face]));
        V_RETURN(g_pConvolveEffect->SetTexture("SH_Coefficients", g_pIrradianceSHCoefficients));
        
        if ( SUCCEEDED(g_pConvolveEffect->Begin(&uPasses,0)) )
        {
            DBG_ASSERT( face<uPasses );
            V_RETURN(g_pConvolveEffect->BeginPass(face));
            V_RETURN(g_pFSQuadMesh->Draw(pd3dDevice, FALSE));
            V_RETURN(g_pConvolveEffect->EndPass());
            g_pConvolveEffect->End();
        }
        pRTSurf->Release();
    }
    return S_OK;
}

HRESULT DisplayDebugImage( LPDIRECT3DDEVICE9 pd3dDevice, LPDIRECT3DTEXTURE9 pTexture )
{
    UINT uPasses;
    HRESULT hr = S_OK;
    V_RETURN(pd3dDevice->SetRenderTarget(0, g_pBackBuffer));
    V_RETURN(pd3dDevice->SetDepthStencilSurface(g_pZBuffer));
    V_RETURN(g_pDisplayEffect->SetTechnique("Simple2DRender"));
    V_RETURN(g_pDisplayEffect->SetTexture("CubeTexture", pTexture));

    if ( SUCCEEDED(g_pDisplayEffect->Begin(&uPasses, 0)) )
    {
        V_RETURN(g_pDisplayEffect->BeginPass(0));
        V_RETURN(g_pFSQuadMesh->Draw(pd3dDevice, TRUE));
        V_RETURN(g_pDisplayEffect->EndPass());
        V_RETURN(g_pDisplayEffect->End());
    }

    return S_OK;

}

//--------------------------------------------------------------------------------------
// Render the help and statistics text. This function uses the ID3DXFont interface for 
// efficient text rendering.
//--------------------------------------------------------------------------------------
void RenderText()
{
    // The helper object simply helps keep track of text position, and color
    // and then it calls pFont->DrawText( m_pSprite, strMsg, -1, &rc, DT_NOCLIP, m_clr );
    // If NULL is passed in as the sprite object, then it will work however the 
    // pFont->DrawText() will not be batched together.  Batching calls will improves performance.
    CDXUTTextHelper txtHelper( g_pFont, g_pTextSprite, 15 );
    const D3DSURFACE_DESC* pd3dsdBackBuffer = DXUTGetBackBufferSurfaceDesc();

	// Output statistics
	txtHelper.Begin();
	txtHelper.SetInsertionPos( 5, 15 );
	if( g_bShowUI )
	{
		txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
		txtHelper.DrawTextLine( DXUTGetFrameStats() );
		txtHelper.DrawTextLine( DXUTGetDeviceStats() );
        txtHelper.DrawTextLine( g_LightingMethods[g_uTechnique].name );
        txtHelper.DrawTextLine( g_DisplayCubemaps[g_uDisplayCubeMap].name );

		// Display any additional information text here

		if( !g_bShowHelp )
		{
			txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ));
			txtHelper.DrawTextLine( TEXT("F1      - Toggle help text") );
		}
	}

	if( g_bShowHelp )
	{
		// Display help text here
		txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
		txtHelper.DrawTextLine( TEXT("F1      - Toggle help text") );
		txtHelper.DrawTextLine( TEXT("H       - Toggle UI") );
		txtHelper.DrawTextLine( TEXT("ESC  - Quit") );
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
    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

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
		case VK_F1:
			g_bShowHelp = !g_bShowHelp;
			break;

		case 'H':
		case 'h':
			g_bShowUI = !g_bShowUI;
			for( int i = 0; i < IDC_LAST; i++ )
				g_HUD.GetControl(i)->SetVisible( g_bShowUI );
			break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl )
{
	switch( nControlID )
    {
        case IDC_TOGGLEFULLSCREEN: DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:        DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:     DXUTSetShowSettingsDialog( !DXUTGetShowSettingsDialog() ); break;
        case IDC_CYCLECUBEMAPS:    g_uDisplayCubeMap = (g_uDisplayCubeMap + 1)%3; break;
        case IDC_CYCLELIGHTING:    g_uTechnique = (g_uTechnique + 1) % 2; break;
//        case IDC_DEBUGIMAGE: g_bShowDebug = !g_bShowDebug; break;
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
    if( g_pFont )
        g_pFont->OnLostDevice();

    if ( g_pCubeMesh )
        g_pCubeMesh->InvalidateDeviceObjects();

    if ( g_pFSQuadMesh )
        g_pFSQuadMesh->InvalidateDeviceObjects();

    for ( UINT i=0; i<2; i++ )
    {
        SAFE_RELEASE( g_pParaboloidMap[i] );
        SAFE_RELEASE( g_pParaboloidSHWeights[i] );
    }
    for ( UINT i=0; i<6; i++ )
    {
        SAFE_RELEASE( g_pLambertSHEval[i] );
        SAFE_RELEASE( g_pPhongSHEval[i] );
    }

    SAFE_DELETE( g_pDragon );
    SAFE_RELEASE( g_pDragonVBDecl );
    SAFE_RELEASE( g_pDragonEffect );
	SAFE_RELEASE( g_pTextSprite );
    SAFE_RELEASE( g_pBaseCubeMap );
    SAFE_RELEASE( g_pConvolveEffect );
    SAFE_RELEASE( g_pDisplayEffect );
    SAFE_RELEASE( g_pParaboloidEffect );
    SAFE_RELEASE( g_pBackBuffer );
    SAFE_RELEASE( g_pZBuffer );
    SAFE_RELEASE( g_pDiffuseCubeMap );
    SAFE_RELEASE( g_pSpecularCubeMap );
    SAFE_RELEASE( g_pIrradianceSHCoefficients );
}


//--------------------------------------------------------------------------------------
// This callback function will be called immediately after the Direct3D device has 
// been destroyed, which generally happens as a result of application termination or 
// windowed/full screen toggles. Resources created in the OnCreateDevice callback 
// should be released here, which generally includes all D3DPOOL_MANAGED resources. 
//--------------------------------------------------------------------------------------
void CALLBACK OnDestroyDevice()
{
    // TODO: Cleanup any objects created in InitDeviceObjects()
    SAFE_RELEASE(g_pFont);
    SAFE_DELETE( g_pCubeMesh );
    SAFE_DELETE( g_pFSQuadMesh );
}