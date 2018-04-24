//----------------------------------------------------------------------------------
// File:   SkinnedInstancing.cpp
// Author: Bryan Dudash
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
//
// Demonstrates using instancing with an army of skinned animated m_Characters.
//
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "resource.h"

#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTSettingsDlg.h"
#include "sdkmesh_old.h"

#include <vector>

#include "ArmyManager.h"
#include "AnimatedCharacter.h"
#include "TextureLibray.h"

#include "SDKmisc.h"

int g_NumInstances = 4000;
int g_TargetLoad = 0;
float g_Time = 0.0f;
unsigned int g_Frame = 0;
unsigned int g_width, g_height;

CFirstPersonCamera      g_Camera;               // A model viewing camera
ID3DX10Font*            g_pFont10 = NULL;       
ID3DX10Sprite*          g_pSprite10 = NULL;
CDXUTTextHelper*        g_pTxtHelper = NULL;

CDXUTDialogResourceManager   g_DialogResourceManager; // manager for shared resources of dialogs
CDXUTDialog                  g_SampleUI;             // dialog for sample specific controls
CDXUTDialog             g_HUD;
CD3DSettingsDlg         g_SettingsDlg;
bool                    g_bShowHelp = false;

#define IDC_TOGGLEFULLSCREEN        1
#define IDC_TOGGLEREF               2
#define IDC_CHANGEDEVICE            3


#define IDC_NUM_INSTANCES_STATIC    5
#define IDC_NUM_INSTANCES           6
#define IDC_SINGLEDRAW              7
#define IDC_HYBRID_INSTANCING       8
#define IDC_CPU_LOAD_STATIC         10
#define IDC_CPU_LOAD                11
#define IDC_VISUALIZE_LOD           12

void InitApp();
void    CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );

//--------------------------------------------------------------------------------------
// Reject any D3D10 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D10DeviceAcceptable( UINT Adapter, UINT Output, D3D10_DRIVER_TYPE DeviceType, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
#ifdef _DEBUG
    pDeviceSettings->d3d10.CreateFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

    pDeviceSettings->d3d10.SyncInterval = 0;
//    pDeviceSettings->d3d10.sd.SampleDesc.Count = 4;
//    pDeviceSettings->d3d10.sd.SampleDesc.Quality = 16;
    return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10CreateDevice( ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;
    V_RETURN( g_DialogResourceManager.OnD3D10CreateDevice( pd3dDevice ) );
    V_RETURN( g_SettingsDlg.OnD3D10CreateDevice( pd3dDevice ) );
    V_RETURN( D3DX10CreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
        OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
        L"Arial", &g_pFont10 ) );
    V_RETURN( D3DX10CreateSprite( pd3dDevice, 512, &g_pSprite10 ) );
    g_pTxtHelper = new CDXUTTextHelper( NULL, NULL, g_pFont10, g_pSprite10, 15 );

    // Setup the camera's view parameters
    D3DXVECTOR3 vecEye(0.0f, 50.0f, -130.0f);
    D3DXVECTOR3 vecAt (0.0f, 0.0f, 0.0f);
    g_Camera.SetViewParams( &vecEye, &vecAt );
    g_Camera.SetEnablePositionMovement(true);
    g_Camera.SetScalers(0.01f,25.f);

    // Now load and process all that data create the d3d objects.
    V_RETURN(ArmyManager::singleton()->Initialize(pd3dDevice,ArmyManager::singleton()->GetMaxCharacters()));
    ArmyManager::singleton()->SetNumCharacters(g_NumInstances);

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10ResizedSwapChain( ID3D10Device* pd3dDevice, IDXGISwapChain *pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;
    V_RETURN( g_DialogResourceManager.OnD3D10ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_SettingsDlg.OnD3D10ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    g_width = pBackBufferSurfaceDesc->Width;
    g_height = pBackBufferSurfaceDesc->Height;

    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0);
    g_HUD.SetSize( 170, 170 );

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( D3DX_PI/4, fAspectRatio, 0.1f, 2400.0f );

    g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width-250, pBackBufferSurfaceDesc->Height-100 );
    g_SampleUI.SetSize( 250, 300 );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    g_Camera.FrameMove( fElapsedTime );
    ArmyManager::singleton()->Update(fElapsedTime,*g_Camera.GetEyePt());

    // Simulate other game related load
    if(g_TargetLoad > 0)
    {
        DWORD start = GetTickCount();
        while(GetTickCount() - start < (DWORD)g_TargetLoad);
    }
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 2, 0 );
    g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats(true) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
    int polys = ArmyManager::singleton()->GetPolysDrawn();
    int draws = ArmyManager::singleton()->GetDrawCalls();
    LPWSTR sStatus = new WCHAR[MAX_PATH];
    swprintf_s((WCHAR*)sStatus,MAX_PATH,L"%d Polys, %d DrawCalls",polys,draws);
    g_pTxtHelper->DrawTextLine( sStatus );

    if(g_bShowHelp)
    {
        g_pTxtHelper->DrawTextLine( L"Arrow keys move camera.  ");
        g_pTxtHelper->DrawTextLine( L"Mouse Left Click and Drag rotates camera." );
        g_pTxtHelper->DrawTextLine( L"No Instancing = Single Draw Calls");
        g_pTxtHelper->DrawTextLine( L"Hybrid Instancing = The main technique");
        g_pTxtHelper->DrawTextLine( L"Visualize LOD = Color characters based on LOD" );
    }
    else
    {
        g_pTxtHelper->DrawTextLine( L"Hit F1 for Help." );
    }

    g_pTxtHelper->End();
    delete [] sStatus;
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D10 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender( ID3D10Device* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
    // Clear Render target and the depth stencil 
    //float ClearColor[4] = { 0.176f, 0.196f, 0.667f, 0.0f };
    float ClearColor[4] = { 0.f, 0.f, 0.f, 0.0f };
    pd3dDevice->ClearRenderTargetView( DXUTGetD3D10RenderTargetView(), ClearColor );
    pd3dDevice->ClearDepthStencilView( DXUTGetD3D10DepthStencilView(), D3D10_CLEAR_DEPTH, 1.0, 0 );

    D3DXMATRIX  mView = *g_Camera.GetViewMatrix();
    D3DXMATRIX  mProj = *g_Camera.GetProjMatrix();

    // This will set its own effect technique and everything....
    ArmyManager::singleton()->Render(pd3dDevice,mView,mProj,(float)fTime);

    g_HUD.OnRender( fElapsedTime );
    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.OnRender( fElapsedTime );
        return;
    }


    RenderText();
    g_SampleUI.OnRender( fElapsedTime ); 
}

//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10ReleasingSwapChain( void* pUserContext )
{
    g_DialogResourceManager.OnD3D10ReleasingSwapChain();
    g_SettingsDlg.OnD3D10DestroyDevice();
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D10DestroyDevice();
    g_SampleUI.RemoveAllControls();
    TextureLibrary::singleton()->Release();
    ArmyManager::singleton()->Release();
    TextureLibrary::destroy();
    ArmyManager::destroy();
    SAFE_RELEASE( g_pFont10 );
    SAFE_RELEASE( g_pSprite10 );
    SAFE_DELETE( g_pTxtHelper );
}

//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, 
                          bool* pbNoFurtherProcessing, void* pUserContext )
{

    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    if(g_SampleUI.MsgProc(hWnd,uMsg,wParam,lParam))
    {
        return 1;
    }

    if( g_HUD.MsgProc( hWnd, uMsg, wParam, lParam ) )
    {
        return 1;
    }

    if(g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam ))
    {
        return 1;
    }

    if(uMsg == WM_KEYDOWN)
    {
        switch( (char)wParam )
        {
        case 'I':
        case 'i':

            ArmyManager::singleton()->m_iInstancingMode = (ArmyManager::singleton()->m_iInstancingMode+1)%2;
            g_SampleUI.GetRadioButton(IDC_SINGLEDRAW+ArmyManager::singleton()->m_iInstancingMode)->SetChecked(true);
            break;


        case 'k':
        case 'l':
        case 'K':
        case 'L':

            if(wParam == 'k' || wParam == 'K')

                g_NumInstances -= 1;
            else
                g_NumInstances += 1;

            // Update the UI, etc
            g_SampleUI.GetSlider(IDC_NUM_INSTANCES)->SetValue(g_NumInstances); 
            ArmyManager::singleton()->SetNumCharacters(g_NumInstances);
            
            WCHAR sNumInstances[MAX_PATH];
            wsprintf(sNumInstances,L"Instances: %d",g_NumInstances);
            g_SampleUI.GetStatic(IDC_NUM_INSTANCES_STATIC)->SetText(sNumInstances);
            break;

        case VK_F1:
            g_bShowHelp = !g_bShowHelp;
            break;
        }
        return 1;
    }

    return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
}


//--------------------------------------------------------------------------------------
// Handle mouse button presses
//--------------------------------------------------------------------------------------
void CALLBACK OnMouse( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, 
                       bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, 
                       int xPos, int yPos, void* pUserContext )
{
}


//--------------------------------------------------------------------------------------
// Call if device was removed.  Return true to find a new device, false to quit
//--------------------------------------------------------------------------------------
bool CALLBACK OnDeviceRemoved( void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Initialize everything and go into a Render loop
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // DXUT will create and use the best device (either D3D9 or D3D10) 
    // that is available on the system depending on which D3D callbacks are set below

    // Set general DXUT callbacks
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackKeyboard( OnKeyboard );
    DXUTSetCallbackMouse( OnMouse );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackDeviceRemoved( OnDeviceRemoved );

   // Set the D3D10 DXUT callbacks. Remove these sets if the app doesn't need to support D3D10
    DXUTSetCallbackD3D10DeviceAcceptable( IsD3D10DeviceAcceptable );
    DXUTSetCallbackD3D10DeviceCreated( OnD3D10CreateDevice );
    DXUTSetCallbackD3D10SwapChainResized( OnD3D10ResizedSwapChain );
    DXUTSetCallbackD3D10FrameRender( OnD3D10FrameRender );
    DXUTSetCallbackD3D10SwapChainReleasing( OnD3D10ReleasingSwapChain );
    DXUTSetCallbackD3D10DeviceDestroyed( OnD3D10DestroyDevice );

    HRESULT hr;
    V_RETURN(DXUTSetMediaSearchPath(L"..\\Source\\SkinnedInstancing"));

    // Perform any application-level initialization here
    InitApp();
    DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"SkinnedInstancing" );
    DXUTCreateDevice( true, 640, 480 );  
    DXUTMainLoop(); // Enter into the DXUT Render loop

    // Perform any application-level cleanup here


    return DXUTGetExitCode();
}

void InitApp()
{
    int iX = 0;
    int iY = 0;
    g_SampleUI.Init( &g_DialogResourceManager );
    g_SampleUI.SetCallback( OnGUIEvent );
    WCHAR sNumInstances[MAX_PATH];
    wsprintf(sNumInstances,L"Instances: %d",g_NumInstances);
    WCHAR sCPULoad[MAX_PATH];
    wsprintf(sCPULoad,L"Reserved CPU(Non-Graphics): %d ms",g_TargetLoad);

    g_SampleUI.AddStatic(IDC_NUM_INSTANCES_STATIC,sNumInstances,iX,iY,220,15);
    g_SampleUI.AddSlider(IDC_NUM_INSTANCES,iX,iY+=15,220,15,1,ArmyManager::singleton()->GetMaxCharacters(),g_NumInstances);
//    g_SampleUI.AddStatic(IDC_CPU_LOAD_STATIC,sCPULoad,iX,iY+=15,220,15);
//    g_SampleUI.AddSlider(IDC_CPU_LOAD,iX,iY+=15,220,15,0,100,g_TargetLoad);
    g_SampleUI.AddRadioButton(IDC_SINGLEDRAW,0,L"No Instancing",iX,iY+=15,220,15,false);
    g_SampleUI.AddRadioButton(IDC_HYBRID_INSTANCING,0,L"Hybrid Instancing",iX,iY+=15,220,15,true);
    g_SampleUI.AddCheckBox(IDC_VISUALIZE_LOD,L"Visualize LOD?",iX,iY+=15,220,15,false);

    g_SettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent ); iY = 10;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, 125, 22 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 35, iY += 24, 125, 22, VK_F3 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 35, iY += 24, 125, 22, VK_F2 );

}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
    switch( nControlID )
    {
    case IDC_TOGGLEFULLSCREEN: DXUTToggleFullScreen(); break;
    case IDC_TOGGLEREF:        DXUTToggleREF(); break;
    case IDC_CHANGEDEVICE:     g_SettingsDlg.SetActive( !g_SettingsDlg.IsActive() ); break;

    case IDC_NUM_INSTANCES:    
        {
            g_NumInstances = g_SampleUI.GetSlider(IDC_NUM_INSTANCES)->GetValue(); 
            ArmyManager::singleton()->SetNumCharacters(g_NumInstances);
            
            WCHAR sNumInstances[MAX_PATH];
            wsprintf(sNumInstances,L"Instances: %d",g_NumInstances);
            g_SampleUI.GetStatic(IDC_NUM_INSTANCES_STATIC)->SetText(sNumInstances);
        }
        break;
    case IDC_CPU_LOAD:    
        {
            g_TargetLoad = g_SampleUI.GetSlider(IDC_CPU_LOAD)->GetValue(); 
            
            WCHAR sCPULoad[MAX_PATH];
            wsprintf(sCPULoad,L"Reserved CPU(Non-Graphics): %d ms",g_TargetLoad);
            g_SampleUI.GetStatic(IDC_CPU_LOAD_STATIC)->SetText(sCPULoad);
        }
        break;
    case IDC_SINGLEDRAW:    
        {
            if(g_SampleUI.GetRadioButton(IDC_SINGLEDRAW)->GetChecked())
                ArmyManager::singleton()->m_iInstancingMode = 0;
        }
        break;
    case IDC_HYBRID_INSTANCING:    
        {
            if(g_SampleUI.GetRadioButton(IDC_HYBRID_INSTANCING)->GetChecked())
                ArmyManager::singleton()->m_iInstancingMode = 1;
        }
        break;
    case IDC_VISUALIZE_LOD:
        {
            ArmyManager::singleton()->m_bVisualizeLOD = g_SampleUI.GetCheckBox(IDC_VISUALIZE_LOD)->GetChecked();
        }
        break;
    }
}


