//--------------------------------------------------------------------------------------
// This sample illustrates procedural wind animation technique for trees as
// described in the chapter "GPU Generated, Procedural Wind Animation for Trees"
// of the book GPU Gems III.
//
//
// Author: Renaldas Zioma (rej@scene.lt)
//
#include "DXUT.h"

#include "Cfg.h"

#include <numeric>
#include <memory>
#include <cassert>

#include "Ui_dx10.h"
#include "Parameters.h"
#include "SampleCore.h"

#include "Tree.h"
#include "RenderableTree.h"
#include "Wind.h"

#include "ComPtr.h"
#include "Utils.h"

#include "SDKmisc.h"


//#define DEBUG_SHADER   // Uncomment this line to get debug information from D3D10

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
ID3D10Device*					g_d3dDevice;

com_ptr<ID3DX10Font>			g_pFont;				// Font for drawing text
com_ptr<ID3DX10Sprite>			g_pSprite;				// Sprite for batching draw text calls

com_ptr<ID3D10Effect>			g_pEffect;				// D3DX effect interface

com_ptr<ID3D10InputLayout>		g_pGeometryLayout;
com_ptr<ID3D10InputLayout>		g_pBranchLayout;

enum { StreamOutBuffers = 3 };
com_ptr<ID3D10Buffer>			g_branchInputVB;
com_ptr<ID3D10Buffer>			g_branchMatricesVB[StreamOutBuffers];
com_ptr<ID3D10ShaderResourceView>
								g_branchMatricesRV[StreamOutBuffers];
size_t							g_currentBufferIndex = 0;

ID3D10EffectTechnique*			g_pStreamOutTechnique;
ID3D10EffectTechnique*			g_pStreamInTechnique;

ID3D10EffectShaderResourceVariable*
								g_pBufferBranchMatrix;

ID3D10EffectMatrixVariable*		g_pmWorldViewProjection;
ID3D10EffectMatrixVariable*		g_pmWorld;

ID3D10EffectVectorVariable*		g_pvLightDir;
ID3D10EffectVectorVariable*		g_pvWindDir;
ID3D10EffectVectorVariable*		g_pvWindTangent;
ID3D10EffectVectorVariable*		g_pvAngleShifts;
ID3D10EffectVectorVariable*		g_pvAmplitudes;
ID3D10EffectVectorVariable*		g_pvOriginsAndPhases;
ID3D10EffectVectorVariable*		g_pvDirections;

ID3D10EffectScalarVariable*		g_pfTime;
ID3D10EffectScalarVariable*		g_piHierarchyDepth;
ID3D10EffectScalarVariable*		g_pfFrequencies;
ID3D10EffectScalarVariable*		g_piBranchCount;


UiDx10							g_Ui;
CModelViewerCamera      		g_Camera;
std::auto_ptr<SampleCore>		g_Core;

UINT							g_currentTreeCount = 0;
UINT							g_currentBranchCount = 0;

bool							g_bShowUI;
bool							g_bShowHelp;
bool							g_bFreeze;
//--------------------------------------------------------------------------------------
void RenderText( double fTime )
{
    CDXUTTextHelper txtHelper( g_pFont, g_pSprite, 15 );


    // Output statistics
    txtHelper.Begin();
    txtHelper.SetInsertionPos( 2, 0 );
    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    txtHelper.DrawTextLine( DXUTGetFrameStats(true) );
	txtHelper.DrawTextLine( L"Hide Info: [F1]     Hide UI: [Tab]     Freeze: [Space]\n" 
                            L"Quit: [Esc]\n\n" );

    txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );

	const size_t vertexCount = g_Core->getRenderableTree().mesh->GetVertexCount();
	const size_t instanceCount = g_Core->getParameters().treeCount;

	const size_t boneCount = std::accumulate(
		g_Core->getRenderableTree().branchPerHierarchyLevelCount.begin(), 
		g_Core->getRenderableTree().branchPerHierarchyLevelCount.begin() + g_Core->getParameters().hierarchyDepth + 1, 0); 

	WCHAR sz[100]; sz[0] = 0;
    txtHelper.SetInsertionPos( 2, 48 );
	StringCchPrintf(sz, 100, L"verts per instance = %d, bones per instance = %d, instances = %d", vertexCount, boneCount, instanceCount); 
	txtHelper.DrawTextLine( sz );

    txtHelper.SetInsertionPos( 2, 64 );
	StringCchPrintf(sz, 100, L"sin(inertia_propagation * wind(t - inertia_delay)) = %f", g_Core->getTreeInstances()[0].pseudoInertiaFactor); 
	txtHelper.DrawTextLine( sz );

    UINT nBackBufferHeight = DXUTGetDXGIBackBufferSurfaceDesc()->Height;
    txtHelper.SetInsertionPos( 2, nBackBufferHeight-15*6 );
    txtHelper.SetForegroundColor( D3DXCOLOR(1.0f, 0.75f, 0.0f, 1.0f ) );
    txtHelper.DrawTextLine( L"Controls:" );

    txtHelper.SetInsertionPos( 20, nBackBufferHeight-15*5 );
    txtHelper.DrawTextLine( L"Rotate wind: Left mouse button\n"
                            L"Rotate camera: Right mouse button\n"
                            L"Zoom camera: Mouse wheel scroll\n"
							L"NOTE: Close Info and UI panes for maximum performance\n");

	txtHelper.End();
}

HRESULT OnSceneChange()
{
	g_Ui.update(g_Core->getSceneRadius());

	if (g_Core->getParameters().treeCount == g_currentTreeCount && 
		g_Core->getRenderableTree().branchCount == g_currentBranchCount)
		return S_OK;

	g_currentTreeCount = (UINT)g_Core->getParameters().treeCount;
	g_currentBranchCount = (UINT)g_Core->getRenderableTree().branchCount;

	HRESULT hr;
    // Create VB that holds information about tree instances
    D3D10_BUFFER_DESC vbdesc =
    {
		g_currentTreeCount * sizeof(SampleCore::TreeInstance),
        D3D10_USAGE_DYNAMIC,
        D3D10_BIND_VERTEX_BUFFER,
        D3D10_CPU_ACCESS_WRITE,
        0
    };
    V_RETURN(g_d3dDevice->CreateBuffer(&vbdesc, 0, g_branchInputVB.assignGet()));

    // Create VBs that will hold all of the branch matrices 
	// that need to be streamed out from simulation step
	const int ElementsCount = 5;
    D3D10_BUFFER_DESC vbdescSO =
    {
        g_currentTreeCount * g_currentBranchCount * sizeof(D3DXVECTOR4) * ElementsCount,
        D3D10_USAGE_DEFAULT,
        D3D10_BIND_SHADER_RESOURCE | D3D10_BIND_STREAM_OUTPUT,
        0,
        0
    };

    // Create the resource view for the branch matrices buffer
    D3D10_SHADER_RESOURCE_VIEW_DESC SRVDesc;
    ZeroMemory(&SRVDesc, sizeof(SRVDesc));
    SRVDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    SRVDesc.ViewDimension = D3D10_SRV_DIMENSION_BUFFER;
    SRVDesc.Buffer.ElementOffset = 0;
    SRVDesc.Buffer.ElementWidth = g_currentTreeCount * g_currentBranchCount * ElementsCount;

	for(size_t q = 0; q < StreamOutBuffers; ++q)
	{
		V_RETURN(g_d3dDevice->CreateBuffer(&vbdescSO, NULL, g_branchMatricesVB[q].assignGet()));
		V_RETURN(g_d3dDevice->CreateShaderResourceView(g_branchMatricesVB[q], &SRVDesc, g_branchMatricesRV[q].assignGet()));
	}

	return S_OK;
}

void CALLBACK OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext)
{
	g_Core->onUiEvent(g_Ui, nEvent, nControlID, pControl);
	OnSceneChange();
}

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
bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext)
{
    // Turn vsync off
    pDeviceSettings->d3d10.SyncInterval = 0;

#ifdef DEBUG_SHADER
	// To get debug information from D3D10
    pDeviceSettings->d3d10.CreateFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if(s_bFirstTime)
    {
        s_bFirstTime = false;
        if(DXUT_D3D10_DEVICE == pDeviceSettings->ver && 
			pDeviceSettings->d3d10.DriverType == D3D10_DRIVER_TYPE_REFERENCE)
            DXUTDisplaySwitchingToREFWarning(pDeviceSettings->ver);
    }

    return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10CreateDevice( ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	g_d3dDevice = pd3dDevice;

	HRESULT hr;
    V_RETURN( D3DX10CreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                                OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                                L"Arial", &g_pFont ) );
    V_RETURN( D3DX10CreateSprite( pd3dDevice, 512, &g_pSprite ) );

	assert(pd3dDevice);
	g_Core.reset(new SampleCore(*pd3dDevice));

	g_Ui.init(g_Core->getParameters(), OnGUIEvent);
	g_Ui.OnD3D10CreateDevice(pd3dDevice, pBackBufferSurfaceDesc);
 
    DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
    #if defined( DEBUG ) || defined( _DEBUG ) || defined( DEBUG_SHADER )
    // Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3D10_SHADER_DEBUG;
    #endif

    // Read the D3DX effect file
    WCHAR str[MAX_PATH];
    V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, L"DX10Tree.fx" ) );
	V_RETURN( D3DX10CreateEffectFromFile( str, NULL, NULL, "fx_4_0", dwShaderFlags, 0, pd3dDevice, NULL, NULL, &g_pEffect, NULL, NULL ) );

	// Retrieve parameter handlers
	g_pmWorldViewProjection = g_pEffect->GetVariableByName("mWorldViewProjection")->AsMatrix();
	g_pmWorld = g_pEffect->GetVariableByName("mWorld")->AsMatrix();

	g_pvLightDir = g_pEffect->GetVariableByName("vLightDir")->AsVector();
	g_pvWindDir = g_pEffect->GetVariableByName("vWindDir")->AsVector();
	g_pvWindTangent = g_pEffect->GetVariableByName("vWindTangent")->AsVector();
	g_pvAngleShifts = g_pEffect->GetVariableByName("vAngleShifts")->AsVector();
	g_pvAmplitudes = g_pEffect->GetVariableByName("vAmplitudes")->AsVector();
	g_pvOriginsAndPhases = g_pEffect->GetVariableByName("vOriginsAndPhases")->AsVector();
	g_pvDirections = g_pEffect->GetVariableByName("vDirections")->AsVector();

	g_pfFrequencies = g_pEffect->GetVariableByName("fFrequencies")->AsScalar();
	g_piHierarchyDepth = g_pEffect->GetVariableByName("iHierarchyDepth")->AsScalar();
	g_pfTime = g_pEffect->GetVariableByName("fTime")->AsScalar();

	g_pStreamOutTechnique = g_pEffect->GetTechniqueByName("StreamOutBranches");
	g_pStreamInTechnique = g_pEffect->GetTechniqueByName("StreamInBranches");

	g_piBranchCount = g_pEffect->GetVariableByName( "iBranchCount" )->AsScalar();
    g_pBufferBranchMatrix = g_pEffect->GetVariableByName( "bufferBranchMatrix" )->AsShaderResource();


	// Create stream layout for rendering geometry
	const D3D10_INPUT_ELEMENT_DESC geometryDeclaration[] =
	{
		{ "POSITION",     0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,    D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "BLENDWEIGHT",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 3*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT,      0, 7*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
		{ "NORMAL",       0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 8*4,  D3D10_INPUT_PER_VERTEX_DATA, 0 },
	};

	D3D10_PASS_DESC PassDesc;
	V_RETURN(g_pStreamInTechnique->GetPassByIndex(0)->GetDesc(&PassDesc));
	V_RETURN(g_d3dDevice->CreateInputLayout(geometryDeclaration, 4, 
		PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &g_pGeometryLayout));

	// Create stream layout for simulation step
	const D3D10_INPUT_ELEMENT_DESC branchDeclaration[] =
	{
		{ "WORLDPOS",     0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 0,    D3D10_INPUT_PER_INSTANCE_DATA, 1 },
		{ "WINDROTATION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 3*4,  D3D10_INPUT_PER_INSTANCE_DATA, 1 },
		{ "PHASE",        0, DXGI_FORMAT_R32_FLOAT,          0, 7*4,  D3D10_INPUT_PER_INSTANCE_DATA, 1 },
		{ "INERTIA",      0, DXGI_FORMAT_R32_FLOAT,          0, 8*4,  D3D10_INPUT_PER_INSTANCE_DATA, 1 },
	};

	V_RETURN(g_pStreamOutTechnique->GetPassByIndex( 0 )->GetDesc(&PassDesc));
	V_RETURN(g_d3dDevice->CreateInputLayout(branchDeclaration, 4,
		PassDesc.pIAInputSignature, PassDesc.IAInputSignatureSize, &g_pBranchLayout));

	OnSceneChange();
    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10ResizedSwapChain( ID3D10Device* pd3dDevice, IDXGISwapChain *pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;
	V_RETURN(g_Ui.OnD3D10ResizedSwapChain(pd3dDevice, pBackBufferSurfaceDesc));

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / (FLOAT)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams(D3DX_PI/4, fAspectRatio, 2.0f, 4000.0f);
    g_Camera.SetWindow(pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height);
	g_Camera.SetButtonMasks(MOUSE_RIGHT_BUTTON, MOUSE_WHEEL, MOUSE_RIGHT_BUTTON);

    // Setup the camera's view parameters
	assert(g_Core.get());
	float sceneRadius = g_Core->getSceneRadius();
    D3DXVECTOR3 vecEye(0.0f, 0.0f, -15.0f);
    D3DXVECTOR3 vecAt (0.0f, 0.0f, -0.0f);
    g_Camera.SetViewParams( &vecEye, &vecAt );
    g_Camera.SetRadius(sceneRadius*3.0f, sceneRadius*0.5f, sceneRadius*10.0f);

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
	if(g_bFreeze)
	{
		fTime = 0.0f;
		fElapsedTime = 0.0f;
	}

	g_Camera.FrameMove(static_cast<float>(fTime));
	g_Ui.pull();

	g_Core->update(static_cast<float>(fTime));
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D10 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender( ID3D10Device* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
	if(g_bFreeze)
	{
		fTime = 0.0f;
		fElapsedTime = 0.0f;
	}

	HRESULT hr;

    // Clear render target and the depth stencil 
    float ClearColor[4] = { 0.176f, 0.196f, 0.667f, 0.0f };
    pd3dDevice->ClearRenderTargetView( DXUTGetD3D10RenderTargetView(), ClearColor );
    pd3dDevice->ClearDepthStencilView( DXUTGetD3D10DepthStencilView(), D3D10_CLEAR_DEPTH, 1.0, 0 );

	D3DXMATRIX mWorld;
	D3DXMATRIX mView;
	D3DXMATRIX mProj;

    // Get the projection & view matrix from the camera class
	mWorld = g_Core->getWorldMatrix();
    mProj = *g_Camera.GetProjMatrix();
    mView = *g_Camera.GetViewMatrix();

	D3DXMATRIX mWorldViewProjection;
	mWorldViewProjection = mWorld * mView * mProj;

	D3DXMATRIX mInvWorld;
	D3DXMatrixInverse(&mInvWorld, 0, &mWorld);

	D3DXVECTOR3 vWindTangent;
	vWindTangent = D3DXVECTOR3(-g_Core->getWindDir().y, g_Core->getWindDir().x, g_Core->getWindDir().z);

	{ // Setup shader parameters
		V(g_pmWorldViewProjection->SetMatrix((float*)&mWorldViewProjection));
		V(g_pmWorld->SetMatrix((float*)&mWorld));
		V(g_pfTime->SetFloat((float)fTime));

		V(g_pvWindDir->SetRawValue(&const_cast<D3DXVECTOR3&>(g_Core->getWindDir()), 0, sizeof(g_Core->getWindDir())));
		V(g_pvWindTangent->SetRawValue(&vWindTangent, 0, sizeof(vWindTangent)));

		V(g_pvAngleShifts->SetFloatVectorArray(
			(float*)g_Core->getPackedBranchParameters().angleShift, 0, SimulationParameters::MaxRuleCount));
		V(g_pvAmplitudes->SetFloatVectorArray(
			(float*)g_Core->getPackedBranchParameters().amplitude, 0, SimulationParameters::MaxRuleCount));
		V(g_pfFrequencies->SetFloatArray(
			(float*)g_Core->getPackedBranchParameters().frequency, 0, SimulationParameters::MaxRuleCount));

		V(g_piHierarchyDepth->SetInt(g_Core->getParameters().hierarchyDepth - 1));

		V(g_pvOriginsAndPhases->SetFloatVectorArray((float*)&g_Core->getRenderableTree().originsAndPhases[0], 0,
			static_cast<UINT>(g_Core->getRenderableTree().originsAndPhases.size())));
		V(g_pvDirections->SetFloatVectorArray((float*)&g_Core->getRenderableTree().directions[0], 0,
			static_cast<UINT>(g_Core->getRenderableTree().directions.size())));
		V(g_piBranchCount->SetInt(static_cast<int>(g_Core->getRenderableTree().branchCount)));

		D3DXVECTOR3 vLightDir(1, 1, -1);
		D3DXVec3Normalize(&vLightDir, &vLightDir);
		D3DXVec3TransformNormal(&vLightDir, &vLightDir, &mInvWorld);
		V(g_pvLightDir->SetRawValue(&vLightDir, 0, sizeof(vLightDir)));
	}

	{ // Fill buffer with information for simulation step
		SampleCore::TreeInstance* treeInstances = 0;
		hr = g_branchInputVB->Map(D3D10_MAP_WRITE_DISCARD, 0, reinterpret_cast<void**>(&treeInstances));
		if( SUCCEEDED( hr ) )
		{
			size_t treeIt = 0;
			if(g_Core->getParameters().treeCount == 1)
			{
				SampleCore::TreeInstance srcInstance = g_Core->getTreeInstances()[0];
				srcInstance.worldPos = D3DXVECTOR3(0.0f, 0.0f, 0.0f);		
				treeInstances[0] = srcInstance;
			}
			else
			{
				for(size_t y = 0; y < 100; ++y)
					for( size_t x = 0; x < 10 && treeIt < g_Core->getParameters().treeCount; ++x, ++treeIt)
					{
						SampleCore::TreeInstance srcInstance = g_Core->getTreeInstances()[treeIt];
						srcInstance.worldPos = D3DXVECTOR3((x - 5.0f) * 20.0f, (y - 0.0f) * 20.0f, 0.0f);
				
						treeInstances[treeIt] = srcInstance;
					}
			}
			g_branchInputVB->Unmap();
		}
	}

	size_t prevBufferIndex = g_currentBufferIndex;

	{ // Run simulation step
		// NOTE: results of the simulation step are stored in the intermediate vertex buffer
		// using DirectX10 StreamOut functionality
		g_currentBufferIndex = (g_currentBufferIndex+1) % StreamOutBuffers;

		// Set stream out destination buffers
		ID3D10Buffer *dstBuffers[] = { g_branchMatricesVB[g_currentBufferIndex] };
		UINT dstOffsets[] = { 0 };

		// Turn on stream out
		pd3dDevice->SOSetTargets(1, dstBuffers, dstOffsets);

		// Set vertex Layout
		pd3dDevice->IASetInputLayout(g_pBranchLayout);

		// Set source vertex buffer
		ID3D10Buffer *srcBuffers[] = { g_branchInputVB };
		UINT srcStrides[] = { sizeof(SampleCore::TreeInstance) };
		UINT srcOffsets[] = { 0 };

		pd3dDevice->IASetVertexBuffers(0, 1, srcBuffers, srcStrides, srcOffsets);

		// Treat the entire vertex buffer as list of points
		// where each point represents a single branch 
		pd3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_POINTLIST);

		// Render the vertices as an array of points
		D3D10_TECHNIQUE_DESC techDesc;
		g_pStreamOutTechnique->GetDesc(&techDesc);
		for(UINT passIt = 0; passIt < techDesc.Passes; ++passIt)
		{
			g_pStreamOutTechnique->GetPassByIndex(passIt)->Apply(0);
			pd3dDevice->DrawInstanced(
				static_cast<UINT>(g_Core->getRenderableTree().branchCount),
				static_cast<UINT>(g_Core->getParameters().treeCount),
				0, 0);
		}

		// Turn off stream out
		dstBuffers[0] = 0;
		pd3dDevice->SOSetTargets(1, dstBuffers, dstOffsets);
	}

	{ // Render trees
		pd3dDevice->IASetInputLayout(g_pGeometryLayout);

		// Feed in information from previous simulation step
		g_pBufferBranchMatrix->SetResource(g_branchMatricesRV[prevBufferIndex]);

		D3D10_TECHNIQUE_DESC techDesc;
		g_pStreamInTechnique->GetDesc(&techDesc);
		for (UINT iPass = 0; iPass < techDesc.Passes; ++iPass)
		{
			g_pStreamInTechnique->GetPassByIndex(iPass)->Apply(0);
			V(g_Core->getRenderableTree().mesh->DrawSubsetInstanced(
				0, static_cast<UINT>(g_Core->getParameters().treeCount), 0));
		}
	}

	// UI
	if(g_bShowUI)
		g_Ui.OnRender(mView, mProj, *g_Camera.GetEyePt(), fElapsedTime);
	if(g_bShowHelp)
		RenderText(fTime);
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10ReleasingSwapChain( void* pUserContext )
{
	g_Ui.OnD3D10ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
    DXUTGetGlobalResourceCache().OnDestroyDevice();
	g_Ui.OnD3D10DestroyDevice();
    g_pFont.release();
    g_pSprite.release();
    g_pEffect.release();
	g_pGeometryLayout.release();
	
	g_pBranchLayout.release();
	g_branchInputVB.release();
	for(size_t q = 0; q < StreamOutBuffers; ++q)
	{
		g_branchMatricesVB[q].release();
		g_branchMatricesRV[q].release();
	}

	g_Core.reset();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, 
                          bool* pbNoFurtherProcessing, void* pUserContext )
{
	*pbNoFurtherProcessing = g_Ui.MsgProc(hWnd, uMsg, wParam, lParam);
    if( *pbNoFurtherProcessing )
        return 0;

	g_Camera.HandleMessages(hWnd, uMsg, wParam, lParam);

	return 0;
}

void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	if( bKeyDown )
	{
		switch( nChar )
		{
			case VK_F1: g_bShowHelp = !g_bShowHelp; break;
			case VK_TAB: g_bShowUI = !g_bShowUI; break;
			case VK_SPACE: g_bFreeze = !g_bFreeze; break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Call if device was removed.  Return true to find a new device, false to quit
//--------------------------------------------------------------------------------------
bool CALLBACK OnDeviceRemoved( void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // Set general DXUT callbacks
    DXUTSetCallbackFrameMove(OnFrameMove);
    DXUTSetCallbackKeyboard(OnKeyboard);
    DXUTSetCallbackMsgProc(MsgProc);
    DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
    DXUTSetCallbackDeviceRemoved(OnDeviceRemoved);

    // Set the D3D10 DXUT callbacks
    DXUTSetCallbackD3D10DeviceAcceptable(IsD3D10DeviceAcceptable);
    DXUTSetCallbackD3D10DeviceCreated(OnD3D10CreateDevice);
    DXUTSetCallbackD3D10SwapChainResized(OnD3D10ResizedSwapChain);
    DXUTSetCallbackD3D10FrameRender(OnD3D10FrameRender);
    DXUTSetCallbackD3D10SwapChainReleasing(OnD3D10ReleasingSwapChain);
    DXUTSetCallbackD3D10DeviceDestroyed(OnD3D10DestroyDevice);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(PROFILE) && !defined(DEBUG)
	g_bShowUI = false;
	g_bShowHelp = false;
	g_bFreeze = true;
#else
	g_bShowUI = true;
	g_bShowHelp = true;
	g_bFreeze = false;
#endif
	//////////////////////////////////////////////////////////////////////////////////////////////////////

    DXUTInit(true, true, NULL); // Parse the command line, show msgboxes on error, no extra command line params
    DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
    DXUTCreateWindow(L"GPU Procedural Wind Animation Sample [DirectX10]");
#if defined(PROFILE) && !defined(DEBUG)
    DXUTCreateDevice(true, 320, 240);
#else
    DXUTCreateDevice(true, 640, 480);
#endif
    DXUTMainLoop(); // Enter into the DXUT render loop

    // Perform any application-level cleanup here

    return DXUTGetExitCode();
}


