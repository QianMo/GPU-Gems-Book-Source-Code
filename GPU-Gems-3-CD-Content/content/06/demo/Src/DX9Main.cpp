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

#include "Ui_dx9.h"
#include "Parameters.h"
#include "SampleCore.h"

#include "Tree.h"
#include "RenderableTree.h"
#include "Wind.h"

#include "ComPtr.h"
#include "Utils.h"

#include "SDKmisc.h"


//#define DEBUG_VS   // Uncomment this line to debug vertex shaders 
//#define DEBUG_PS   // Uncomment this line to debug pixel shaders

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
IDirect3DDevice9*				g_d3dDevice;

com_ptr<ID3DXFont>				g_pFont;				// Font for drawing text
com_ptr<ID3DXSprite>			g_pSprite;				// Sprite for batching draw text calls

com_ptr<ID3DXEffect>			g_pEffect;				// D3DX effectx interface
com_ptr<IDirect3DVertexBuffer9>	g_pVBInstanceData;
com_ptr<IDirect3DVertexDeclaration9>
								g_pVertexDeclHardware;

D3DXHANDLE						g_mWorldViewProjection;
D3DXHANDLE						g_mWorld;
D3DXHANDLE						g_mWindRotation;
D3DXHANDLE						g_fTime;
D3DXHANDLE						g_vWindDir;
D3DXHANDLE						g_vWindTangent;
D3DXHANDLE						g_vAngleShifts;
D3DXHANDLE						g_vAmplitudes;
D3DXHANDLE						g_fFrequencies;
D3DXHANDLE						g_iHierarchyDepth;
D3DXHANDLE						g_vOriginsAndPhases;
D3DXHANDLE						g_fPseudoInertiaFactor;
D3DXHANDLE						g_vLightDir;
D3DXHANDLE						g_vWorldPos;


UiDx9							g_Ui;
CModelViewerCamera      		g_Camera;
std::auto_ptr<
	SampleCore>					g_Core;

UINT							g_currentTreeCount = 0;

bool							g_bShowUI;
bool							g_bShowHelp;
bool							g_bFreeze;
//--------------------------------------------------------------------------------------
void RenderText(double fTime)
{
    CDXUTTextHelper txtHelper(g_pFont, g_pSprite, 15);


    // Output statistics
    txtHelper.Begin();
    txtHelper.SetInsertionPos(2, 0);
    txtHelper.SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 0.0f, 1.0f));
    txtHelper.DrawTextLine(DXUTGetFrameStats(true));
	txtHelper.DrawTextLine(L"Hide Info: [F1]     Hide UI: [Tab]     Freeze: [Space]\n" 
                            L"Quit: [Esc]\n\n");

    txtHelper.SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 1.0f, 1.0f));

	const size_t vertexCount = g_Core->getRenderableTree().mesh->GetNumVertices();
	const size_t instanceCount = g_Core->getParameters().treeCount;

	const size_t boneCount = std::accumulate(
		g_Core->getRenderableTree().branchPerHierarchyLevelCount.begin(), 
		g_Core->getRenderableTree().branchPerHierarchyLevelCount.begin() + g_Core->getParameters().hierarchyDepth + 1, 0); 

	WCHAR sz[100]; sz[0] = 0;
    txtHelper.SetInsertionPos(2, 48);
	StringCchPrintf(sz, 100, L"verts per instance = %d, bones per instance = %d, instances = %d", vertexCount, boneCount, instanceCount); 
	txtHelper.DrawTextLine(sz);

    txtHelper.SetInsertionPos(2, 64);
	StringCchPrintf(sz, 100, L"sin(inertia_propagation * wind(t - inertia_delay)) = %f", g_Core->getTreeInstances()[0].pseudoInertiaFactor); 
	txtHelper.DrawTextLine(sz);

    UINT nBackBufferHeight = DXUTGetD3D9BackBufferSurfaceDesc()->Height;
    txtHelper.SetInsertionPos(2, nBackBufferHeight-15*6);
    txtHelper.SetForegroundColor(D3DXCOLOR(1.0f, 0.75f, 0.0f, 1.0f));
    txtHelper.DrawTextLine(L"Controls:");

    txtHelper.SetInsertionPos(20, nBackBufferHeight-15*5);
    txtHelper.DrawTextLine(L"Rotate wind: Left mouse button\n"
                            L"Rotate camera: Right mouse button\n"
                            L"Zoom camera: Mouse wheel scroll\n"
							L"NOTE: Hide Info and UI panes for maximum performance\n");

	txtHelper.End();
}

HRESULT OnSceneChange()
{
	g_Ui.update(g_Core->getSceneRadius());

	if(g_Core->getParameters().treeCount == g_currentTreeCount)
		return S_OK;

	g_currentTreeCount = (UINT)g_Core->getParameters().treeCount;

	HRESULT hr;
    // Create a VB for the instancing data
	V_RETURN(g_d3dDevice->CreateVertexBuffer(
		static_cast<UINT>(g_currentTreeCount * sizeof(SampleCore::TreeInstance)),
		0, 0, D3DPOOL_DEFAULT, g_pVBInstanceData.assignGet(), 0));

	return S_OK;
}

void CALLBACK OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext)
{
	g_Core->onUiEvent(g_Ui, nEvent, nControlID, pControl);
	OnSceneChange();
}

//--------------------------------------------------------------------------------------
// Rejects any devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsDeviceAcceptable(D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, 
                                  D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext)
{
    // No fallback defined by this app, so reject any device that 
    // doesn't support at least ps1.1
    if(pCaps->PixelShaderVersion < D3DPS_VERSION(1,1))
        return false;

    // Typically want to skip backbuffer formats that don't support alpha blending
    IDirect3D9* pD3D = DXUTGetD3D9Object(); 
    if(FAILED(pD3D->CheckDeviceFormat(pCaps->AdapterOrdinal, pCaps->DeviceType,
                    AdapterFormat, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, 
                    D3DRTYPE_TEXTURE, BackBufferFormat)))
        return false;

    return true;
}


//--------------------------------------------------------------------------------------
// Before a device is created, modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext)
{
    // Turn vsync off
    pDeviceSettings->d3d9.pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;

    D3DCAPS9 caps;
    DXUTGetD3D9Object()->GetDeviceCaps(
		pDeviceSettings->d3d9.AdapterOrdinal, pDeviceSettings->d3d9.DeviceType, &caps);

    // If device doesn't support HW T&L or doesn't support 1.1 vertex shaders in HW 
    // then switch to SWVP.
    if((caps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT) == 0 ||
         caps.VertexShaderVersion < D3DVS_VERSION(1,1))
    {
        pDeviceSettings->d3d9.BehaviorFlags = D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }

	// Debugging vertex shaders requires either REF or software vertex processing 
    // and debugging pixel shaders requires REF.  
#ifdef DEBUG_VS
    if(pDeviceSettings->d3d9.DeviceType != D3DDEVTYPE_REF)
    {
        pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_HARDWARE_VERTEXPROCESSING;
        pDeviceSettings->d3d9.BehaviorFlags &= ~D3DCREATE_PUREDEVICE;                            
        pDeviceSettings->d3d9.BehaviorFlags |= D3DCREATE_SOFTWARE_VERTEXPROCESSING;
    }
#endif
#ifdef DEBUG_PS
    pDeviceSettings->d3d9.DeviceType = D3DDEVTYPE_REF;
#endif
    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if(s_bFirstTime)
    {
        s_bFirstTime = false;
        if(pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF)
            DXUTDisplaySwitchingToREFWarning(pDeviceSettings->ver);
    }

    return true;
}

//--------------------------------------------------------------------------------------
// Create any D3DPOOL_MANAGED resources here 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnCreateDevice(IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
	HRESULT hr;
    // Initialize the font
    V_RETURN(D3DXCreateFont(pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                              OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                              L"Arial", &g_pFont));

	assert(pd3dDevice);
	g_Core.reset(new SampleCore(*pd3dDevice));

	g_Ui.init(g_Core->getParameters(), OnGUIEvent);
	g_Ui.OnCreateDevice(pd3dDevice, pBackBufferSurfaceDesc);

	
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
    DWORD dwShaderFlags = D3DXFX_NOT_CLONEABLE;

    #if defined(DEBUG) || defined(_DEBUG)
    // Set the D3DXSHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DXSHADER_DEBUG;
    #endif

    #ifdef DEBUG_VS
        dwShaderFlags |= D3DXSHADER_FORCE_VS_SOFTWARE_NOOPT;
    #endif
    #ifdef DEBUG_PS
        dwShaderFlags |= D3DXSHADER_FORCE_PS_SOFTWARE_NOOPT;
    #endif

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	g_d3dDevice = pd3dDevice;

    // Read the D3DX effect file
    WCHAR str[MAX_PATH];
    V_RETURN(DXUTFindDXSDKMediaFileCch(str, MAX_PATH, L"DX9Tree.fx"));

    // If this fails, there should be debug output as to 
    // why the .fx file failed to compile
    V_RETURN(D3DXCreateEffectFromFile(pd3dDevice, str, NULL, NULL, dwShaderFlags, NULL, &g_pEffect, NULL));

	// Retrieve parameter handlers
	g_mWindRotation = g_pEffect->GetParameterByName(0, "mWindRotation");
	g_fPseudoInertiaFactor = g_pEffect->GetParameterByName(0, "fPseudoInertiaFactor");
	g_mWorldViewProjection = g_pEffect->GetParameterByName(0, "mWorldViewProjection");
	g_mWorld = g_pEffect->GetParameterByName(0, "mWorld");
	g_mWindRotation = g_pEffect->GetParameterByName(0, "mWindRotation");
	g_fTime = g_pEffect->GetParameterByName(0, "fTime");
	g_vWindDir = g_pEffect->GetParameterByName(0, "vWindDir");
	g_vWindTangent = g_pEffect->GetParameterByName(0, "vWindTangent");
	g_vAngleShifts = g_pEffect->GetParameterByName(0, "vAngleShifts");
	g_vAmplitudes = g_pEffect->GetParameterByName(0, "vAmplitudes");
	g_fFrequencies = g_pEffect->GetParameterByName(0, "fFrequencies");
	g_iHierarchyDepth = g_pEffect->GetParameterByName(0, "iHierarchyDepth");
	g_vOriginsAndPhases = g_pEffect->GetParameterByName(0, "vOriginsAndPhases");
	g_fPseudoInertiaFactor = g_pEffect->GetParameterByName(0, "fPseudoInertiaFactor");
	g_vLightDir = g_pEffect->GetParameterByName(0, "vLightDir");
	g_vWorldPos = g_pEffect->GetParameterByName(0, "vWorldPos");

    // Create vertex declaration
	D3DVERTEXELEMENT9 g_VertexElemHardware[] = 
	{
		{ 0, 0,			D3DDECLTYPE_FLOAT3,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_POSITION,		0 },
		{ 0, 3 * 4,		D3DDECLTYPE_FLOAT4,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_BLENDWEIGHT,	0 },
		{ 0, 7 * 4,		D3DDECLTYPE_UBYTE4,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_BLENDINDICES,	0 },
		{ 0, 8 * 4,		D3DDECLTYPE_FLOAT3,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_NORMAL,		0 },

		{ 1, 0,			D3DDECLTYPE_FLOAT3,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_TEXCOORD,		0 },
		{ 1, 3 * 4,		D3DDECLTYPE_FLOAT4,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_TEXCOORD,		1 },
		{ 1, 7 * 4,		D3DDECLTYPE_FLOAT1,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_TEXCOORD,		2 },
		{ 1, 8 * 4,		D3DDECLTYPE_FLOAT1,	D3DDECLMETHOD_DEFAULT,	D3DDECLUSAGE_TEXCOORD,		3 },
		D3DDECL_END()
	};
    V_RETURN(pd3dDevice->CreateVertexDeclaration(g_VertexElemHardware, &g_pVertexDeclHardware));

    // If device doesn't support ShaderModel3 switch to CPU instancing
	if(DXUTGetD3D9DeviceCaps()->VertexShaderVersion < D3DVS_VERSION(3,0))
		g_Core->getParameters().instancingType = Parameters::InstancingCPU;
	else
		g_Core->getParameters().instancingType = Parameters::InstancingGPU;

	OnSceneChange();
	return S_OK;
}

//--------------------------------------------------------------------------------------
// Create any D3DPOOL_DEFAULT resources here 
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnResetDevice(IDirect3DDevice9* pd3dDevice, 
                                const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	HRESULT hr;
	if(g_pFont)
        V_RETURN(g_pFont->OnResetDevice());
    V_RETURN(D3DXCreateSprite(pd3dDevice, &g_pSprite));
    if(g_pEffect)
        V_RETURN(g_pEffect->OnResetDevice());

	g_Ui.OnResetDevice(pBackBufferSurfaceDesc);

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
    g_Camera.SetViewParams(&vecEye, &vecAt);
    g_Camera.SetRadius(sceneRadius*3.0f, sceneRadius*0.5f, sceneRadius*10.0f);

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove(double fTime, float fElapsedTime, void* pUserContext)
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
// Render the scene 
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameRender(IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext)
{
	if(g_bFreeze)
	{
		fTime = 0.0f;
		fElapsedTime = 0.0f;
	}

    HRESULT hr;

    // Clear the render target and the zbuffer 
    V(pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB(0, 45, 50, 170), 1.0f, 0));

    // Render the scene
    if(SUCCEEDED(pd3dDevice->BeginScene()))
    {
		D3DXMATRIXA16 mWorldViewProjection;
		D3DXMATRIXA16 mWorld;
		D3DXMATRIXA16 mView;
		D3DXMATRIXA16 mProj;
		D3DXMATRIXA16 mInvWorld;
		D3DXVECTOR3 vWindTangent;


		// Get the projection & view matrix from the camera class
		mWorld = g_Core->getWorldMatrix();
        mProj = *g_Camera.GetProjMatrix();
        mView = *g_Camera.GetViewMatrix();

        mWorldViewProjection = mWorld * mView * mProj;

		D3DXMatrixInverse(&mInvWorld, 0, &mWorld);

		vWindTangent = D3DXVECTOR3(-g_Core->getWindDir().y, g_Core->getWindDir().x, g_Core->getWindDir().z);
        
		{ // Setup shader parameters
			V(g_pEffect->SetMatrix(g_mWorldViewProjection, &mWorldViewProjection));
			V(g_pEffect->SetMatrix(g_mWorld, &mWorld));
			V(g_pEffect->SetFloat(g_fTime, (float)fTime));

			V(g_pEffect->SetValue(g_vWindDir, &g_Core->getWindDir(), sizeof(g_Core->getWindDir())));
			V(g_pEffect->SetValue(g_vWindTangent, &vWindTangent, sizeof(vWindTangent)));

			V(g_pEffect->SetVectorArray(g_vAngleShifts,
				g_Core->getPackedBranchParameters().angleShift, SimulationParameters::MaxRuleCount));
			V(g_pEffect->SetVectorArray(g_vAmplitudes,
				g_Core->getPackedBranchParameters().amplitude, SimulationParameters::MaxRuleCount));
			V(g_pEffect->SetFloatArray(g_fFrequencies,
				g_Core->getPackedBranchParameters().frequency, SimulationParameters::MaxRuleCount));

			V(g_pEffect->SetInt(g_iHierarchyDepth, g_Core->getParameters().hierarchyDepth - 1));
		}

		if(g_Core->getParameters().treeCount > 1 &&
			g_Core->getParameters().instancingType == Parameters::InstancingGPU)
		{
			// Lock and fill the instancing buffer
			SampleCore::TreeInstance* treeInstances = 0;
			hr = g_pVBInstanceData->Lock(0, NULL, (void**) &treeInstances, D3DLOCK_DISCARD);
			if(SUCCEEDED(hr))
			{
				size_t treeIt = 0;
				for(size_t y = 0; y < 100; ++y)
					for(size_t x = 0; x < 10 && treeIt < g_Core->getParameters().treeCount; ++x, ++treeIt)
					{
						SampleCore::TreeInstance& treeInstance = treeInstances[treeIt];
						treeInstance = g_Core->getTreeInstances()[treeIt];
						treeInstance.worldPos = D3DXVECTOR3((x - 5.0f) * 20.0f, (y - 0.0f) * 20.0f, 0.0f);
					}
				g_pVBInstanceData->Unlock();
			}

			com_ptr<IDirect3DVertexBuffer9> vb = 0;
			g_Core->getRenderableTree().mesh->GetVertexBuffer(&vb);

			com_ptr<IDirect3DIndexBuffer9> ib = 0;
			g_Core->getRenderableTree().mesh->GetIndexBuffer(&ib);

			V(pd3dDevice->SetVertexDeclaration(g_pVertexDeclHardware));

			V(pd3dDevice->SetStreamSource(0, vb, 0, sizeof(BranchVertex)));
			V(pd3dDevice->SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | (UINT)g_Core->getParameters().treeCount));

			// Stream one is the instancing buffer, so this advances to the next value
			// after each box instance has been drawn, so the divider is 1.
			V(pd3dDevice->SetStreamSource(1, g_pVBInstanceData, 0, sizeof(SampleCore::TreeInstance)));
			V(pd3dDevice->SetStreamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1UL));

		    V(pd3dDevice->SetIndices(ib));

			V(g_pEffect->SetTechnique("Instancing"));
		}
		else if(g_Core->getParameters().treeCount == 1 || 
			g_Core->getParameters().instancingType == Parameters::InstancingCPU)
		{
			V(pd3dDevice->SetStreamSource(0, 0, 0, 0));
			V(pd3dDevice->SetStreamSource(1, 0, 0, 0));
			V(pd3dDevice->SetStreamSourceFreq(0, 1));
			V(pd3dDevice->SetStreamSourceFreq(1, 1));

			V(g_pEffect->SetTechnique("Default"));
		}

		D3DXVECTOR3 vLightDir(1, 1, -1);
		D3DXVec3Normalize(&vLightDir, &vLightDir);
		D3DXVec3TransformNormal(&vLightDir, &vLightDir, &mInvWorld);
        V(g_pEffect->SetValue(g_vLightDir, &vLightDir, sizeof(vLightDir)));

		V(g_pEffect->SetVectorArray(g_vOriginsAndPhases, &g_Core->getRenderableTree().originsAndPhases[0],
			static_cast<UINT>(g_Core->getRenderableTree().originsAndPhases.size())));

		UINT iPass, cPasses;
        V(g_pEffect->Begin(&cPasses, 0));

        for (iPass = 0; iPass < cPasses; iPass++)
        {
            V(g_pEffect->BeginPass(iPass));

			if(g_Core->getParameters().treeCount == 1)
			{
				// Render single tree
				D3DXMATRIXA16 mWindRotation;
				D3DXMatrixRotationQuaternion(&mWindRotation, &g_Core->getTreeInstances()[0].windRotation);
				V(g_pEffect->SetMatrix(g_mWindRotation, &mWindRotation));
				V(g_pEffect->SetFloat(g_fPseudoInertiaFactor, g_Core->getTreeInstances()[0].pseudoInertiaFactor));
				D3DXVECTOR4 vWorldPos(0.0f, 0.0f, 0.0f, 0.0f);
				V(g_pEffect->SetVector(g_vWorldPos, &vWorldPos));

				DXUT_BeginPerfEvent(DXUT_PERFEVENTCOLOR, L"Render_Default / Stats");
				V(g_pEffect->CommitChanges());
				V(g_Core->getRenderableTree().mesh->DrawSubset(0));
				DXUT_EndPerfEvent();
			}
			else if(g_Core->getParameters().instancingType == Parameters::InstancingCPU)
			{
				// Render multiple trees via loop
				DXUT_BeginPerfEvent(DXUT_PERFEVENTCOLOR, L"Render_InstancingCPU / Stats");
				D3DXMATRIXA16 mWindRotation;
				D3DXVECTOR4 vWorldPos;

				size_t treeIt = 0;
				for(size_t y = 0; y < 100; ++y)
					for(size_t x = 0; x < 10 && treeIt < g_Core->getParameters().treeCount; ++x, ++treeIt)
					{
						V(g_pEffect->SetVectorArray(g_vOriginsAndPhases, &g_Core->getRenderableTree().originsAndPhases[0],
							static_cast<UINT>(g_Core->getRenderableTree().originsAndPhases.size())));

						vWorldPos = D3DXVECTOR4((x - 5.0f) * 20.0f, (y - 0.0f) * 20.0f, 0.0f, 0.0f);
						D3DXMatrixRotationQuaternion(&mWindRotation, &g_Core->getTreeInstances()[treeIt].windRotation);
						V(g_pEffect->SetMatrix(g_mWindRotation, &mWindRotation));
						V(g_pEffect->SetVector(g_vWorldPos, &vWorldPos));
						V(g_pEffect->SetFloat(g_fPseudoInertiaFactor, g_Core->getTreeInstances()[treeIt].pseudoInertiaFactor));
						V(g_pEffect->CommitChanges());
						V(g_Core->getRenderableTree().mesh->DrawSubset(0));
					}
				DXUT_EndPerfEvent();
			}
			else if (g_Core->getParameters().instancingType == Parameters::InstancingGPU)
			{ 
				// Render trees using hardware instancing
				DXUT_BeginPerfEvent(DXUT_PERFEVENTCOLOR, L"Render_InstancingGPU / Stats");
				V(pd3dDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0,
					g_Core->getRenderableTree().mesh->GetNumVertices(), 0, g_Core->getRenderableTree().mesh->GetNumFaces()));
				DXUT_EndPerfEvent();
			}

            V(g_pEffect->EndPass());
        }
        V(g_pEffect->End());
		

		// UI
	    DXUT_BeginPerfEvent(DXUT_PERFEVENTCOLOR, L"HUD / Stats");
		if(g_bShowUI)
			g_Ui.OnRender(mView, mProj, *g_Camera.GetEyePt(), fElapsedTime);
		if(g_bShowHelp)
			RenderText(fTime);
		DXUT_EndPerfEvent();

        V(pd3dDevice->EndScene());
    }
}


//--------------------------------------------------------------------------------------
// Handle messages to the application 
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, 
                          bool* pbNoFurtherProcessing, void* pUserContext)
{
	*pbNoFurtherProcessing = g_Ui.MsgProc(hWnd, uMsg, wParam, lParam);
    if(*pbNoFurtherProcessing)
        return 0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////	
	g_Camera.HandleMessages(hWnd, uMsg, wParam, lParam);

    return 0;
}

void CALLBACK OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext)
{
	if(bKeyDown)
	{
		switch(nChar)
		{
			case VK_F1: g_bShowHelp = !g_bShowHelp; break;
			case VK_TAB: g_bShowUI = !g_bShowUI; break;
			case VK_SPACE: g_bFreeze = !g_bFreeze; break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Release resources created in the OnResetDevice callback here 
//--------------------------------------------------------------------------------------
void CALLBACK OnLostDevice(void* pUserContext)
{
	g_Ui.OnLostDevice();

    if(g_pFont)
        g_pFont->OnLostDevice();
	g_pSprite.release();

    if(g_pEffect)
        g_pEffect->OnLostDevice();
}


//--------------------------------------------------------------------------------------
// Release resources created in the OnCreateDevice callback here
//--------------------------------------------------------------------------------------
void CALLBACK OnDestroyDevice(void* pUserContext)
{
    g_Ui.OnDestroyDevice();

	g_pFont.release();
	g_pEffect.release();
	g_pVBInstanceData.release();
	g_pVertexDeclHardware.release();
	g_d3dDevice = 0;

	g_Core.reset();
}



//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
INT WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    // Set general DXUT callbacks
	DXUTSetCallbackMsgProc(MsgProc);
    DXUTSetCallbackKeyboard(OnKeyboard);
    DXUTSetCallbackFrameMove(OnFrameMove);
    DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);

    // Set the D3D9 DXUT callbacks
    DXUTSetCallbackD3D9DeviceCreated(OnCreateDevice);
    DXUTSetCallbackD3D9DeviceReset(OnResetDevice);
    DXUTSetCallbackD3D9DeviceLost(OnLostDevice);
    DXUTSetCallbackD3D9DeviceDestroyed(OnDestroyDevice);
    DXUTSetCallbackD3D9FrameRender(OnFrameRender);
   
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

    // Initialize DXUT and create the desired Win32 window and Direct3D device for the application
    DXUTInit(true, true, NULL); // Parse the command line, handle the default hotkeys, and show msgboxes
    DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
    DXUTCreateWindow(L"GPU Procedural Wind Animation Sample [DirectX9]");
#if defined(PROFILE) && !defined(DEBUG)
    DXUTCreateDevice(true, 320, 240);
#else
    DXUTCreateDevice(true, 640, 480);
#endif

    // Start the render loop
    DXUTMainLoop();

    return DXUTGetExitCode();
}
