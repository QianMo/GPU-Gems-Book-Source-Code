//-----------------------------------------------------------------------------
// File: bLois.cpp
//
// Desc: DirectX window application created by the DirectX AppWizard
//-----------------------------------------------------------------------------
#define STRICT
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <basetsd.h>
#include <direct.h>
#include <math.h>
#include <stdio.h>
#include <d3dx9.h>
#include <dxerr9.h>
#include <tchar.h>
#include "DXUtil.h"
#include "D3DEnumeration.h"
#include "D3DSettings.h"
#include "D3DApp.h"
#include "D3DFont.h"
#include "D3DFile.h"
#include "D3DUtil.h"
#include "resource.h"
#include "bLois.h"
#include <algorithm>
#include <functional>

static float RandMinusOneToOne()
{
	return float( double(rand()) / double(RAND_MAX) * 2.0 - 1.0 );
}

static float RandZeroToOne()
{
	return float( double(rand()) / double(RAND_MAX) );
}

static const float kGravConst = 30.f;

class ClearVert
{
public:
	D3DXVECTOR3		m_Pos;
	D3DXVECTOR2		m_Uv;
};
static const int kVSize = sizeof(ClearVert);
static const DWORD kClearVertFVF = D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE2(0);


//-----------------------------------------------------------------------------
// Global access to the app (needed for the global WndProc())
//-----------------------------------------------------------------------------
CMyD3DApplication* g_pApp  = NULL;
HINSTANCE          g_hInst = NULL;




//-----------------------------------------------------------------------------
// Name: WinMain()
// Desc: Entry point to the program. Initializes everything, and goes into a
//       message-processing loop. Idle time is used to render the scene.
//-----------------------------------------------------------------------------
INT WINAPI WinMain( HINSTANCE hInst, HINSTANCE, LPSTR, INT )
{
    CMyD3DApplication d3dApp;

    g_pApp  = &d3dApp;
    g_hInst = hInst;

    InitCommonControls();
    if( FAILED( d3dApp.Create( hInst ) ) )
        return 0;

    return d3dApp.Run();
}




//-----------------------------------------------------------------------------
// Name: CMyD3DApplication()
// Desc: Application constructor.   Paired with ~CMyD3DApplication()
//       Member variables should be initialized to a known state here.  
//       The application window has not yet been created and no Direct3D device 
//       has been created, so any initialization that depends on a window or 
//       Direct3D should be deferred to a later stage. 
//-----------------------------------------------------------------------------
CMyD3DApplication::CMyD3DApplication()
{
    m_dwCreationWidth           = 500;
    m_dwCreationHeight          = 375;
    m_strWindowTitle            = TEXT( "bLois" );
    m_d3dEnumeration.AppUsesDepthBuffer   = TRUE;
	m_bStartFullscreen			= false;
	m_bShowCursorWhenFullscreen	= false;

	m_bShowHelp = false;
	m_bSortWater = true;
	m_bDrawBump = false;
	m_LastToggle = 0.f;

    // Create a D3D font using d3dfont.cpp
    m_pFont                     = new CD3DFont( _T("Arial"), 12, D3DFONT_BOLD );
    m_bLoadingApp               = TRUE;

    memset(m_bKey, 0x00, sizeof(m_bKey));

    D3DXMatrixIdentity(&m_matView);
    D3DXMatrixIdentity(&m_matPosition);
    D3DXMatrixIdentity(&m_matProjection);

	m_CompCosinesEff = NULL;
	m_WaterEff = NULL;
	m_WaterMesh = NULL;
	m_LandMesh = NULL;
	m_PillarsMesh = NULL;

	m_EnvMap = NULL;
	m_LandTex = NULL;

	m_CosineLUT = NULL;
	m_BiasNoiseMap = NULL;
	
	m_BumpTex = NULL;
	m_BumpSurf = NULL;
	m_BumpRender = NULL;
	m_BumpVBuffer = NULL;

	m_WaterIndices = NULL;
	m_WaterFacePos = NULL;
	m_WaterSortData = NULL;


	ResetWater();
}




//-----------------------------------------------------------------------------
// Name: ~CMyD3DApplication()
// Desc: Application destructor.  Paired with CMyD3DApplication()
//-----------------------------------------------------------------------------
CMyD3DApplication::~CMyD3DApplication()
{
	delete [] m_WaterIndices;
	delete [] m_WaterFacePos;
	delete [] m_WaterSortData;
}




//-----------------------------------------------------------------------------
// Name: OneTimeSceneInit()
// Desc: Paired with FinalCleanup().
//       The window has been created and the IDirect3D9 interface has been
//       created, but the device has not been created yet.  Here you can
//       perform application-related initialization and cleanup that does
//       not depend on a device.
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::OneTimeSceneInit()
{
    // TODO: perform one time initialization

    // Drawing loading status message until app finishes loading
    SendMessage( m_hWnd, WM_PAINT, 0, 0 );

    m_bLoadingApp = FALSE;

    // Misc stuff

	D3DXMatrixLookAtLH(&m_matView, 
		&D3DXVECTOR3(0.f, -150.f, 50.f),
		&D3DXVECTOR3(0.f, 0.f, 0.f),
		&D3DXVECTOR3(0.f, 0.f, 1.f));

    D3DXMatrixInverse(&m_matPosition, NULL, &m_matView);

    return S_OK;
}









//-----------------------------------------------------------------------------
// Name: ConfirmDevice()
// Desc: Called during device initialization, this code checks the display device
//       for some minimum set of capabilities
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::ConfirmDevice( D3DCAPS9* pCaps, DWORD dwBehavior,
                                          D3DFORMAT Format )
{
    UNREFERENCED_PARAMETER( Format );
    UNREFERENCED_PARAMETER( dwBehavior );
    UNREFERENCED_PARAMETER( pCaps );
    
    BOOL bCapsAcceptable;

    // TODO: Perform checks to see if these display caps are acceptable.
    bCapsAcceptable = TRUE;

    if( bCapsAcceptable )         
        return S_OK;
    else
        return E_FAIL;
}




//-----------------------------------------------------------------------------
// Name: InitDeviceObjects()
// Desc: Paired with DeleteDeviceObjects()
//       The device has been created.  Resources that are not lost on
//       Reset() can be created here -- resources in D3DPOOL_MANAGED,
//       D3DPOOL_SCRATCH, or D3DPOOL_SYSTEMMEM.  Image surfaces created via
//       CreateImageSurface are never lost and can be created here.  Vertex
//       shaders and pixel shaders can also be created here as they are not
//       lost on Reset().
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::InitDeviceObjects()
{
    // TODO: create device objects

    HRESULT hr;

    // Init the font
    m_pFont->InitDeviceObjects( m_pd3dDevice );

	if( FAILED(hr = CreateCosineLUT()) )
		return hr;

	if( FAILED(hr = CreateBiasNoiseMap()) )
		return hr;

	if( FAILED(hr = D3DXCreateCubeTextureFromFile( m_pd3dDevice, TEXT("nvlobby_new_cube_mipmap.dds"), &m_EnvMap)) )
        return DXTRACE_ERR( "EnvMap load", hr );

	if( FAILED(hr = D3DXCreateTextureFromFile( m_pd3dDevice, "Sandy.dds", &m_LandTex)) )
		return hr;

    if( FAILED(hr = D3DXLoadMeshFromX(TEXT("LandMesh.x"), D3DXMESH_MANAGED, m_pd3dDevice, NULL, NULL, NULL, NULL, &m_LandMesh)) )
		return hr;

    if( FAILED(hr = D3DXLoadMeshFromX(TEXT("pillars.x"), D3DXMESH_MANAGED, m_pd3dDevice, NULL, NULL, NULL, NULL, &m_PillarsMesh)) )
		return hr;

	ID3DXMesh* waterMesh = NULL;
    if( FAILED(hr = D3DXLoadMeshFromX(TEXT("WaterMesh.x"), D3DXMESH_MANAGED, m_pd3dDevice, NULL, NULL, NULL, NULL, &waterMesh)) )
		return hr;

	if( FAILED( hr = CreateWaterMesh(waterMesh)) )
		return hr;

	if( FAILED(hr = CreateClearBuffer()) )
		return hr;

    LPD3DXBUFFER pBufferErrors = NULL;
    if( FAILED( hr = D3DXCreateEffectFromFile( m_pd3dDevice, "CompCosines.fx", NULL, NULL,
                                            0, NULL, &m_CompCosinesEff, &pBufferErrors) ) )
    {
		char* errStr = (char*)pBufferErrors->GetBufferPointer();
		OutputDebugString(errStr);
        return hr;
    }
	GetCompCosineEffParams();

	if( FAILED(hr =  D3DXCreateEffectFromFile( m_pd3dDevice, "WaterRip.fx", NULL, NULL,
                                            0, NULL, &m_WaterEff, &pBufferErrors) ) )
    {
		char* errStr = (char*)pBufferErrors->GetBufferPointer();
		OutputDebugString(errStr);
        return hr;
    }
	GetWaterParams();

    return S_OK;
}

HRESULT CMyD3DApplication::CreateWaterMesh(ID3DXMesh* waterMesh)
{
	// All this should happen off line. In particular, note the assumption of even
	// spacing in calculating the edge lengths, which defeats the whole purpose.
	// Still, it's a demo, not a game.
	struct OutVert
	{
		D3DXVECTOR3		m_Pos;
		DWORD			m_Color;
	};

	const DWORD kWaterFVF = D3DFVF_XYZ | D3DFVF_DIFFUSE;
	HRESULT hr = waterMesh->CloneMeshFVF(D3DXMESH_MANAGED,
							kWaterFVF,
							m_pd3dDevice,
							&m_WaterMesh);

	SAFE_RELEASE(waterMesh);

	if( FAILED(hr) )
		return hr;

	const int nVerts = m_WaterMesh->GetNumVertices();

	OutVert* oVert;
	if( FAILED(hr = m_WaterMesh->LockVertexBuffer(0, (void**)&oVert)) )
		return hr;
	OutVert* origVert = oVert;

	D3DXVECTOR3 del = oVert[0].m_Pos - oVert[1].m_Pos;
	float dist = D3DXVec3Length(&del);
	if( dist < 1.f )
		dist = 1.f;
	UINT alpha = UINT(255.9f / dist);
	int i;
	for( i = 0; i < nVerts; i++ )
	{
		oVert->m_Color = alpha << 24 | 0x00ffffff;
		oVert++;
	}

	oVert = origVert;

	const int nFaces = m_WaterMesh->GetNumFaces();

	m_WaterIndices = new FaceIndices[nFaces];
	m_WaterFacePos = new D3DXVECTOR3[nFaces];
	m_WaterSortData = new FaceSortData[nFaces];

	DWORD* oIdx;
	if( FAILED(hr = m_WaterMesh->LockIndexBuffer(0, (void**)&oIdx)) )
	{
		m_WaterMesh->UnlockVertexBuffer();
		return hr;
	}
	memcpy(m_WaterIndices, oIdx, nFaces * sizeof(*m_WaterIndices));

	for( i = 0; i < nFaces; i++ )
	{
		D3DXVECTOR3 pos = oVert[m_WaterIndices[i].m_Idx[0]].m_Pos;

		pos += oVert[m_WaterIndices[i].m_Idx[1]].m_Pos;
		pos += oVert[m_WaterIndices[i].m_Idx[2]].m_Pos;
		pos.z = 0;

		pos /= 3.f;

		m_WaterFacePos[i] = pos;
	}


	m_WaterMesh->UnlockIndexBuffer();

	m_WaterMesh->UnlockVertexBuffer();

	return hr;
}

struct CompSortFace : public std::binary_function<FaceSortData, FaceSortData, bool>
{
	bool operator()( const FaceSortData& lhs, const FaceSortData& rhs) const
	{
		return lhs.m_Dist < rhs.m_Dist;
	}
};


void CMyD3DApplication::SortWaterMesh()
{
	const D3DXVECTOR3 eyePos(m_matPosition._41, m_matPosition._42, 0.f);
	const int nFaces = m_WaterMesh->GetNumFaces();
	int i;
	for( i = 0; i < nFaces; i++ )
	{
		D3DXVECTOR3 del(eyePos - m_WaterFacePos[i]);
		m_WaterSortData[i].m_Dist = D3DXVec3LengthSq(&del);
		m_WaterSortData[i].m_Idx = i;
	}

	FaceSortData* begin = m_WaterSortData;
	FaceSortData* end = begin + nFaces;

	std::sort(begin, end, CompSortFace());

	FaceIndices* oIdx;
	if( FAILED(m_WaterMesh->LockIndexBuffer(0, (void**)&oIdx)) )
	{
		m_WaterMesh->UnlockVertexBuffer();
		return;
	}

	for( i = 0; i < nFaces; i++ )
	{
		oIdx[i].m_Idx[0] = m_WaterIndices[m_WaterSortData[i].m_Idx].m_Idx[0];
		oIdx[i].m_Idx[1] = m_WaterIndices[m_WaterSortData[i].m_Idx].m_Idx[1];
		oIdx[i].m_Idx[2] = m_WaterIndices[m_WaterSortData[i].m_Idx].m_Idx[2];
	}

	m_WaterMesh->UnlockIndexBuffer();
}

HRESULT CMyD3DApplication::CreateClearBuffer()
{
	HRESULT hr;
	if( FAILED( hr = m_pd3dDevice->CreateVertexBuffer( 4 * kVSize,
												D3DUSAGE_WRITEONLY, 
												0,
												D3DPOOL_MANAGED, 
												&m_BumpVBuffer,
												NULL) ) )
	{
		return hr;
	}


	ClearVert* ptr;
	if( FAILED( hr = m_BumpVBuffer->Lock( 0, 0, (void **)&ptr, 0 ) ) )
		return hr;

	ptr[2].m_Pos.x = -1.f;
	ptr[2].m_Pos.y = -1.f;
	ptr[2].m_Pos.z = 0.5f;

	ptr[2].m_Uv.x = 0.5f / kBumpTexSize;
	ptr[1].m_Uv.y = 0.5f / kBumpTexSize;

	ptr[0] = ptr[2];
	ptr[0].m_Pos.y += 2.f;
	ptr[0].m_Uv.y += 1.f;

	ptr[1] = ptr[2];
	ptr[1].m_Pos.x += 2.f;
	ptr[1].m_Pos.y += 2.f;
	ptr[1].m_Uv.x += 1.f;
	ptr[1].m_Uv.y += 1.f;

	ptr[3] = ptr[2];
	ptr[3].m_Pos.x += 2.f;
	ptr[3].m_Uv.x += 1.f;

	m_BumpVBuffer->Unlock();

	return S_OK;
}


//-----------------------------------------------------------------------------
// Name: RestoreDeviceObjects()
// Desc: Paired with InvalidateDeviceObjects()
//       The device exists, but may have just been Reset().  Resources in
//       D3DPOOL_DEFAULT and any other device state that persists during
//       rendering should be set here.  Render states, matrices, textures,
//       etc., that don't change during rendering can be set once here to
//       avoid redundant state setting during Render() or FrameMove().
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::RestoreDeviceObjects()
{
    // TODO: setup render states

    // Setup a material
    D3DMATERIAL9 mtrl;
    D3DUtil_InitMaterial( mtrl, 1.0f, 1.0f, 1.0f );
    m_pd3dDevice->SetMaterial( &mtrl );

    // Set up the textures
    m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_MODULATE );
    m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
    m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );
    m_pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP,   D3DTOP_MODULATE );
    m_pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE );
    m_pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE );
    m_pd3dDevice->SetSamplerState( 0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR );
    m_pd3dDevice->SetSamplerState( 0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR );

    // Set miscellaneous render states
	m_pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    m_pd3dDevice->SetRenderState( D3DRS_DITHERENABLE,   FALSE );
    m_pd3dDevice->SetRenderState( D3DRS_SPECULARENABLE, FALSE );
    m_pd3dDevice->SetRenderState( D3DRS_ZENABLE,        TRUE );
    m_pd3dDevice->SetRenderState( D3DRS_AMBIENT,        0x000F0F0F );
	m_pd3dDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

    // Set the world matrix
    D3DXMATRIX matIdentity;
    D3DXMatrixIdentity( &matIdentity );
    m_pd3dDevice->SetTransform( D3DTS_WORLD,  &matIdentity );


    // Set the projection matrix
    FLOAT fAspect = ((FLOAT)m_d3dsdBackBuffer.Width) / m_d3dsdBackBuffer.Height;
    D3DXMatrixPerspectiveFovLH( &m_matProjection, D3DX_PI/4, fAspect, 1.0f, 10000.0f );
    m_pd3dDevice->SetTransform( D3DTS_PROJECTION, &m_matProjection);

    // Set up lighting states
    D3DLIGHT9 light;
    D3DUtil_InitLight( light, D3DLIGHT_DIRECTIONAL, -1.0f, -1.0f, -2.0f );
    m_pd3dDevice->SetLight( 0, &light );
    m_pd3dDevice->LightEnable( 0, TRUE );
    m_pd3dDevice->SetRenderState( D3DRS_LIGHTING, TRUE );

    // Restore the font
    m_pFont->RestoreDeviceObjects();

	HRESULT hr;
	// Create our bump map. We'll composite the normals of our texture waves into this with renders.
    if(FAILED(hr = D3DXCreateTexture(m_pd3dDevice, kBumpTexSize, kBumpTexSize, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &m_BumpTex)) &&
       FAILED(hr = D3DXCreateTexture(m_pd3dDevice, kBumpTexSize, kBumpTexSize, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &m_BumpTex)))
    {
        return hr;
    }

    D3DSURFACE_DESC desc;
    m_BumpTex->GetSurfaceLevel(0, &m_BumpSurf);
    m_BumpSurf->GetDesc(&desc);

    if(FAILED(hr = D3DXCreateRenderToSurface(m_pd3dDevice, desc.Width, desc.Height, 
									        desc.Format, FALSE, D3DFMT_UNKNOWN, &m_BumpRender)))
        return hr;

    m_CompCosinesEff->OnResetDevice();
	m_WaterEff->OnResetDevice();

    return S_OK;
}

static void Clamp(float& val, float lo, float hi)
{
	if( val < lo )
		val = lo;
	else if( val > hi )
		val = hi;
}

void CMyD3DApplication::MoveOnInput()
{
    //
    // Process keyboard input
    //

    D3DXVECTOR3 vecT(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 vecR(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 vecZ(0.0f, 0.0f, 0.0f);

    if(m_bKey[VK_NUMPAD1] || m_bKey[VK_LEFT])  vecT.x -= 1.0f; // Slide Left
    if(m_bKey[VK_NUMPAD3] || m_bKey[VK_RIGHT]) vecT.x += 1.0f; // Slide Right
    if(m_bKey[VK_DOWN])                        vecT.y -= 1.0f; // Slide Down
    if(m_bKey[VK_UP])                          vecT.y += 1.0f; // Slide Up
    if(m_bKey['W'])                            vecT.z += 2.0f; // Move Forward
    if(m_bKey['S'])                            vecT.z -= 2.0f; // Move Backward
    if(m_bKey['A'] || m_bKey[VK_NUMPAD8])      vecR.x -= 1.0f; // Pitch Down
    if(m_bKey['Z'] || m_bKey[VK_NUMPAD2])      vecR.x += 1.0f; // Pitch Up
    if(m_bKey['E'] || m_bKey[VK_NUMPAD6])      vecZ.z -= 1.0f; // Turn Right
    if(m_bKey['Q'] || m_bKey[VK_NUMPAD4])      vecZ.z += 1.0f; // Turn Left
    if(m_bKey[VK_NUMPAD9])                     vecR.z -= 2.0f; // Roll CW
    if(m_bKey[VK_NUMPAD7])                     vecR.z += 2.0f; // Roll CCW

	const float speed = 10.f;
	const float angSpeed = D3DX_PI / 5.f;
	vecT *= speed * m_fElapsedTime;
	vecR *= angSpeed * m_fElapsedTime;
	vecZ *= angSpeed * m_fElapsedTime;

	if(m_bKey[VK_SHIFT])
	{
		vecT *= 4.f;
		vecR *= 4.f;
		vecZ *= 4.f;
	}

    //
    // Update position and view matricies
    //

    D3DXMATRIXA16 matT, matR, matZ;
    D3DXQUATERNION qR;
    D3DXQUATERNION qZ;


    D3DXMatrixTranslation(&matT, vecT.x, vecT.y, vecT.z);
    D3DXMatrixMultiply(&m_matPosition, &matT, &m_matPosition);

    D3DXQuaternionRotationYawPitchRoll(&qR, vecR.y, vecR.x, vecR.z);
    D3DXMatrixRotationQuaternion(&matR, &qR);

    D3DXQuaternionRotationYawPitchRoll(&qZ, vecZ.y, vecZ.x, vecZ.z);
    D3DXMatrixRotationQuaternion(&matZ, &qZ);

    D3DXMatrixMultiply(&m_matPosition, &matR, &m_matPosition);
    D3DXMatrixMultiply(&m_matPosition, &m_matPosition, &matZ);

    D3DXMatrixInverse(&m_matView, NULL, &m_matPosition);

	float timeScale = m_bKey[VK_SHIFT] ? m_fElapsedTime : -m_fElapsedTime;
	// MoreOptions
	// Debounce the toggles
	const float kMinToggle = 0.5f;
	if( m_fTime - m_LastToggle > kMinToggle )
	{
		// Reset
		if(m_bKey['R'])
		{
			if( m_bKey[VK_SHIFT] )
				ResetWater();
			else
				InitWaves();
			m_LastToggle = m_fTime;
		}
		// Show bump
		if(m_bKey['B'])
		{
			m_bDrawBump = !m_bDrawBump;
			m_LastToggle = m_fTime;
		}

		if(m_bKey[VK_SPACE])
		{
			m_bShowHelp = !m_bShowHelp;
			m_LastToggle = m_fTime;
		}
	}
	// Water Height
	if(m_bKey['H'])
		m_GeoState.m_WaterLevel += timeScale * 3.f;
	// Geo Wave Height
	if(m_bKey['J'])
	{
		m_GeoState.m_AmpOverLen += timeScale * 0.05f;
		Clamp(m_GeoState.m_AmpOverLen, 0.f, 0.1f);
	}
	// Geo Chop
	if(m_bKey['K'])
	{
		m_GeoState.m_Chop += timeScale * 0.5f;
		Clamp(m_GeoState.m_Chop, 0.f, 4.f);
	}
	// Tex Wave Height
	if(m_bKey['U'])
	{
		m_TexState.m_AmpOverLen += timeScale * 0.05f;
		Clamp(m_TexState.m_AmpOverLen, 0.f, 0.5f);
	}
	// Tex scale
	if(m_bKey['Y'])
	{
		m_TexState.m_RippleScale += timeScale * 1.f;
		Clamp(m_TexState.m_RippleScale, 5.f, 50.f);
	}
	if(m_bKey['N'])
	{
		m_TexState.m_Noise += timeScale * 0.1f;
		Clamp(m_TexState.m_Noise, 0.f, 1.f);
	}
	if(m_bKey['O'])
	{
		m_GeoState.m_AngleDeviation += timeScale * 10.f;
		Clamp(m_GeoState.m_AngleDeviation, 0.f, 180.f);
	}
	if(m_bKey['P'])
	{
		m_TexState.m_AngleDeviation += timeScale * 10.f;
		Clamp(m_TexState.m_AngleDeviation, 0.f, 180.f);
	}
	if(m_bKey['G'])
	{
		m_GeoState.m_EnvRadius *= 1.f + timeScale * 0.5f;
		Clamp(m_GeoState.m_EnvRadius, 100.f, 10000.f);
	}
	if(m_bKey['F'])
	{
		m_GeoState.m_EnvHeight += timeScale * 10.f;
		Clamp(m_GeoState.m_EnvHeight, -100.f, 100.f);
	}
}


//-----------------------------------------------------------------------------
// Name: FrameMove()
// Desc: Called once per frame, the call is the entry point for animating
//       the scene.
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::FrameMove()
{
    // TODO: update world

    // Update user input state
	MoveOnInput();

	UpdateTexWaves(m_fElapsedTime);
	UpdateGeoWaves(m_fElapsedTime);

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name: UpdateInput()
// Desc: Update the user input.  Called once per frame 
//-----------------------------------------------------------------------------
void CMyD3DApplication::UpdateInput( UserInput* pUserInput )
{
    pUserInput->bRotateUp    = ( m_bActive && (GetAsyncKeyState( VK_UP )    & 0x8000) == 0x8000 );
    pUserInput->bRotateDown  = ( m_bActive && (GetAsyncKeyState( VK_DOWN )  & 0x8000) == 0x8000 );
    pUserInput->bRotateLeft  = ( m_bActive && (GetAsyncKeyState( VK_LEFT )  & 0x8000) == 0x8000 );
    pUserInput->bRotateRight = ( m_bActive && (GetAsyncKeyState( VK_RIGHT ) & 0x8000) == 0x8000 );

	pUserInput->bMoveUp = ( m_bActive && (GetAsyncKeyState( VK_UP )    & 0x8000) == 0x8000 );
}




//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Called once per frame, the call is the entry point for 3d
//       rendering. This function sets up render states, clears the
//       viewport, and renders the scene.
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::Render()
{
	RenderTexture();

    // Clear the viewport
    m_pd3dDevice->Clear( 0L, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER,
                         0x00707070, 1.0f, 0L );

    // Begin the scene
    if( SUCCEEDED( m_pd3dDevice->BeginScene() ) )
    {
        // TODO: render world
		m_pd3dDevice->SetTransform(D3DTS_PROJECTION, &m_matProjection);
		m_pd3dDevice->SetTransform(D3DTS_VIEW, &m_matView);
        
		m_pd3dDevice->SetTexture(0, m_LandTex);
		m_LandMesh->DrawSubset(0);
		m_pd3dDevice->SetTexture(0, NULL);

		m_PillarsMesh->DrawSubset(0);

		RenderWater();

		if( m_bDrawBump )
		{
			D3DXMATRIXA16 matIdent;
			D3DXMatrixIdentity(&matIdent);

			m_pd3dDevice->SetTransform(D3DTS_VIEW, &matIdent);
			matIdent._11 = 0.5f;
			matIdent._22 = 0.5f;
			m_pd3dDevice->SetTransform(D3DTS_PROJECTION, &matIdent);

			m_pd3dDevice->SetRenderState(D3DRS_SRCBLEND,  D3DBLEND_ONE);
			m_pd3dDevice->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ZERO);
			m_pd3dDevice->SetRenderState(D3DRS_ALPHAFUNC, D3DCMP_ALWAYS);

			m_pd3dDevice->SetTexture(0, m_BumpTex);
			m_pd3dDevice->SetFVF(kClearVertFVF);
			m_pd3dDevice->SetStreamSource(0, m_BumpVBuffer, 0, kVSize);

			m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_SELECTARG1 );
			m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
			m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );

			m_pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

			m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_MODULATE );
			m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
			m_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );
}

        // Render stats and help text  
        RenderText();

        // End the scene.
        m_pd3dDevice->EndScene();
    }

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name: RenderText()
// Desc: Renders stats and help text to the scene.
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::RenderText()
{
    D3DCOLOR fontColor        = D3DCOLOR_ARGB(255,255,255,0);
    TCHAR szMsg[MAX_PATH] = TEXT("");

    // Output display stats
    FLOAT fNextLine = 40.0f; 

    lstrcpy( szMsg, m_strDeviceStats );
    fNextLine -= 20.0f;
    m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

    lstrcpy( szMsg, m_strFrameStats );
    fNextLine -= 20.0f;
    m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

    // Output statistics & help
    fNextLine = (FLOAT) m_d3dsdBackBuffer.Height; 


    lstrcpy( szMsg, TEXT("Spacebar toggles help") );
    fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

	if( m_bShowHelp )
	{
		lstrcpy( szMsg, TEXT("Press 'F2' to configure display") );
		fNextLine -= 40.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'F' - Environment map height") );
		fNextLine -= 40.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'G' - Environment map radius") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'P' - Texture wave angle deviation") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'O' - Geometric wave angle deviation") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'N' - Texture Noise") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'Y' - Texture scaling") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'U' - Texture wave height") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'K' - Geometric wave choppiness") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'J' - Geometric wave height") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'H' - Water level") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("Parameters - Shift goes up, non-shift goes down") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'B' - Show bump map") );
		fNextLine -= 40.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'r' - Reinitializes system with current settings") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("'R' - Resets system settings to default and re-init") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("Toggles:") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("Left/right arrow - slide left/right") );
		fNextLine -= 40.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("Up/down arrow - move up/down") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("W/S - move forward/backward") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("A/Z - pitch up/down") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("E/Q - rotate scene right/left") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );

		lstrcpy( szMsg, TEXT("Movement - Shift is faster:") );
		fNextLine -= 20.0f; m_pFont->DrawText( 2, fNextLine, fontColor, szMsg );


	}

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: Overrrides the main WndProc, so the sample can do custom message
//       handling (e.g. processing mouse, keyboard, or menu commands).
//-----------------------------------------------------------------------------
LRESULT CMyD3DApplication::MsgProc( HWND hWnd, UINT msg, WPARAM wParam,
                                    LPARAM lParam )
{
    switch( msg )
    {
        case WM_KEYDOWN:
            m_bKey[wParam] = TRUE;
            break;

        case WM_KEYUP:
            m_bKey[wParam] = FALSE;
            break;
        case WM_PAINT:
        {
            if( m_bLoadingApp )
            {
                // Draw on the window tell the user that the app is loading
                // TODO: change as needed
                HDC hDC = GetDC( hWnd );
                TCHAR strMsg[MAX_PATH];
                wsprintf( strMsg, TEXT("Loading... Please wait") );
                RECT rct;
                GetClientRect( hWnd, &rct );
                DrawText( hDC, strMsg, -1, &rct, DT_CENTER|DT_VCENTER|DT_SINGLELINE );
                ReleaseDC( hWnd, hDC );
            }
            break;
        }

    }

    return CD3DApplication::MsgProc( hWnd, msg, wParam, lParam );
}




//-----------------------------------------------------------------------------
// Name: InvalidateDeviceObjects()
// Desc: Invalidates device objects.  Paired with RestoreDeviceObjects()
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::InvalidateDeviceObjects()
{
    // TODO: Cleanup any objects created in RestoreDeviceObjects()
    m_pFont->InvalidateDeviceObjects();
    m_CompCosinesEff->OnLostDevice();
	m_WaterEff->OnLostDevice();

	SAFE_RELEASE(m_BumpTex);
	SAFE_RELEASE(m_BumpSurf);
	SAFE_RELEASE(m_BumpRender);

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name: DeleteDeviceObjects()
// Desc: Paired with InitDeviceObjects()
//       Called when the app is exiting, or the device is being changed,
//       this function deletes any device dependent objects.  
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::DeleteDeviceObjects()
{
    // TODO: Cleanup any objects created in InitDeviceObjects()
    m_pFont->DeleteDeviceObjects();

	SAFE_RELEASE(m_CompCosinesEff);
	SAFE_RELEASE(m_WaterEff);
	SAFE_RELEASE(m_CosineLUT);
	SAFE_RELEASE(m_BiasNoiseMap);
	SAFE_RELEASE(m_EnvMap);
	SAFE_RELEASE(m_LandTex);

	SAFE_RELEASE(m_WaterMesh);
	SAFE_RELEASE(m_LandMesh);
	SAFE_RELEASE(m_PillarsMesh);

	SAFE_RELEASE(m_BumpVBuffer);

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name: FinalCleanup()
// Desc: Paired with OneTimeSceneInit()
//       Called before the app exits, this function gives the app the chance
//       to cleanup after itself.
//-----------------------------------------------------------------------------
HRESULT CMyD3DApplication::FinalCleanup()
{
    // TODO: Perform any final cleanup needed
    // Cleanup D3D font
    SAFE_DELETE( m_pFont );

    return S_OK;
}

HRESULT CMyD3DApplication::CreateCosineLUT()
{
	HRESULT hr = D3DXCreateTexture(m_pd3dDevice,
		kBumpTexSize,
		1,
		1,
		0,
		D3DFMT_A8R8G8B8,
		D3DPOOL_MANAGED,
		&m_CosineLUT);

	if( FAILED(hr) )
		return hr;

	D3DLOCKED_RECT lockedRect;
	hr = m_CosineLUT->LockRect(0, &lockedRect, NULL, 0);
	if( FAILED(hr) )
		return hr;

	unsigned long* pDat = (unsigned long*)lockedRect.pBits;
	int i;
	for( i = 0; i < kBumpTexSize; i++ )
	{
		float dist = float(i) / float(kBumpTexSize-1) * 2.f * D3DX_PI;
		float c = float(cos(dist));
		float s = float(sin(dist));
		s *= 0.5f;
		s += 0.5f;
		s = float(pow(s, m_TexState.m_Chop));
		c *= s;
		unsigned char cosDist = (unsigned char)((c * 0.5 + 0.5) * 255.999f);
		pDat[i] = (0xff << 24)
					| (cosDist << 16)
					| (cosDist << 8)
					| 0xff;
	}

	m_CosineLUT->UnlockRect(0);

	return S_OK;
}

HRESULT CMyD3DApplication::CreateBiasNoiseMap()
{
	HRESULT hr = D3DXCreateTexture(m_pd3dDevice,
		kBumpTexSize,
		kBumpTexSize,
		1,
		0,
		D3DFMT_A8R8G8B8,
		D3DPOOL_MANAGED,
		&m_BiasNoiseMap);

	if( FAILED(hr) )
		return hr;

	D3DLOCKED_RECT lockedRect;
	hr = m_BiasNoiseMap->LockRect(0, &lockedRect, NULL, 0);
	if( FAILED(hr) )
		return hr;

	unsigned long* pDat = (unsigned long*)lockedRect.pBits;

	const int size = kBumpTexSize;
	int i;
	for( i = 0; i < size; i++ )
	{
		int j;
		for( j = 0; j < size; j++ )
		{
			float x = RandZeroToOne();
			float y = RandZeroToOne();

			unsigned char r = (unsigned char)(x * 255.999f);
			unsigned char g = (unsigned char)(y * 255.999f);
	
			pDat[j] = (0xff << 24)
				| (r << 16)
				| (g << 8)
				| 0xff;
		}

		pDat += lockedRect.Pitch / 4;
	}

	m_BiasNoiseMap->UnlockRect(0);

	return S_OK;
}

void CMyD3DApplication::InitTexWaves()
{
	int i;
	for( i = 0; i < kNumTexWaves; i++ )
		InitTexWave(i);
}

void CMyD3DApplication::UpdateTexWaves(FLOAT dt)
{
	int i;
	for( i = 0; i < kNumTexWaves; i++ )
		UpdateTexWave(i, dt);
}

void CMyD3DApplication::UpdateTexWave(int i, FLOAT dt)
{
	if( i == m_TexState.m_TransIdx )
	{
		m_TexWaves[i].m_Fade += m_TexState.m_TransDel * dt;
		if( m_TexWaves[i].m_Fade < 0 )
		{
			// This wave is faded out. Re-init and fade it back up.
			InitTexWave(i);
			m_TexWaves[i].m_Fade = 0;
			m_TexState.m_TransDel = -m_TexState.m_TransDel;
		}
		else if( m_TexWaves[i].m_Fade > 1.f )
		{
			// This wave is faded back up. Start fading another down.
			m_TexWaves[i].m_Fade = 1.f;
			m_TexState.m_TransDel = -m_TexState.m_TransDel;
			if( ++m_TexState.m_TransIdx >= kNumTexWaves )
				m_TexState.m_TransIdx = 0;
		}
	}
	m_TexWaves[i].m_Phase -= dt * m_TexWaves[i].m_Speed;
	m_TexWaves[i].m_Phase -= int(m_TexWaves[i].m_Phase);
}

void CMyD3DApplication::InitTexWave(int i)
{
	float rads = RandMinusOneToOne() * m_TexState.m_AngleDeviation * D3DX_PI / 180.f;
	float dx = float(sin(rads));
	float dy = float(cos(rads));


	float tx = dx;
	dx = m_TexState.m_WindDir.y * dx - m_TexState.m_WindDir.x * dy;
	dy = m_TexState.m_WindDir.x * tx + m_TexState.m_WindDir.y * dy;

	float maxLen = m_TexState.m_MaxLength * kBumpTexSize / m_TexState.m_RippleScale;
	float minLen = m_TexState.m_MinLength * kBumpTexSize / m_TexState.m_RippleScale;
	float len = float(i) / float(kNumTexWaves-1) * (maxLen - minLen) + minLen;

	float reps = float(kBumpTexSize) / len;

	dx *= reps;
	dy *= reps;
	dx = float(int(dx >= 0 ? dx + 0.5f : dx - 0.5f));
	dy = float(int(dy >= 0 ? dy + 0.5f : dy - 0.5f));

	m_TexWaves[i].m_RotScale.x = dx;
	m_TexWaves[i].m_RotScale.y = dy;

	float effK = float(1.0 / sqrt(dx*dx + dy*dy));
	m_TexWaves[i].m_Len = float(kBumpTexSize) * effK;
	m_TexWaves[i].m_Freq = D3DX_PI * 2.f / m_TexWaves[i].m_Len;
	m_TexWaves[i].m_Amp = m_TexWaves[i].m_Len * m_TexState.m_AmpOverLen;
	m_TexWaves[i].m_Phase = RandZeroToOne();
	
	m_TexWaves[i].m_Dir.x = dx * effK;
	m_TexWaves[i].m_Dir.y = dy * effK;

	m_TexWaves[i].m_Fade = 1.f;

	float speed = float( 1.0 / sqrt(m_TexWaves[i].m_Len / (2.f * D3DX_PI * kGravConst)) ) / 3.f;
	speed *= 1.f + RandMinusOneToOne() * m_TexState.m_SpeedDeviation;
	m_TexWaves[i].m_Speed = speed;
}

void CMyD3DApplication::InitTexState()
{
	m_TexState.m_Noise = 0.2f;
	m_TexState.m_Chop = 1.f;
	m_TexState.m_AngleDeviation = 15.f;
	m_TexState.m_WindDir.x = 0;
	m_TexState.m_WindDir.y = 1.f;
	m_TexState.m_MaxLength = 10.f;
	m_TexState.m_MinLength = 1.f;
	m_TexState.m_AmpOverLen = 0.1f;
	m_TexState.m_RippleScale = 25.f;
	m_TexState.m_SpeedDeviation = 0.1f;

	m_TexState.m_TransIdx = 0;
	m_TexState.m_TransDel = -1.f / 5.f;
}

void CMyD3DApplication::GetCompCosineEffParams()
{
	m_CompCosineParams.m_UTrans[0] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans0");
	m_CompCosineParams.m_UTrans[1] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans1");
	m_CompCosineParams.m_UTrans[2] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans2");
	m_CompCosineParams.m_UTrans[3] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans3");
	m_CompCosineParams.m_UTrans[4] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans4");
	m_CompCosineParams.m_UTrans[5] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans5");
	m_CompCosineParams.m_UTrans[6] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans6");
	m_CompCosineParams.m_UTrans[7] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans7");
	m_CompCosineParams.m_UTrans[8] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans8");
	m_CompCosineParams.m_UTrans[9] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans9");
	m_CompCosineParams.m_UTrans[10] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans10");
	m_CompCosineParams.m_UTrans[11] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans11");
	m_CompCosineParams.m_UTrans[12] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans12");
	m_CompCosineParams.m_UTrans[13] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans13");
	m_CompCosineParams.m_UTrans[14] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans14");
	m_CompCosineParams.m_UTrans[15] = m_CompCosinesEff->GetParameterByName(NULL, "cUTrans15");

	m_CompCosineParams.m_Coef[0] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef0");
	m_CompCosineParams.m_Coef[1] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef1");
	m_CompCosineParams.m_Coef[2] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef2");
	m_CompCosineParams.m_Coef[3] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef3");
	m_CompCosineParams.m_Coef[4] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef4");
	m_CompCosineParams.m_Coef[5] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef5");
	m_CompCosineParams.m_Coef[6] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef6");
	m_CompCosineParams.m_Coef[7] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef7");
	m_CompCosineParams.m_Coef[8] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef8");
	m_CompCosineParams.m_Coef[9] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef9");
	m_CompCosineParams.m_Coef[10] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef10");
	m_CompCosineParams.m_Coef[11] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef11");
	m_CompCosineParams.m_Coef[12] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef12");
	m_CompCosineParams.m_Coef[13] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef13");
	m_CompCosineParams.m_Coef[14] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef14");
	m_CompCosineParams.m_Coef[15] = m_CompCosinesEff->GetParameterByName(NULL, "cCoef15");

	m_CompCosineParams.m_ReScale = m_CompCosinesEff->GetParameterByName(NULL, "cReScale");

	m_CompCosineParams.m_NoiseXform[0] = m_CompCosinesEff->GetParameterByName(NULL, "cNoiseXForm0_00");
	m_CompCosineParams.m_NoiseXform[1] = m_CompCosinesEff->GetParameterByName(NULL, "cNoiseXForm0_10");
	m_CompCosineParams.m_NoiseXform[2] = m_CompCosinesEff->GetParameterByName(NULL, "cNoiseXForm1_00");
	m_CompCosineParams.m_NoiseXform[3] = m_CompCosinesEff->GetParameterByName(NULL, "cNoiseXForm1_10");

	D3DXVECTOR4 init(20.f, 0.f, 0.f, 0.f);
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_NoiseXform[0], &init);
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_NoiseXform[2], &init);
	init.x = 0;
	init.y = 20.f;
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_NoiseXform[1], &init);
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_NoiseXform[3], &init);

	m_CompCosineParams.m_ScaleBias = m_CompCosinesEff->GetParameterByName(NULL, "cScaleBias");

	m_CompCosineParams.m_CosineLUT = m_CompCosinesEff->GetParameterByName(NULL, "tCosineLUT");
	m_CompCosineParams.m_BiasNoise = m_CompCosinesEff->GetParameterByName(NULL, "tBiasNoise");
}

void CMyD3DApplication::SetCompCosineEffParams()
{
	int i;
	for( i = 0; i < 16; i++ )
	{
		D3DXVECTOR4 UTrans(m_TexWaves[i].m_RotScale.x, m_TexWaves[i].m_RotScale.y, 0.f, m_TexWaves[i].m_Phase);
		m_CompCosinesEff->SetVector(m_CompCosineParams.m_UTrans[i], &UTrans);

		float normScale = m_TexWaves[i].m_Fade / float(kNumBumpPasses);
		D3DXVECTOR4 Coef(m_TexWaves[i].m_Dir.x * normScale, m_TexWaves[i].m_Dir.y * normScale, 1.f, 1.f);
		m_CompCosinesEff->SetVector(m_CompCosineParams.m_Coef[i], &Coef);

	}

	D3DXVECTOR4 xform;
	
	const FLOAT kRate = 0.1f;
	m_CompCosinesEff->GetVector(m_CompCosineParams.m_NoiseXform[0], &xform);
	xform.w += m_fElapsedTime * kRate;
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_NoiseXform[0], &xform);

	m_CompCosinesEff->GetVector(m_CompCosineParams.m_NoiseXform[3], &xform);
	xform.w += m_fElapsedTime * kRate;
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_NoiseXform[3], &xform);

	float s = 0.5f / (float(kNumBumpPerPass) + m_TexState.m_Noise);
	D3DXVECTOR4 reScale(s, s, 1.f, 1.f);
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_ReScale, &reScale);

	float scaleBias = 0.5f * m_TexState.m_Noise / (float(kNumBumpPasses) + m_TexState.m_Noise);
	D3DXVECTOR4 scaleBiasVec(scaleBias, scaleBias, 0.f, 1.f);
	m_CompCosinesEff->SetVector(m_CompCosineParams.m_ScaleBias, &scaleBiasVec);

	m_CompCosinesEff->SetTexture(m_CompCosineParams.m_CosineLUT, m_CosineLUT);
	m_CompCosinesEff->SetTexture(m_CompCosineParams.m_BiasNoise, m_BiasNoiseMap);
}

void CMyD3DApplication::InitGeoWaves()
{
	int i;
	for( i = 0; i < kNumGeoWaves; i++ )
		InitGeoWave(i);
}

void CMyD3DApplication::InitGeoWave(int i)
{
	m_GeoWaves[i].m_Phase = RandZeroToOne() * D3DX_PI * 2.f;
	m_GeoWaves[i].m_Len = m_GeoState.m_MinLength + RandZeroToOne() * (m_GeoState.m_MaxLength - m_GeoState.m_MinLength);
	m_GeoWaves[i].m_Amp = m_GeoWaves[i].m_Len * m_GeoState.m_AmpOverLen / float(kNumGeoWaves);
	m_GeoWaves[i].m_Freq = 2.f * D3DX_PI / m_GeoWaves[i].m_Len;
	m_GeoWaves[i].m_Fade = 1.f;

	float rotBase = m_GeoState.m_AngleDeviation * D3DX_PI / 180.f;

	float rads = rotBase * RandMinusOneToOne();
	float rx = float(cosf(rads));
	float ry = float(sinf(rads));

	float x = m_GeoState.m_WindDir.x;
	float y = m_GeoState.m_WindDir.y;
	m_GeoWaves[i].m_Dir.x = x * rx + y * ry;
	m_GeoWaves[i].m_Dir.y = x * -ry + y * rx;
}

void CMyD3DApplication::UpdateGeoWave(int i, FLOAT dt)
{
	if( i == m_GeoState.m_TransIdx )
	{
		m_GeoWaves[i].m_Fade += m_GeoState.m_TransDel * dt;
		if( m_GeoWaves[i].m_Fade < 0 )
		{
			// This wave is faded out. Re-init and fade it back up.
			InitGeoWave(i);
			m_GeoWaves[i].m_Fade = 0;
			m_GeoState.m_TransDel = -m_GeoState.m_TransDel;
		}
		else if( m_GeoWaves[i].m_Fade > 1.f )
		{
			// This wave is faded back up. Start fading another down.
			m_GeoWaves[i].m_Fade = 1.f;
			m_GeoState.m_TransDel = -m_GeoState.m_TransDel;
			if( ++m_GeoState.m_TransIdx >= kNumGeoWaves )
				m_GeoState.m_TransIdx = 0;
		}
	}

	const float speed = float(1.0 / sqrt(m_GeoWaves[i].m_Len / (2.f * D3DX_PI * kGravConst)));

	m_GeoWaves[i].m_Phase += speed * dt;
	m_GeoWaves[i].m_Phase = float(fmod(m_GeoWaves[i].m_Phase, 2.f*D3DX_PI));

	m_GeoWaves[i].m_Amp = m_GeoWaves[i].m_Len * m_GeoState.m_AmpOverLen / float(kNumGeoWaves) * m_GeoWaves[i].m_Fade;
}

void CMyD3DApplication::UpdateGeoWaves(FLOAT dt)
{
	int i;
	for( i = 0; i < kNumGeoWaves; i++ )
		UpdateGeoWave(i, dt);
}

void CMyD3DApplication::InitGeoState()
{
	m_GeoState.m_Chop = 2.5f;
	m_GeoState.m_AngleDeviation = 15.f;
	m_GeoState.m_WindDir.x = 0;
	m_GeoState.m_WindDir.y = 1.f;

	m_GeoState.m_MinLength = 15.f;
	m_GeoState.m_MaxLength = 25.f;
	m_GeoState.m_AmpOverLen = 0.1f;

	m_GeoState.m_EnvHeight = -50.f;
	m_GeoState.m_EnvRadius = 100.f;
	m_GeoState.m_WaterLevel = -2.f;

	m_GeoState.m_TransIdx = 0;
	m_GeoState.m_TransDel = -1.f / 6.f;

	m_GeoState.m_SpecAtten = 1.f;
	m_GeoState.m_SpecEnd = 200.f;
	m_GeoState.m_SpecTrans = 100.f;
}

void CMyD3DApplication::GetWaterParams()
{
	m_WaterParams.m_cWorld2NDC = m_WaterEff->GetParameterByName(NULL, "cWorld2NDC");
	m_WaterParams.m_cWaterTint = m_WaterEff->GetParameterByName(NULL, "cWaterTint");
	m_WaterParams.m_cFrequency = m_WaterEff->GetParameterByName(NULL, "cFrequency");
	m_WaterParams.m_cPhase = m_WaterEff->GetParameterByName(NULL, "cPhase");
	m_WaterParams.m_cAmplitude = m_WaterEff->GetParameterByName(NULL, "cAmplitude");
	m_WaterParams.m_cDirX = m_WaterEff->GetParameterByName(NULL, "cDirX");
	m_WaterParams.m_cDirY = m_WaterEff->GetParameterByName(NULL, "cDirY");
	m_WaterParams.m_cSpecAtten = m_WaterEff->GetParameterByName(NULL, "cSpecAtten");
	m_WaterParams.m_cCameraPos = m_WaterEff->GetParameterByName(NULL, "cCameraPos");
	m_WaterParams.m_cEnvAdjust = m_WaterEff->GetParameterByName(NULL, "cEnvAdjust");
	m_WaterParams.m_cEnvTint = m_WaterEff->GetParameterByName(NULL, "cEnvTint");
	m_WaterParams.m_cLocal2World = m_WaterEff->GetParameterByName(NULL, "cLocal2World");
	m_WaterParams.m_cLengths = m_WaterEff->GetParameterByName(NULL, "cLengths");
	m_WaterParams.m_cDepthOffset = m_WaterEff->GetParameterByName(NULL, "cDepthOffset");
	m_WaterParams.m_cDepthScale = m_WaterEff->GetParameterByName(NULL, "cDepthScale");
	m_WaterParams.m_cFogParams = m_WaterEff->GetParameterByName(NULL, "cFogParams");
	m_WaterParams.m_cDirXK = m_WaterEff->GetParameterByName(NULL, "cDirXK");
	m_WaterParams.m_cDirYK = m_WaterEff->GetParameterByName(NULL, "cDirYK");
	m_WaterParams.m_cDirXW = m_WaterEff->GetParameterByName(NULL, "cDirXW");
	m_WaterParams.m_cDirYW = m_WaterEff->GetParameterByName(NULL, "cDirYW");
	m_WaterParams.m_cKW = m_WaterEff->GetParameterByName(NULL, "cKW");
	m_WaterParams.m_cDirXSqKW = m_WaterEff->GetParameterByName(NULL, "cDirXSqKW");
	m_WaterParams.m_cDirXDirYKW = m_WaterEff->GetParameterByName(NULL, "cDirXDirYKW");
	m_WaterParams.m_cDirYSqKW = m_WaterEff->GetParameterByName(NULL, "cDirYSqKW");

	m_WaterParams.m_tEnvMap = m_WaterEff->GetParameterByName(NULL, "tEnvMap");
	m_WaterParams.m_tBumpMap = m_WaterEff->GetParameterByName(NULL, "tBumpMap");
}

void CMyD3DApplication::SetWaterParams()
{
	D3DXMATRIXA16 world2NDC = m_matView * m_matProjection;
	m_WaterEff->SetMatrixTranspose(m_WaterParams.m_cWorld2NDC, &world2NDC);

	D3DXVECTOR4 waterTint(0.05f, 0.1f, 0.1f, 0.5f);
	m_WaterEff->SetVector(m_WaterParams.m_cWaterTint, &waterTint);

	D3DXVECTOR4 freq(m_GeoWaves[0].m_Freq, m_GeoWaves[1].m_Freq, m_GeoWaves[2].m_Freq, m_GeoWaves[3].m_Freq);
	m_WaterEff->SetVector(m_WaterParams.m_cFrequency, &freq);

	D3DXVECTOR4 phase(m_GeoWaves[0].m_Phase, m_GeoWaves[1].m_Phase, m_GeoWaves[2].m_Phase, m_GeoWaves[3].m_Phase);
	m_WaterEff->SetVector(m_WaterParams.m_cPhase, &phase);

	D3DXVECTOR4 amp(m_GeoWaves[0].m_Amp, m_GeoWaves[1].m_Amp, m_GeoWaves[2].m_Amp, m_GeoWaves[3].m_Amp);
	m_WaterEff->SetVector(m_WaterParams.m_cAmplitude, &amp);

	D3DXVECTOR4 dirX(m_GeoWaves[0].m_Dir.x, m_GeoWaves[1].m_Dir.x, m_GeoWaves[2].m_Dir.x, m_GeoWaves[3].m_Dir.x);
	m_WaterEff->SetVector(m_WaterParams.m_cDirX, &dirX);

	D3DXVECTOR4 dirY(m_GeoWaves[0].m_Dir.y, m_GeoWaves[1].m_Dir.y, m_GeoWaves[2].m_Dir.y, m_GeoWaves[3].m_Dir.y);
	m_WaterEff->SetVector(m_WaterParams.m_cDirY, &dirY);

	FLOAT normScale = m_GeoState.m_SpecAtten * m_TexState.m_AmpOverLen * 2.f * D3DX_PI;
	normScale *= (float(kNumBumpPasses) + m_TexState.m_Noise);
	normScale *= (m_TexState.m_Chop + 1.f);

	D3DXVECTOR4 specAtten(m_GeoState.m_SpecEnd, 1.f / m_GeoState.m_SpecTrans, normScale, 1.f / m_TexState.m_RippleScale);
	m_WaterEff->SetVector(m_WaterParams.m_cSpecAtten, &specAtten);

	D3DXVECTOR3 camPos(m_matPosition._41, m_matPosition._42, m_matPosition._43);
	m_WaterEff->SetVector(m_WaterParams.m_cCameraPos, &D3DXVECTOR4(camPos.x, camPos.y, camPos.z, 1.f));


	D3DXVECTOR3 envCenter(0.f, 0.f, m_GeoState.m_EnvHeight); // Just happens to be centered at origin.
	D3DXVECTOR3 camToCen = envCenter - camPos;
	float G = D3DXVec3LengthSq(&camToCen) - m_GeoState.m_EnvRadius * m_GeoState.m_EnvRadius;
	m_WaterEff->SetVector(m_WaterParams.m_cEnvAdjust, &D3DXVECTOR4(camToCen.x, camToCen.y, camToCen.z, G));

	D3DXVECTOR4 envTint(1.f, 1.f, 1.f, 1.f);
	m_WaterEff->SetVector(m_WaterParams.m_cEnvTint, &envTint);

	D3DXMATRIXA16 matIdent;
	D3DXMatrixIdentity(&matIdent);
	m_WaterEff->SetMatrixTranspose(m_WaterParams.m_cLocal2World, &matIdent);

	D3DXVECTOR4 lengths(m_GeoWaves[0].m_Len, m_GeoWaves[1].m_Len, m_GeoWaves[2].m_Len, m_GeoWaves[3].m_Len);
	m_WaterEff->SetVector(m_WaterParams.m_cLengths, &lengths);

	D3DXVECTOR4 depthOffset(m_GeoState.m_WaterLevel + 1.f, 
		m_GeoState.m_WaterLevel + 1.f, 
		m_GeoState.m_WaterLevel + 0.f, 
		m_GeoState.m_WaterLevel);
	m_WaterEff->SetVector(m_WaterParams.m_cDepthOffset, &depthOffset);

	D3DXVECTOR4 depthScale(1.f / 2.f, 1.f / 2.f, 1.f / 2.f, 1.f);
	m_WaterEff->SetVector(m_WaterParams.m_cDepthScale, &depthScale);

	D3DXVECTOR4 fogParams(-200.f, 1.f / (100.f - 200.f), 0.f, 1.f);
	m_WaterEff->SetVector(m_WaterParams.m_cFogParams, &fogParams);

	float K = 5.f;
	if( m_GeoState.m_AmpOverLen > m_GeoState.m_Chop / (2.f * D3DX_PI * kNumGeoWaves * K) )
		K = m_GeoState.m_Chop / (2.f*D3DX_PI* m_GeoState.m_AmpOverLen * kNumGeoWaves);
	D3DXVECTOR4 dirXK(m_GeoWaves[0].m_Dir.x * K, 
		m_GeoWaves[1].m_Dir.x * K, 
		m_GeoWaves[2].m_Dir.x * K, 
		m_GeoWaves[3].m_Dir.x * K);
	D3DXVECTOR4 dirYK(m_GeoWaves[0].m_Dir.y * K, 
		m_GeoWaves[1].m_Dir.y * K, 
		m_GeoWaves[2].m_Dir.y * K, 
		m_GeoWaves[3].m_Dir.y * K);
	m_WaterEff->SetVector(m_WaterParams.m_cDirXK, &dirXK);
	m_WaterEff->SetVector(m_WaterParams.m_cDirYK, &dirYK);

	D3DXVECTOR4 dirXW(m_GeoWaves[0].m_Dir.x * m_GeoWaves[0].m_Freq, 
		m_GeoWaves[1].m_Dir.x * m_GeoWaves[1].m_Freq, 
		m_GeoWaves[2].m_Dir.x * m_GeoWaves[2].m_Freq, 
		m_GeoWaves[3].m_Dir.x * m_GeoWaves[3].m_Freq);
	D3DXVECTOR4 dirYW(m_GeoWaves[0].m_Dir.y * m_GeoWaves[0].m_Freq, 
		m_GeoWaves[1].m_Dir.y * m_GeoWaves[1].m_Freq, 
		m_GeoWaves[2].m_Dir.y * m_GeoWaves[2].m_Freq, 
		m_GeoWaves[3].m_Dir.y * m_GeoWaves[3].m_Freq);
	m_WaterEff->SetVector(m_WaterParams.m_cDirXW, &dirXW);
	m_WaterEff->SetVector(m_WaterParams.m_cDirYW, &dirYW);

	D3DXVECTOR4 KW(K * m_GeoWaves[0].m_Freq,
		K * m_GeoWaves[1].m_Freq,
		K * m_GeoWaves[2].m_Freq,
		K * m_GeoWaves[3].m_Freq);
	m_WaterEff->SetVector(m_WaterParams.m_cKW, &KW);

	D3DXVECTOR4 dirXSqKW(m_GeoWaves[0].m_Dir.x * m_GeoWaves[0].m_Dir.x * K * m_GeoWaves[0].m_Freq,
		m_GeoWaves[1].m_Dir.x * m_GeoWaves[1].m_Dir.x * K * m_GeoWaves[1].m_Freq,
		m_GeoWaves[2].m_Dir.x * m_GeoWaves[2].m_Dir.x * K * m_GeoWaves[2].m_Freq,
		m_GeoWaves[3].m_Dir.x * m_GeoWaves[3].m_Dir.x * K * m_GeoWaves[3].m_Freq);
	m_WaterEff->SetVector(m_WaterParams.m_cDirXSqKW, &dirXSqKW);

	D3DXVECTOR4 dirYSqKW(m_GeoWaves[0].m_Dir.y * m_GeoWaves[0].m_Dir.y * K * m_GeoWaves[0].m_Freq,
		m_GeoWaves[1].m_Dir.y * m_GeoWaves[1].m_Dir.y * K * m_GeoWaves[1].m_Freq,
		m_GeoWaves[2].m_Dir.y * m_GeoWaves[2].m_Dir.y * K * m_GeoWaves[2].m_Freq,
		m_GeoWaves[3].m_Dir.y * m_GeoWaves[3].m_Dir.y * K * m_GeoWaves[3].m_Freq);
	m_WaterEff->SetVector(m_WaterParams.m_cDirYSqKW, &dirYSqKW);

	D3DXVECTOR4 dirXdirYKW(m_GeoWaves[0].m_Dir.y * m_GeoWaves[0].m_Dir.x * K * m_GeoWaves[0].m_Freq,
		m_GeoWaves[1].m_Dir.x * m_GeoWaves[1].m_Dir.y * K * m_GeoWaves[1].m_Freq,
		m_GeoWaves[2].m_Dir.x * m_GeoWaves[2].m_Dir.y * K * m_GeoWaves[2].m_Freq,
		m_GeoWaves[3].m_Dir.x * m_GeoWaves[3].m_Dir.y * K * m_GeoWaves[3].m_Freq);
	m_WaterEff->SetVector(m_WaterParams.m_cDirXDirYKW, &dirXdirYKW);

	m_WaterEff->SetTexture(m_WaterParams.m_tEnvMap, m_EnvMap);
	m_WaterEff->SetTexture(m_WaterParams.m_tBumpMap, m_BumpTex);
}

void CMyD3DApplication::ResetWater()
{
	InitTexState();
	InitGeoState();

	InitWaves();
}

void CMyD3DApplication::InitWaves()
{
	InitTexWaves();
	InitGeoWaves();
}

void CMyD3DApplication::RenderWater()
{
	SetWaterParams();

	if( m_bSortWater )
		SortWaterMesh();

	UINT nPass;
	m_WaterEff->Begin(&nPass, 0);

	UINT i;
	for( i = 0; i < nPass; i++ )
	{
		m_WaterEff->Pass(i);

		m_WaterMesh->DrawSubset(0);

	}
	m_WaterEff->End();
}

void CMyD3DApplication::RenderTexture()
{
	if( SUCCEEDED(m_BumpRender->BeginScene(m_BumpSurf, NULL)) )
	{

		D3DXMATRIXA16 matIdent;
		D3DXMatrixIdentity(&matIdent);
		m_pd3dDevice->SetTransform(D3DTS_PROJECTION, &matIdent);
		m_pd3dDevice->SetTransform(D3DTS_VIEW, &matIdent);
		m_pd3dDevice->SetTransform(D3DTS_WORLD, &matIdent);

		m_pd3dDevice->SetRenderState(D3DRS_ZENABLE, FALSE);

		SetCompCosineEffParams();

		UINT nPass;
		m_CompCosinesEff->Begin(&nPass, 0);

		UINT i;
		for( i = 0; i < nPass; i++ )
		{
			m_CompCosinesEff->Pass(i);

			m_pd3dDevice->SetFVF(kClearVertFVF);
			m_pd3dDevice->SetStreamSource(0, m_BumpVBuffer, 0, kVSize);
			m_pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
		}
		m_CompCosinesEff->End();

		m_pd3dDevice->SetRenderState(D3DRS_ZENABLE, TRUE);

		m_BumpRender->EndScene( 0 );
	}
}


