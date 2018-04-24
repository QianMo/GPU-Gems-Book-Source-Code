#include "gpu_reliefmap_preprocess.h"
#include "resource.h"

ID3DXSprite*			g_Sprite = NULL; 

ID3DXEffect*			g_Effect = NULL;

char					g_ReliefFileName[MAX_PATH] = "tile1.tga";
LPDIRECT3DTEXTURE9		g_ReliefTexture = NULL;
UINT					g_ReliefSizeX = 0;
UINT					g_ReliefSizeY = 0;

UINT					g_OffsetX = 0;
UINT					g_OffsetY = 0;
UINT					g_SamplingGroupX = 0;
UINT					g_SamplingGroupY = 0;
UINT					g_SamplingGroupPos = 0;
UINT					g_SamplingGroupSize = 8;
UINT					g_SamplingOrder[] = 
							{ 0,36,4,32,18,54,22,50,
							  2,38,6,34,16,52,20,48,
							  0+9,36+9,4+9,32+9,18+9,54+9,22+9,50+9,
							  2+9,38+9,6+9,34+9,16+9,52+9,20+9,48+9,
							  0+1,36+1,4+1,32+1,18+1,54+1,22+1,50+1,
							  2+1,38+1,6+1,34+1,16+1,52+1,20+1,48+1,
							  0+8,36+8,4+8,32+8,18+8,54+8,22+8,50+8,
							  2+8,38+8,6+8,34+8,16+8,52+8,20+8,48+8 };

enum					{ RUNMODE_STOPED=0,RUNMODE_RUNNING=1 };
UINT					g_RunMode = 0;

enum					{ RENDERMODE_CONE=0,RENDERMODE_QUADCONE=1,RENDERMODE_RELAXEDCONE=2 };
UINT					g_RenderMode = 0;

bool					g_ResetTex = true;
LPDIRECT3DTEXTURE9		g_ResultTex = NULL;
LPDIRECT3DSURFACE9		g_ResultSurface = NULL;
LPDIRECT3DSURFACE9		g_DepthSurf = NULL;
LPDIRECT3DTEXTURE9		g_ColorTex[2] = { NULL, NULL };
LPDIRECT3DSURFACE9		g_ColorSurf[2] = { NULL, NULL };
UINT					g_CurrentTex = 0;

PDIRECT3DDEVICE9			g_d3dDevice = NULL;

CDXUTDialogResourceManager	g_DialogManager;
CDXUTDialog					g_DlgOptions;

enum 
{ 
	IDC_STATIC,
	IDC_LOAD,
	IDC_SAVE,
	IDC_RESET,
	IDC_START,
	IDC_PROGRESS,
	IDC_CONEMAP,
	IDC_QUADCONEMAP,
	IDC_RELAXEDCONEMAP
};

const char *BrowseFile(const char *title, const char* filter, const char* defExt,const char *initial_file)
{
	static char filename[MAX_PATH];
	filename[0] = 0;

	OPENFILENAME ofn;
 	memset(&ofn,0,sizeof(ofn));

	ofn.lpstrDefExt		  = defExt;
	ofn.lStructSize       = sizeof(ofn);
	ofn.hwndOwner         = DXUTGetHWNDFocus();
	ofn.lpstrFilter       = filter;
	ofn.nFilterIndex      = 1;
	ofn.lpstrFile         = filename;
	ofn.lpstrTitle		  = title;
	ofn.nMaxFile          = MAX_PATH;
	ofn.Flags			  = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;

	if (initial_file)
	{
		char str[MAX_PATH]="";
		DXUTFindDXSDKMediaFileCch( str, MAX_PATH, initial_file  );
		ofn.lpstrInitialDir	= str;

		StringCchCopy(filename,MAX_PATH,initial_file);
	}

	if (TRUE == GetOpenFileName(&ofn))
		return filename;

	return NULL;
}

void CreateData(IDirect3DDevice9* d3dDevice)
{
	char str[MAX_PATH];

	DXUTFindDXSDKMediaFileCch( str, MAX_PATH, g_ReliefFileName );
	D3DXCreateTextureFromFileEx( 
		d3dDevice, str, 
		D3DX_DEFAULT, D3DX_DEFAULT, 1,
		D3DUSAGE_DYNAMIC,D3DFMT_A8R8G8B8,
		D3DPOOL_DEFAULT, 
		D3DX_DEFAULT ,D3DX_DEFAULT,
		0xFF000000, NULL, NULL, &g_ReliefTexture );
	if (g_ReliefTexture==0)
		return;

	D3DSURFACE_DESC desc;
	g_ReliefTexture->GetLevelDesc(0,&desc);
	g_ReliefSizeX = desc.Width;
	g_ReliefSizeY = desc.Height;

	d3dDevice->CreateTexture( g_ReliefSizeX, g_ReliefSizeY, 1,
		D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &g_ResultTex, NULL );
	g_ResultTex->GetSurfaceLevel( 0, &g_ResultSurface );

	d3dDevice->CreateTexture( g_ReliefSizeX, g_ReliefSizeY, 1,
		D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_ColorTex[0], NULL );
	g_ColorTex[0]->GetSurfaceLevel( 0, &g_ColorSurf[0] );

	d3dDevice->CreateTexture( g_ReliefSizeX, g_ReliefSizeY, 1,
		D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_ColorTex[1], NULL );
	g_ColorTex[1]->GetSurfaceLevel( 0, &g_ColorSurf[1] );

	d3dDevice->CreateDepthStencilSurface( g_ReliefSizeX, g_ReliefSizeY, D3DFMT_D24X8,
		D3DMULTISAMPLE_NONE, 0, TRUE, &g_DepthSurf, NULL );

	g_ResetTex = true;
}

void ReleaseData()
{
	SAFE_RELEASE( g_ReliefTexture );

	SAFE_RELEASE( g_ResultSurface );
	SAFE_RELEASE( g_ColorSurf[0] );
	SAFE_RELEASE( g_ColorSurf[1] );
    SAFE_RELEASE( g_DepthSurf );

	SAFE_RELEASE( g_ResultTex );
	SAFE_RELEASE( g_ColorTex[0] );
	SAFE_RELEASE( g_ColorTex[1] );
}

void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	switch( nControlID )
	{
		case IDC_RESET:
			{
				g_ResetTex = true;

				g_RunMode = RUNMODE_STOPED;
				CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
				Button->SetText("Start");
			}
			break;
		case IDC_START:
			{
				CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
				if ( g_RunMode == RUNMODE_STOPED )
				{
					g_RunMode = RUNMODE_RUNNING;
					Button->SetText("Stop");
				}
				else
				{
					g_RunMode = RUNMODE_STOPED;
					Button->SetText("Start");
				}
			}
			break;
		case IDC_CONEMAP:
			if (g_RenderMode!=0)
			{
				g_ResetTex = true;
				g_RenderMode = 0;
				g_RunMode = RUNMODE_STOPED;
				CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
				Button->SetText("Start");
			}
			break;
		case IDC_QUADCONEMAP:
			if (g_RenderMode!=1)
			{
				g_ResetTex = true;
				g_RenderMode = 1;
				g_RunMode = RUNMODE_STOPED;
				CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
				Button->SetText("Start");
			}
			break;
		case IDC_RELAXEDCONEMAP:
			if (g_RenderMode!=2)
			{
				g_ResetTex = true;
				g_RenderMode = 2;
				g_RunMode = RUNMODE_STOPED;
				CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
				Button->SetText("Start");
			}
			break;
		case IDC_LOAD:
			{
				const char *filename = BrowseFile( "Open Relief Map",
						"Targa 32 bits/pixel image files (.TGA)\0*.TGA\0\0", "TGA", g_ReliefFileName );
				if (filename)
				{
					ReleaseData();

					StringCchCopy( g_ReliefFileName, MAX_PATH, filename );

					CreateData(g_d3dDevice);

					CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
					Button->SetText("Start");
					g_RunMode = RUNMODE_STOPED;
				}
			}
			break;
		case IDC_SAVE:
			if (g_ReliefTexture)
			{
				D3DLOCKED_RECT LockedRectDst;
				D3DLOCKED_RECT LockedRectSrc;
				if (S_OK==g_d3dDevice->GetRenderTargetData(g_ColorSurf[!g_CurrentTex],g_ResultSurface))
				{
					if (S_OK==g_ReliefTexture->LockRect(0,&LockedRectDst,NULL,0))
					{
						if (S_OK==g_ResultTex->LockRect(0,&LockedRectSrc,NULL,0))
						{
							unsigned char *dst = (unsigned char *)LockedRectDst.pBits;
							unsigned char *src = (unsigned char *)LockedRectSrc.pBits;

							src+=4*g_ReliefSizeX*(g_ReliefSizeY-1);

							UINT i,j;
							for( j=0;j<g_ReliefSizeY;j++ )
							{
								for( i=0;i<g_ReliefSizeX;i++ )
								{
									dst[0]=src[0];
									if (g_RenderMode==RENDERMODE_QUADCONE)
									{
										dst[1]=src[1];
										dst[2]=src[2];
										dst[3]=src[3];
									}
									dst+=4;
									src+=4;
								}
								src-=8*g_ReliefSizeX;
							}

							g_ResultTex->UnlockRect(0);
						}

						g_ReliefTexture->UnlockRect(0);
					}
				}

				char str[MAX_PATH];
				StringCchCopy(str,MAX_PATH,g_ReliefFileName);
				char *c = strrchr(str,'.');
				if (c)
					switch( g_RenderMode )
					{
						case 0: strcpy(c,"_cone.tga"); break;
						case 1: strcpy(c,"_quadcone.tga"); break;
						case 2: strcpy(c,"_relaxedcone.tga"); break;
					}
				D3DXSaveTextureToFile(str,D3DXIFF_TGA,g_ReliefTexture,NULL);

				SAFE_RELEASE( g_ReliefTexture );

				DXUTFindDXSDKMediaFileCch( str, MAX_PATH, g_ReliefFileName );
				D3DXCreateTextureFromFileEx( 
					g_d3dDevice, str, 
					D3DX_DEFAULT, D3DX_DEFAULT, 1,
					D3DUSAGE_DYNAMIC,D3DFMT_A8R8G8B8,
					D3DPOOL_DEFAULT, 
					D3DX_DEFAULT ,D3DX_DEFAULT,
					0xFF000000, NULL, NULL, &g_ReliefTexture );
			}
			break;
	}
}

void CALLBACK OnFrameRender( IDirect3DDevice9* d3dDevice, double Time, float ElapsedTime, void* UserContext )
{
	d3dDevice->SetRenderState(D3DRS_ZWRITEENABLE,FALSE);
	d3dDevice->SetRenderState(D3DRS_ZENABLE,FALSE);
	d3dDevice->SetRenderState(D3DRS_CULLMODE,D3DCULL_NONE);
	d3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,FALSE);

	if (g_Effect && g_ReliefTexture)
	{
		D3DXMATRIX mOrthoProj;
		D3DXMatrixOrthoLH( &mOrthoProj, (float)g_ReliefSizeX, (float)g_ReliefSizeY, -1, 1 );

		D3DXMATRIX mTranslate;
		D3DXMatrixTranslation( &mTranslate, -(float)g_ReliefSizeX/2.0f, -(float)g_ReliefSizeY/2.0f, 0 );

		D3DXMATRIX mViewOrthoProj = mTranslate * mOrthoProj;

		g_Effect->SetMatrix( "g_WorldViewProj", &mViewOrthoProj );

		if (g_ResetTex)
		{
			LPDIRECT3DSURFACE9 oldRT;
			d3dDevice->GetRenderTarget( 0, &oldRT );
			LPDIRECT3DSURFACE9 oldDS;
			d3dDevice->GetDepthStencilSurface( &oldDS );

			d3dDevice->SetDepthStencilSurface( g_DepthSurf );

			for( int i=0;i<2;i++ )
			{
				d3dDevice->SetRenderTarget( 0, g_ColorSurf[i] );
				d3dDevice->Clear(0,0,D3DCLEAR_TARGET,0xFFFFFFFF,1.0f,0);
			}

			g_ResetTex = false;
			g_SamplingGroupX = 0;
			g_SamplingGroupY = 0;
			g_SamplingGroupPos = 0;
			g_OffsetX = 0;
			g_OffsetY = 0;

			CDXUTStatic *ProgressText = (CDXUTStatic *)g_DlgOptions.GetControl(IDC_PROGRESS);
			char Text[32];
			StringCchPrintf(Text,32,"Progress: 0.00%% (1/%i)",g_SamplingGroupSize*g_SamplingGroupSize);
			ProgressText->SetText(Text);
		
			d3dDevice->SetDepthStencilSurface( oldDS );
			d3dDevice->SetRenderTarget( 0, oldRT );
			SAFE_RELEASE( oldRT );
			SAFE_RELEASE( oldDS );
		}

		if ( g_RunMode == RUNMODE_RUNNING )
		{
			LPDIRECT3DSURFACE9 oldRT;
			d3dDevice->GetRenderTarget( 0, &oldRT );
			LPDIRECT3DSURFACE9 oldDS;
			d3dDevice->GetDepthStencilSurface( &oldDS );

			d3dDevice->SetDepthStencilSurface( g_DepthSurf );
			d3dDevice->SetRenderTarget( 0, g_ColorSurf[g_CurrentTex] );

			switch( g_RenderMode )
			{
				case 0: g_Effect->SetTechnique("depth2cone"); break;
				case 1: g_Effect->SetTechnique("depth2quadcone"); break;
				case 2: g_Effect->SetTechnique("depth2relaxedcone"); break;
			}

			g_Effect->SetTexture( "g_ReliefMap", g_ReliefTexture );
			g_Effect->SetTexture( "g_ColorMap", g_ColorTex[!g_CurrentTex] );
			
			#define FRAND (((rand()%30000)-15000)/15000.0f)
			#define FABSRAND ((rand()%30000)/30000.0f)

			int OffsetX = g_SamplingGroupX*g_SamplingGroupSize+(g_SamplingOrder[g_SamplingGroupPos]%g_SamplingGroupSize);
			int OffsetY = g_SamplingGroupY*g_SamplingGroupSize+(g_SamplingOrder[g_SamplingGroupPos]/g_SamplingGroupSize);
			D3DXVECTOR3 Offset;
			Offset.x=(OffsetX-(int)g_ReliefSizeX/2)/(float)g_ReliefSizeX+0.5f/g_ReliefSizeX;
			Offset.y=(OffsetY-(int)g_ReliefSizeY/2)/(float)g_ReliefSizeY+0.5f/g_ReliefSizeY;
			Offset.z=0;

			g_SamplingGroupX++;
			if (g_SamplingGroupX>=g_ReliefSizeX/g_SamplingGroupSize)
			{
				g_SamplingGroupX=0;

				UINT s2 = g_SamplingGroupSize*g_SamplingGroupSize;
				UINT sx = g_ReliefSizeX/g_SamplingGroupSize;
				UINT sy = g_ReliefSizeY/g_SamplingGroupSize;

				float progress1 = 100.0f*g_SamplingGroupPos/s2;
				float progress2 = 100.0f/s2*(g_SamplingGroupX+g_SamplingGroupY*sx)/(sx*sy);

				CDXUTStatic *ProgressText = (CDXUTStatic *)g_DlgOptions.GetControl(IDC_PROGRESS);
				char Text[32];
				StringCchPrintf(Text,32,"Progress: %3.2f%% (%i/%i)",
					progress1+progress2,
					g_SamplingGroupPos+1,g_SamplingGroupSize*g_SamplingGroupSize);
				ProgressText->SetText(Text);

				g_SamplingGroupY++;
				if (g_SamplingGroupY>=g_ReliefSizeY/g_SamplingGroupSize)
				{
					g_SamplingGroupY=0;
					g_SamplingGroupPos++;
					if (g_SamplingGroupPos>=g_SamplingGroupSize*g_SamplingGroupSize)
					{
						g_SamplingGroupPos=0;
						g_RunMode=0;
						CDXUTStatic *ProgressText = (CDXUTStatic *)g_DlgOptions.GetControl(IDC_PROGRESS);
						ProgressText->SetText("Progress: 100%");
						CDXUTButton *Button = (CDXUTButton *)g_DlgOptions.GetControl(IDC_START);
						Button->SetText("Start");
					}
				}
			}
			g_Effect->SetValue("g_Offset",&Offset.x,sizeof(float)*3);

			g_Effect->CommitChanges();

			UINT NumPasses,Pass;
			g_Effect->Begin(&NumPasses, 0);
			for (Pass = 0; Pass < NumPasses; Pass++)
			{
				g_Effect->BeginPass(Pass);

				g_Sprite->Begin( D3DXSPRITE_DONOTMODIFY_RENDERSTATE );
				g_Sprite->Draw( g_ReliefTexture, NULL, NULL, NULL, 0xffffffff );
				g_Sprite->End();

				g_Effect->EndPass();
			}
			g_Effect->End();

			d3dDevice->SetDepthStencilSurface( oldDS );
			d3dDevice->SetRenderTarget( 0, oldRT );
			SAFE_RELEASE( oldRT );
			SAFE_RELEASE( oldDS );

			g_CurrentTex = !g_CurrentTex;
		}

		d3dDevice->Clear(0,0,D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER,0x00000000,1.0f,0);

		g_Effect->SetTechnique("viewtexture");
		g_Effect->SetTexture("g_ColorMap", g_ColorTex[!g_CurrentTex] );

		UINT NumPasses,Pass;
		g_Effect->Begin(&NumPasses, 0);
		for (Pass = 0; Pass < NumPasses; Pass++)
		{
			g_Effect->BeginPass(Pass);

			g_Sprite->Begin( D3DXSPRITE_DONOTMODIFY_RENDERSTATE );
			g_Sprite->Draw( g_ColorTex[g_CurrentTex], NULL, NULL, NULL, 0xffffffff );
			g_Sprite->End();

			g_Effect->EndPass();
		}
		g_Effect->End();

		d3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,TRUE);
		d3dDevice->SetRenderState(D3DRS_SRCBLEND,D3DBLEND_SRCALPHA);
		d3dDevice->SetRenderState(D3DRS_DESTBLEND,D3DBLEND_INVSRCALPHA);

		D3DVIEWPORT9 Viewport;
		d3dDevice->GetViewport(&Viewport);

		RECT rect={0,0,256,80};
		g_DlgOptions.DrawRect(&rect,0x80808080);
	}
	else
		d3dDevice->Clear(0,0,D3DCLEAR_TARGET,0,1.0f,0);

	g_DlgOptions.OnRender( 0.01f );
}

void CALLBACK OnFrameMove( IDirect3DDevice9* d3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
}

BOOL IsDepthFormatOk( D3DFORMAT DepthFormat,
                      D3DFORMAT AdapterFormat,
                      D3DFORMAT BackBufferFormat )
{
    HRESULT hr = DXUTGetD3DObject()->CheckDeviceFormat( D3DADAPTER_DEFAULT,
                                                        D3DDEVTYPE_HAL,
                                                        AdapterFormat,
                                                        D3DUSAGE_DEPTHSTENCIL,
                                                        D3DRTYPE_SURFACE,
                                                        DepthFormat );
    if( FAILED( hr ) ) return FALSE;

    hr = DXUTGetD3DObject()->CheckDeviceFormat( D3DADAPTER_DEFAULT,
                                                D3DDEVTYPE_HAL,
                                                AdapterFormat,
                                                D3DUSAGE_RENDERTARGET,
                                                D3DRTYPE_SURFACE,
                                                BackBufferFormat );
    if( FAILED( hr ) ) return FALSE;

    hr = DXUTGetD3DObject()->CheckDepthStencilMatch( D3DADAPTER_DEFAULT,
                                                     D3DDEVTYPE_HAL,
                                                     AdapterFormat,
                                                     BackBufferFormat,
                                                     DepthFormat );

    return SUCCEEDED(hr);
}

bool CALLBACK IsDeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, 
                                  D3DFORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    IDirect3D9* pD3D = DXUTGetD3DObject(); 
    if( FAILED( pD3D->CheckDeviceFormat( pCaps->AdapterOrdinal, pCaps->DeviceType,
                    AdapterFormat, D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING, 
                    D3DRTYPE_TEXTURE, BackBufferFormat ) ) )
        return false;

    if( pCaps->PixelShaderVersion < D3DPS_VERSION( 2, 0 ) )
        return false;

    if( !IsDepthFormatOk( D3DFMT_D24S8,
                          AdapterFormat,
                          BackBufferFormat ) &&
        !IsDepthFormatOk( D3DFMT_D24X4S4,
                          AdapterFormat,
                          BackBufferFormat ) &&
        !IsDepthFormatOk( D3DFMT_D15S1,
                          AdapterFormat,
                          BackBufferFormat ) &&
        !IsDepthFormatOk( D3DFMT_D24FS8,
                          AdapterFormat,
                          BackBufferFormat ) )
        return false;

    return true;
}

bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, const D3DCAPS9* pCaps, void* pUserContext )
{
    if( IsDepthFormatOk( D3DFMT_D24S8,
                         pDeviceSettings->AdapterFormat,
                         pDeviceSettings->pp.BackBufferFormat ) )
        pDeviceSettings->pp.AutoDepthStencilFormat = D3DFMT_D24S8;
    else
    if( IsDepthFormatOk( D3DFMT_D24X4S4,
                         pDeviceSettings->AdapterFormat,
                         pDeviceSettings->pp.BackBufferFormat ) )
        pDeviceSettings->pp.AutoDepthStencilFormat = D3DFMT_D24X4S4;
    else
    if( IsDepthFormatOk( D3DFMT_D24FS8,
                         pDeviceSettings->AdapterFormat,
                         pDeviceSettings->pp.BackBufferFormat ) )
        pDeviceSettings->pp.AutoDepthStencilFormat = D3DFMT_D24FS8;
    else
    if( IsDepthFormatOk( D3DFMT_D15S1,
                         pDeviceSettings->AdapterFormat,
                         pDeviceSettings->pp.BackBufferFormat ) )
        pDeviceSettings->pp.AutoDepthStencilFormat = D3DFMT_D15S1;

	return true;
}

LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext )
{
	*pbNoFurtherProcessing = g_DialogManager.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;
	
	*pbNoFurtherProcessing = g_DlgOptions.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	return 0;
}

void CALLBACK KeyboardProc( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
}

HRESULT CALLBACK OnCreateDevice( IDirect3DDevice9* d3dDevice, const D3DSURFACE_DESC* BackBufferSurfaceDesc, void* pUserContext )
{
	g_d3dDevice = d3dDevice;

	g_DialogManager.OnCreateDevice( d3dDevice );

	DWORD ShaderFlags = D3DXFX_NOT_CLONEABLE;
    #ifdef DEBUG_VS
        ShaderFlags |= D3DXSHADER_FORCE_VS_SOFTWARE_NOOPT;
    #endif
    #ifdef DEBUG_PS
        ShaderFlags |= D3DXSHADER_FORCE_PS_SOFTWARE_NOOPT;
    #endif

	char str[MAX_PATH];

	DXUTFindDXSDKMediaFileCch( str, MAX_PATH, "FX\\gpu_reliefmap_preprocess.fx" );
	D3DXCreateEffectFromFile( d3dDevice, str, NULL, NULL, 
			ShaderFlags, NULL, &g_Effect, NULL );

	D3DXCreateSprite( d3dDevice, &g_Sprite );

	return S_OK;
}

HRESULT CALLBACK OnResetDevice( IDirect3DDevice9* d3dDevice, 
                                const D3DSURFACE_DESC* BackBufferSurfaceDesc, void* UserContext )
{
	g_DialogManager.OnResetDevice();

	if (g_Effect)
		g_Effect->OnResetDevice();

	if (g_Sprite)
		g_Sprite->OnResetDevice();

	CreateData(d3dDevice);

	return S_OK;
}

void CALLBACK OnLostDevice( void* pUserContext )
{
	g_DialogManager.OnLostDevice();
	
	if (g_Effect)
		g_Effect->OnLostDevice();

	if (g_Sprite)
		g_Sprite->OnLostDevice();

	ReleaseData();
}

void CALLBACK OnDestroyDevice( void* pUserContext )
{
	g_DialogManager.OnDestroyDevice();

	SAFE_RELEASE( g_Effect );

	SAFE_RELEASE( g_Sprite );
}

INT WINAPI WinMain( HINSTANCE hInst, HINSTANCE, LPSTR lpCmdLine, INT )
{
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif
	
	if (lpCmdLine && lpCmdLine[0]!=0)
		StringCchCopy(g_ReliefFileName,MAX_PATH,lpCmdLine);

    DXUTSetCursorSettings( true, true );
    DXUTSetCallbackDeviceCreated( OnCreateDevice );
    DXUTSetCallbackDeviceReset( OnResetDevice );
    DXUTSetCallbackDeviceLost( OnLostDevice );
    DXUTSetCallbackDeviceDestroyed( OnDestroyDevice );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( KeyboardProc );
    DXUTSetCallbackFrameRender( OnFrameRender );
    DXUTSetCallbackFrameMove( OnFrameMove );

	g_DlgOptions.Init( &g_DialogManager );
	g_DlgOptions.SetCallback( OnGUIEvent );

   	CDXUTElement* pElement = g_DlgOptions.GetDefaultElement( DXUT_CONTROL_STATIC, 0 );
    if( pElement )
        pElement->dwTextFormat = DT_LEFT | DT_VCENTER;

	g_DlgOptions.AddStatic( IDC_STATIC, "Mode:", 10, 5, 150, 20 );

	g_DlgOptions.AddRadioButton( IDC_CONEMAP,0,"Single",50,5,60,20,true);
	g_DlgOptions.AddRadioButton( IDC_QUADCONEMAP,0,"Quad",115,5,60,20);
	g_DlgOptions.AddRadioButton( IDC_RELAXEDCONEMAP,0,"Relaxed",180,5,65,20);

	g_DlgOptions.AddButton( IDC_LOAD, "Load", 5, 32, 50, 20 );
	g_DlgOptions.AddButton( IDC_SAVE, "Save", 60, 32, 50, 20 );
	g_DlgOptions.AddButton( IDC_RESET, "Reset", 145, 32, 50, 20 );
	g_DlgOptions.AddButton( IDC_START, "Start", 200, 32, 50, 20 );

	g_DlgOptions.AddStatic( IDC_PROGRESS, "Progress: 0.00%", 10, 55, 150, 20 );

    DXUTInit( true, true, true );
    DXUTCreateWindow( "GPU Relief Map Pre-Process" );
    DXUTCreateDevice( D3DADAPTER_DEFAULT, true, 512, 512, IsDeviceAcceptable, ModifyDeviceSettings );

	DXUTMainLoop();

    return 0;
}

