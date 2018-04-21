//-----------------------------------------------------------------------------
// File: bLois.h
//
// Desc: Header file bLois sample app
//-----------------------------------------------------------------------------
#pragma once




//-----------------------------------------------------------------------------
// Defines, and constants
//-----------------------------------------------------------------------------
// TODO: change "DirectX AppWizard Apps" to your name or the company name
#define DXAPP_KEY        TEXT("Software\\DirectX AppWizard Apps\\bLois")

// Struct to store the current input state
struct UserInput
{
    // TODO: change as needed
    BOOL bRotateUp;
    BOOL bRotateDown;
    BOOL bRotateLeft;
    BOOL bRotateRight;
	BOOL bMoveForward;
	BOOL bMoveBack;
	BOOL bMoveUp;
	BOOL bMoveDown;
	BOOL bMoveLeft;
	BOOL bMoveRight;
};


class CompCosineParams
{
public:
	D3DXHANDLE			m_UTrans[16];
	D3DXHANDLE			m_Coef[16];
	D3DXHANDLE			m_ReScale;
	D3DXHANDLE			m_NoiseXform[4];
	D3DXHANDLE			m_ScaleBias;

	D3DXHANDLE			m_CosineLUT;
	D3DXHANDLE			m_BiasNoise;
};

class WaterParams
{
public:
	D3DXHANDLE			m_cWorld2NDC;
	D3DXHANDLE			m_cWaterTint;
	D3DXHANDLE			m_cFrequency;
	D3DXHANDLE			m_cPhase;
	D3DXHANDLE			m_cAmplitude;
	D3DXHANDLE			m_cDirX;
	D3DXHANDLE			m_cDirY;
	D3DXHANDLE			m_cSpecAtten;
	D3DXHANDLE			m_cCameraPos;
	D3DXHANDLE			m_cEnvAdjust;
	D3DXHANDLE			m_cEnvTint;
	D3DXHANDLE			m_cLocal2World;
	D3DXHANDLE			m_cLengths;
	D3DXHANDLE			m_cDepthOffset;
	D3DXHANDLE			m_cDepthScale;
	D3DXHANDLE			m_cFogParams;
	D3DXHANDLE			m_cDirXK;
	D3DXHANDLE			m_cDirYK;
	D3DXHANDLE			m_cDirXW;
	D3DXHANDLE			m_cDirYW;
	D3DXHANDLE			m_cKW;
	D3DXHANDLE			m_cDirXSqKW;
	D3DXHANDLE			m_cDirXDirYKW;
	D3DXHANDLE			m_cDirYSqKW;

	D3DXHANDLE			m_tEnvMap;
	D3DXHANDLE			m_tBumpMap;
};

struct FaceIndices
{
	unsigned short		m_Idx[3];
};
struct FaceSortData
{
	unsigned short		m_Idx;
	FLOAT				m_Dist;
};

//-----------------------------------------------------------------------------
// Name: class CMyD3DApplication
// Desc: Application class. The base class (CD3DApplication) provides the 
//       generic functionality needed in all Direct3D samples. CMyD3DApplication 
//       adds functionality specific to this sample program.
//-----------------------------------------------------------------------------
class CMyD3DApplication : public CD3DApplication
{
    BOOL                    m_bLoadingApp;          // TRUE, if the app is loading
    CD3DFont*               m_pFont;                // Font for drawing text

	BOOL					m_bShowHelp;
	BOOL					m_bDrawBump;
	FLOAT					m_LastToggle;
	BOOL					m_bSortWater;

    BYTE					m_bKey[256];
    D3DXMATRIXA16			m_matView;
    D3DXMATRIXA16			m_matPosition;
    D3DXMATRIXA16			m_matProjection;

    FLOAT                   m_fWorldRotX;           // World rotation state X-axis
    FLOAT                   m_fWorldRotY;           // World rotation state Y-axis

	D3DXVECTOR3				m_CamPos;
	D3DXVECTOR3				m_CamAt;
	D3DXVECTOR3				m_CamUp;

	// Water demo specifics follow
	ID3DXMesh*				m_WaterMesh;
	ID3DXMesh*				m_LandMesh;
	LPDIRECT3DTEXTURE9		m_LandTex;
	ID3DXMesh*				m_PillarsMesh;

	LPDIRECT3DCUBETEXTURE9	m_EnvMap;

	LPD3DXEFFECT			m_WaterEff;
	WaterParams				m_WaterParams;

	CompCosineParams		m_CompCosineParams;
	LPD3DXEFFECT			m_CompCosinesEff;
	LPDIRECT3DTEXTURE9		m_CosineLUT;
	LPDIRECT3DTEXTURE9		m_BiasNoiseMap;
	
	LPDIRECT3DTEXTURE9		m_BumpTex;
	LPDIRECT3DSURFACE9		m_BumpSurf;
	LPD3DXRENDERTOSURFACE	m_BumpRender;

	IDirect3DVertexBuffer9* m_BumpVBuffer;

	FaceIndices*			m_WaterIndices;
	D3DXVECTOR3*			m_WaterFacePos;
	FaceSortData*			m_WaterSortData;


	enum {
		kNumGeoWaves	= 4,
		kNumBumpPerPass = 4,
		kNumTexWaves	= 16,
		kNumBumpPasses	= kNumTexWaves / kNumBumpPerPass,
		kBumpTexSize	= 256
	};
	class TexWaveDesc
	{
	public:
		FLOAT		m_Phase;
		FLOAT		m_Amp;
		FLOAT		m_Len;
		FLOAT		m_Speed;
		FLOAT		m_Freq;
		D3DXVECTOR2	m_Dir;
		D3DXVECTOR2	m_RotScale;
		FLOAT		m_Fade;
	};
	TexWaveDesc		m_TexWaves[kNumTexWaves];

	class TexState
	{
	public:
		FLOAT		m_Noise;
		FLOAT		m_Chop;
		FLOAT		m_AngleDeviation;
		D3DXVECTOR2	m_WindDir;
		FLOAT		m_MaxLength;
		FLOAT		m_MinLength;
		FLOAT		m_AmpOverLen;
		FLOAT		m_RippleScale;
		FLOAT		m_SpeedDeviation;

		int			m_TransIdx;
		FLOAT		m_TransDel;
	};
	TexState		m_TexState;
	
	class GeoWaveDesc
	{
	public:
		FLOAT		m_Phase;
		FLOAT		m_Amp;
		FLOAT		m_Len;
		FLOAT		m_Freq;
		D3DXVECTOR2	m_Dir;
		FLOAT		m_Fade;
	};
	GeoWaveDesc		m_GeoWaves[kNumGeoWaves];

	class GeoState
	{
	public:
		FLOAT		m_Chop;
		FLOAT		m_AngleDeviation;
		D3DXVECTOR2	m_WindDir;
		FLOAT		m_MinLength;
		FLOAT		m_MaxLength;
		FLOAT		m_AmpOverLen;

		FLOAT		m_SpecAtten;
		FLOAT		m_SpecEnd;
		FLOAT		m_SpecTrans;

		FLOAT		m_EnvHeight;
		FLOAT		m_EnvRadius;
		FLOAT		m_WaterLevel;

		int			m_TransIdx;
		FLOAT		m_TransDel;
	};
	GeoState		m_GeoState;

	HRESULT			CreateCosineLUT();
	HRESULT			CreateBiasNoiseMap();
	HRESULT			CreateClearBuffer();

	HRESULT			CreateWaterMesh(ID3DXMesh* waterMesh);

	void GetCompCosineEffParams();
	void SetCompCosineEffParams();

	void GetWaterParams();
	void SetWaterParams();

	void InitTexState();
	void InitTexWave(int i);
	void InitTexWaves();
	void UpdateTexWave(int i, FLOAT dt);
	void UpdateTexWaves(FLOAT dt);

	void InitGeoState();
	void InitGeoWave(int i);
	void InitGeoWaves();
	void UpdateGeoWave(int i, FLOAT dt);
	void UpdateGeoWaves(FLOAT dt);

	void MoveOnInput();

	void ResetWater();
	void InitWaves();

	void RenderTexture();
	void RenderWater();
	void SortWaterMesh();

protected:
    virtual HRESULT OneTimeSceneInit();
    virtual HRESULT InitDeviceObjects();
    virtual HRESULT RestoreDeviceObjects();
    virtual HRESULT InvalidateDeviceObjects();
    virtual HRESULT DeleteDeviceObjects();
    virtual HRESULT Render();
    virtual HRESULT FrameMove();
    virtual HRESULT FinalCleanup();
    virtual HRESULT ConfirmDevice( D3DCAPS9*, DWORD, D3DFORMAT );

    HRESULT RenderText();

    void    UpdateInput( UserInput* pUserInput );

public:
    LRESULT MsgProc( HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam );
    CMyD3DApplication();
    virtual ~CMyD3DApplication();
};

