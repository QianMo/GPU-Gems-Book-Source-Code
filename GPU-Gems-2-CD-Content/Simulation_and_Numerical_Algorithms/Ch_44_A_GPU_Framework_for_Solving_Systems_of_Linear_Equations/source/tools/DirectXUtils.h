/**************************************************************
 *                                                            *
 * description: tools for use with DirectX                    *
 * version    : 2.84                                          *
 * date       : 01.Jul.2003                                   *
 * modified   : 03.Dec.2004                                   *
 * author     : Jens Krüger                                   *
 * e-mail     : mail@jens-krueger.com                         *
 *                                                            *
 **************************************************************/
#pragma once

#include <windows.h>
#include <d3dx9.h>
#include <dxerr9.h>
#include <stdio.h>
#include <tchar.h>
#include <tchar.h>

#include <limits>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "vectors.h"

#  define TIMER_VAR(x)  struct _timeb ts_ ## x, te_ ## x
#  define TIMER_START(x) _ftime (&ts_ ## x)
#  define TIMER_PROBE(x)  _ftime (&te_ ## x)
#  define TIMER_RESULT(x) ((double)te_ ## x.time - (double)ts_ ## x.time + ((double)te_ ## x.millitm - (double)ts_ ## x.millitm)/1e3)

// other "CONSTANT" declarations
#define PI    3.1415926535897932384626433832795 // PI 
#define root2 1.4142135623730950488016887242097 // sqrt(2)
#define root3 1.7320508075688772935274463415059	// sqrt(3)	
#define SIN(x) sin(x*0.017453292)				// sin for degrees
#define COS(x) cos(x*0.017453292)				// cos for degrees
#define ASIN(x) asin(x)*57.29577951				// arcsin for degrees
#define ACOS(x) acos(x)*57.29577951				// arccos for degrees

//-----------------------------------------------------------------------------
// Miscellaneous helper functions
//-----------------------------------------------------------------------------
#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=NULL; } }
#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=NULL; } }
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }


#ifdef _DEBUG
	#define CHECK_NULL(p) {if((p)==NULL ) {\
							_TCHAR strError[300];\
							_stprintf(strError,_T("DEBUG: Call returned NULL in file %s in line %d."),_T(__FILE__), __LINE__);\
							MessageBox(NULL,strError,_T("NULL Returned"),MB_OK);\
                            return E_FAIL;\
						}}
#else
	#define CHECK_NULL(p)	{if((p)==NULL) return E_FAIL;}
#endif

#ifdef _DEBUG
	#define CHECK_HR(p) {if( FAILED(hr=(p))) {\
							_TCHAR strError[300];\
							_stprintf(strError,_T("DEBUG: HRESULT FAILED (%s) in file %s in line %d."),DXGetErrorString9(hr),_T(__FILE__), __LINE__);\
							MessageBox(NULL,strError,_T("HRESULT FAILED"),MB_OK);\
                            return hr;\
						}}
#else
	#define CHECK_HR(p)	{if( FAILED(hr=(p))) return hr;}
#endif

#ifdef _DEBUG
	#define CHECK_HR_(p) {if( FAILED(hr=(p))) {\
							_TCHAR strError[300];\
							_stprintf(strError,_T("DEBUG: HRESULT FAILED (%s) in file %s in line %d."),DXGetErrorString9(hr),_T(__FILE__), __LINE__);\
							MessageBox(NULL,strError,_T("HRESULT FAILED"),MB_OK);\
						}}
#else
	#define CHECK_HR_(p) p
#endif

namespace DirectXUtils {
	HRESULT createRenderTexture(PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface, LPDIRECT3DDEVICE9 &pDevice,
								int iWidth, int iHeight, D3DFORMAT format);
	HRESULT createRenderTexture(LPD3DXRENDERTOSURFACE &pRenderToSurface, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface,
								LPDIRECT3DDEVICE9 &pDevice, D3DXMATRIXA16 &pOffScreenProjMat, int iWidth, int iHeight, D3DFORMAT format);
	HRESULT createRenderTexture(LPD3DXRENDERTOSURFACE &pRenderToSurface, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface,
								LPDIRECT3DDEVICE9 &pDevice, int iWidth, int iHeight, D3DFORMAT format);
	bool fileExists(const _TCHAR *fileName);
	_TCHAR* findPath(_TCHAR *fileName, _TCHAR* path);
	HRESULT fileExistsVerbose(const _TCHAR *fileName);
	HRESULT fileExistsVerbose(const _TCHAR *fileName, const _TCHAR *printFileName);
	HRESULT checkLoadTexture(_TCHAR *fileName, LPDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DDEVICE9 &pDevice, _TCHAR*path=_T(".\\"));
	HRESULT checkLoadShader(_TCHAR *fileName, LPD3DXEFFECT &pShader, LPDIRECT3DDEVICE9 &pDevice, _TCHAR* path=_T(".\\"), DWORD dwFlags=0, const D3DXMACRO* pDefines = NULL);
	HRESULT checkLoadCubemap(_TCHAR *pSrcPath, IDirect3DCubeTexture9 **ppCubeTexture,
		                     D3DFORMAT format, LPDIRECT3DDEVICE9 &pDevice,
							 int iSize=256, _TCHAR *ppFilenames[6] = NULL);
	HRESULT checkLoadCubemap(_TCHAR *pSrcPath, IDirect3DCubeTexture9 **ppCubeTexture,
		                     D3DFORMAT format, LPDIRECT3DDEVICE9 &pDevice,
							 int iSize, _TCHAR *ppFilename);
	void swapSurfaceAndTexture(LPDIRECT3DSURFACE9 &pS1,PDIRECT3DTEXTURE9 &pT1, 
										 LPDIRECT3DSURFACE9 &pS2,PDIRECT3DTEXTURE9 &pT2);
		                  
	HRESULT loadPaletteFromFile(_TCHAR *pstrFile, BYTE pbPallette[256][4]);
	int clamp(int i, int iMin=0, int iMax=255);
	float fClamp(float f, float fMin=0, float fMax=1);

	HRESULT createFilledIndexBuffer(LPDIRECT3DDEVICE9 &pd3dDevice, VOID* hIndexArray, int iCount, DWORD dwFormat, D3DPOOL d3dPool, LPDIRECT3DINDEXBUFFER9 &pIB);
	HRESULT createFilledVertexBuffer(LPDIRECT3DDEVICE9 &pDevice, VOID* hDataArray, int iSize, DWORD dwFormat, D3DPOOL d3dPool, LPDIRECT3DVERTEXBUFFER9	&pVB);
	static double fps = 0.0;       // for fps-display. Updated every second.
	static double last;
	static double lastfps;
	static double elapsed_time;    // to hold queried system time

	void fpsTimerStart();
	void fpsTimerStartFrame();
	_TCHAR* fpsTimerGetFPS();
	_TCHAR* fpsTimerGetElapsed();

	inline float smoothstep(float x);
	inline bool floatEqual(float a, float b);

	inline void findMinMax(int a, int b, int &min, int &max);
	inline void findMinMax(int a, int b, int c, int &min, int &max);
	inline void findMinMax(int a, int b, int c, int d, int &min, int &max);

	DWORD FtoDW( FLOAT f );

	int log2(int n);
	int pow2(int e);
	int gaussianSum(int n);

	void normalizeScaleByte(BYTE &x, BYTE &y, BYTE &z);

	_TCHAR* limitString(_TCHAR* strIn, unsigned int iMaxLength);

	D3DXVECTOR4 PlaneFromPoints(D3DXVECTOR3 v1, D3DXVECTOR3 v2, D3DXVECTOR3 v3);
	bool isTextureFormatOk( D3DFORMAT CheckFormat, D3DFORMAT AdapterFormat, LPDIRECT3D9 pD3D);
	bool GetFileName(LPCTSTR lpstrTitle, LPCTSTR lpstrFilter, TCHAR **filename, bool save);

	void OutputDebugFloat(float f);
	void OutputDebugInt(int i);
}

class MESH {
	private:
		LPD3DXMESH              m_pMesh;			// Our mesh object in sysmem
		D3DMATERIAL9*           m_pMeshMaterials;	// Materials for our mesh
		LPDIRECT3DTEXTURE9*     m_ppMeshTextures;	// Textures for our mesh
		DWORD                   m_dwNumMaterials;	// Number of mesh materials
		D3DXVECTOR3				m_d3vPosition;		// postion of the object
		D3DXVECTOR3				m_d3vOffset;		// offset added to position
		float					m_fRotAngle;		// rotation angle
		D3DXVECTOR3				m_vRotAxis;			// axis of rotation

	public:
		MESH();
		virtual ~MESH();
		HRESULT LoadFromFile(LPDIRECT3DDEVICE9 pd3dDevice, _TCHAR *strFilename, DWORD dwOptions=D3DXMESH_SYSTEMMEM);
		void Shift(D3DXVECTOR3 d3vOffset);
		void Place(D3DXVECTOR3 d3vPosition);
		void SetRotAngle(float fRotAngle);
		void SetRotAxis(D3DXVECTOR3 vRotAxis);
		HRESULT Render(LPDIRECT3DDEVICE9 pd3dDevice);
		HRESULT Render(LPDIRECT3DDEVICE9 pd3dDevice, D3DXMATRIX matWorld);
		void DeleteDeviceObjects();
};

class LOGFILE {
	private:
		bool m_bSecure;
		FILE *m_stream;
		_TCHAR *m_pcFilename;
		void createFile(_TCHAR *filename);
	public:
		LOGFILE();
		LOGFILE(_TCHAR *filename, bool bSecure);
		~LOGFILE();
		void writeLog(_TCHAR *data);
		void writeLog(int i);
		void writeLog(float f);
		void testValue(_TCHAR *comment, float s, float h);
		void show();
};

class KEYSHIFT {
	protected:
		static WPARAM	keys[3][2];
		D3DXMATRIXA16	m_Mat;
		D3DXVECTOR3		m_vParam;
		INTVECTOR3		m_iDir;
		int				m_iSign[3];
		float			m_fStep[3];

		virtual void computeMatrix();
	public:
		KEYSHIFT();
		KEYSHIFT(D3DXMATRIX mat);
		virtual void SetMatrix(D3DXMATRIX mat);

		void SetDirections(int x, int y, int z);
		void SetDirSign(int x, int y, int z);
		void SetStep(float x, float y, float z);
		D3DXMATRIXA16 getMatrix();
		void FireKeyUp(float step=1.0f) {MsgProc(NULL,WM_KEYDOWN,VK_UP,NULL,step);}
		void FireKeyDown(float step=1.0f) {MsgProc(NULL,WM_KEYDOWN,VK_DOWN,NULL,step);}
		void FireKeyLeft(float step=1.0f) {MsgProc(NULL,WM_KEYDOWN,VK_LEFT,NULL,step);}
		void FireKeyRight(float step=1.0f) {MsgProc(NULL,WM_KEYDOWN,VK_RIGHT,NULL,step);}
		void FireKeyBack(float step=1.0f) {MsgProc(NULL,WM_KEYDOWN,VK_PRIOR,NULL,step);}
		void FireKeyFront(float step=1.0f) {MsgProc(NULL,WM_KEYDOWN,VK_NEXT,NULL,step);}
		bool MsgProc( HWND, UINT uMsg, WPARAM wParam, LPARAM, float step=1.0f);
};

class KEYSCALE : public KEYSHIFT {
	protected:
		virtual void computeMatrix();
	public:
		KEYSCALE();
		KEYSCALE(D3DXMATRIX	mat);
		virtual void SetMatrix(D3DXMATRIX mat);
};