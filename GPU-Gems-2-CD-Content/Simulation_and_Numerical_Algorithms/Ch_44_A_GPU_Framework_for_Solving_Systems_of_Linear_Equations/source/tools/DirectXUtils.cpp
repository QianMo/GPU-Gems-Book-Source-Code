/**************************************************************
 *                                                            *
 * description: tools for use with DirectX                    *
 * version    : 2.84                                          *
 * date       : 01.Jul.2003                                   *
 * modified   : 08.Oct.2004                                   *
 * author     : Jens Krüger                                   *
 * e-mail     : mail@jens-krueger.com                         *
 *                                                            *
 **************************************************************/
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "directxUtils.h"

void DirectXUtils::normalizeScaleByte(BYTE &x, BYTE &y, BYTE &z) {
	float fLength = sqrt((float)x*(float)x+(float)y*(float)y+(float)z*(float)z);

	x = (BYTE)((x*255)/fLength);
	y = (BYTE)((y*255)/fLength);
	z = (BYTE)((z*255)/fLength);
}


DWORD DirectXUtils::FtoDW( FLOAT f ) {
	return *((DWORD*)&f); 
}

int DirectXUtils::log2(int n) {
	int iLog=0;
	while( n>>=1 ) iLog++;
	return iLog;
}

int DirectXUtils::pow2(int e) {
	return 1<<e;
}

int DirectXUtils::gaussianSum(int n) {
	return n*(n+1)/2;
}

void DirectXUtils::swapSurfaceAndTexture(LPDIRECT3DSURFACE9 &pS1,PDIRECT3DTEXTURE9 &pT1, 
										 LPDIRECT3DSURFACE9 &pS2,PDIRECT3DTEXTURE9 &pT2) {
	LPDIRECT3DSURFACE9 pSTemp = pS1;
	PDIRECT3DTEXTURE9 pTTemp = pT1;

	pS1 = pS2;	    pT1 = pT2;
	pS2 = pSTemp;	pT2 = pTTemp;
}



//-----------------------------------------------------------------------------
// Name: createRenderTexture()
// Desc: setup everything necessary to render to a texture
//-----------------------------------------------------------------------------
HRESULT DirectXUtils::createRenderTexture(PDIRECT3DTEXTURE9 &pTexture, 
										  LPDIRECT3DSURFACE9 &pTextureSurface, 
										  LPDIRECT3DDEVICE9 &pDevice,
										  int iWidth,
										  int iHeight,
										  D3DFORMAT format) {
	HRESULT hr;

	// Create Texture
	CHECK_HR( pDevice->CreateTexture(iWidth, iHeight, 1, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT, &pTexture,0));
    pTexture->GetSurfaceLevel( 0, &pTextureSurface );

	return S_OK;
}

HRESULT DirectXUtils::createRenderTexture(LPD3DXRENDERTOSURFACE &pRenderToSurface, 
										  PDIRECT3DTEXTURE9 &pTexture, 
										  LPDIRECT3DSURFACE9 &pTextureSurface, 
										  LPDIRECT3DDEVICE9 &pDevice,
										  int iWidth,
										  int iHeight,
										  D3DFORMAT format) {
	HRESULT hr;

	// Create Texture
	CHECK_HR( pDevice->CreateTexture(iWidth, iHeight, 1, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT, &pTexture,0));

	// Create an off-screen "render to" surface
    D3DSURFACE_DESC desc;
    pTexture->GetSurfaceLevel( 0, &pTextureSurface );
    pTextureSurface ->GetDesc( &desc );
	return D3DXCreateRenderToSurface( pDevice, desc.Width, desc.Height, desc.Format, TRUE, D3DFMT_D24S8, &pRenderToSurface );
}

HRESULT DirectXUtils::createRenderTexture(LPD3DXRENDERTOSURFACE &pRenderToSurface, 
										  PDIRECT3DTEXTURE9 &pTexture, 
										  LPDIRECT3DSURFACE9 &pTextureSurface, 
										  LPDIRECT3DDEVICE9 &pDevice,
										  D3DXMATRIXA16 &pOffScreenProjMat,
										  int iWidth,
										  int iHeight,
										  D3DFORMAT format) {

	HRESULT hr;

	// Create Texture
	CHECK_HR( pDevice->CreateTexture(iWidth, iHeight, 1, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT, &pTexture,0));

	// Create an off-screen "render to" surface
    D3DSURFACE_DESC desc;
    pTexture->GetSurfaceLevel( 0, &pTextureSurface );
    pTextureSurface ->GetDesc( &desc );
	CHECK_HR( D3DXCreateRenderToSurface( pDevice, desc.Width, desc.Height, desc.Format, TRUE, D3DFMT_D24S8, &pRenderToSurface ));

	// set projection matrix
	D3DXMatrixPerspectiveFovLH( &pOffScreenProjMat, D3DX_PI/4, (FLOAT)desc.Width/desc.Height, 0.1f, 30.0f );

	return S_OK;
}

										  
bool DirectXUtils::fileExists(const _TCHAR *fileName) {
	FILE *stream;	
	stream = _tfopen(fileName, _T("r"));
	if (stream != NULL) {
		fclose( stream );
		return true;
	} else return false;
}

//-----------------------------------------------------------------------------
// Name: fileExistsVerbose()
// Desc: checks if a file exists and outputs an error message if not
//-----------------------------------------------------------------------------
HRESULT DirectXUtils::fileExistsVerbose(const _TCHAR *fileName, const _TCHAR *printFileName) {
	if (_tcslen(fileName) <= 0 || !fileExists(fileName)){
		_TCHAR sMsg[255];	
		_stprintf(sMsg,_T("Critical Error: File '%s' not found."),printFileName);
        MessageBox(NULL, sMsg, _T("Critical Error"), MB_ICONERROR);
		return E_FAIL;
    } else return S_OK;
}

//-----------------------------------------------------------------------------
// Name: fileExistsVerbose()
// Desc: checks if a file exists and outputs an error message if not
//-----------------------------------------------------------------------------
HRESULT DirectXUtils::fileExistsVerbose(const _TCHAR *fileName) {
	return fileExistsVerbose(fileName, fileName);
}

_TCHAR* DirectXUtils::findPath(_TCHAR *fileName, _TCHAR* path) {
	_TCHAR *searchFile = new _TCHAR[_tcslen(path)+_tcslen(fileName)+2];

	// search in the given path
	_stprintf(searchFile,_T("%s%s"),path,fileName);
	if (fileExists(searchFile))	return searchFile;

	// search in the current directory
	_stprintf(searchFile,_T(".\\%s"),fileName);
	if (fileExists(searchFile)) return searchFile;

	// search in the parent directory
	_stprintf(searchFile,_T("..\\%s"),fileName);
	if (fileExists(searchFile)) return searchFile;

	_stprintf(searchFile,_T(""));
	return searchFile;
}

//-----------------------------------------------------------------------------
// Name: checkLoadTexture()
// Desc: checks if a texture file exists and loads it
//-----------------------------------------------------------------------------
HRESULT DirectXUtils::checkLoadTexture(_TCHAR *fileName, LPDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DDEVICE9 &pDevice, _TCHAR*path) {
	HRESULT hr;

	_TCHAR *pcCompleteFilename = findPath(fileName,path);

	// check for the most common error: file does not exist
	if (FAILED(hr = fileExistsVerbose(pcCompleteFilename,fileName))) {
		SAFE_DELETE_ARRAY( pcCompleteFilename );
		return hr;
	}

	hr = D3DXCreateTextureFromFile(pDevice,pcCompleteFilename,&pTexture);

	SAFE_DELETE_ARRAY( pcCompleteFilename );
    return hr;
}

//-----------------------------------------------------------------------------
// Name: checkLoadShader()
// Desc: checks if a shader file exists and loads it doing additional checking
//-----------------------------------------------------------------------------
HRESULT DirectXUtils::checkLoadShader(_TCHAR *fileName, LPD3DXEFFECT	&pShader, LPDIRECT3DDEVICE9 &pDevice, _TCHAR*path, DWORD dwFlags, const D3DXMACRO *pDefines) {

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
	#ifdef DEBUG_VS		
		dwFlags = 0;
    #endif
    #ifdef DEBUG_PS
		dwFlags = 0;
    #endif

	#ifdef DEBUG_VS		
        dwFlags |= D3DXSHADER_SKIPOPTIMIZATION | D3DXSHADER_FORCE_VS_SOFTWARE_NOOPT | D3DXSHADER_DEBUG;
    #endif

    #ifdef DEBUG_PS
        dwFlags |= D3DXSHADER_SKIPOPTIMIZATION | D3DXSHADER_FORCE_PS_SOFTWARE_NOOPT | D3DXSHADER_DEBUG;
	#endif
	
	
	LPD3DXBUFFER pError = NULL;
	HRESULT hr;

	_TCHAR *pcCompleteFilename = findPath(fileName,path);

	// check for the most common error: file does not exist
	if (FAILED(hr = fileExistsVerbose(pcCompleteFilename,fileName))) {
		SAFE_DELETE_ARRAY( pcCompleteFilename );
		return hr;
	}

	hr = D3DXCreateEffectFromFile( pDevice,				// the directX device
								   pcCompleteFilename,	// srcFile
								   pDefines,			// pDefines: defines preprocessor instructions
								   NULL,				// pIncludes: use for handling #include directives
								   dwFlags,				// additional compiler flags
								   NULL,				// pool for shared parameters
								   &pShader,			// returns buffer containing the compiled effect
								   &pError);			// returns buffer containing a listing of compile errors

	SAFE_DELETE_ARRAY( pcCompleteFilename );

	// handle compile errors
    if (pError != NULL)  {

		#ifdef  _UNICODE
			wchar_t *errortxt = new wchar_t[pError->GetBufferSize()];
			MultiByteToWideChar(CP_ACP, 0, (char*)pError->GetBufferPointer(), -1, errortxt, pError->GetBufferSize());
	        MessageBox(NULL, errortxt, L"HLSL Compile Error", MB_ICONERROR);
			delete [] errortxt;
		#else
			char* errortxt = (char*)pError->GetBufferPointer();
	        MessageBox(NULL, errortxt, "HLSL Compile Error", MB_ICONERROR);
		#endif

	    SAFE_RELEASE(pError);
    }

	CHECK_HR(hr);

	return S_OK;
}


TIMER_VAR(sys_timer);

void DirectXUtils::fpsTimerStart() {
	TIMER_START(sys_timer);    // start timer
	TIMER_PROBE(sys_timer);
	last=TIMER_RESULT(sys_timer);
}

void DirectXUtils::fpsTimerStartFrame() {	
	TIMER_PROBE(sys_timer);               // Query timer
	elapsed_time=TIMER_RESULT(sys_timer);
}

_TCHAR* DirectXUtils::fpsTimerGetFPS() {	
	static _TCHAR text[255] = _T("\0");

	if (elapsed_time-last>=1.0) {
		lastfps=fps/(elapsed_time-last);		

		_sntprintf(text,254,_T("@ %.2f fps"),lastfps);
		
		last=elapsed_time;
		fps=0.0;
	}
	fps+=1.0; 

	return text;
}

_TCHAR* DirectXUtils::fpsTimerGetElapsed() {	
	static _TCHAR text[255] = _T("\0");
	_sntprintf(text,254,_T("%.2f"),elapsed_time);
	return text;
}

inline float DirectXUtils::smoothstep(float x) {return 3*x*x-2*x*x*x;}

inline bool DirectXUtils::floatEqual(float a, float b) {
	return fabs(a-b) < 0.0000000000001;
}

inline void DirectXUtils::findMinMax(int a, int b, int &min, int &max) {
	if (a<b) {min = a; max = b;} else {min = b; max = a;}
}

inline void DirectXUtils::findMinMax(int a, int b, int c, int &min, int &max) {
	int min1, max1, dummy;
	findMinMax(a,b,min1,max1);
	findMinMax(min1,c,min,dummy);
	findMinMax(max1,c,dummy,max);
}

inline void DirectXUtils::findMinMax(int a, int b, int c, int d, int &min, int &max) {
	int min1, min2, max1, max2, dummy;
	findMinMax(a,b,min1,max1);
	findMinMax(c,d,min2,max2);
	findMinMax(min1,min2,min,dummy);
	findMinMax(max1,max2,dummy,max);
}

HRESULT DirectXUtils::checkLoadCubemap(_TCHAR *pSrcPath, IDirect3DCubeTexture9 **ppCubeTexture, D3DFORMAT format, LPDIRECT3DDEVICE9 &pDevice, int iSize, _TCHAR *ppcFilename) {

	HRESULT hr;
    CHECK_HR( pDevice->CreateCubeTexture(iSize,1,0,format,D3DPOOL_DEFAULT,ppCubeTexture,NULL)); 
	_TCHAR pcFilename[2000];
    _sntprintf(pcFilename,1999, _T("%s\\%s"), pSrcPath, ppcFilename);

	LPDIRECT3DSURFACE9 mainMemSurface=NULL;
	PDIRECT3DTEXTURE9 pTexture=NULL;
	CHECK_HR( pDevice->CreateTexture(iSize, iSize, 1, NULL, format, D3DPOOL_SYSTEMMEM, &pTexture,0));
	pTexture->GetSurfaceLevel(0,&mainMemSurface);

	CHECK_HR( fileExistsVerbose(pcFilename));
	CHECK_HR( D3DXLoadSurfaceFromFile(mainMemSurface,NULL,NULL,pcFilename,NULL,D3DX_DEFAULT,0xFF000000,NULL));

	LPDIRECT3DSURFACE9 pCubeMapFace;

    // load the six faces of the cube map
    for( DWORD i=0; i<6; i++ ) {
        (*ppCubeTexture)->GetCubeMapSurface( (D3DCUBEMAP_FACES)i, 0, &pCubeMapFace );
		CHECK_HR( pDevice->UpdateSurface(mainMemSurface,NULL,pCubeMapFace,NULL));
        pCubeMapFace->Release();
    }

	SAFE_RELEASE(mainMemSurface);
	SAFE_RELEASE(pTexture);
    return S_OK;
}

HRESULT DirectXUtils::checkLoadCubemap(_TCHAR *pSrcPath, IDirect3DCubeTexture9 **ppCubeTexture, D3DFORMAT format, LPDIRECT3DDEVICE9 &pDevice, int iSize, _TCHAR *ppcFilenames[6]) {

	_TCHAR *ppcDefaultCubeMapFaces[6] = {_T("POS_X.BMP"),_T("NEG_X.BMP"),_T("POS_Y.BMP"),_T("NEG_Y.BMP"),_T("POS_Z.BMP"),_T("NEG_Z.BMP")};


	if (ppcFilenames == NULL) ppcFilenames = ppcDefaultCubeMapFaces;

	HRESULT hr;

    CHECK_HR( pDevice->CreateCubeTexture( 
        iSize,                        // edge length
        1,                            // mip map levels (0 = complete till 1x1)
        0,                            // usage
        format,						  // format
        D3DPOOL_MANAGED,              // memory pool
        ppCubeTexture,                // cube texture
        NULL ));	                  // handle

    
	_TCHAR pcFilename[2000];
	LPDIRECT3DSURFACE9 pCubeMapFace;

    // load the six faces of the cube map
    for( DWORD i=0; i<6; i++ ) {

        _sntprintf(pcFilename,1999, _T("%s\\%s"), pSrcPath, ppcFilenames[i]);

		// check for the most common error: file does not exist
		CHECK_HR( fileExistsVerbose(pcFilename));

        (*ppCubeTexture)->GetCubeMapSurface( (D3DCUBEMAP_FACES)i, 0, &pCubeMapFace );

        CHECK_HR( D3DXLoadSurfaceFromFile(
            pCubeMapFace,           // DestSurface
            NULL,                   // pDestPalette (NULL is no palette)
            NULL,                   // pDestRect (NULL is entire rectangle)
            pcFilename,        // pSrcFile
            NULL,                   // pSrcRect (NULL is entire rectangle)
            D3DX_DEFAULT,		    // filter
            0xFF000000,             // color key (0xFF000000 is none)
            NULL                    // pSrcInfo           
            ));

        pCubeMapFace->Release();
    }
    return S_OK;
}

HRESULT DirectXUtils::createFilledIndexBuffer(LPDIRECT3DDEVICE9 &pd3dDevice, VOID* hIndexArray, int iCount, DWORD dwFormat, D3DPOOL d3dPool, LPDIRECT3DINDEXBUFFER9 &pIB) {
	HRESULT hr;	
	
	// Create the indexbuffer.
	if (dwFormat == D3DFMT_INDEX32) {
		CHECK_HR(pd3dDevice->CreateIndexBuffer(sizeof(DWORD)*iCount,D3DUSAGE_WRITEONLY,D3DFMT_INDEX32,d3dPool,&pIB, NULL));

		DWORD* pIndices;
		pIB->Lock( 0, 0, (void**)&pIndices, NULL );
			memcpy(pIndices,hIndexArray,sizeof(DWORD)*iCount);
		pIB->Unlock();
	} else {
		CHECK_HR(pd3dDevice->CreateIndexBuffer(sizeof(WORD)*iCount,D3DUSAGE_WRITEONLY,D3DFMT_INDEX16,d3dPool,&pIB, NULL));

		WORD* pIndices;
		pIB->Lock( 0, 0, (void**)&pIndices, NULL );
			memcpy(pIndices,hIndexArray,sizeof(WORD)*iCount);
		pIB->Unlock();
	}

	return S_OK;
}

HRESULT DirectXUtils::createFilledVertexBuffer(LPDIRECT3DDEVICE9 &pDevice, VOID* hDataArray, int iSize, DWORD dwFormat, D3DPOOL d3dPool, LPDIRECT3DVERTEXBUFFER9 &pVB) {
	HRESULT hr;	
	
	// Create the vertex buffer.
	CHECK_HR(pDevice->CreateVertexBuffer(iSize,D3DUSAGE_WRITEONLY,dwFormat,d3dPool,&pVB, NULL));

	// Fill the vertex buffer.
	VOID* pVertices;
	CHECK_HR(pVB->Lock( 0, 0, (void**)&pVertices, NULL ));
		memcpy( pVertices, hDataArray, iSize );
	pVB->Unlock();

	return S_OK;
}

HRESULT DirectXUtils::loadPaletteFromFile(_TCHAR *pstrFile, BYTE pbPallette[256][4]) {
	DWORD  read;
	_TCHAR str[10000];
	int  r,g,b,a;

	HANDLE file = CreateFile(pstrFile, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL); 
	ReadFile(file, str, 10000, &read, NULL);
	CloseHandle(file);

	DWORD index = 0;

	for (int i=0; i<256; i++) {
		if (index > read) {
			MessageBox(NULL, _T("Unexpected end of pallette file."), _T("ERROR"), MB_ICONHAND);
			return E_FAIL;
		}

		while(str[index++] == ' ');
		index--;
		r = _ttoi(str+index);
		r = clamp(r);
		
		while(str[index++] != ' ');
		while(str[index++] == ' ');
		index--;			
		g = _ttoi(str+index);
		g = clamp(g);

		while(str[index++] != ' ');
		while(str[index++] == ' ');
		index--;			
		b = _ttoi(str+index);
		b = clamp(b);

		while(str[index++] != ' ');
		while(str[index++] == ' ');
		index--;			
		a = _ttoi(str+index);
		a = clamp(a);

		while(str[index++] != '\n');

		pbPallette[i][0] = (BYTE)r;
		pbPallette[i][1] = (BYTE)g;
		pbPallette[i][2] = (BYTE)b;
		pbPallette[i][3] = (BYTE)a;
	}

	return S_OK;;
}

int DirectXUtils::clamp(int i, int iMin, int iMax) {
	return MAX(iMin,MIN(i,iMax));
}

float DirectXUtils::fClamp(float f, float fMin, float fMax) {
	return MAX(fMin,MIN(f,fMax));
}


void DirectXUtils::OutputDebugFloat(float f) {
	TCHAR str[1024];
	_stprintf(str, _T("%g"), f);
	OutputDebugString(str);
}

void DirectXUtils::OutputDebugInt(int i) {
	TCHAR str[1024];
	_stprintf(str, _T("%i"), i);
	OutputDebugString(str);
}


_TCHAR* DirectXUtils::limitString(_TCHAR* strIn, unsigned int iMaxLength) {
	_TCHAR* strResult = new _TCHAR[iMaxLength+1];

	if (_tcslen(strIn) > iMaxLength) {	
		_stprintf(strResult,_T("...%s"),strIn+_tcslen(strIn)-(iMaxLength-4));
	} else {
		_tcscpy(strResult,strIn);
	}

	return strResult;
}

D3DXVECTOR4 DirectXUtils::PlaneFromPoints(D3DXVECTOR3 v1, D3DXVECTOR3 v2, D3DXVECTOR3 v3) {
	D3DXVECTOR3 normal;
	D3DXVECTOR3 dir1 = (v1-v2);
	D3DXVECTOR3 dir2 = (v1-v3);
	D3DXVec3Cross( &normal, &dir1, &dir2 );	
	float d = - D3DXVec3Dot(&normal, &v1);
	return D3DXVECTOR4(normal.x,normal.y,normal.z,d);
}

bool DirectXUtils::isTextureFormatOk( D3DFORMAT CheckFormat, D3DFORMAT AdapterFormat, LPDIRECT3D9 pD3D) 
{
	if (pD3D)
		return SUCCEEDED(pD3D->CheckDeviceFormat( D3DADAPTER_DEFAULT,D3DDEVTYPE_HAL,AdapterFormat,0,D3DRTYPE_TEXTURE,CheckFormat)) ||
			   SUCCEEDED(pD3D->CheckDeviceFormat( D3DADAPTER_DEFAULT,D3DDEVTYPE_REF,AdapterFormat,0,D3DRTYPE_TEXTURE,CheckFormat));
	else
		return false;
}

bool DirectXUtils::GetFileName(LPCTSTR lpstrTitle, LPCTSTR lpstrFilter, TCHAR **filename, bool save) {
	BOOL result;
	OPENFILENAME ofn;
	ZeroMemory(&ofn,sizeof(OPENFILENAME));
	
	static TCHAR szFile[MAX_PATH];
	szFile[0] = 0;

	//====== Dialog parameters
	ofn.lStructSize   = sizeof(OPENFILENAME);
	ofn.lpstrFilter   = lpstrFilter;
	ofn.nFilterIndex  = 1;
	ofn.lpstrFile     = szFile;
	ofn.nMaxFile      = sizeof(szFile);
	ofn.lpstrTitle    = lpstrTitle;
	ofn.nMaxFileTitle = sizeof (ofn.lpstrTitle);
	ofn.hwndOwner     = NULL;

	if (save) {
		ofn.Flags = OFN_NOCHANGEDIR | OFN_HIDEREADONLY | OFN_EXPLORER | OFN_OVERWRITEPROMPT;
		result = GetSaveFileName(&ofn);
	} else {
		ofn.Flags = OFN_NOCHANGEDIR | OFN_HIDEREADONLY | OFN_EXPLORER | OFN_FILEMUSTEXIST;
		result = GetOpenFileName(&ofn);
	}
	if (result)	{
		*filename = (TCHAR*)szFile;
		return true;
	} else {
		*filename = NULL;
		return false;
	}

}

/********************************************************
				MESH
********************************************************/
MESH::MESH() {
	m_pMesh          = NULL;
	m_pMeshMaterials = NULL;
	m_ppMeshTextures = NULL;
	m_dwNumMaterials = 0L;
	m_d3vPosition	 = D3DXVECTOR3(0.0f,0.0f,0.0f);
	m_d3vOffset		 = D3DXVECTOR3(0.0f,0.0f,0.0f);
	m_fRotAngle		 = 0.0f;
	m_vRotAxis		 = D3DXVECTOR3(1.0f,0.0f,0.0f);
}

MESH::~MESH() {
	DeleteDeviceObjects();
}

HRESULT MESH::LoadFromFile(LPDIRECT3DDEVICE9 pd3dDevice, _TCHAR *strFilename, DWORD dwOptions) {
	HRESULT hr;
	LPD3DXBUFFER pD3DXMtrlBuffer;

	// Load the mesh from the specified file
	CHECK_HR( D3DXLoadMeshFromX(strFilename, dwOptions, pd3dDevice, NULL, &pD3DXMtrlBuffer, NULL, &m_dwNumMaterials, &m_pMesh ) );

	// We need to extract the material properties and texture names from the 
	// pD3DXMtrlBuffer
	D3DXMATERIAL* d3dxMaterials = (D3DXMATERIAL*)pD3DXMtrlBuffer->GetBufferPointer();
	m_pMeshMaterials = new D3DMATERIAL9[m_dwNumMaterials];
	m_ppMeshTextures = new LPDIRECT3DTEXTURE9[m_dwNumMaterials];

	for( DWORD i = 0; i < m_dwNumMaterials; i++ )
    {
        m_pMeshMaterials[i] = d3dxMaterials[i].MatD3D;
        m_pMeshMaterials[i].Ambient = m_pMeshMaterials[i].Diffuse;
        m_ppMeshTextures[i]  = NULL;

        // Get a path to the texture
        if( d3dxMaterials[i].pTextureFilename != NULL && d3dxMaterials[i].pTextureFilename > 0)
        {
			#ifdef UNICODE
				WCHAR wszBuf[MAX_PATH];
				MultiByteToWideChar( CP_ACP, 0, d3dxMaterials[i].pTextureFilename, -1, wszBuf, MAX_PATH );
				wszBuf[MAX_PATH - 1] = L'\0';

				// Load the texture
				CHECK_HR(D3DXCreateTextureFromFile( pd3dDevice, wszBuf, &m_ppMeshTextures[i] ));
			#else
			    CHECK_HR(D3DXCreateTextureFromFile( pd3dDevice, d3dxMaterials[i].pTextureFilename, &m_ppMeshTextures[i] ) );
			#endif

        }
    }


	// Done with the material buffer
    SAFE_RELEASE( pD3DXMtrlBuffer );

	return S_OK;
}

HRESULT MESH::Render(LPDIRECT3DDEVICE9 pd3dDevice) {
	HRESULT hr;
    // Meshes are divided into subsets, one for each material. Render them in a loop
    for( DWORD i=0; i<m_dwNumMaterials; i++ )  {
        // Set the material and texture for this subset
        CHECK_HR(pd3dDevice->SetMaterial( &m_pMeshMaterials[i] ));
        CHECK_HR(pd3dDevice->SetTexture( 0, m_ppMeshTextures[i] ));
    
        // Draw the mesh subset
        CHECK_HR(m_pMesh->DrawSubset( i ));
    }

	return S_OK;
}

HRESULT MESH::Render(LPDIRECT3DDEVICE9 pd3dDevice, D3DXMATRIX matWorld) {
	HRESULT hr;

	D3DXMATRIXA16 matLocalWorld, matRotate, matTranslate;
	D3DXMatrixTranslation(&matTranslate, m_d3vPosition.x+m_d3vOffset.x,m_d3vPosition.y+m_d3vOffset.y,m_d3vPosition.z+m_d3vOffset.z);
	D3DXMatrixRotationAxis(&matRotate, &m_vRotAxis, m_fRotAngle);
	D3DXMatrixMultiply( &matLocalWorld, &matTranslate, &matWorld);
	D3DXMatrixMultiply( &matLocalWorld, &matRotate, &matLocalWorld);
	pd3dDevice->SetTransform( D3DTS_WORLD, &matLocalWorld );

	hr = Render(pd3dDevice);

	pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );

	return hr;
}

void MESH::DeleteDeviceObjects() {
	SAFE_DELETE_ARRAY(m_pMeshMaterials);
	if( m_ppMeshTextures ) {
		for( DWORD i = 0; i < m_dwNumMaterials; i++ ) SAFE_RELEASE(m_ppMeshTextures[i]);
		SAFE_DELETE_ARRAY(m_ppMeshTextures);
	}
	SAFE_RELEASE(m_pMesh);
}

void MESH::Place(D3DXVECTOR3 d3vPosition) {
	m_d3vPosition = d3vPosition;
}

void MESH::Shift(D3DXVECTOR3 d3vOffset) {
	m_d3vOffset = d3vOffset;
}

void MESH::SetRotAngle(float fRotAngle) {
	m_fRotAngle = fRotAngle*0.017453292f;
}

void MESH::SetRotAxis(D3DXVECTOR3 vRotAxis) {
	m_vRotAxis = vRotAxis;
}

/********************************************************
				LOGFILE
********************************************************/
LOGFILE::LOGFILE() {
	m_bSecure = true;
	createFile(_T("logfile.txt"));
}

LOGFILE::LOGFILE(_TCHAR *filename, bool bSecure) {
	m_bSecure = bSecure;
	createFile(filename);
}

LOGFILE::~LOGFILE() {
	if (!m_bSecure) fclose( m_stream );
	delete [] m_pcFilename;
}

void LOGFILE::writeLog(_TCHAR *data) {
	if (m_bSecure) m_stream = _tfopen( m_pcFilename, _T("a") );
	_ftprintf( m_stream, _T("%s\n"), data );
	if (m_bSecure) fclose( m_stream );
}

void LOGFILE::writeLog(int i) {
	if (m_bSecure) m_stream = _tfopen( m_pcFilename, _T("a") );
	_ftprintf( m_stream, _T("%i\n"), i);
	if (m_bSecure) fclose( m_stream );
}

void LOGFILE::writeLog(float f) {
	if (m_bSecure) m_stream = _tfopen( m_pcFilename, _T("a") );
	_ftprintf( m_stream, _T("%g\n"), f);
	if (m_bSecure) fclose( m_stream );
}

void LOGFILE::testValue(_TCHAR *comment, float s, float h) {
	if (m_bSecure) m_stream = _tfopen( m_pcFilename, _T("a") );

	if (DirectXUtils::floatEqual(s,h))
		_ftprintf( m_stream, _T("%s: OK   (%g)\n"), comment,s-h);
	else
		_ftprintf( m_stream, _T("%s: DIFF (%g) %.15g != %.15g\n"), comment,s-h,s,h);

	if (m_bSecure) fclose( m_stream );
}

void LOGFILE::createFile(_TCHAR *filename) {
	m_pcFilename = new _TCHAR[_tcslen(filename)+1];
	_tcscpy(m_pcFilename,filename);

	m_stream = _tfopen( m_pcFilename, _T("w") );

	if (m_bSecure) fclose( m_stream );
}

void LOGFILE::show() {
	if (!m_bSecure) fclose( m_stream );

	_TCHAR *s = new _TCHAR[_tcslen(m_pcFilename)+9];
	_stprintf(s,_T("notepad %s"),m_pcFilename);
	_tsystem( s );
	delete [] s;

	if (!m_bSecure) m_stream = _tfopen( m_pcFilename, _T("a") );
}

WPARAM KEYSHIFT::keys[3][2] = { {VK_RIGHT, VK_LEFT}, {VK_UP, VK_DOWN}, {VK_PRIOR, VK_NEXT}};

KEYSHIFT::KEYSHIFT(){
	m_vParam = D3DXVECTOR3(0,0,0);
	m_iDir	 = INTVECTOR3(0,1,2);
	m_fStep[0] = 0.01f; m_fStep[1] = 0.01f; m_fStep[2] = 0.01f;
	m_iSign[0] = 0;	m_iSign[1] = 0;	m_iSign[2] = 0;
	D3DXMatrixIdentity( &m_Mat );
}

KEYSHIFT::KEYSHIFT(D3DXMATRIX mat){
	m_iDir	 = INTVECTOR3(0,1,2);
	m_fStep[0] = 0.01f; m_fStep[1] = 0.01f; m_fStep[2] = 0.01f;
	m_iSign[0] = 0;	m_iSign[1] = 0;	m_iSign[2] = 0;

	SetMatrix(mat);
}

void KEYSHIFT::SetMatrix(D3DXMATRIX mat) {
	m_vParam = D3DXVECTOR3(mat._41,mat._42,mat._43);
	computeMatrix();
}

D3DXMATRIXA16 KEYSHIFT::getMatrix() {
	return m_Mat;
}

bool KEYSHIFT::MsgProc( HWND, UINT uMsg, WPARAM wParam, LPARAM, float step) {
	bool bHandled = false;

	if ( uMsg == WM_KEYDOWN) {
		bHandled = true;
		if (wParam == keys[m_iDir.x][m_iSign[0]])	m_vParam.x += step*m_fStep[0]; else
		if (wParam == keys[m_iDir.x][1-m_iSign[0]])	m_vParam.x -= step*m_fStep[0]; else
		if (wParam == keys[m_iDir.y][m_iSign[1]])	m_vParam.y += step*m_fStep[1]; else
		if (wParam == keys[m_iDir.y][1-m_iSign[1]])	m_vParam.y -= step*m_fStep[1]; else
		if (wParam == keys[m_iDir.z][m_iSign[2]])	m_vParam.z += step*m_fStep[2]; else
		if (wParam == keys[m_iDir.z][1-m_iSign[2]])	m_vParam.z -= step*m_fStep[2]; else bHandled = false;
	}
	
	if (bHandled) computeMatrix();
	return bHandled;
}

void KEYSHIFT::SetStep(float x, float y, float z) {
	m_fStep[0] = x;
	m_fStep[1] = y;
	m_fStep[2] = z;
}

void KEYSHIFT::SetDirSign(int x, int y, int z) {
	m_iSign[0] = x;
	m_iSign[1] = y;
	m_iSign[2] = z;
}

void KEYSHIFT::SetDirections(int x, int y, int z) {
	m_iDir.x = x;
	m_iDir.y = y;
	m_iDir.z = z;
}

void KEYSHIFT::computeMatrix() {
	D3DXMatrixTranslation(&m_Mat,m_vParam.x,m_vParam.y,m_vParam.z);
}

KEYSCALE::KEYSCALE(){
	m_vParam = D3DXVECTOR3(1,1,1);
	m_fStep[0] = 0.01f; m_fStep[1] = 0.01f; m_fStep[2] = 0.01f;
	m_iSign[0] = 0;	m_iSign[1] = 0;	m_iSign[2] = 0;
	D3DXMatrixIdentity( &m_Mat );
}

KEYSCALE::KEYSCALE(D3DXMATRIX mat){
	m_iDir	 = INTVECTOR3(0,1,2);
	m_fStep[0] = 0.01f; m_fStep[1] = 0.01f; m_fStep[2] = 0.01f;
	m_iSign[0] = 0;	m_iSign[1] = 0;	m_iSign[2] = 0;

	SetMatrix(mat);
}

void KEYSCALE::SetMatrix(D3DXMATRIX mat) {
	m_vParam = D3DXVECTOR3(mat._11,mat._22,mat._33);
	computeMatrix();
}

void KEYSCALE::computeMatrix() {
	D3DXMatrixScaling(&m_Mat,m_vParam.x,m_vParam.y,m_vParam.z);
}