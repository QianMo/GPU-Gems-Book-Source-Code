/************************************************************
 *															*
 * decr     : A single float value stored on the GPU		*
 * version  : 1.10											*
 * author   : Jens Krüger									*
 * date     : 08.02.2004									*
 * modified	: 10.06.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#include "clfloat.h"

int						clFloat::ms_iClFloatCount=0;
LPD3DXEFFECT			clFloat::ms_pShaderClFloat=NULL;
LPDIRECT3DVERTEXBUFFER9	clFloat::ms_pCoverPoint=NULL;

D3DXHANDLE				clFloat::ms_fScalar;
D3DXHANDLE				clFloat::ms_tFloatA;
D3DXHANDLE				clFloat::ms_tFloatB;
D3DXHANDLE				clFloat::ms_tMulClFloat;
D3DXHANDLE				clFloat::ms_tAddClFloat;
D3DXHANDLE				clFloat::ms_tDivClFloat;
D3DXHANDLE				clFloat::ms_tDivZClFloat;
D3DXHANDLE				clFloat::ms_tInvert;
D3DXHANDLE				clFloat::ms_tAddScalar;
D3DXHANDLE				clFloat::ms_tMultilyScalar;

const DWORD CLFLOATVERTEX::FVF = D3DFVF_XYZ;

clFloat::clFloat()
{
}

clFloat::clFloat(LPDIRECT3DDEVICE9 pd3dDevice)
{
	init(pd3dDevice);
}

clFloat::clFloat(LPDIRECT3DDEVICE9 pd3dDevice, float fValue) 
{
	init(pd3dDevice);
	setData(fValue);
}

/*******************************************************************************************
   Name:    loadShaders
   Purpose: loads all shaders required for this clfloat, if loaded already use the old one
********************************************************************************************/
void clFloat::loadShaders() {
	if (ms_iClFloatCount < 1) {
		if (FAILED(DirectXUtils::checkLoadShader(_T("clFloat.fx"), ms_pShaderClFloat, m_pd3dDevice, clFrameworkShaderPath, NULL, clClass::ms_clPSProfile))) exit(-1);
		// prefetch shader techniques & parameters 

		// parameters 
		ms_fScalar			= ms_pShaderClFloat->GetParameterByName(NULL,"fScalar");
		ms_tFloatA			= ms_pShaderClFloat->GetParameterByName(NULL,"tFloatA");
		ms_tFloatB			= ms_pShaderClFloat->GetParameterByName(NULL,"tFloatB");

		// techniques
		ms_tMulClFloat		= ms_pShaderClFloat->GetTechniqueByName("tMulClFloat");
		ms_tAddClFloat		= ms_pShaderClFloat->GetTechniqueByName("tAddClFloat");
		ms_tDivClFloat		= ms_pShaderClFloat->GetTechniqueByName("tDivClFloat");
		ms_tDivZClFloat		= ms_pShaderClFloat->GetTechniqueByName("tDivZClFloat");
		ms_tInvert			= ms_pShaderClFloat->GetTechniqueByName("tInvert");
		ms_tAddScalar		= ms_pShaderClFloat->GetTechniqueByName("tAddScalar");
		ms_tMultilyScalar	= ms_pShaderClFloat->GetTechniqueByName("tMultilyScalar");
	}
}

void clFloat::init(LPDIRECT3DDEVICE9 pd3dDevice) 
{
	m_pd3dDevice = pd3dDevice;
	pd3dDevice->GetRenderTarget(0, &m_lpBackBuffer);


	// init data for temp render-to texture
	m_TempID = -1;
	m_pFloatTexture = NULL;
	m_pFloatTextureSurface = NULL;

	// create a 1x1 texture and surface
	m_iMemID = ms_memoryMananger->getTextureTarget(clMemDescr(1,1,FLOAT_TEX_R,D3DUSAGE_RENDERTARGET),m_pFloatTexture,m_pFloatTextureSurface);

	// init static variables
	loadShaders();
	if (ms_iClFloatCount < 1)	{

		// Create a one-point vertex buffer
		CLFLOATVERTEX hCoverPoint[]= {{0, 0, 0}};
		m_pd3dDevice->CreateVertexBuffer(sizeof(hCoverPoint),D3DUSAGE_WRITEONLY,CLFLOATVERTEX::FVF,D3DPOOL_DEFAULT,&ms_pCoverPoint, NULL);
		VOID* pVertices;
		ms_pCoverPoint->Lock( 0, 0, (void**)&pVertices, NULL );
			memcpy( pVertices, hCoverPoint, sizeof(hCoverPoint) );
		ms_pCoverPoint->Unlock();
	}

	// count the active incarnations of this class
	ms_iClFloatCount++;
}

clFloat::~clFloat(void)
{
	// if this is the last clFloat object alive, destroy static members
	ms_iClFloatCount--;
	if (ms_iClFloatCount==0) 	{
		SAFE_RELEASE( ms_pShaderClFloat);
		SAFE_RELEASE( ms_pCoverPoint );
	}

	// destroy non static members
	ms_memoryMananger->releaseTextureTarget(m_iMemID,m_pFloatTexture,m_pFloatTextureSurface);
	SAFE_RELEASE(m_lpBackBuffer);
}


void clFloat::setData(float fVectorData)
{
    D3DLOCKED_RECT rfTextureLock;
	LPDIRECT3DSURFACE9 pSurf;
	PDIRECT3DTEXTURE9  pTex;
	int id = ms_memoryMananger->getSysmemTexture(clMemDescr(1,1,FLOAT_TEX_R,NULL),pTex, pSurf);

    pTex->LockRect(0, &rfTextureLock, NULL, 0);
		#ifdef use16BitPrecOnly
			D3DXFLOAT16 temp(fVectorData);
			((D3DXFLOAT16*)rfTextureLock.pBits)[0] = temp;
		#else
			((float*)rfTextureLock.pBits)[0] = fVectorData;
		#endif
    pTex->UnlockRect(0);

	m_pd3dDevice->UpdateTexture(pTex, m_pFloatTexture);

	ms_memoryMananger->releaseSysmemTexture(id);
}

void clFloat::getData(float *fVectorData)
{
	D3DLOCKED_RECT rfTextureLock;
	LPDIRECT3DSURFACE9 pSurf;
	PDIRECT3DTEXTURE9  pTex;
	int id = ms_memoryMananger->getSysmemTexture(clMemDescr(1,1,FLOAT_TEX_R,NULL),pTex, pSurf);

	m_pd3dDevice->GetRenderTargetData(m_pFloatTextureSurface, pSurf); 	// transfer data from GPU to CPU

	// copy data into user pointer
	pTex->LockRect(0, &rfTextureLock, NULL, D3DLOCK_READONLY);
		#ifdef use16BitPrecOnly
			D3DXFLOAT16 temp = ((D3DXFLOAT16*)rfTextureLock.pBits)[0];
			(*fVectorData) = (FLOAT)temp;
		#else
			(*fVectorData) = ((float*)rfTextureLock.pBits)[0];
		#endif
	pTex->UnlockRect(0);

	ms_memoryMananger->releaseSysmemTexture(id);
}

float clFloat::getData() {
	float f;
	getData(&f);
	return f;
}


/*******************************************************************************************
   Name:    add
   Purpose: compute this = a*this+other*b
********************************************************************************************/
void clFloat::add(clFloat *other,clFloat *target,float a, float b) {	
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(a,b,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, m_pFloatTexture);
		ms_pShaderClFloat->SetTexture(ms_tFloatB, other->m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique("tAddClFloat");

		renderPoint();
	target->EndScene();
}

/*******************************************************************************************
   Name:    mul
   Purpose: compute this = a*this*other
********************************************************************************************/
void clFloat::mul(clFloat *other,clFloat *target, float a) {
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(a,0,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, m_pFloatTexture);
		ms_pShaderClFloat->SetTexture(ms_tFloatB, other->m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique("tMulClFloat");
		renderPoint();
	target->EndScene();
}

/*******************************************************************************************
   Name:    div
   Purpose: compute this = (a*this)/(other*b)
********************************************************************************************/
void clFloat::div(clFloat *other,clFloat *target, float a, float b) {
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(a,b,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, this->m_pFloatTexture);
		ms_pShaderClFloat->SetTexture(ms_tFloatB, other->m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique("tDivClFloat");
		renderPoint();
	target->EndScene();
}

/*******************************************************************************************
   Name:    divZ
   Purpose: compute this = (b == 0) ? 0 : (a*this)/(other*b) 
********************************************************************************************/
void clFloat::divZ(clFloat *other, clFloat *target, float a, float b) {
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(a,b,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, this->m_pFloatTexture);
		ms_pShaderClFloat->SetTexture(ms_tFloatB, other->m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique(ms_tDivZClFloat);
		renderPoint();
	target->EndScene();
}

/*******************************************************************************************
   Name:    invert
   Purpose: compute this = a/(this*b)
********************************************************************************************/
void clFloat::invert(clFloat *target, float a, float b) {
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(a,b,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique("tInvert");
		renderPoint();
	target->EndScene();
}

/*******************************************************************************************
   Name:    add
   Purpose: compute this = this+scalar
********************************************************************************************/
void clFloat::add(float scalar,clFloat *target) {
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(scalar,0,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique("tAddScalar");
		renderPoint();
	target->EndScene();
}

/*******************************************************************************************
   Name:    mul
   Purpose: compute this = this*scalar
********************************************************************************************/
void clFloat::mul(float scalar,clFloat *target) {
	target->BeginScene();
		D3DXVECTOR4 f4Scalar = D3DXVECTOR4(scalar,0,0,0);
		ms_pShaderClFloat->SetVector(ms_fScalar,&f4Scalar);
		ms_pShaderClFloat->SetTexture(ms_tFloatA, m_pFloatTexture);
		ms_pShaderClFloat->SetTechnique("tMultilyScalar");

		renderPoint();
	target->EndScene();
}

void clFloat::renderPoint() {
	UINT cPasses;
	ms_pShaderClFloat->Begin(&cPasses, 0);
		ms_pShaderClFloat->BeginPass(0);
		m_pd3dDevice->SetStreamSource( 0, ms_pCoverPoint, 0, sizeof(CLFLOATVERTEX) );
		m_pd3dDevice->SetFVF( CLFLOATVERTEX::FVF );
		m_pd3dDevice->DrawPrimitive(D3DPT_POINTLIST,0,1);
		ms_pShaderClFloat->EndPass();
	ms_pShaderClFloat->End();
}

char* clFloat::toString()
{
  float f;
  getData(&f);
  char* cpTemp   = new char[20];
  sprintf(cpTemp,"%g",f);
  return cpTemp;
}


/*******************************************************************************************
   Name:    BeginScene
   Purpose: initialazes the scene for rendering i.e. allocating a render to surace,
            retrieving a write surface, calling begin scene on the render to surface
********************************************************************************************/
HRESULT clFloat::BeginScene() {
	HRESULT hr;
	CHECK_HR(m_pd3dDevice->SetRenderTarget(0,getWriteSurface()));
	return S_OK;
}

/*******************************************************************************************
   Name:    EndScene
   Purpose: finalizes a rendering pass i.e. calling end-scene on the render to surface
            freeing the render to surace, swapping the wrte surface and the read surface
********************************************************************************************/
HRESULT clFloat::EndScene() {
	HRESULT hr;
	CHECK_HR(m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer));
	swapSurfaces();
	return S_OK;
}

LPDIRECT3DSURFACE9 clFloat::getWriteSurface() {
	m_TempID = ms_memoryMananger->getTextureTarget(clMemDescr(1,1,FLOAT_TEX_R,D3DUSAGE_RENDERTARGET),m_pTempTexture,m_pTempTextureSurface);
	return m_pTempTextureSurface;
}

void clFloat::swapSurfaces() {
	// swap temp and vector surface and texture
	PDIRECT3DTEXTURE9	swapTex		= m_pFloatTexture;
	LPDIRECT3DSURFACE9	swapSurface = m_pFloatTextureSurface;
	m_pFloatTexture					= m_pTempTexture;
	m_pFloatTextureSurface			= m_pTempTextureSurface;
	ms_memoryMananger->releaseTextureTarget(m_TempID,swapTex,swapSurface);
	m_TempID = -1;
}