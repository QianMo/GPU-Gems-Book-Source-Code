/************************************************************
 *															*
 * decr     : RGBA ecnoded vector class						*
 * version  : 1.2											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 10.06.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#include "clFragmentVector.h"

#include "clPackedVector.h"

/*******************************************************************************************
   Name:    init
   Purpose: init for a vector of length iSize, automaticaly finds the best layout
********************************************************************************************/
HRESULT clPackedVector::init(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant, D3DFORMAT pD3DFormat) {
	m_pUnpackedTexture = NULL;
	return clFragmentVector::init(pd3dDevice, iSize, bConstant, pD3DFormat);
}


/*******************************************************************************************
   Name:    init
   Purpose: initializes this vector, should be called explitly only if default constructor
            has been used to create this object
********************************************************************************************/
HRESULT clPackedVector::init(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, bool bConstant, D3DFORMAT pD3DFormat) {
	m_pUnpackedTexture = NULL;
	return clFragmentVector::init(pd3dDevice, iSizeX, iSizeY, bConstant, pD3DFormat);
}


/*******************************************************************************************
   Name:    default destructor
   Purpose: destroyes this object
********************************************************************************************/
clPackedVector::~clPackedVector(void)	{

	// parent constructor called implicitly

	// delete memenbers needed to unpack this vector
	if (m_pUnpackedTexture != NULL) {
		SAFE_RELEASE( m_pUnpackedRenderSurface );
		SAFE_RELEASE( m_pUnpackedTexture );
		SAFE_RELEASE( m_pUnpackedTextureSurface );
	}
}

/*******************************************************************************************
   Name:    combineLastPoints
   Purpose: short version of combineLastPointsGeneral, that decides what shader pass to use
            for this packed vector, this function is called by the clFragmentVector to reduce
			into a clFloat structure
********************************************************************************************/
void clPackedVector::combineLastPoints(int iDepth, LPDIRECT3DSURFACE9 pMiniTextureSurface) {
	combineLastPointsGeneral(iDepth, pMiniTextureSurface,1);
}

/*******************************************************************************************
   Name:    evalLastPoint
   Purpose: transfers the last texel via GetRenderTargetData back to the main mem and 
            combines it to a single float value
********************************************************************************************/
float clPackedVector::evalLastPoint(LPDIRECT3DSURFACE9 pMiniTextureSurface) {
	// read data fron graphics board to main mem
	float*         pfTexture;
	D3DLOCKED_RECT rfTextureLock;
	LPDIRECT3DSURFACE9 pSurf;
	PDIRECT3DTEXTURE9  pTex;
	int id = ms_memoryMananger->getSysmemTexture(clMemDescr(1,1,FLOAT_TEX_RGBA,NULL),pTex, pSurf);

	// transfer data from GPU to CPU
	m_pd3dDevice->GetRenderTargetData(pMiniTextureSurface, pSurf);

	// copy data into user pointer
	pTex->LockRect(0, &rfTextureLock, NULL, D3DLOCK_READONLY);	// lock whole region in the vector texture
		pfTexture = (float*)rfTextureLock.pBits;								// start address of texture
 		
		float fResult = pfTexture[0]+pfTexture[1]+pfTexture[2]+pfTexture[3];
    pTex->UnlockRect(0);

	ms_memoryMananger->releaseSysmemTexture(id);

	return fResult;
}


/*******************************************************************************************
   Name:    unpack
   Purpose: convert the packed vector into an unpacked representation
********************************************************************************************/
PDIRECT3DTEXTURE9 clPackedVector::unpack(clFragmentVector *vTarget) {

	vTarget->BeginScene();

		ms_pShaderClVector->SetVector(ms_fSize,&m_vSize);
		ms_pShaderClVector->SetTexture(ms_tVector, m_pVectorTexture);
		ms_pShaderClVector->SetTechnique(ms_tUnpackVector);

		RenderViewPortCover(ms_pShaderClVector);
	vTarget->EndScene();

	return m_pUnpackedTexture;
}

/*******************************************************************************************
   Name:    repack
   Purpose: read the data of the unpacked vector vTarget into this packed vector
********************************************************************************************/
PDIRECT3DTEXTURE9 clPackedVector::repack(clFragmentVector *vSource) {

	// REMARK: using different constant definitions in this function
	// fShift  = packed size in X
	// fSize.x = unpacked size in X
	// fSize.y = unpacked size in Y
	// fSize.b = half the packed size in X
	// fSize.a = 1/(unpacked size in X)


	D3DXVECTOR4 vConstants = D3DXVECTOR4((float)m_memDesc.m_iWidth*2.0f,
										 (float)m_memDesc.m_iHeight*2.0f,
										 (float)m_memDesc.m_iWidth/2.0f,
										 0.5f/(float)m_memDesc.m_iWidth);


	m_pd3dDevice->SetRenderTarget(0,m_pVectorTextureSurface);
		
		ms_pShaderClVector->SetTechnique(ms_tPackVector);

		ms_pShaderClVector->SetVector(ms_fSize,&vConstants);
		ms_pShaderClVector->SetFloat(ms_fShift,(float)m_memDesc.m_iWidth);
		ms_pShaderClVector->SetTexture(ms_tVector, vSource->m_pVectorTexture);

		RenderViewPortCover(ms_pShaderClVector);
	m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer);

	return m_pUnpackedTexture;
}

/*******************************************************************************************
   Name:    multiplyVector
   Purpose: computes a vector vector product while NOT shifting the second vector:
	          vTarget[i] = fScalar*this[i]*vSource[i]
********************************************************************************************/
void clPackedVector::multiplyVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScalar) {
	multiplyVector(vSource, vTarget, fScalar,0,0);
}


/*******************************************************************************************
   Name:    multiplyVector
   Purpose: computes a vector vector product while shifting the second vector:
	          vTarget[i] = fScalar*this[i]*vSource[i+iShift*4+iSubShift]
********************************************************************************************/
void clPackedVector::multiplyVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScalar, signed int iShift, int iSubShift) {

	// render to target's temp texture
	vTarget->BeginScene();
		if (iShift < 0) 
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatAllCases_noadd);
		else
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatPosOnly_noadd);
	
		ms_pShaderClVector->SetTexture(ms_tVector,    this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tVector2,   vSource->m_pVectorTexture);
        D3DXVECTOR4 temp(m_vSize.x, m_vSize.y, float(iShift), fScalar);
        ms_pShaderClVector->SetVector(ms_fSize,&temp);

		RenderViewPortCover(ms_pShaderClVector,iSubShift);
	vTarget->EndScene();
}

/*******************************************************************************************
   Name:    multiplyAddVector
   Purpose: computes a vector vector product while NOT shifting the second vector:
	          vTarget[i] = vTarget[i]+(fScalar*this[i]*vSource[i])
********************************************************************************************/
void clPackedVector::multiplyAddVector(clPackedVector* vSource, clPackedVector* vTarget, float fScalar) {
	multiplyAddVector(vSource, vTarget, fScalar, 0, 0);
}

/*******************************************************************************************
   Name:    multiplyAddVector
   Purpose: computes a vector vector product while shifting the second vector:
	          vTarget[i] = vTarget[i]+(fScalar*this[i]*vSource[i+iShift*4+iSubShift])
********************************************************************************************/
void clPackedVector::multiplyAddVector(clPackedVector* vSource, clPackedVector* vTarget, float fScalar, signed int iShift, int iSubShift) {

	// render to target's temp texture
	vTarget->BeginScene();
		if (iShift < 0)
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatAllCases);
		else 
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatPosOnly);
	
		ms_pShaderClVector->SetTexture(ms_tVector,    this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tVector2,   vSource->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tLastPass,  vTarget->m_pVectorTexture);
		
        D3DXVECTOR4 temp(m_vSize.x, m_vSize.y, float(iShift), fScalar);
        ms_pShaderClVector->SetVector(ms_fSize,&temp);

		RenderViewPortCover(ms_pShaderClVector,iSubShift);
	vTarget->EndScene();
}
