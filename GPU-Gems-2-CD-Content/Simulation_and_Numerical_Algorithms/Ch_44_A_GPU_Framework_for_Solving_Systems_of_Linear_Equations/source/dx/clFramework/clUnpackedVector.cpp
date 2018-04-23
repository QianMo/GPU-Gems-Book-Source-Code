/************************************************************
 *															*
 * decr     : R encoded (non packed) vector class			*
 * version  : 1.1											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 10.06.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#include "clFragmentVector.h"
#include "clUnpackedVector.h"

/*******************************************************************************************
   Name:    combineLastPoints
   Purpose: short version of combineLastPointsGeneral, that decides what shader pass to use
            for this unpacked vector, this function is called by the clFragmentVector to
			reduce into a clFloat structure
********************************************************************************************/
void clUnpackedVector::combineLastPoints(int iDepth, LPDIRECT3DSURFACE9 pMiniTextureSurface) {
	combineLastPointsGeneral(iDepth, pMiniTextureSurface,0);
}

/*******************************************************************************************
   Name:    evalLastPoint
   Purpose: transfers the last point via GetRenderTargetData back to the main mem
********************************************************************************************/
float clUnpackedVector::evalLastPoint(LPDIRECT3DSURFACE9 pMiniTextureSurface) {
	// read data fron graphics board to main mem
	float*         pfTexture;
	D3DLOCKED_RECT rfTextureLock;
	LPDIRECT3DSURFACE9 pSurf;
	PDIRECT3DTEXTURE9  pTex;
	int id = ms_memoryMananger->getSysmemTexture(clMemDescr(1,1,FLOAT_TEX_R,NULL),pTex, pSurf);

	// transfer data from GPU to CPU
	m_pd3dDevice->GetRenderTargetData(pMiniTextureSurface, pSurf);

	// copy data into user pointer
	pTex->LockRect(0, &rfTextureLock, NULL, D3DLOCK_READONLY);	// lock whole region in the vector texture
		pfTexture = (float*)rfTextureLock.pBits;							    // start address of texture
 		float fResult = pfTexture[0];
    pTex->UnlockRect(0);

	ms_memoryMananger->releaseSysmemTexture(id);

	return fResult;
}


/*******************************************************************************************
   Name:    multiplyAddVector
   Purpose: computes a vector vector product while shifting the second vector:
	          vTarget[i] = vTarget[i]+(fScalar*this[i]*vSource[i+iShift])
********************************************************************************************/
void clUnpackedVector::multiplyAddVector(clUnpackedVector* vSource, clUnpackedVector* vTarget, float fScalar, signed int iShift) {

	// render to target's temp texture
	vTarget->BeginScene();
		ms_pShaderClVector->SetTexture(ms_tVector,    this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tVector2,   vSource->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tLastPass,  vTarget->m_pVectorTexture);

		if (iShift < 0)
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatAllCases);
		else
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatPosOnly);

        D3DXVECTOR4 temp(m_vSize.x, m_vSize.y, float(iShift), fScalar);
        ms_pShaderClVector->SetVector(ms_fSize,&temp);

		RenderViewPortCover(ms_pShaderClVector);
	vTarget->EndScene();
}


/*******************************************************************************************
   Name:    multiplyVector
   Purpose: computes a vector vector product while shifting the second vector:
	          vTarget[i] = fScalar*this[i]*vSource[i+iShift]
********************************************************************************************/
void clUnpackedVector::multiplyVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScalar, signed int iShift) {

	// render to target's temp texture
	vTarget->BeginScene();
		ms_pShaderClVector->SetTexture(ms_tVector,    this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tVector2,   vSource->m_pVectorTexture);

		if (iShift < 0)
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatAllCases_noadd);
		else
			ms_pShaderClVector->SetTechnique(ms_tVectorMultiplyMatPosOnly_noadd);

        D3DXVECTOR4 temp(m_vSize.x, m_vSize.y, float(iShift), fScalar);
        ms_pShaderClVector->SetVector(ms_fSize,&temp);

		RenderViewPortCover(ms_pShaderClVector);
	vTarget->EndScene();
}
