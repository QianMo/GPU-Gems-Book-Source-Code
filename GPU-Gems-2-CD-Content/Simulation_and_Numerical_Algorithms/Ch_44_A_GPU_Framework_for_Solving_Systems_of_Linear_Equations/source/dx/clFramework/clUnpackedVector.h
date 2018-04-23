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

#pragma once

#include "clFragmentVector.h"

class clUnpackedVector : public clFragmentVector {

public:
	clUnpackedVector(void) {}
	clUnpackedVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, bool bConstant=false) {init(pd3dDevice, iSizeX, iSizeY, bConstant, FLOAT_TEX_R);}
	clUnpackedVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant=false) {init(pd3dDevice, iSize, bConstant, FLOAT_TEX_R);}

	virtual ~clUnpackedVector(void){}

	virtual int getSize() {return m_memDesc.m_iWidth*m_memDesc.m_iHeight;}
	virtual int getSizeX() {return m_memDesc.m_iWidth;}
	virtual int getSizeY() {return m_memDesc.m_iHeight;}

	virtual void multiplyAddVector(clUnpackedVector* vSource, clUnpackedVector* vTarget, float fScalar=1, signed int iShift=0);
	virtual void multiplyVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScalar=1, signed int iShift=0);

	static void preOrder(int iSizeX, int iSizeY, int iCount, bool bConstant=false) {
		ms_memoryMananger->preOrderTextureTarget(clMemDescr(iSizeX, iSizeY, FLOAT_TEX_R,(bConstant) ? NULL : D3DUSAGE_RENDERTARGET),iCount);
	}

protected:
	virtual float evalLastPoint(LPDIRECT3DSURFACE9 pMiniTextureSurface);
	virtual void combineLastPoints(int iDepth, LPDIRECT3DSURFACE9 pMiniTextureSurface);
};
