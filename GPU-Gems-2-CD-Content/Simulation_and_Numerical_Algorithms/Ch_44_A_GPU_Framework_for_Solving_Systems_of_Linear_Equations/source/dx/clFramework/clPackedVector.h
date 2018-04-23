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
#pragma once



#include "clFragmentVector.h"

class clPackedVector : public clFragmentVector {
	public:
		clPackedVector(void)	{}
		clPackedVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, bool bConstant=false) {init(pd3dDevice, iSizeX/2, iSizeY/2, bConstant, FLOAT_TEX_RGBA);}
		clPackedVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant=false)			   {init(pd3dDevice, iSize/4, bConstant, FLOAT_TEX_RGBA);}
		virtual HRESULT init(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant, D3DFORMAT pD3DFormat);
		virtual HRESULT init(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, bool bConstant, D3DFORMAT pD3DFormat);
		~clPackedVector(void);

		virtual void multiplyAddVector(clPackedVector* vSource, clPackedVector* vTarget, float fScalar=1);
		virtual void multiplyAddVector(clPackedVector* vSource, clPackedVector* vTarget, float fScalar, signed int iShift, int iSubShift);
		virtual void multiplyVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScalar=1);
		virtual void multiplyVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScalar, signed int iShift, int iSubShift);

		virtual int getSize() {return m_memDesc.m_iWidth*m_memDesc.m_iHeight*4;}
		virtual int getSizeX() {return 2*m_memDesc.m_iWidth;}
		virtual int getSizeY() {return 2*m_memDesc.m_iHeight;}

		virtual PDIRECT3DTEXTURE9 unpack(clFragmentVector *vTarget);
		virtual PDIRECT3DTEXTURE9 repack(clFragmentVector *vSource);

		static void preOrder(int iSizeX, int iSizeY, int iCount, bool bConstant=false) {
			ms_memoryMananger->preOrderTextureTarget(clMemDescr(iSizeX/2, iSizeY/2, FLOAT_TEX_RGBA,(bConstant) ? NULL : D3DUSAGE_RENDERTARGET),iCount);
		}

	protected:
		virtual float evalLastPoint(LPDIRECT3DSURFACE9 pMiniTextureSurface);
		virtual void combineLastPoints(int iDepth, LPDIRECT3DSURFACE9 pMiniTextureSurface);

		LPD3DXRENDERTOSURFACE	m_pUnpackedRenderSurface;
		PDIRECT3DTEXTURE9		m_pUnpackedTexture;
		LPDIRECT3DSURFACE9		m_pUnpackedTextureSurface;
};
