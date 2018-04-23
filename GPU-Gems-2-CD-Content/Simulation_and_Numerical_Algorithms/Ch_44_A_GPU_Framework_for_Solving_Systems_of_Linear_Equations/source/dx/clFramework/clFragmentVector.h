/************************************************************
 *															*
 * decr     : abstract base class for all fragment based	*
 *            vector clases									*
 * version  : 1.2											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 10.06.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#pragma once

#include "clvector.h"
#include "clFloat.h"
#include "clMemMan.h"

class clFragmentVector : public clVector {
	public:
		clFragmentVector(void){}
		virtual ~clFragmentVector(void);
		virtual HRESULT init(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, bool bConstant, D3DFORMAT pD3DFormat);
		virtual HRESULT init(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant, D3DFORMAT pD3DFormat);

		virtual int getSize() = NULL;
		virtual int getSizeX() = NULL;
		virtual int getSizeY() = NULL;
		virtual clMemDescr getDecription() {return m_memDesc;}

		virtual void getData(float* fVectorData);
		virtual void setData(float* fVectorData);
		virtual HRESULT setData(LPDIRECT3DSURFACE9 m_pSurfSystem);
		virtual HRESULT getData(LPDIRECT3DSURFACE9 m_pSurfSystem);

		virtual void clear();
		virtual float reduceAdd();
		virtual float reduceAdd(clFragmentVector* clvSecond);
		virtual void reduceAdd(clFloat *clfResult);
		virtual void reduceAdd(clFragmentVector* clvSecond,clFloat *clfResult);

		virtual void multiplyScalar(float fScalar);
		virtual void copyVector(clFragmentVector *clvSource);
		virtual void addVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScal1=1, float fScal2=1);
		virtual void subtractVector(clFragmentVector* clvSource, clFragmentVector* clvTarget, float fScal1=1, float fScal2=1) {addVector(clvSource, clvTarget, fScal1, -fScal2);};
		virtual void vectorOp(CL_enum eOpType, clFragmentVector *clvSource, clFragmentVector *clvTarget, float fScal1=1, float fScal2=1);

		virtual void addVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScal, clFloat *clfScal);
		virtual void subtractVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScal, clFloat *clfScal);

		void freeRenderSurface();
		static LPD3DXEFFECT		ms_pShaderClVector;

		virtual void swapSurfaces();
		virtual LPDIRECT3DSURFACE9 getWriteSurface();
		virtual LPDIRECT3DSURFACE9 getReadSurface();
		PDIRECT3DTEXTURE9		m_pVectorTexture;

		virtual TCHAR* toString(int iOutputCount);
		virtual TCHAR* toShortString(int iOutputCount);

		HRESULT BeginScene();
		HRESULT EndScene();

	protected:
		static int				s_iFragmentVectorCount;
		int						m_iTempActive;
		clMemDescr				m_memDesc;
		int						m_iMemID;

		D3DXVECTOR4				m_vSize;

		static D3DXHANDLE				ms_fMultiply;
		static D3DXHANDLE				ms_fMultiply2;
		static D3DXHANDLE				ms_fReduceStep;
		static D3DXHANDLE				ms_f4ReduceStep;
		static D3DXHANDLE				ms_f4TexShift;
		static D3DXHANDLE				ms_fShift;
		static D3DXHANDLE				ms_fSize;
		static D3DXHANDLE				ms_tVector;
		static D3DXHANDLE				ms_tVector2;
		static D3DXHANDLE				ms_tLastPass;
		static D3DXHANDLE				ms_tMultiply;

		static D3DXHANDLE				ms_tMultiplyScal;
		static D3DXHANDLE				ms_tReduceAddFirst;
		static D3DXHANDLE				ms_tReduceAddRest;
		static D3DXHANDLE				ms_tReduceAddLast;
		static D3DXHANDLE				ms_tReduceAddRestX;
		static D3DXHANDLE				ms_tReduceAddRestY;
		static D3DXHANDLE				ms_tVectorAdd;
		static D3DXHANDLE				ms_tVectorMultiply;
		static D3DXHANDLE				ms_tVectorMultiplyMatAllCases;
		static D3DXHANDLE				ms_tVectorMultiplyMatPosOnly;
		static D3DXHANDLE				ms_tVectorMultiplyMatAllCases_noadd;
		static D3DXHANDLE				ms_tVectorMultiplyMatPosOnly_noadd;
		static D3DXHANDLE				ms_tUnpackVector;
		static D3DXHANDLE				ms_tPackVector;

		int						m_ID[2];
		PDIRECT3DTEXTURE9		m_pTempTexture[2];
		LPDIRECT3DSURFACE9		m_pTempTextureSurface[2];

		LPDIRECT3DSURFACE9		m_pVectorTextureSurface;

		virtual void loadShaders();

		virtual void splitQuadX(CL_TEX2D_VERTEX *hQuad);
		virtual void splitQuadY(CL_TEX2D_VERTEX *hQuad);
		virtual void splitQuad(CL_TEX2D_VERTEX *hQuad);
		virtual int reduceAddInternal();
		virtual void reduceInXDirection(int &iDepth, CL_TEX2D_VERTEX *hQuad);
		virtual void reduceInYDirection(int &iDepth, CL_TEX2D_VERTEX *hQuad);
		virtual void combineLastPointsGeneral(int iDepth, LPDIRECT3DSURFACE9 pMiniTextureSurface, int iType);
		virtual void combineLastPoints(int iDepth, LPDIRECT3DSURFACE9 pMiniTextureSurface) = NULL;
		virtual float evalLastPoint(LPDIRECT3DSURFACE9 pMiniTextureSurface) = NULL;
};
