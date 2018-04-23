/************************************************************
 *															*
 * decr     : vertex based matrix class						*
 * version  : 1.01											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 06.10.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#pragma once

#include "clAbstractMatrix.h"
#include "clPackedVector.h"
#include "clUnpackedVector.h"

struct CLVECMATRIXVERTEX {
    FLOAT      x,y,z;					// position    (encodes row)
	FLOAT      tu_0, tv_0;				// tex-coords0 (encodes column)
	FLOAT      tu_1, tv_1;				// tex-coords1 (encodes column)
	FLOAT      tu_2, tv_2;				// tex-coords2 (encodes column)
	FLOAT      tu_3, tv_3;				// tex-coords3 (encodes column)
	FLOAT      val0, val1, val2, val3;	// tex-coords4 (encodes values)
	FLOAT      posX, posY;				// tex-coords5 (encodes row in tex coords)
	static const DWORD FVF;
};

struct clVertexMatrixElement {
	int iX, iY;
	float fVal;
};

class clVertexMatrix : public clAbstractMatrix {
	public:
		clVertexMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize);
		clVertexMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY);
		virtual ~clVertexMatrix(void);

		virtual int setData(clVertexMatrixElement *meData, int iElemCount);
		virtual void setDataSorted(clVertexMatrixElement *meData, int iElemCount, int iVertexBufferCount);

		virtual void matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);
		virtual void matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);

		virtual void clPreparePackedUse();

		// public for debuf only
		LPDIRECT3DVERTEXBUFFER9 *m_ppVertexBuffers;
		int *m_iElemCounter;
		int m_iVertexBufferCount;
		static clUnpackedVector *ms_pUnpackedVectorRes;
		static clUnpackedVector *ms_pUnpackedVectorX;

	protected:
		clVertexMatrix() {} // block default constructor
		
		static LPD3DXEFFECT ms_pShaderVMatrix;
		static int ms_iVertexMatrixCount;

		static D3DXHANDLE ms_tVector;
		static D3DXHANDLE ms_tLastPass;
		static D3DXHANDLE ms_tMatrixMultiplyNoLast;
		static D3DXHANDLE ms_tMatrixMultiply;


		virtual void init(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY);
		virtual void deleteVertexBuffers();
		virtual int sortData(clVertexMatrixElement *meData, int iElemCount);
		virtual void insertElement(clVertexMatrixElement &meData, CLVECMATRIXVERTEX &vbData, int iRGBAIndex);
		virtual void completeLastRow(CLVECMATRIXVERTEX &vbData, int iRGBAIndex);
		virtual void coordsTo2D(int i1DCoord, float &f2Dx, float &f2Dy);
		virtual void coordsToPos(int i1DCoord, float &f2Dx, float &f2Dy);
		virtual void loadShaders();

};
