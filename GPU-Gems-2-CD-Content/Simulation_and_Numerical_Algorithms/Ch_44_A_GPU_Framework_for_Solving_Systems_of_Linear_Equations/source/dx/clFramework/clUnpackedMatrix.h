/************************************************************
 *															*
 * decr     : R-only encoded (non packed) matrix class		*
 * version  : 1.01											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 06.10.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#pragma once



#include "clFragmentMatrix.h"
#include "clUnpackedVector.h"

class clUnpackedMatrix : public clFragmentMatrix {
public:
	clUnpackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize);
	clUnpackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY);
	clUnpackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, int *piRowIndices, int iRowCount);
	virtual ~clUnpackedMatrix();

	virtual void matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);
	virtual void matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);
	virtual clUnpackedVector* getRow(int iIndex);
	virtual HRESULT createRowVectors(int* piRowIndices, int iRowCount);
	virtual void deleteRowVectors();

	static void preOrder(int iSizeX, int iSizeY, int iDiags, int iCount) {
		clUnpackedVector::preOrder(iSizeX, iSizeY,iDiags*iCount,true);
	}

protected:
	clUnpackedMatrix(void) {} // block default constructor

	clUnpackedVector *m_clvRows;
};
