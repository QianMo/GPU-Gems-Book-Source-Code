/************************************************************
 *															*
 * decr     : RGBA packed matrix class						*
 * version  : 1.01											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 06.10.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#pragma once



#include "clPackedVector.h"
#include "clFragmentMatrix.h"

class clPackedMatrix : public clFragmentMatrix {
public:
	clPackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize);	
	clPackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY);	
	clPackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, int *piRowIndices, int iRowCount);
	virtual ~clPackedMatrix();

	virtual bool setRow(clPackedVector *clvRow, int iIndex);
	virtual clPackedVector* getRow(int iIndex);
	virtual void matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);
	virtual void matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);

	virtual HRESULT createRowVectors(int *piRowIndices, int iRowCount);
	virtual void deleteRowVectors();

	static void preOrder(int iSizeX, int iSizeY, int iDiags, int iCount) {
		clPackedVector::preOrder(iSizeX, iSizeY,iDiags*iCount,true);
	}


protected:
	clPackedMatrix(void) {} // block default constructor
	virtual int findArrayIndex(int iIndex);

	clPackedVector *m_clvRows;

};
