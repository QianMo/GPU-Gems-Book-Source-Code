/************************************************************
 *															*
 * decr     : abstract base class for all fragment based	*
 *            matrix clases									*
 * version  : 1.01											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 06.10.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#include "clAbstractMatrix.h"
#include "clFragmentMatrix.h"

int clFragmentMatrix::findArrayIndex(int iIndex) {
	for (int i=0;i<m_iRowCount;i++) 
		if (m_piRowIndices[i] == iIndex) return i;
	return -1; // element not found
}

/*******************************************************************************************
   Name:    matrixVectorOp 
   Purpose: computes matrix vector operations of the kind:
	          clvResult = this*clvX OP clvY
	          where OP (eOpType) can be one of the following: CL_NULL,CL_ADD,CL_SUB,CL_MULT
********************************************************************************************/
void clFragmentMatrix::matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) {
	clvResult->clear();
	matrixVectorOpAdd(eOpType, clvX, clvY, clvResult);
}
