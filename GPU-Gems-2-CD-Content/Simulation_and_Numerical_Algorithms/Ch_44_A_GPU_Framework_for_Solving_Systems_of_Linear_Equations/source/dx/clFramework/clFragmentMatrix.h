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

#pragma once



#include "clAbstractMatrix.h"

class clFragmentMatrix : public clAbstractMatrix {
public:
	virtual void matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult);
	virtual void matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) = NULL;


protected:
	virtual int findArrayIndex(int iIndex);
};
