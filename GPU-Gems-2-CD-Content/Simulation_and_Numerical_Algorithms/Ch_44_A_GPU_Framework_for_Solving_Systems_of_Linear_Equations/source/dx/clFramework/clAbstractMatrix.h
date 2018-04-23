/************************************************************
 *															*
 * decr     : abstract base class for all clMatrix classes	*
 * version  : 1.11											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 06.10.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#pragma once



#include "clClass.h"
#include "clFragmentVector.h"

class clAbstractMatrix : public clClass {
	public:
		clAbstractMatrix(LPDIRECT3DDEVICE9, int) {};	

		virtual void matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) = NULL;
		virtual void matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) = NULL;

		virtual int getSizeX() {return m_iSizeX;}
		virtual int getSizeY() {return m_iSizeY;}
	protected:
		int	m_iSizeX,m_iSizeY;
		int m_iRowCount;
		int *m_piRowIndices;

		clAbstractMatrix(void){}  // block default constructor

};
