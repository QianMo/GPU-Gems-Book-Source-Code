/************************************************************
 *															*
 * decr     : conjugate gradient solver class				*
 * version  : 1.1											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 30.09.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#pragma once

#define CL_PACKED   0
#define CL_UNPACKED 1
#define CL_VERTEX   2

#include "clUnpackedVector.h"
#include "clUnpackedMatrix.h"
#include "clPackedVector.h"
#include "clPackedMatrix.h"

class clCGSolver : public clClass {
public:
	clCGSolver(void) {m_pd3dDevice = NULL; m_iSizeX = NULL; m_iSizeY = NULL;}
	clCGSolver(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, int iRepresentation);
	clCGSolver(clAbstractMatrix *clM, clFragmentVector *clvX, clFragmentVector *clvB, int iRepresentation);
	clCGSolver(clAbstractMatrix *clM, clFragmentVector *clvX, clFragmentVector *clvB, clFragmentVector *clvP, clFragmentVector *clvQ, clFragmentVector *clvR);
	~clCGSolver(void);

	clFragmentVector* getResult() {	return m_clvX;}
	void setResult(clFragmentVector* clvResult);

	clFragmentVector* getRHS() {return m_clvB;}
	void setRHS(clFragmentVector* clvRHS);

	void getTemp(clFragmentVector** clvP, clFragmentVector** clvQ, clFragmentVector** clvR);
	void setTemp(clFragmentVector* clvP, clFragmentVector* clvQ, clFragmentVector* clvR);

	clAbstractMatrix* getMatrix() {return m_clMatrix;}
	void setMatrix(clAbstractMatrix* clM);

	int solve(float rhoTresh=0.001, int iter=1);
	float solveInit();
	float solveIteration(float rho);

	int solveNT(int iter=1);
	void solveInitNT();
	void solveIterationNT();

	static void preOrder(int iSizeX, int iSizeY, int iRepresentation, int iCount, bool bSupplyAllVectors=false) {

		if (!bSupplyAllVectors) {
			if (iRepresentation == CL_PACKED) clPackedVector::preOrder(iSizeX, iSizeY, iCount*3); else
			if (iRepresentation == CL_UNPACKED) clUnpackedVector::preOrder(iSizeX, iSizeY, iCount*3);
		}

		clFloat::preOrder(5);
	}
	
protected:	
	clAbstractMatrix *m_clMatrix;
	clFragmentVector *m_clvX, *m_clvB, *m_clvP, *m_clvQ, *m_clvR;
	clFloat *clfRho, *clfAlpha, *clfBeta, *clfTemp, *clfNewRho;
	int m_iSizeX,m_iSizeY;

	void initClFloats();
};
