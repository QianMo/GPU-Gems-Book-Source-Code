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

#include "clCGSolver.h"

clCGSolver::clCGSolver(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, int iRepresentation) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX    = iSizeX;
	m_iSizeY    = iSizeY;

	switch (iRepresentation) {
		case CL_PACKED :		m_clvB		= new clPackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvP		= new clPackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvQ		= new clPackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvR		= new clPackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvX      = new clPackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clMatrix	= new clPackedMatrix(pd3dDevice, iSizeX, iSizeY);
								break;
		case CL_UNPACKED :		m_clvB		= new clUnpackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvP		= new clUnpackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvQ		= new clUnpackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvR		= new clUnpackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clvX      = new clUnpackedVector(pd3dDevice, iSizeX, iSizeY);
								m_clMatrix	= new clUnpackedMatrix(pd3dDevice, iSizeX, iSizeY);
								break;
	}

	initClFloats();
}

clCGSolver::clCGSolver(clAbstractMatrix *clM, clFragmentVector *clvX, clFragmentVector *clvB, clFragmentVector *clvP, clFragmentVector *clvQ, clFragmentVector *clvR) {
	m_clvB			= clvB;

	m_pd3dDevice	= m_clvB->getDevice();
	m_iSizeX		= m_clvB->getSizeX();
	m_iSizeY		= m_clvB->getSizeY();

	m_clvP			= clvP;
	m_clvQ			= clvQ;
	m_clvR			= clvR;
	m_clvX			= clvX;
	m_clMatrix		= clM;

	initClFloats();
}

clCGSolver::clCGSolver(clAbstractMatrix *clM, clFragmentVector *clvX, clFragmentVector *clvB, int iRepresentation) {
	m_clvB			= clvB;
	m_clvX			= clvX;
	m_clMatrix		= clM;

	m_pd3dDevice	= m_clvB->getDevice();
	m_iSizeX		= m_clvB->getSizeX();
	m_iSizeY		= m_clvB->getSizeY();

	switch (iRepresentation) {
		case CL_PACKED :		m_clvP = new clPackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
								m_clvQ = new clPackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
								m_clvR = new clPackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
								break;
		case CL_UNPACKED :		m_clvP = new clUnpackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
								m_clvQ = new clUnpackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
								m_clvR = new clUnpackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
								break;
	}

	initClFloats();
}

clCGSolver::~clCGSolver(void) {
	SAFE_DELETE( m_clvB );
	SAFE_DELETE( m_clvP );
	SAFE_DELETE( m_clvQ );
	SAFE_DELETE( m_clvR );
	SAFE_DELETE( m_clvX );
	SAFE_DELETE( m_clMatrix );

	SAFE_DELETE( clfRho );
	SAFE_DELETE( clfAlpha );
	SAFE_DELETE( clfBeta );
	SAFE_DELETE( clfTemp );
	SAFE_DELETE( clfNewRho );
}

void clCGSolver::initClFloats() {
	clfRho		= new clFloat(m_pd3dDevice); clfRho->setData(0);
	clfAlpha	= new clFloat(m_pd3dDevice); clfAlpha->setData(0);
	clfBeta		= new clFloat(m_pd3dDevice); clfBeta->setData(0);
	clfTemp		= new clFloat(m_pd3dDevice); clfTemp->setData(0);
	clfNewRho	= new clFloat(m_pd3dDevice); clfNewRho->setData(0);
}

void clCGSolver::setResult(clFragmentVector* clvResult){
	// TODO: check device and size values
	m_clvX = clvResult;
}

void clCGSolver::setRHS(clFragmentVector* clvRHS){
	// TODO: check device and size values
	m_clvB = clvRHS;
}

void clCGSolver::getTemp(clFragmentVector** clvP, clFragmentVector** clvQ, clFragmentVector** clvR){
	*clvP = m_clvP;
	*clvQ = m_clvQ;
	*clvR = m_clvR;
}

void clCGSolver::setTemp(clFragmentVector* clvP, clFragmentVector* clvQ, clFragmentVector* clvR){
	// TODO: check device and size values

	m_clvP = clvP;
	m_clvQ = clvQ;
	m_clvR = clvR;
}


void clCGSolver::setMatrix(clAbstractMatrix* clM) {
	// TODO: check device and size values
	m_clMatrix = clM;
}

int clCGSolver::solveNT(int iter) {
	solveInitNT();
	for (int i = 0;i<iter;i++)	solveIterationNT();
	return i;
}

void clCGSolver::solveInitNT() {
	m_clMatrix->matrixVectorOp(CL_SUB,m_clvX,m_clvB,m_clvR);	// R = A*x-b (use last result as inital guess)
	m_clvR->multiplyScalar(-1);									// R = -R
	m_clvP->copyVector(m_clvR);									// P =  R
	m_clvR->reduceAdd(m_clvR, clfRho);							// rho = sum(R*R);
}

void clCGSolver::solveIterationNT() {
	m_clMatrix->matrixVectorOp(CL_NULL,m_clvP,NULL,m_clvQ);	// Q = Ap;

	m_clvP->reduceAdd(m_clvQ,clfTemp);						// temp  = sum(P*Q);
	clfRho->divZ(clfTemp,clfAlpha);							// alpha = rho/temp;

	m_clvX->addVector(m_clvP,m_clvX,1.0f,clfAlpha);			// X = X + alpha*P
	m_clvR->subtractVector(m_clvQ,m_clvR,1.0f,clfAlpha);	// R = R - alpha*Q

	m_clvR->reduceAdd(m_clvR,clfNewRho);					// newrho = sum(R*R);
	clfNewRho->divZ(clfRho,clfBeta);                        // beta = newrho/rho

	m_clvR->addVector(m_clvP,m_clvP,1,clfBeta);				// P = R+beta*P;

	// swap rho and newrho pointes
	clFloat *temp;	temp=clfNewRho; clfNewRho=clfRho; clfRho=temp;
}

int clCGSolver::solve(float rhoTresh, int iter) {
	float rho = solveInit();
	for (int i = 0;i<iter && rho > rhoTresh;i++) rho = solveIteration(rho);
	return i;
}

float clCGSolver::solveInit() {
	m_clMatrix->matrixVectorOp(CL_SUB,m_clvX,m_clvB,m_clvR);	// R = A*x-b (use last result as inital guess)
	m_clvR->multiplyScalar(-1);									// R = -R
	m_clvP->copyVector(m_clvR);									// P =  R
	return m_clvR->reduceAdd(m_clvR);							// rho = sum(R*R);
}

float clCGSolver::solveIteration(float rho) {
	m_clMatrix->matrixVectorOp(CL_NULL,m_clvP,NULL,m_clvQ);	// Q = Ap;

	float alpha = rho/m_clvP->reduceAdd(m_clvQ);			// alpha = rho/sum(P*Q);

	m_clvX->addVector(m_clvP,m_clvX,1.0f,alpha);			// X = X + alpha*P
	m_clvR->subtractVector(m_clvQ,m_clvR,1.0f,alpha);		// R = R - alpha*Q

	float newrho = m_clvR->reduceAdd(m_clvR);				// newrho = sum(R*R);
	float beta   = newrho / rho;							// beta = newrho/rho

	m_clvR->addVector(m_clvP,m_clvP,1.0f,beta);				// P = R+beta*P;

	return newrho;
}
