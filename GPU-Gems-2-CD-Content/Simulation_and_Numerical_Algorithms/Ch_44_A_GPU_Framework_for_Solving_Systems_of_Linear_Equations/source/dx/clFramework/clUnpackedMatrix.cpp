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

#include "clFragmentMatrix.h"
#include "clUnpackedVector.h"

#include "clUnpackedMatrix.h"

clUnpackedMatrix::clUnpackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize) {
	float fLog = DirectXUtils::log2(iSize)/2.0f;

	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = 1<<(int)ceil(fLog);
	m_iSizeY	 = 1<<(int)floor(fLog);

	m_piRowIndices = NULL;
    m_iRowCount  = 0;;
}

clUnpackedMatrix::clUnpackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;
	
	m_piRowIndices = NULL;
    m_iRowCount  = 0;
}

clUnpackedMatrix::clUnpackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, int *piRowIndices, int iRowCount) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;

	m_iRowCount  = 0;

	createRowVectors(piRowIndices, iRowCount);
}


HRESULT clUnpackedMatrix::createRowVectors(int* piRowIndices, int iRowCount) {
	HRESULT hr;

	#ifdef _DEBUG
	if (m_iRowCount >0) {
		MessageBox(NULL,_T("(clUnpackedMatrix::createRowVectors) Non zero row count, call deleteRowVectors first"),_T("DEBUG Error"),MB_OK);
		exit(-1);
	}
	#endif

	m_iRowCount  = iRowCount;

	m_piRowIndices = new int[iRowCount];
    memcpy( m_piRowIndices, piRowIndices, sizeof(int)*iRowCount );

	clUnpackedVector *clvRows = new clUnpackedVector[m_iRowCount];  
	for (int i = 0;i<m_iRowCount;i++) {
		CHECK_HR(clvRows[i].init(m_pd3dDevice,m_iSizeX,m_iSizeY,true,FLOAT_TEX_R));
	}
	m_clvRows = clvRows;

	return S_OK;
}

clUnpackedMatrix::~clUnpackedMatrix(void){
	deleteRowVectors();
}

clUnpackedVector* clUnpackedMatrix::getRow(int iIndex) {
	int iArrayIndex = findArrayIndex(iIndex);

	if (iArrayIndex != -1)
		return &(static_cast<clUnpackedVector*>(m_clvRows))[iArrayIndex];
	else
		return NULL;
}

/*******************************************************************************************
   Name:    matrixVectorOpAdd 
   Purpose: computes matrix vector operations of the kind:
	          clvResult = clvResult + this*clvX OP clvY
	          where OP (eOpType) can be one of the following: CL_NULL,CL_ADD,CL_SUB,CL_MULT
********************************************************************************************/
void clUnpackedMatrix::matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) {
	signed int iShiftIndex;	int	iSqrSize = m_iSizeX*m_iSizeY;

	// compute Ax first
	for (int i=0;i<m_iRowCount;i++) {
		if (m_piRowIndices[i] < iSqrSize) iShiftIndex = m_piRowIndices[i]-iSqrSize; else iShiftIndex = m_piRowIndices[i]%iSqrSize;
		m_clvRows[i].multiplyAddVector((static_cast<clUnpackedVector*>(clvX)),(static_cast<clUnpackedVector*>(clvResult)),1,iShiftIndex);
	}

	// now compute OP y
	if (clvY != NULL) clvResult->vectorOp(eOpType,clvY,clvResult);
}

void clUnpackedMatrix::matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) {
	signed int iShiftIndex;	int	iSqrSize = m_iSizeX*m_iSizeY;

	// compute Ax first
	for (int i=0;i<m_iRowCount;i++) {
		if (m_piRowIndices[i] < iSqrSize) iShiftIndex = m_piRowIndices[i]-iSqrSize; else iShiftIndex = m_piRowIndices[i]%iSqrSize;
		if (i == 0)
			m_clvRows[i].multiplyVector((static_cast<clUnpackedVector*>(clvX)),(static_cast<clUnpackedVector*>(clvResult)),1,iShiftIndex);
		else
			m_clvRows[i].multiplyAddVector((static_cast<clUnpackedVector*>(clvX)),(static_cast<clUnpackedVector*>(clvResult)),1,iShiftIndex);
	}

	// now compute OP y
	if (clvY != NULL) clvResult->vectorOp(eOpType,clvY,clvResult);
}
void clUnpackedMatrix::deleteRowVectors() {
	delete [] m_piRowIndices;
	delete [] m_clvRows;
	m_piRowIndices = NULL;
    m_iRowCount  = 0;;
}