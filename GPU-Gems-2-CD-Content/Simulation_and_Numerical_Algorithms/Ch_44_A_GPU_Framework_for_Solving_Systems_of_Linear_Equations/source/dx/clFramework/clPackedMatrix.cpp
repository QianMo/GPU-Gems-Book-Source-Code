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

#include "clPackedMatrix.h"

clPackedMatrix::clPackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize) {
	float fLog = DirectXUtils::log2(iSize)/2.0f;

	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = 1<<(int)ceil(fLog);
	m_iSizeY	 = 1<<(int)floor(fLog);

	m_piRowIndices = NULL;
    m_iRowCount  = 0;;
}

clPackedMatrix::clPackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;

	m_piRowIndices = NULL;
    m_iRowCount  = 0;;
}

clPackedMatrix::clPackedMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, int *piRowIndices, int iRowCount) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;
	m_iRowCount  = 0;

	createRowVectors(piRowIndices, iRowCount);
}

void clPackedMatrix::deleteRowVectors() {
	delete [] m_piRowIndices;
	delete [] m_clvRows;
	m_piRowIndices = NULL;
    m_iRowCount  = 0;;
}

HRESULT clPackedMatrix::createRowVectors(int* piRowIndices, int iRowCount) {
	HRESULT hr;

	#ifdef _DEBUG
	if (m_iRowCount >0) {
		MessageBox(NULL,_T("(clPackedMatrix::createRowVectors) Non zero row count, call deleteRowVectors first"),_T("DEBUG Error"),MB_OK);
		exit(-1);
	}
	#endif

	m_iRowCount  = iRowCount;

	m_piRowIndices = new int[iRowCount];
    memcpy( m_piRowIndices, piRowIndices, sizeof(int)*iRowCount );

	m_clvRows = new clPackedVector[m_iRowCount];  
	for (int i = 0;i<m_iRowCount;i++) {
		CHECK_HR(m_clvRows[i].init(m_pd3dDevice,m_iSizeX/2,m_iSizeY/2,true,FLOAT_TEX_RGBA));
	}

	return S_OK;
}

clPackedMatrix::~clPackedMatrix(void){
	deleteRowVectors();
}


int clPackedMatrix::findArrayIndex(int iIndex) {
	for (int i=0;i<m_iRowCount;i++) if (m_piRowIndices[i] == iIndex) return i;
	return -1; // element not found
}

bool clPackedMatrix::setRow(clPackedVector *clvRow, int iIndex) {
	if (m_piRowIndices == NULL) {
		m_clvRows[iIndex] = *clvRow;
		return true;
	} else {
		// find index
		int iArrayIndex = findArrayIndex(iIndex);

		// if index was found, assign row vector
		if (iArrayIndex != -1) {
			m_clvRows[iArrayIndex] = *clvRow;
			return true;
		}
		return false;
	}
}

clPackedVector* clPackedMatrix::getRow(int iIndex) {
	int iArrayIndex = findArrayIndex(iIndex);

	if (iArrayIndex != -1) {
		return &m_clvRows[iArrayIndex];
	}
	return NULL;
}


/*******************************************************************************************
   Name:    matrixVectorOpAdd 
   Purpose: computes matrix vector operations of the kind:
	          clvResult = clvResult + this*clvX OP clvY
	          where OP (eOpType) can be one of the following: CL_NULL,CL_ADD,CL_SUB,CL_MULT
********************************************************************************************/
void clPackedMatrix::matrixVectorOpAdd(CL_enum eOpType, clFragmentVector *clvX, clFragmentVector *clvY, clFragmentVector *clvResult) {
	signed int iShiftIndex;	int	iSqrSize = m_iSizeX*m_iSizeY;

	// compute Ax first, handle diagonals "shift-case" by "shift-case"
	for (int iCurrentShiftCase=0; iCurrentShiftCase<4;iCurrentShiftCase++) {
		for (int i=0;i<m_iRowCount;i++) {
			// shift lower diagonals by a negative value
			if (m_piRowIndices[i] < iSqrSize) iShiftIndex = m_piRowIndices[i]-iSqrSize; else iShiftIndex = m_piRowIndices[i]%iSqrSize;

			// compute shift-case
			int iShiftCase = (m_iSizeX*m_iSizeY+iShiftIndex)%4;
			
			// test if the current diagonal matches the currrent "shift-case"
			if (iCurrentShiftCase==iShiftCase) {
				iShiftIndex = (iShiftIndex-iShiftCase)/4;
				m_clvRows[i].multiplyAddVector((static_cast<clPackedVector*>(clvX)),(static_cast<clPackedVector*>(clvResult)),1,iShiftIndex,iCurrentShiftCase);
			}
		}
	}

	// now compute OP y
	if (clvY != NULL && eOpType != CL_NULL) clvResult->vectorOp(eOpType,clvY,clvResult);
}

/*******************************************************************************************
   Name:    matrixVectorOp 
   Purpose: computes matrix vector operations of the kind:
	          clvResult = this*clvX OP clvY
	          where OP (eOpType) can be one of the following: CL_NULL,CL_ADD,CL_SUB,CL_MULT
********************************************************************************************/
void clPackedMatrix::matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) {
	signed int iShiftIndex;	
	int		   iSqrSize = m_iSizeX*m_iSizeY;
	bool	   bFirst = true;

	// compute Ax first, handle diagonals "shift-case" by "shift-case"
	for (int iCurrentShiftCase=0; iCurrentShiftCase<4;iCurrentShiftCase++) {
		for (int i=0;i<m_iRowCount;i++) {
			// shift lower diagonals by a negative value
			if (m_piRowIndices[i] < iSqrSize) iShiftIndex = m_piRowIndices[i]-iSqrSize; else iShiftIndex = m_piRowIndices[i]%iSqrSize;

			// compute shift-case
			int iShiftCase = (m_iSizeX*m_iSizeY+iShiftIndex)%4;
			
			// test if the current diagonal matches the currrent "shift-case"
			if (iCurrentShiftCase==iShiftCase) {
				iShiftIndex = (iShiftIndex-iShiftCase)/4;
				if (!bFirst) 
					m_clvRows[i].multiplyAddVector((static_cast<clPackedVector*>(clvX)),(static_cast<clPackedVector*>(clvResult)),1,iShiftIndex,iCurrentShiftCase);
				else {
					m_clvRows[i].multiplyVector((static_cast<clPackedVector*>(clvX)),(static_cast<clPackedVector*>(clvResult)),1,iShiftIndex,iCurrentShiftCase);
					bFirst=false;
				}
			}
		}
	}

	// now compute OP y
	if (clvY != NULL && eOpType != CL_NULL) clvResult->vectorOp(eOpType,clvY,clvResult);
}
