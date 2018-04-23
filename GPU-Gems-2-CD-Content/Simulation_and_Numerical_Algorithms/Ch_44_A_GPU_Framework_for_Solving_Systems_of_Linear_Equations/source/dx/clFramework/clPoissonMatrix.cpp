#include "clPackedMatrix.h"

#include "clPoissonMatrix.h"

void clPoissonMatrix::setDeltaX(float fDeltaX)	{m_fDeltaX	= fDeltaX;	fillRows();}
void clPoissonMatrix::setDeltaY(float fDeltaY)	{m_fDeltaY	= fDeltaY;	fillRows();}

float clPoissonMatrix::getDeltaX()	{return m_fDeltaX;}
float clPoissonMatrix::getDeltaY()	{return m_fDeltaY;}


void clPoissonMatrix::fillRows() {
	int i;

	// todo: extend this for dx!=dy
	float fInv = 1.0f/(m_fDeltaX*m_fDeltaX);

	float *pfVectorData = new float[m_iSizeX*m_iSizeY];

	// setup diagonal-1
	for (i=0;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = (i%m_iSizeX) ? 1*fInv : 0;
	getRow(m_iSizeX*m_iSizeY-1)->setData(pfVectorData);

	// setup diagonal-m_iSizeY
	ZeroMemory(pfVectorData, m_iSizeX*m_iSizeY*sizeof(float));
	for (i=m_iSizeY;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = 1*fInv;
	getRow(m_iSizeX*(m_iSizeY-1))->setData(pfVectorData);

	// setup diagonal
	for (i=0;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = -4.0f*fInv;
	getRow(m_iSizeX*m_iSizeY)->setData(pfVectorData);

	// setup diagonal+1
	for (i=0;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = ((i+1)%m_iSizeX) ? 1*fInv : 0;
	getRow(m_iSizeX*m_iSizeY+1)->setData(pfVectorData);

	// setup diagonal+m_iSizeY
	ZeroMemory(pfVectorData, m_iSizeX*m_iSizeY*sizeof(float));
	for (i=0;i<m_iSizeX*(m_iSizeY-1);i++) pfVectorData[i] = 1*fInv;
	getRow(m_iSizeX*(m_iSizeY+1))->setData(pfVectorData);

	delete [] pfVectorData;
}


clPoissonMatrix::clPoissonMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, float fDeltaX, float fDeltaY) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;
	
	m_fDeltaX	= fDeltaX;
	m_fDeltaY	= fDeltaY;

	// define the rows
	int piRows[] = {m_iSizeX*(m_iSizeY-1),
					m_iSizeX* m_iSizeY-1,
					m_iSizeX* m_iSizeY,
					m_iSizeX* m_iSizeY+1,
					m_iSizeX*(m_iSizeY+1)}; 
	
	// define rows
	m_iRowCount = 0;
	createRowVectors(piRows,5);

	// fillup the rows
	fillRows();
}
