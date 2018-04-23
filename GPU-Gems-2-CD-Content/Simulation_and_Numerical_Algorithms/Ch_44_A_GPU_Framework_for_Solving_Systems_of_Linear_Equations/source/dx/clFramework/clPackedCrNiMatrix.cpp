#include "clPackedCrNiMatrix.h"

void clPackedCrNiMatrix::setDeltaT(float fDeltaT)	{m_fDeltaT	= fDeltaT;	fillRows();}
void clPackedCrNiMatrix::setC(float fC)				{m_fC		= fC;		fillRows();}
void clPackedCrNiMatrix::setDeltaX(float fDeltaX)	{m_fDeltaX	= fDeltaX;	fillRows();}
void clPackedCrNiMatrix::setDeltaY(float fDeltaY)	{m_fDeltaY	= fDeltaY;	fillRows();}

float clPackedCrNiMatrix::getDeltaT()	{return m_fDeltaT;}
float clPackedCrNiMatrix::getC()		{return m_fC;}
float clPackedCrNiMatrix::getDeltaX()	{return m_fDeltaX;}
float clPackedCrNiMatrix::getDeltaY()	{return m_fDeltaY;}


void clPackedCrNiMatrix::fillRows() {
	int i;

	float fPreFactor = (m_fDeltaT*m_fDeltaT*m_fC*m_fC) / (2.0f*m_fDeltaX*m_fDeltaY);
	float *pfVectorData = new float[m_iSizeX*m_iSizeY];

	// setup diagonal-m_iSizeY
	ZeroMemory(pfVectorData, m_iSizeX*m_iSizeY*sizeof(float));
	for (i=m_iSizeY;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = -fPreFactor;
	getRow(m_iSizeX*(m_iSizeY-1))->setData(pfVectorData);

	// setup diagonal-1
	ZeroMemory(pfVectorData, m_iSizeX*m_iSizeY*sizeof(float));
	for (i=0;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = (i%m_iSizeX) ? -fPreFactor : 0;
	getRow(m_iSizeX*m_iSizeY-1)->setData(pfVectorData);

	// setup diagonal
	float fDiagFactor = 1+4*fPreFactor;
	for (i=0;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = fDiagFactor;
	getRow(m_iSizeX*m_iSizeY)->setData(pfVectorData);

	// setup diagonal+1
	ZeroMemory(pfVectorData, m_iSizeX*m_iSizeY*sizeof(float));
	for (i=0;i<m_iSizeX*m_iSizeY;i++) pfVectorData[i] = ((i+1)%m_iSizeX) ? -fPreFactor : 0;
	getRow(m_iSizeX*m_iSizeY+1)->setData(pfVectorData);

	// setup diagonal+m_iSizeY
	ZeroMemory(pfVectorData, m_iSizeX*m_iSizeY*sizeof(float));
	for (i=0;i<m_iSizeX*(m_iSizeY-1);i++) pfVectorData[i] = -fPreFactor;
	getRow(m_iSizeX*(m_iSizeY+1))->setData(pfVectorData);

	delete [] pfVectorData;
}


clPackedCrNiMatrix::clPackedCrNiMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, float fDeltaT, float fC, float fDeltaX, float fDeltaY) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;

	
	m_fDeltaT	= fDeltaT;
	m_fC		= fC;
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
	createRowVectors(piRows,sizeof(piRows)/sizeof(int));

	// fillup the rows
	fillRows();
}
