/****************************************************************
 *																*
 * decr     : Crank-Nicholson matrix for the water surface demo	*
 * version  : 1.01												*
 * author   : Jens Krüger										*
 * date     : 16.09.2003										*
 * modified	: 06.10.2003										*
 * e-mail   : jens.krueger@in.tum.de							*
 *																*
 ****************************************************************/

#include "clunpackedmatrix.h"

#include "clCrNiMatrix.h"

void clCrNiMatrix::setDeltaT(float fDeltaT)	{m_fDeltaT	= fDeltaT;	fillRows();}
void clCrNiMatrix::setC(float fC)			{m_fC		= fC;		fillRows();}
void clCrNiMatrix::setDeltaX(float fDeltaX)	{m_fDeltaX	= fDeltaX;	fillRows();}
void clCrNiMatrix::setDeltaY(float fDeltaY)	{m_fDeltaY	= fDeltaY;	fillRows();}

float clCrNiMatrix::getDeltaT()	{return m_fDeltaT;}
float clCrNiMatrix::getC()		{return m_fC;}
float clCrNiMatrix::getDeltaX()	{return m_fDeltaX;}
float clCrNiMatrix::getDeltaY()	{return m_fDeltaY;}

void clCrNiMatrix::fillRows() {
	int i;

	float fPreFactor = (m_fDeltaT*m_fDeltaT*m_fC*m_fC) / (2*m_fDeltaX*m_fDeltaY);
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
	float fDiagFactor = 4*fPreFactor+1;
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


clCrNiMatrix::clCrNiMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, float fDeltaT, float fC, float fDeltaX, float fDeltaY) {
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
	createRowVectors(piRows,5);

	// fillup the rows
	fillRows();
}
