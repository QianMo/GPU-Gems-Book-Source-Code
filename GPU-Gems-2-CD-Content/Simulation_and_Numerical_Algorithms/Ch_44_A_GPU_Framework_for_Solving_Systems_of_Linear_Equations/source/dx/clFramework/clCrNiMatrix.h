#pragma once



#include "clunpackedmatrix.h"

class clCrNiMatrix : public clUnpackedMatrix {
public:
	clCrNiMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, float fDeltaT, float fC, float fDeltaX, float fDeltaY);

	void setDeltaT(float fDeltaT);
	void setC(float fC);
	void setDeltaX(float fDeltaX);
	void setDeltaY(float fDeltaY);

	float getDeltaT();
	float getC();
	float getDeltaX();
	float getDeltaY();

	static void preOrder(int iSizeX, int iSizeY, int iCount) {
		clUnpackedVector::preOrder(iSizeX, iSizeY,5*iCount,true);
	}

protected:
	float m_fDeltaT;
	float m_fC;
	float m_fDeltaX;
	float m_fDeltaY;

	// block the other constructors
	clCrNiMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize) {
		UNREFERENCED_PARAMETER( pd3dDevice );
		UNREFERENCED_PARAMETER( iSize );
	};
	clCrNiMatrix() {};

	void fillRows();
};

