#pragma once


#ifdef useUpacked
	#include "clUnpackedMatrix.h"
	class clPoissonMatrix : public clUnpackedMatrix {
#else
	#include "clPackedMatrix.h"
	class clPoissonMatrix : public clPackedMatrix {
#endif

public:
	clPoissonMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, float fDeltaX, float fDeltaY);

	void setDeltaX(float fDeltaX);
	void setDeltaY(float fDeltaY);

	float getDeltaX();
	float getDeltaY();

	static void preOrder(int iSizeX, int iSizeY, int iCount) {
		#ifdef useUpacked
			clUnpackedVector::preOrder(iSizeX, iSizeY,5*iCount, true);
		#else
			clPackedVector::preOrder(iSizeX, iSizeY,5*iCount);
		#endif
	}

protected:
	float m_fDeltaX;
	float m_fDeltaY;

	// block the other constructors
	clPoissonMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize) {
		UNREFERENCED_PARAMETER( pd3dDevice );
		UNREFERENCED_PARAMETER( iSize );
	};
	clPoissonMatrix() {};

	void fillRows();
};
