#pragma once



#include "clunpackedvector.h"

class clCrNiVector : public clUnpackedVector {
public:
	clCrNiVector(void) {}
	clCrNiVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX,int iSizeY, bool bConstant=false);
	clCrNiVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant=false);

	virtual ~clCrNiVector(void);

	void setC(float fC);
	void setSimulParam(float fDt,float fC,float fDX,float fDY);
	void getSimulParam(float &fDt,float &fC,float &fDX,float &fDY);
	void computeRHS(clUnpackedVector *cluULast, clUnpackedVector *cluUCurrent);


protected:
	static int s_iShaderClCrNiUser;
	static LPD3DXEFFECT ms_pShaderclCrNi;

	static D3DXHANDLE ms_f4Shift;
	static D3DXHANDLE ms_fPreFac;
	static D3DXHANDLE ms_tLast;
	static D3DXHANDLE ms_tCurrent;
	static D3DXHANDLE ms_tCompRHS;

	float m_fDt, m_fC, m_fDX, m_fDY; 
	float m_fPreFactor;	

	virtual void loadShaders();
};
