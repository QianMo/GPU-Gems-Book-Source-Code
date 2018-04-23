/****************************************************************
 *																*
 * decr     : Crank-Nicholson vector for the water surface demo	*
 * version  : 1.01												*
 * author   : Jens Krüger										*
 * date     : 16.09.2003										*
 * modified	: 06.10.2003										*
 * e-mail   : jens.krueger@in.tum.de							*
 *																*
 ****************************************************************/

#include "clunpackedvector.h"

#include "clCrNiVector.h"


int				clCrNiVector::s_iShaderClCrNiUser=0;
LPD3DXEFFECT	clCrNiVector::ms_pShaderclCrNi=NULL;
D3DXHANDLE		clCrNiVector::ms_f4Shift;
D3DXHANDLE		clCrNiVector::ms_fPreFac;
D3DXHANDLE		clCrNiVector::ms_tLast;
D3DXHANDLE		clCrNiVector::ms_tCurrent;
D3DXHANDLE		clCrNiVector::ms_tCompRHS;

clCrNiVector::~clCrNiVector(void){
	s_iShaderClCrNiUser--;
	if (s_iShaderClCrNiUser==0) 	SAFE_RELEASE( ms_pShaderclCrNi );
}

clCrNiVector::clCrNiVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX,int iSizeY, bool bConstant) 
{
	init(pd3dDevice, iSizeX,iSizeY, bConstant, FLOAT_TEX_R);
}

clCrNiVector::clCrNiVector(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant) 
{
	init(pd3dDevice, iSize, bConstant, FLOAT_TEX_R);
}


void clCrNiVector::setC(float fC) {
	setSimulParam(m_fDt, fC, m_fDX, m_fDY);
}

void clCrNiVector::setSimulParam(float fDt,float fC,float fDX,float fDY) {
	m_fDt = fDt;
	m_fC  = fC;
	m_fDX = fDX;
	m_fDY = fDY;

	m_fPreFactor = (m_fDt*m_fDt*m_fC*m_fC) / (2.0f*m_fDX*m_fDY);  
}

void clCrNiVector::getSimulParam(float &fDt,float &fC,float &fDX,float &fDY) {
	fDt = m_fDt;
	fC  = m_fC;
	fDX = m_fDX;
	fDY = m_fDY;
}

void clCrNiVector::computeRHS(clUnpackedVector *cluULast, clUnpackedVector *cluUCurrent) {
	m_pd3dDevice->SetRenderTarget(0,m_pVectorTextureSurface);

		ms_pShaderclCrNi->SetVector(ms_f4Shift,&m_vSize);
		ms_pShaderclCrNi->SetFloat(ms_fPreFac,m_fPreFactor);
		ms_pShaderclCrNi->SetTexture(ms_tLast,     cluULast->m_pVectorTexture);
		ms_pShaderclCrNi->SetTexture(ms_tCurrent,  cluUCurrent->m_pVectorTexture);
		
		ms_pShaderclCrNi->SetTechnique(ms_tCompRHS);

		RenderViewPortCover(ms_pShaderclCrNi);
	m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer);
}

void clCrNiVector::loadShaders() {
	// load the standart shaders
	clUnpackedVector::loadShaders();
	// load the aditional shaders
	if (s_iShaderClCrNiUser < 1) {
		if (FAILED(DirectXUtils::checkLoadShader(_T("clCrNiVector.fx"), ms_pShaderclCrNi, m_pd3dDevice,clFrameworkShaderPath, NULL, clClass::ms_clPSProfile))) exit(-1);
		// precache shader parameters 

		// parameters 
		ms_f4Shift			= ms_pShaderclCrNi->GetParameterByName(NULL,"f4Shift");
		ms_fPreFac			= ms_pShaderclCrNi->GetParameterByName(NULL,"fPreFac");
		ms_tLast			= ms_pShaderclCrNi->GetParameterByName(NULL,"tLast");
		ms_tCurrent			= ms_pShaderclCrNi->GetParameterByName(NULL,"tCurrent");

		// techniques
		ms_tCompRHS			= ms_pShaderclCrNi->GetTechniqueByName("tCompRHS");
	}
	s_iShaderClCrNiUser++;
}
