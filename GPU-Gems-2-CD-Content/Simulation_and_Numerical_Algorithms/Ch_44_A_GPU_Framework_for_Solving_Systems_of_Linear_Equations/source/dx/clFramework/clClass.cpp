/************************************************************
 *															*
 * decr     : base class for al clClasses					*
 * version  : 1.0											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 16.09.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#include "clClass.h"


const DWORD clClass::CL_TEX2D_VERTEX_FVF = D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE2(0);
CL_TEX2D_VERTEX clClass::ms_hCoverQuad[4]= {{ -1,  1, 0, 0, 0},
											{  1,  1, 0, 1, 0},
											{ -1, -1, 0, 0, 1},
											{  1, -1, 0, 1, 1}};

CL_TEX2D_VERTEX clClass::ms_hCoverTriangle[3]= {{ -1,  1, 0, 0, 0},
												{  3,  1, 0, 2, 0},
												{ -1, -3, 0, 0, 2}};

LPDIRECT3DDEVICE9		clClass::m_pd3dDevice=NULL;
LPDIRECT3DVERTEXBUFFER9	clClass::ms_pCoverVB=NULL;
UINT					clClass::ms_cPasses=0;
clMemMan*				clClass::ms_memoryMananger=NULL;

#ifdef optNVIDIA
	#define PS_PROFILE "ps_2_a"
#else
	#define PS_PROFILE "ps_2_0"
#endif

D3DXMACRO				clClass::ms_clPSProfile[2] = { { "PS_PROFILE", PS_PROFILE }, {NULL,NULL} };




HRESULT clClass::StartupCLFrameWork(LPDIRECT3DDEVICE9 pd3dDevice) {
	HRESULT hr;

	m_pd3dDevice = pd3dDevice;

	if (ms_pCoverVB == NULL) {
		#ifdef useCoverQuad
			CHECK_HR(DirectXUtils::createFilledVertexBuffer(m_pd3dDevice,ms_hCoverQuad,sizeof(CL_TEX2D_VERTEX)*4,CL_TEX2D_VERTEX_FVF ,D3DPOOL_DEFAULT,ms_pCoverVB));
		#else
			CHECK_HR(DirectXUtils::createFilledVertexBuffer(m_pd3dDevice,ms_hCoverTriangle,sizeof(CL_TEX2D_VERTEX)*3,CL_TEX2D_VERTEX_FVF, D3DPOOL_DEFAULT,ms_pCoverVB));
		#endif
	}

	if (ms_memoryMananger == NULL) ms_memoryMananger = new clMemMan(pd3dDevice);

	return S_OK;
}

void clClass::ShutdownCLFrameWork() {
	SAFE_RELEASE(ms_pCoverVB);
	SAFE_DELETE(ms_memoryMananger);
}


HRESULT clClass::RenderViewPortCover() {
	HRESULT hr;

	CHECK_HR(m_pd3dDevice->SetStreamSource( 0, ms_pCoverVB, 0, sizeof(CL_TEX2D_VERTEX) ));
	CHECK_HR(m_pd3dDevice->SetFVF( CL_TEX2D_VERTEX_FVF ));

#ifdef useCoverQuad
	CHECK_HR(m_pd3dDevice->DrawPrimitive(D3DPT_TRIANGLESTRIP,0,2));
#else
	CHECK_HR(m_pd3dDevice->DrawPrimitive(D3DPT_TRIANGLELIST,0,1));
#endif

	return S_OK;
}

HRESULT clClass::RenderViewPortCover(LPD3DXEFFECT effect, int iPass) {
	HRESULT hr;

	CHECK_HR(effect->Begin(&ms_cPasses, 0));
		CHECK_HR(effect->BeginPass(iPass));
		CHECK_HR(RenderViewPortCover());
		CHECK_HR(effect->EndPass());
	CHECK_HR(effect->End());

	return S_OK;
}