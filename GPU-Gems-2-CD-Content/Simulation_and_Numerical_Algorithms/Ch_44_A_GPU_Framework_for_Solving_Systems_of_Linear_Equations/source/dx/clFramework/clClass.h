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
#pragma once

// this define controls whether a screen filling quad
// or a screen filling triangle is rendered
// if useCoverQuad is defined a quad is used, otherwise a
// triangle is used
#define useCoverQuad 1

// this define controls what precision to use, usually the framework
// uses 32 bit precision but setting this define reduced it ti 16bit
//#define use16BitPrecOnly 1

#ifdef use16BitPrecOnly
	#define FLOAT_TEX_RGBA D3DFMT_A16B16G16R16F
	#define FLOAT_TEX_R D3DFMT_R16F
#else
	#define FLOAT_TEX_RGBA D3DFMT_A32B32G32R32F
	#define FLOAT_TEX_R D3DFMT_R32F
#endif

#include <stdio.h>
#include <d3dx9.h>
#include <dxerr9.h>
#include <typeinfo>

#include "clMemMan.h"

#define CL_NULL  0
#define CL_ADD   1
#define CL_SUB   2
#define CL_MULT  3
#define CL_enum int


struct CL_TEX2D_VERTEX {
    FLOAT      x,y,z;		// position
    FLOAT      tu, tv;		// tex-coords
};

#ifndef clFrameworkShaderPath
	#ifndef _DEBUG
		#define clFrameworkShaderPath _T("shader/")
	#else
		#define clFrameworkShaderPath _T("../clFramework/shader/")
	#endif
#endif

class clClass {
public:
	clClass(void) {}
	virtual ~clClass(void) {}
	LPDIRECT3DDEVICE9 getDevice() {return m_pd3dDevice;};
	static void ShutdownCLFrameWork();
	static HRESULT StartupCLFrameWork(LPDIRECT3DDEVICE9 pd3dDevice);

	static const DWORD				CL_TEX2D_VERTEX_FVF;
	static clMemMan*				ms_memoryMananger;
	static D3DXMACRO				ms_clPSProfile[2];

protected:
	static UINT						ms_cPasses;
	static CL_TEX2D_VERTEX			ms_hCoverQuad[4];
	static CL_TEX2D_VERTEX			ms_hCoverTriangle[3];
	static LPDIRECT3DDEVICE9		m_pd3dDevice;
	static LPDIRECT3DVERTEXBUFFER9	ms_pCoverVB;

	LPDIRECT3DSURFACE9				m_lpBackBuffer;

	HRESULT RenderViewPortCover();
	HRESULT RenderViewPortCover(LPD3DXEFFECT effect, int iPass=0);
};
