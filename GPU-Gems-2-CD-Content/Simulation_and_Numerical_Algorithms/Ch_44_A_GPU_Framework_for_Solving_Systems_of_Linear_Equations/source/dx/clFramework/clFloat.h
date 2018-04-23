/************************************************************
 *															*
 * decr     : A single float value stored on the GPU		*
 * version  : 1.10											*
 * author   : Jens Krüger									*
 * date     : 08.02.2004									*
 * modified	: 10.06.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#pragma once

#include "clclass.h"

struct CLFLOATVERTEX {
    FLOAT      x,y,z;		// position
    static const DWORD FVF;
};

class clFloat :	public clClass {
public:
	clFloat();
	clFloat(LPDIRECT3DDEVICE9 pd3dDevice);
	clFloat(LPDIRECT3DDEVICE9 pd3dDevice, float fValue);
	virtual ~clFloat(void);

	virtual void init(LPDIRECT3DDEVICE9 pd3dDevice);
	virtual void setData(float fVectorData);
	virtual void getData(float *fVectorData);
	virtual float getData();

	virtual void add(clFloat *other, float a=1, float b=1) {add(other,this,a,b);}
	virtual void mul(clFloat *other, float a=1) {mul(other,this,a);}
	virtual void sub(clFloat *other, float a=1, float b=1) {add(other,this,a,-b);}
	virtual void div(clFloat *other, float a=1, float b=1) {div(other,this,a,b);}
	virtual void invert(float a=1, float b=1) {invert(this,a,b);}
	virtual void add(float scalar) {add(scalar,this);}
	virtual void mul(float scalar) {mul(scalar,this);}	
	virtual void sub(float scalar) {add(-scalar,this);};
	virtual void div(float scalar) {mul(1.0f/scalar,this);};

	virtual void add(clFloat *other, clFloat *target, float a=1, float b=1);
	virtual void mul(clFloat *other, clFloat *target, float a=1);
	virtual void sub(clFloat *other, clFloat *target, float a=1, float b=1) {add(other,target,a,-b);};
	virtual void div(clFloat *other, clFloat *target, float a=1, float b=1);
	virtual void divZ(clFloat *other,clFloat *target, float a=1, float b=1);
	virtual void invert(clFloat *target, float a=1, float b=1);
	virtual void add(float scalar, clFloat *target);
	virtual void mul(float scalar, clFloat *target);	
	virtual void sub(float scalar, clFloat *target) {add(target, -scalar);};
	virtual void div(float scalar, clFloat *target) {mul(target, 1.0f/scalar);};

	virtual char* toString();

	static void preOrder(int iCount) {ms_memoryMananger->preOrderTextureTarget(clMemDescr(1, 1, FLOAT_TEX_R,D3DUSAGE_RENDERTARGET),iCount);}

	LPDIRECT3DSURFACE9 readSurface() {return m_pFloatTextureSurface;}
	PDIRECT3DTEXTURE9  readTexture() {return m_pFloatTexture;}

protected:
	static int						ms_iClFloatCount;
	static LPD3DXEFFECT				ms_pShaderClFloat;
	static LPDIRECT3DVERTEXBUFFER9	ms_pCoverPoint;

	static D3DXHANDLE				ms_fScalar;
	static D3DXHANDLE				ms_tFloatA;
	static D3DXHANDLE				ms_tFloatB;
	static D3DXHANDLE				ms_tMulClFloat;
	static D3DXHANDLE				ms_tAddClFloat;
	static D3DXHANDLE				ms_tDivClFloat;
	static D3DXHANDLE				ms_tDivZClFloat;
	static D3DXHANDLE				ms_tInvert;
	static D3DXHANDLE				ms_tAddScalar;
	static D3DXHANDLE				ms_tMultilyScalar;

	void loadShaders();
	void renderPoint();

	PDIRECT3DTEXTURE9				m_pFloatTexture;
	LPDIRECT3DSURFACE9				m_pFloatTextureSurface;
	int								m_iActiveTexture;
	int								m_iMemID;

	int								m_TempID;
	PDIRECT3DTEXTURE9				m_pTempTexture;
	LPDIRECT3DSURFACE9				m_pTempTextureSurface;

	HRESULT BeginScene();
	HRESULT EndScene();
	LPDIRECT3DSURFACE9 getWriteSurface();
	void swapSurfaces();
};
