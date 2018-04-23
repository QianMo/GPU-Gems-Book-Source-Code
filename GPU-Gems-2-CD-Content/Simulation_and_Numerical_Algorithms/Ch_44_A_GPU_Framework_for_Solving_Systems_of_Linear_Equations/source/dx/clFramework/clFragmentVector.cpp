/************************************************************
 *															*
 * decr     : abstract base class for all fragment based	*
 *            vector clases									*
 * version  : 1.2											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 10.06.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/
#include "clFragmentVector.h"


int						clFragmentVector::s_iFragmentVectorCount=0;
LPD3DXEFFECT			clFragmentVector::ms_pShaderClVector=NULL;

D3DXHANDLE				clFragmentVector::ms_fMultiply;
D3DXHANDLE				clFragmentVector::ms_fMultiply2;
D3DXHANDLE				clFragmentVector::ms_fReduceStep;
D3DXHANDLE				clFragmentVector::ms_f4ReduceStep;
D3DXHANDLE				clFragmentVector::ms_f4TexShift;
D3DXHANDLE				clFragmentVector::ms_fShift;
D3DXHANDLE				clFragmentVector::ms_fSize;
D3DXHANDLE				clFragmentVector::ms_tVector;
D3DXHANDLE				clFragmentVector::ms_tVector2;
D3DXHANDLE				clFragmentVector::ms_tLastPass;
D3DXHANDLE				clFragmentVector::ms_tMultiply;

D3DXHANDLE				clFragmentVector::ms_tMultiplyScal;
D3DXHANDLE				clFragmentVector::ms_tReduceAddFirst;
D3DXHANDLE				clFragmentVector::ms_tReduceAddRest;
D3DXHANDLE				clFragmentVector::ms_tReduceAddLast;
D3DXHANDLE				clFragmentVector::ms_tReduceAddRestX;
D3DXHANDLE				clFragmentVector::ms_tReduceAddRestY;
D3DXHANDLE				clFragmentVector::ms_tVectorAdd;
D3DXHANDLE				clFragmentVector::ms_tVectorMultiply;
D3DXHANDLE				clFragmentVector::ms_tVectorMultiplyMatPosOnly;
D3DXHANDLE				clFragmentVector::ms_tVectorMultiplyMatAllCases;
D3DXHANDLE				clFragmentVector::ms_tVectorMultiplyMatPosOnly_noadd;
D3DXHANDLE				clFragmentVector::ms_tVectorMultiplyMatAllCases_noadd;
D3DXHANDLE				clFragmentVector::ms_tUnpackVector;
D3DXHANDLE				clFragmentVector::ms_tPackVector;


/*******************************************************************************************
   Name:    init
   Purpose: init for a vector of length iSize, automaticly finds the best layout
********************************************************************************************/
HRESULT clFragmentVector::init(LPDIRECT3DDEVICE9 pd3dDevice, int iSize, bool bConstant, D3DFORMAT pD3DFormat) {
	float fLog = DirectXUtils::log2(iSize)/2.0f;
	return init(pd3dDevice,1<<(int)ceil(fLog),1<<(int)floor(fLog),bConstant,pD3DFormat);
}

/*******************************************************************************************
   Name:    loadShaders
   Purpose: loads all shaders required for this vector, if loaded already use the old one
********************************************************************************************/
void clFragmentVector::loadShaders() {
    

	if (s_iFragmentVectorCount < 1)	{
		if (FAILED(DirectXUtils::checkLoadShader(_T("clVector.fx"), ms_pShaderClVector, m_pd3dDevice, clFrameworkShaderPath, NULL, clClass::ms_clPSProfile))) exit(-1);;
		// precache shader parameters 

		// parameters 
		ms_fMultiply 			= ms_pShaderClVector->GetParameterByName(NULL,"fMultiply");
		ms_fMultiply2 			= ms_pShaderClVector->GetParameterByName(NULL,"fMultiply2");
		ms_fReduceStep 			= ms_pShaderClVector->GetParameterByName(NULL,"fReduceStep");
		ms_f4ReduceStep 		= ms_pShaderClVector->GetParameterByName(NULL,"f4ReduceStep");
		ms_f4TexShift 			= ms_pShaderClVector->GetParameterByName(NULL,"f4TexShift");
		ms_fShift 				= ms_pShaderClVector->GetParameterByName(NULL,"fShift");
		ms_fSize 				= ms_pShaderClVector->GetParameterByName(NULL,"fSize");

		// textures
		ms_tVector 				= ms_pShaderClVector->GetParameterByName(NULL,"tVector");
		ms_tVector2 			= ms_pShaderClVector->GetParameterByName(NULL,"tVector2");
		ms_tLastPass 			= ms_pShaderClVector->GetParameterByName(NULL,"tLastPass");
		ms_tMultiply 			= ms_pShaderClVector->GetParameterByName(NULL,"tMultiply");

		// techniques
		ms_tMultiplyScal						= ms_pShaderClVector->GetTechniqueByName("tMultiplyScal");
		ms_tReduceAddFirst						= ms_pShaderClVector->GetTechniqueByName("tReduceAddFirst");
		ms_tReduceAddRest						= ms_pShaderClVector->GetTechniqueByName("tReduceAddRest");
		ms_tReduceAddLast						= ms_pShaderClVector->GetTechniqueByName("tReduceAddLast");
		ms_tReduceAddRestX						= ms_pShaderClVector->GetTechniqueByName("tReduceAddRestX");
		ms_tReduceAddRestY						= ms_pShaderClVector->GetTechniqueByName("tReduceAddRestY");
		ms_tVectorAdd							= ms_pShaderClVector->GetTechniqueByName("tVectorAdd");
		ms_tVectorMultiply						= ms_pShaderClVector->GetTechniqueByName("tVectorMultiply");
		ms_tVectorMultiplyMatPosOnly			= ms_pShaderClVector->GetTechniqueByName("tVectorMultiplyMatPosOnly");
		ms_tVectorMultiplyMatAllCases			= ms_pShaderClVector->GetTechniqueByName("tVectorMultiplyMatAllCases");
		ms_tVectorMultiplyMatPosOnly_noadd		= ms_pShaderClVector->GetTechniqueByName("tVectorMultiplyMatPosOnly_noadd");
		ms_tVectorMultiplyMatAllCases_noadd		= ms_pShaderClVector->GetTechniqueByName("tVectorMultiplyMatAllCases_noadd");
		ms_tUnpackVector						= ms_pShaderClVector->GetTechniqueByName("tUnpackVector");
		ms_tPackVector							= ms_pShaderClVector->GetTechniqueByName("tPackVector");
	}
}

/*******************************************************************************************
   Name:    init
   Purpose: initializes this vector, should be called explitly only if default constructor
            has been used to create this object
********************************************************************************************/
HRESULT clFragmentVector::init(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY, bool bConstant, D3DFORMAT pD3DFormat) {
	// store values
	m_pd3dDevice		  = pd3dDevice;
	pd3dDevice->GetRenderTarget(0, &m_lpBackBuffer);

	m_memDesc.m_iWidth    = iSizeX;
	m_memDesc.m_iHeight	  = iSizeY;
	m_memDesc.m_d3dFormat = pD3DFormat;

	#ifdef optNVIDIA
		m_memDesc.m_dUsage	  = (bConstant) ? NULL : D3DUSAGE_RENDERTARGET;
	#else
		UNREFERENCED_PARAMETER(bConstant);
		m_memDesc.m_dUsage	  = D3DUSAGE_RENDERTARGET;
	#endif

	// needed for various shaders
	m_vSize	     = D3DXVECTOR4((float)iSizeX,(float)iSizeY,1.0f/(float)iSizeX,1.0f/(float)iSizeY);

	// setup variables
	m_iTempActive = 0;

	// create render target, textures and surface
	m_iMemID = ms_memoryMananger->getTextureTarget(m_memDesc, m_pVectorTexture, m_pVectorTextureSurface);

	for (int i = 0;i<2;i++) {
		m_pTempTexture[i] = NULL;
		m_pTempTextureSurface[i] = NULL;
		m_ID[i] = -1;
	}

	// init static variables
	loadShaders();

	// count the active incarnations of this class
	s_iFragmentVectorCount++;

	return S_OK;
}


/*******************************************************************************************
   Name:    BeginScene
   Purpose: initialazes the scene for rendering i.e. allocating a render to surace,
            retrieving a write surface, calling begin scene on the render to surface
********************************************************************************************/
HRESULT clFragmentVector::BeginScene() {
	HRESULT hr;
	CHECK_HR(m_pd3dDevice->SetRenderTarget(0,getWriteSurface()));
	return S_OK;
}

/*******************************************************************************************
   Name:    EndScene
   Purpose: finalizes a rendering pass i.e. calling end-scene on the render to surface
            freeing the render to surace, swapping the wrte surface and the read surface
********************************************************************************************/
HRESULT clFragmentVector::EndScene() {
	HRESULT hr;
	CHECK_HR(m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer));
	swapSurfaces();
	return S_OK;
}


/*******************************************************************************************
   Name:    clear
   Purpose: sets all vector elements to zero
********************************************************************************************/
void clFragmentVector::clear() {
	m_pd3dDevice->SetRenderTarget(0,m_pVectorTextureSurface);
		m_pd3dDevice->Clear( 0L, NULL, D3DCLEAR_TARGET,0x00000000, 1.0f, 0L );
	m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer);

}

/*******************************************************************************************
   Name:    default destructor
   Purpose: destroyes this object
********************************************************************************************/
clFragmentVector::~clFragmentVector(void)	{

	ms_memoryMananger->releaseTextureTarget(m_iMemID, m_pVectorTexture, m_pVectorTextureSurface);
	SAFE_RELEASE( m_lpBackBuffer );

	// if this is the last vector object alive, destroy static members
	s_iFragmentVectorCount--;
	if (s_iFragmentVectorCount==0) SAFE_RELEASE( ms_pShaderClVector);
}

/*******************************************************************************************
   Name:    reduceInXDirection
   Purpose: reduces only in y-diretion, this method is required with rectangular textures
********************************************************************************************/
void clFragmentVector::reduceInXDirection(int &iDepth, CL_TEX2D_VERTEX *hQuad) {
	/* try to reduce in x-direction
	 precon: FVF is set to CL_TEX2D_VERTEX_FVF
	         because reduceInXDirection is allways called afer reduceAdd */

	while (iDepth*2 < m_memDesc.m_iWidth) {
		splitQuadX(hQuad);

		ms_pShaderClVector->SetTechnique(ms_tReduceAddRestX);
		m_pd3dDevice->SetRenderTarget(0, m_pTempTextureSurface[m_iTempActive]);

			ms_pShaderClVector->SetFloat(ms_fReduceStep, 1.0f/(iDepth*2.0f));
			UINT cPasses;
			ms_pShaderClVector->Begin(&cPasses, 0);
				ms_pShaderClVector->BeginPass(0);
				m_pd3dDevice->DrawPrimitiveUP( D3DPT_TRIANGLESTRIP, 2, hQuad, sizeof( CL_TEX2D_VERTEX ) );
				ms_pShaderClVector->EndPass();
			ms_pShaderClVector->End();
		iDepth *= 2;

		ms_pShaderClVector->SetTexture(ms_tLastPass, m_pTempTexture[m_iTempActive]);
		m_iTempActive = 1-m_iTempActive; // swap temp-surface
	}
}


/*******************************************************************************************
   Name:    reduceInYDirection
   Purpose: reduces only in y-diretion, this method is required with rectangular textures
********************************************************************************************/
void clFragmentVector::reduceInYDirection(int &iDepth, CL_TEX2D_VERTEX *hQuad) {
	/* try to reduce in y-direction
	 precon: FVF is set to CL_TEX2D_VERTEX_FVF
	         because reduceInYDirection is allways called afer reduceAdd */


	while (iDepth*2 < m_memDesc.m_iHeight) {
		splitQuadY(hQuad);

		ms_pShaderClVector->SetTechnique(ms_tReduceAddRestY);
		m_pd3dDevice->SetRenderTarget(0, m_pTempTextureSurface[m_iTempActive]);

			ms_pShaderClVector->SetFloat(ms_fReduceStep, 1.0f/(iDepth*2.0f));
			UINT cPasses;
			ms_pShaderClVector->Begin(&cPasses, 0);
				ms_pShaderClVector->BeginPass(0);
				m_pd3dDevice->DrawPrimitiveUP( D3DPT_TRIANGLESTRIP, 2, hQuad, sizeof( CL_TEX2D_VERTEX ) );
				ms_pShaderClVector->EndPass();
			ms_pShaderClVector->End();

		iDepth *= 2;

		ms_pShaderClVector->SetTexture(ms_tLastPass, m_pTempTexture[m_iTempActive]);
		m_iTempActive = 1-m_iTempActive; // swap temp-surface
	}
}

/*******************************************************************************************
   Name:    combineLastPoints
   Purpose: the last step in the reduction process requires reduction to a specified 1x1 
			texture, because only whole textures can be read back into main mem
********************************************************************************************/
void clFragmentVector::combineLastPointsGeneral(int iDepth, 
										 LPDIRECT3DSURFACE9 pMiniTextureSurface,
										 int iType) {

	CL_TEX2D_VERTEX hQuad[]		= {	{ -1, -1, 0, 0, 0},
									{  1, -1, 0, 0, 0},
									{ -1,  1, 0, 0, 0},
									{  1,  1, 0, 0, 0}};

	ms_pShaderClVector->SetTechnique(ms_tReduceAddLast);

	m_pd3dDevice->SetRenderTarget(0, pMiniTextureSurface);

		// if the vector was not "quadratic" take the aspect ratio into account for the texture coordinates
		D3DXVECTOR4 f4ReduceStep = D3DXVECTOR4((float)MAX(1,(m_memDesc.m_iHeight/m_memDesc.m_iWidth))/(iDepth*2.0f),
			                                   (float)MAX(1,(m_memDesc.m_iWidth/m_memDesc.m_iHeight))/(iDepth*2.0f),0,0);

		ms_pShaderClVector->SetVector(ms_f4ReduceStep, &f4ReduceStep);
		UINT cPasses;
		ms_pShaderClVector->Begin(&cPasses, 0);
			ms_pShaderClVector->BeginPass(iType);
			m_pd3dDevice->DrawPrimitiveUP( D3DPT_TRIANGLESTRIP, 2, hQuad, sizeof( CL_TEX2D_VERTEX ) );
			ms_pShaderClVector->EndPass();
		ms_pShaderClVector->End();
	m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer);
}

/*******************************************************************************************
   Name:    reduceAdd
   Purpose: compute clfResult=sum(this)
********************************************************************************************/
void clFragmentVector::reduceAdd(clFloat *clfResult) {
	m_ID[0] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[0],m_pTempTextureSurface[0]);
	m_ID[1] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[1],m_pTempTextureSurface[1]);

	ms_pShaderClVector->SetTexture(ms_tLastPass, m_pVectorTexture);
	ms_pShaderClVector->SetTechnique(ms_tReduceAddRest);

	int iDepth = reduceAddInternal();
	// write directly to the readSurface to avoid a flipTexture call
	combineLastPoints(iDepth,clfResult->readSurface());

	ms_memoryMananger->releaseTextureTarget(m_ID[0]);
	ms_memoryMananger->releaseTextureTarget(m_ID[1]);
}

/*******************************************************************************************
   Name:    reduceAdd
   Purpose: compute clfResult=sum(this*clvSecond)
********************************************************************************************/
void clFragmentVector::reduceAdd(clFragmentVector* clvSecond, clFloat *clfResult) {
	m_ID[0] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[0],m_pTempTextureSurface[0]);
	m_ID[1] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[1],m_pTempTextureSurface[1]);

	ms_pShaderClVector->SetTexture(ms_tVector,  m_pVectorTexture);
	ms_pShaderClVector->SetTexture(ms_tVector2, clvSecond->m_pVectorTexture);
	ms_pShaderClVector->SetTechnique(ms_tReduceAddFirst);

	int iDepth = reduceAddInternal();
	// write directly to the readSurface to avoid a flipTexture call
	combineLastPoints(iDepth,clfResult->readSurface());

	ms_memoryMananger->releaseTextureTarget(m_ID[0]);
	ms_memoryMananger->releaseTextureTarget(m_ID[1]);
}

/*******************************************************************************************
   Name:    reduceAdd
   Purpose: return sum(this)
********************************************************************************************/
float clFragmentVector::reduceAdd() {
	LPDIRECT3DSURFACE9 pSurface;
	PDIRECT3DTEXTURE9  pTexture;
	int iMiniTexID = ms_memoryMananger->getTextureTarget(clMemDescr(1,1,m_memDesc.m_d3dFormat,D3DUSAGE_RENDERTARGET),pTexture,pSurface);
	m_ID[0] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[0],m_pTempTextureSurface[0]);
	m_ID[1] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[1],m_pTempTextureSurface[1]);

	ms_pShaderClVector->SetTexture(ms_tLastPass, m_pVectorTexture);
	ms_pShaderClVector->SetTechnique(ms_tReduceAddRest);

	int iDepth = reduceAddInternal();
	combineLastPointsGeneral(iDepth,pSurface,0);

	float result = evalLastPoint(pSurface);
	ms_memoryMananger->releaseTextureTarget(m_ID[0]);
	ms_memoryMananger->releaseTextureTarget(m_ID[1]);
	ms_memoryMananger->releaseTextureTarget(iMiniTexID);
	return result;
}

/*******************************************************************************************
   Name:    reduceAdd
   Purpose: return sum(this*clvSecond)
********************************************************************************************/
float clFragmentVector::reduceAdd(clFragmentVector* clvSecond) {
	LPDIRECT3DSURFACE9 pSurface;
	PDIRECT3DTEXTURE9  pTexture;
	int iMiniTexID = ms_memoryMananger->getTextureTarget(clMemDescr(1,1,m_memDesc.m_d3dFormat,D3DUSAGE_RENDERTARGET),pTexture,pSurface);
	m_ID[0] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[0],m_pTempTextureSurface[0]);
	m_ID[1] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[1],m_pTempTextureSurface[1]);

	ms_pShaderClVector->SetTexture(ms_tVector,  m_pVectorTexture);
	ms_pShaderClVector->SetTexture(ms_tVector2, clvSecond->m_pVectorTexture);
	ms_pShaderClVector->SetTechnique(ms_tReduceAddFirst);

	int iDepth = reduceAddInternal();
	combineLastPointsGeneral(iDepth,pSurface,0);

	float result = evalLastPoint(pSurface);
	ms_memoryMananger->releaseTextureTarget(m_ID[0]);
	ms_memoryMananger->releaseTextureTarget(m_ID[1]);
	ms_memoryMananger->releaseTextureTarget(iMiniTexID);
	return result;
}

/*******************************************************************************************
   Name:    splitQuadX
   Purpose: reduce a quad to half of it's size in x-dimension
********************************************************************************************/
void clFragmentVector::splitQuadX(CL_TEX2D_VERTEX *hQuad) {
	hQuad[1].x = (hQuad[1].x+1.0f)/2.0f-1.0f;
	hQuad[3].x = (hQuad[3].x+1.0f)/2.0f-1.0f;

	hQuad[1].tu /= 2.0;
	hQuad[3].tu /= 2.0;
}

/*******************************************************************************************
   Name:    splitQuadY
   Purpose: reduce a quad to half of it's size in y-dimension
********************************************************************************************/
void clFragmentVector::splitQuadY(CL_TEX2D_VERTEX *hQuad) {
	hQuad[2].y = (hQuad[2].y-1.0f)/2.0f+1.0f;
	hQuad[3].y = (hQuad[3].y-1.0f)/2.0f+1.0f;

	hQuad[2].tv /= 2.0;
	hQuad[3].tv /= 2.0;
}

/*******************************************************************************************
   Name:    splitQuad
   Purpose: reduce a quad to half of it's size in both dimensions
********************************************************************************************/
void clFragmentVector::splitQuad(CL_TEX2D_VERTEX *hQuad) {
	splitQuadX(hQuad);
	splitQuadY(hQuad);
}


/*******************************************************************************************
   Name:    reduceAddInternal
   Purpose: called by the reduceAdd methods
********************************************************************************************/
int clFragmentVector::reduceAddInternal() {
	int iDepth = 1;

	// a quad covering the entire viewport
	CL_TEX2D_VERTEX hQuad[4];	memcpy(hQuad,ms_hCoverQuad,sizeof(CL_TEX2D_VERTEX)*4);

	m_pd3dDevice->SetFVF( CL_TEX2D_VERTEX_FVF );

	while (iDepth*2 < m_memDesc.m_iWidth && iDepth*2 < m_memDesc.m_iHeight) {
		splitQuad(hQuad);
		m_pd3dDevice->SetRenderTarget(0, m_pTempTextureSurface[m_iTempActive]);

			ms_pShaderClVector->SetFloat(ms_fReduceStep, 1.0f/(iDepth*2.0f));
			UINT cPasses;
			ms_pShaderClVector->Begin(&cPasses, 0);
				ms_pShaderClVector->BeginPass(0);
				m_pd3dDevice->DrawPrimitiveUP( D3DPT_TRIANGLESTRIP, 2, hQuad, sizeof( CL_TEX2D_VERTEX ) );
				ms_pShaderClVector->EndPass();
			ms_pShaderClVector->End();
		iDepth *= 2;

		ms_pShaderClVector->SetTexture(ms_tLastPass, m_pTempTexture[m_iTempActive]);
		m_iTempActive = 1-m_iTempActive; // swap temp-surface

		if (iDepth == 2) ms_pShaderClVector->SetTechnique(ms_tReduceAddRest);
	}
	
	// further reduce non-quadratic vectors
	if (m_memDesc.m_iWidth != m_memDesc.m_iHeight) {
		reduceInXDirection(iDepth,hQuad);	// try to reduce in x-direction
		reduceInYDirection(iDepth,hQuad);	// try to reduce in y-direction
	}

	m_pd3dDevice->SetRenderTarget(0,m_lpBackBuffer);

	return iDepth;
}


/*******************************************************************************************
   Name:    multiplyScalar
   Purpose: multiply this vector with a scalar (this = this*fScalar)
********************************************************************************************/
void clFragmentVector::multiplyScalar(float fScalar){
	BeginScene();
		ms_pShaderClVector->SetFloat(ms_fMultiply,fScalar);
		ms_pShaderClVector->SetTexture(ms_tVector, m_pVectorTexture);
		ms_pShaderClVector->SetTechnique(ms_tMultiplyScal);

		RenderViewPortCover(ms_pShaderClVector);
	EndScene();
}

LPDIRECT3DSURFACE9 clFragmentVector::getWriteSurface() {
	m_ID[0] = ms_memoryMananger->getTextureTarget(m_memDesc,m_pTempTexture[0],m_pTempTextureSurface[0]);
	return m_pTempTextureSurface[0];
}

LPDIRECT3DSURFACE9 clFragmentVector::getReadSurface() {
	return m_pVectorTextureSurface;
}

void clFragmentVector::swapSurfaces() {
	// swap temp and vector surface and texture
	PDIRECT3DTEXTURE9	swapTex		= m_pVectorTexture;
	LPDIRECT3DSURFACE9	swapSurface = m_pVectorTextureSurface;
	m_pVectorTexture				= m_pTempTexture[0];
	m_pVectorTextureSurface			= m_pTempTextureSurface[0];
	ms_memoryMananger->releaseTextureTarget(m_ID[0],swapTex,swapSurface);
	m_ID[0] = -1;
}

/*******************************************************************************************
   Name:    addVector
   Purpose: adds this vector to another vector storing the result in target 
            vTarget = (this*fScal1)+(vSource*fScal2)
********************************************************************************************/
void clFragmentVector::addVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScal1, float fScal2) {
	vTarget->BeginScene();
		ms_pShaderClVector->SetTechnique(ms_tVectorAdd);
		ms_pShaderClVector->SetFloat(ms_fMultiply,fScal1);
		ms_pShaderClVector->SetFloat(ms_fMultiply2,fScal2);
		ms_pShaderClVector->SetTexture(ms_tVector,   this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tLastPass, vSource->m_pVectorTexture);

		RenderViewPortCover(ms_pShaderClVector);
	vTarget->EndScene();
}

/*******************************************************************************************
   Name:    vectorOp
   Purpose: executes different vector-vector ops depending on eOpType
********************************************************************************************/
void clFragmentVector::vectorOp(CL_enum eOpType, clFragmentVector *clvSource, clFragmentVector *clvTarget, float fScal1, float fScal2) {
	switch (eOpType) {
		case CL_ADD: addVector(clvSource, clvTarget, fScal1, fScal2);
			break;
		case CL_SUB: addVector(clvSource, clvTarget, fScal1, -fScal2);
			break;
	}
}

void clFragmentVector::addVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScal, clFloat *clfScal) {
	vTarget->BeginScene();
		ms_pShaderClVector->SetTechnique(ms_tVectorAdd);

		ms_pShaderClVector->SetFloat(ms_fMultiply,fScal);
		ms_pShaderClVector->SetTexture(ms_tMultiply, clfScal->readTexture());
		ms_pShaderClVector->SetTexture(ms_tVector,   this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tLastPass, vSource->m_pVectorTexture);

		RenderViewPortCover(ms_pShaderClVector,1);
	vTarget->EndScene();
}

void clFragmentVector::subtractVector(clFragmentVector* vSource, clFragmentVector* vTarget, float fScal, clFloat *clfScal) {
	vTarget->BeginScene();
		ms_pShaderClVector->SetTechnique(ms_tVectorAdd);

		ms_pShaderClVector->SetFloat(ms_fMultiply,fScal);
		ms_pShaderClVector->SetTexture(ms_tMultiply, clfScal->readTexture());
		ms_pShaderClVector->SetTexture(ms_tVector,   this->m_pVectorTexture);
		ms_pShaderClVector->SetTexture(ms_tLastPass, vSource->m_pVectorTexture);

		RenderViewPortCover(ms_pShaderClVector,2);
	vTarget->EndScene();
}


/*******************************************************************************************
   Name:    copyVector
   Purpose: dublicate a vector (this.data = clvSource.data)
********************************************************************************************/
void clFragmentVector::copyVector(clFragmentVector *clvSource) {
	BeginScene();
		ms_pShaderClVector->SetFloat(ms_fMultiply,1);
		ms_pShaderClVector->SetTexture(ms_tVector, clvSource->m_pVectorTexture);
		ms_pShaderClVector->SetTechnique(ms_tMultiplyScal);

		RenderViewPortCover(ms_pShaderClVector);
	EndScene();
}

/*******************************************************************************************
   Name:    setData
   Purpose: copy vector to a texture on the GPU
********************************************************************************************/
void clFragmentVector::setData(float* fVectorData) {
	D3DLOCKED_RECT rfTextureLock;
	LPDIRECT3DSURFACE9 pSurf;
	PDIRECT3DTEXTURE9  pTex;
	int id = ms_memoryMananger->getSysmemTexture(m_memDesc,pTex,pSurf);

	pTex->LockRect(0, &rfTextureLock, NULL, 0);					// lock whole region in the vector texture
		#ifdef use16BitPrecOnly
			D3DXFLOAT16* temp = new D3DXFLOAT16[getSize()]; // convert float32 to float16
			for (int i=0;i<getSize();i++) temp[i] = fVectorData[i];
			D3DXFLOAT16* pfTexture = (D3DXFLOAT16*)rfTextureLock.pBits;
			memcpy(pfTexture,temp,getSize()*sizeof(D3DXFLOAT16));
			delete [] temp;
		#else
			float* pfTexture = (float*)rfTextureLock.pBits;
			memcpy(pfTexture,fVectorData,getSize()*sizeof(float));
		#endif
	pTex->UnlockRect(0);
	setData(pSurf);

	ms_memoryMananger->releaseSysmemTexture(id);
}

/*******************************************************************************************
   Name:    getData
   Purpose: read vector data back into main mem
********************************************************************************************/
void clFragmentVector::getData(float* fVectorData) {
	D3DLOCKED_RECT rfTextureLock;
	LPDIRECT3DSURFACE9 pSurf;
	PDIRECT3DTEXTURE9  pTex;
	int id = ms_memoryMananger->getSysmemTexture(m_memDesc,pTex,pSurf);

	// transfer data from GPU to CPU
	getData(pSurf);

	// copy data into user pointer
	pTex->LockRect(0, &rfTextureLock, NULL, D3DLOCK_READONLY);	// lock whole region in the vector texture
		#ifdef use16BitPrecOnly
			D3DXFLOAT16* temp = new D3DXFLOAT16[getSize()]; // convert float32 to float16
			D3DXFLOAT16* pfTexture = (D3DXFLOAT16*)rfTextureLock.pBits;
 			memcpy(temp,pfTexture,getSize()*sizeof(D3DXFLOAT16));
			for (int i=0;i<getSize();i++) fVectorData[i] = temp[i];
			delete [] temp;
		#else
			float* pfTexture = (float*)rfTextureLock.pBits;
 			memcpy(fVectorData,pfTexture,getSize()*sizeof(float));
		#endif
	pTex->UnlockRect(0);

	ms_memoryMananger->releaseSysmemTexture(id);
}

HRESULT clFragmentVector::setData(LPDIRECT3DSURFACE9 m_pSurfSystem) {
	HRESULT hr;
	CHECK_HR(m_pd3dDevice->UpdateSurface(m_pSurfSystem, NULL, m_pVectorTextureSurface, NULL));
	return S_OK;
}

HRESULT clFragmentVector::getData(LPDIRECT3DSURFACE9 m_pSurfSystem) {
	HRESULT hr;
	CHECK_HR(m_pd3dDevice->GetRenderTargetData(m_pVectorTextureSurface, m_pSurfSystem));
	return S_OK;
}

/*******************************************************************************************
   Name:    toString
   Purpose: convert the vector to a string, outputting only the first iOutputCount values
            iOutputCount==0 (default) outputs all values
********************************************************************************************/
TCHAR* clFragmentVector::toString(int iOutputCount) {
	TCHAR* cpResult = new TCHAR[20*getSize()]; // TODO: check if 20 makes sense here
	TCHAR* cpTemp   = new TCHAR[20];
	float* pfData  = new float[getSize()];
	
	getData(pfData);

	cpResult[0] = 0;
	for (int i = 0;i<getSize();i++) {
		if (iOutputCount > 0 && iOutputCount <= i) return cpResult;

		if ((i+1)%m_memDesc.m_iWidth)
			_stprintf(cpTemp,_T("%7.5f\t "),pfData[i]);
		else
			_stprintf(cpTemp,_T("%7.5f\n"),pfData[i]);
		_tcscat(cpResult,cpTemp);
	}

	return cpResult;
}

/*******************************************************************************************
   Name:    toShortString
   Purpose: convert the vector to a string, but display only non zero entries
********************************************************************************************/
TCHAR* clFragmentVector::toShortString(int iOutputCount) {
	TCHAR* cpResult = new TCHAR[20*getSize()]; // TODO: check if 20 makes sense here
	TCHAR* cpTemp   = new TCHAR[20];
	float* pfData  = new float[getSize()];
	int iOutCounter = 0;
	
	getData(pfData);

	cpResult[0] = 0;
	for (int i = 0;i<getSize();i++) {
		if (fabs(pfData[i]) > 0.00001)	{
			_stprintf(cpTemp,_T("%i:%7.5f\t "),i, pfData[i]);
			_tcscat(cpResult,cpTemp);

			iOutCounter++;
			if (iOutputCount > 0 && iOutputCount < iOutCounter) return cpResult;
		}
	}

	return cpResult;
}

