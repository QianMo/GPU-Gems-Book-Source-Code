/************************************************************
 *															*
 * decr     : vertex based matrix class						*
 * version  : 1.01											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 06.10.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#include "clAbstractMatrix.h"
#include "clPackedVector.h"
#include "clUnpackedVector.h"

#include "clVertexMatrix.h"

const DWORD CLVECMATRIXVERTEX::FVF = D3DFVF_XYZ | D3DFVF_TEX6 | D3DFVF_TEXCOORDSIZE2(0) | D3DFVF_TEXCOORDSIZE2(1) | D3DFVF_TEXCOORDSIZE2(2) | D3DFVF_TEXCOORDSIZE2(3) |D3DFVF_TEXCOORDSIZE4(4) | D3DFVF_TEXCOORDSIZE2(5);

LPD3DXEFFECT		clVertexMatrix::ms_pShaderVMatrix=NULL;
int					clVertexMatrix::ms_iVertexMatrixCount=0;
clUnpackedVector*	clVertexMatrix::ms_pUnpackedVectorRes=NULL;
clUnpackedVector*	clVertexMatrix::ms_pUnpackedVectorX=NULL;

D3DXHANDLE			clVertexMatrix::ms_tVector;
D3DXHANDLE			clVertexMatrix::ms_tLastPass;
D3DXHANDLE			clVertexMatrix::ms_tMatrixMultiplyNoLast;
D3DXHANDLE			clVertexMatrix::ms_tMatrixMultiply;

clVertexMatrix::clVertexMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSize) {
	float fLog = DirectXUtils::log2(iSize)/2.0f;
	init(pd3dDevice,1<<(int)ceil(fLog),1<<(int)floor(fLog));
}

clVertexMatrix::clVertexMatrix(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY) {
	init(pd3dDevice, iSizeX, iSizeY);
}

void clVertexMatrix::init(LPDIRECT3DDEVICE9 pd3dDevice, int iSizeX, int iSizeY) {
	// store values
	m_pd3dDevice = pd3dDevice;
	m_iSizeX	 = iSizeX;
	m_iSizeY	 = iSizeY;
	m_ppVertexBuffers = NULL;
	m_iVertexBufferCount = 0;


	// init static variables
	loadShaders();
	if (ms_iVertexMatrixCount < 1)	{
		// nothing yet
	}

	ms_iVertexMatrixCount++;
}

void clVertexMatrix::clPreparePackedUse() {
	ms_pUnpackedVectorX   = new clUnpackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
	ms_pUnpackedVectorRes = new clUnpackedVector(m_pd3dDevice, m_iSizeX, m_iSizeY);
}

/*******************************************************************************************
   Name:    loadShaders
   Purpose: loads all shaders required for this vector, if loaded already use the old one
********************************************************************************************/
void clVertexMatrix::loadShaders() {
	if (ms_iVertexMatrixCount < 1)	{
		if (FAILED(DirectXUtils::checkLoadShader(_T("clMatrix.fx"), ms_pShaderVMatrix, m_pd3dDevice, clFrameworkShaderPath, NULL, clClass::ms_clPSProfile))) exit(-1);

		ms_tVector					= ms_pShaderVMatrix->GetParameterByName(NULL,"tVector");
		ms_tLastPass				= ms_pShaderVMatrix->GetParameterByName(NULL,"tLastPass");

		ms_tMatrixMultiplyNoLast	= ms_pShaderVMatrix->GetTechniqueByName("tMatrixMultiplyNoLast");
		ms_tMatrixMultiply			= ms_pShaderVMatrix->GetTechniqueByName("tMatrixMultiply");
	}
}

clVertexMatrix::~clVertexMatrix(void) {
	deleteVertexBuffers();

	// if this is the last matrix-object alive, destroy static members
	ms_iVertexMatrixCount--;
	if (ms_iVertexMatrixCount==0) 	{
		SAFE_RELEASE( ms_pShaderVMatrix);

		// maybe the unpacked objects do not exist, but SAFE_DELETE checks that
		SAFE_DELETE ( ms_pUnpackedVectorX );
		SAFE_DELETE ( ms_pUnpackedVectorRes );
	}
}

int clVertexMatrix::sortData(clVertexMatrixElement *meData, int iElemCount) {

	clVertexMatrixElement meTemp;
	int i;

	// sort by row
	for (i = 0;i<iElemCount;i++) {
		for (int j = i+1;j<iElemCount;j++) {
			if (meData[j].iY < meData[i].iY) {
				meTemp.iX    = meData[j].iX;meTemp.iY    = meData[j].iY;meTemp.fVal    = meData[j].fVal;
				meData[j].iX = meData[i].iX;meData[j].iY = meData[i].iY;meData[j].fVal = meData[i].fVal;
				meData[i].iX = meTemp.iX;meData[i].iY    = meTemp.iY;meData[i].fVal    = meTemp.fVal;
			}
		}
	}

	// find longest row --> determins the vertex-buffer count
	int iMaxRowLength     = 0;
	int iCurrentRowLength = 1;
	for (i = 1;i<iElemCount;i++) {
		if (meData[i-1].iY != meData[i].iY) {
			if (iMaxRowLength < iCurrentRowLength) iMaxRowLength = iCurrentRowLength;
			iCurrentRowLength = 1;
		} else iCurrentRowLength++;
	}
	if (iMaxRowLength < iCurrentRowLength) iMaxRowLength = iCurrentRowLength;

	return (int)ceil((float)iMaxRowLength/4.0f);
}

int clVertexMatrix::setData(clVertexMatrixElement *meData, int iElemCount) {
	int iVertexBufferCount = sortData(meData,iElemCount);
	setDataSorted(meData,iElemCount,iVertexBufferCount);
	return iVertexBufferCount;
}



void clVertexMatrix::deleteVertexBuffers() {
	if (m_ppVertexBuffers != NULL) {
		for (int i=0;i<m_iVertexBufferCount;i++) SAFE_RELEASE(m_ppVertexBuffers[i]);
		delete [] m_ppVertexBuffers;
		m_ppVertexBuffers = NULL;
		m_iVertexBufferCount = 0;
		delete [] m_iElemCounter;
	}
}

void clVertexMatrix::coordsTo2D(int i1DCoord, float &f2Dx, float &f2Dy) {
	f2Dx = (float)(i1DCoord%m_iSizeX)/(float)m_iSizeX;
	f2Dy = (float)(i1DCoord/m_iSizeX)/(float)m_iSizeY;
}

void clVertexMatrix::coordsToPos(int i1DCoord, float &f2Dx, float &f2Dy) {
	f2Dx = ((float)(i1DCoord%m_iSizeX)/(float)m_iSizeX)*2.0f-1.0f+1.0f/m_iSizeX;
	f2Dy = ((float)(i1DCoord/m_iSizeX)/(float)m_iSizeY)*2.0f-1.0f+1.0f/m_iSizeY;
}

void clVertexMatrix::insertElement(clVertexMatrixElement &meData, CLVECMATRIXVERTEX &vbData, int iRGBAIndex) {
	switch (iRGBAIndex) {
		case 0 : coordsTo2D(meData.iX,vbData.tu_0,vbData.tv_0);
				 vbData.val0 = meData.fVal;
				 coordsToPos(meData.iY,vbData.x,vbData.y); vbData.z = 1;
			     coordsTo2D(meData.iY,vbData.posX,vbData.posY);
				 break;
		case 1 : coordsTo2D(meData.iX,vbData.tu_1,vbData.tv_1);
				 vbData.val1 = meData.fVal;
				 break;
		case 2 : coordsTo2D(meData.iX,vbData.tu_2,vbData.tv_2);
				 vbData.val2 = meData.fVal;
				 break;
		case 3 : coordsTo2D(meData.iX,vbData.tu_3,vbData.tv_3);
				 vbData.val3 = meData.fVal;
				 break;
		default : throw "invalid index in insertElement"; 
	}
}

void clVertexMatrix::completeLastRow(CLVECMATRIXVERTEX &vbData, int iRGBAIndex) {
	switch (iRGBAIndex) {
		case 1 : vbData.tu_1 = 0;
			     vbData.tv_1 = 0;
				 vbData.val1 = 0;
		case 2 : vbData.tu_2 = 0;
			     vbData.tv_2 = 0;
				 vbData.val2 = 0;
		case 3 : vbData.tu_3 = 0;
			     vbData.tv_3 = 0;
				 vbData.val3 = 0;
	}
}

void clVertexMatrix::setDataSorted(clVertexMatrixElement *meData, int iElemCount,int iVertexBufferCount) {

	int i;
	deleteVertexBuffers();
	if (iElemCount < 1) return;

	// create the vertex buffers
	m_iVertexBufferCount = iVertexBufferCount;
	m_ppVertexBuffers = new LPDIRECT3DVERTEXBUFFER9[m_iVertexBufferCount];
	m_iElemCounter    = new int[m_iVertexBufferCount];

	// create the intermediate buffers
	CLVECMATRIXVERTEX **pVbData = new CLVECMATRIXVERTEX*[m_iVertexBufferCount];
	for (i = 0;i<iVertexBufferCount;i++) {
		pVbData[i]      = new CLVECMATRIXVERTEX[iElemCount];
		m_iElemCounter[i] = 0;
	}

	// fill the intermediate buffers
	int iLastRow    = meData[0].iY;
	int iInRowCount = 0;
	for (i = 0;i<iElemCount;i++) {
		if (iLastRow == meData[i].iY) {
			insertElement(meData[i],pVbData[iInRowCount/4][m_iElemCounter[iInRowCount/4]/4],iInRowCount%4);
			m_iElemCounter[iInRowCount/4]++;
			iInRowCount++;
		} else {
			completeLastRow(pVbData[iInRowCount/4][m_iElemCounter[iInRowCount/4]/4],iInRowCount%4);
			m_iElemCounter[iInRowCount/4] += 3-(iInRowCount-1)%4;
			iInRowCount = 0;
			insertElement(meData[i],pVbData[iInRowCount/4][m_iElemCounter[iInRowCount/4]/4],iInRowCount%4);
			m_iElemCounter[iInRowCount/4]++;
			iLastRow = meData[i].iY;
			iInRowCount++;
		}
	}
	completeLastRow(pVbData[iInRowCount/4][m_iElemCounter[iInRowCount/4]/4],iInRowCount%4);
	m_iElemCounter[iInRowCount/4] += 3-(iInRowCount-1)%4;


	// copy vertices into vertexbuffers and rescale counters
	for (i = 0;i<iVertexBufferCount;i++) {
		m_iElemCounter[i] = m_iElemCounter[i]/4;
		DirectXUtils::createFilledVertexBuffer(m_pd3dDevice,pVbData[i],sizeof(CLVECMATRIXVERTEX)*(m_iElemCounter[i]),
			                                   CLVECMATRIXVERTEX::FVF,D3DPOOL_DEFAULT,m_ppVertexBuffers[i]);
	}

	// delete the intermediate vertexData
	for (i = 0;i<iVertexBufferCount;i++) delete [] pVbData[i];
	delete [] pVbData;
}


/*******************************************************************************************
   Name:    matrixVectorOp 
   Purpose: computes matrix vector operations of the kind:
	          clvResult = this*clvX OP clvY
	          where OP (eOpType) can be one of the following: CL_NULL,CL_ADD,CL_SUB,CL_MULT
********************************************************************************************/
void clVertexMatrix::matrixVectorOp(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) {

	clUnpackedVector* pUnpackedVectorRes;
	clUnpackedVector* pUnpackedVectorX;

	if (typeid(clvResult)==typeid(clPackedVector)) {
		if (clvX == NULL) clPreparePackedUse();
		(static_cast<clPackedVector*>(clvX))->unpack(ms_pUnpackedVectorX);

		pUnpackedVectorRes = ms_pUnpackedVectorRes;
		pUnpackedVectorX   = ms_pUnpackedVectorX;
	} else {
		pUnpackedVectorRes = (static_cast<clUnpackedVector*>(clvResult));
		pUnpackedVectorX   = (static_cast<clUnpackedVector*>(clvX));
	}
	pUnpackedVectorRes->clear();

	// compute Ax first
	for (int i=0;i<m_iVertexBufferCount;i++) {
		pUnpackedVectorRes->BeginScene();
			ms_pShaderVMatrix->SetTexture(ms_tVector, pUnpackedVectorX->m_pVectorTexture);

			if (i==0) {
				ms_pShaderVMatrix->SetTechnique(ms_tMatrixMultiplyNoLast);
			} else {
				ms_pShaderVMatrix->SetTechnique(ms_tMatrixMultiply);
				ms_pShaderVMatrix->SetTexture(ms_tLastPass, pUnpackedVectorRes->m_pVectorTexture);
			}

			UINT cPasses;
			ms_pShaderVMatrix->Begin(&cPasses, 0);
				ms_pShaderVMatrix->BeginPass(0);
				m_pd3dDevice->SetStreamSource( 0, m_ppVertexBuffers[i], 0, sizeof(CLVECMATRIXVERTEX) );
				m_pd3dDevice->SetFVF( CLVECMATRIXVERTEX::FVF );
				m_pd3dDevice->DrawPrimitive(D3DPT_POINTLIST,0,m_iElemCounter[i]);
				ms_pShaderVMatrix->EndPass();
			ms_pShaderVMatrix->End();
		pUnpackedVectorRes->EndScene();
	}

	if (typeid(clvResult)==typeid(clPackedVector)) {
		(static_cast<clPackedVector*>(clvResult))->repack(pUnpackedVectorRes);
	}

	// now compute OP y
	if (clvY != NULL) clvResult->vectorOp(eOpType,clvY,clvResult);
}

/*******************************************************************************************
   Name:    matrixVectorOpAdd 
   Purpose: computes matrix vector operations of the kind:
	          clvResult = clvResult + this*clvX OP clvY
	          where OP (eOpType) can be one of the following: CL_NULL,CL_ADD,CL_SUB,CL_MULT
********************************************************************************************/
void clVertexMatrix::matrixVectorOpAdd(CL_enum eOpType, clFragmentVector* clvX, clFragmentVector* clvY, clFragmentVector *clvResult) {
	clUnpackedVector* pUnpackedVectorRes;
	clUnpackedVector* pUnpackedVectorX;

	if (typeid(clvResult)==typeid(clPackedVector)) {
		if (clvX == NULL) clPreparePackedUse();
		(static_cast<clPackedVector*>(clvResult))->unpack(ms_pUnpackedVectorRes);
		(static_cast<clPackedVector*>(clvX))->unpack(ms_pUnpackedVectorX);
		pUnpackedVectorRes = ms_pUnpackedVectorRes;
		pUnpackedVectorX   = ms_pUnpackedVectorX;
	} else {
		pUnpackedVectorRes = (static_cast<clUnpackedVector*>(clvResult));
		pUnpackedVectorX   = (static_cast<clUnpackedVector*>(clvX));
	}

	// compute Ax first
	for (int i=0;i<m_iVertexBufferCount;i++) {
		pUnpackedVectorRes->BeginScene();
			ms_pShaderVMatrix->SetTexture(ms_tVector,		  pUnpackedVectorX->m_pVectorTexture);
			ms_pShaderVMatrix->SetTechnique(ms_tMatrixMultiply);
			ms_pShaderVMatrix->SetTexture(ms_tLastPass, pUnpackedVectorRes->m_pVectorTexture);

			UINT cPasses;
			ms_pShaderVMatrix->Begin(&cPasses, 0);
				ms_pShaderVMatrix->BeginPass(0);
				m_pd3dDevice->SetStreamSource( 0, m_ppVertexBuffers[i], 0, sizeof(CLVECMATRIXVERTEX) );
				m_pd3dDevice->SetFVF( CLVECMATRIXVERTEX::FVF );
				m_pd3dDevice->DrawPrimitive(D3DPT_POINTLIST,0,m_iElemCounter[i]);
				ms_pShaderVMatrix->EndPass();
			ms_pShaderVMatrix->End();
		pUnpackedVectorRes->EndScene( );
	}

	if (typeid(clvResult)==typeid(clPackedVector)) {
		(static_cast<clPackedVector*>(clvResult))->repack(pUnpackedVectorRes);
	}

	// now compute OP y
	if (clvY != NULL) clvResult->vectorOp(eOpType,clvY,clvResult);
}
