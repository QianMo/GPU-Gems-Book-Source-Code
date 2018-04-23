/************************************************************
 *															*
 * decr     : memory mananger for the cl framework			*
 * version  : 1.0											*
 * author   : Jens Krüger									*
 * date     : 29.02.2004									*
 * modified	: 29.02.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

#include "clMemMan.h"


clMemMan::clMemMan(LPDIRECT3DDEVICE9& pDevice) {
	m_pd3dDevice = pDevice;

	m_RenderSurfaceList.reserve(5);
	m_TextureTargetList.reserve(5);
	m_DSSurfList.reserve(5);
	m_SysTextureList.reserve(5);
}

clMemMan::~clMemMan(void) {
	unsigned int i;

	for (i = 0; i<m_RenderSurfaceList.size();++i) {
		#ifdef _DEBUG
			if (m_RenderSurfaceList[i].lock) MessageBox(NULL,_T("unfreed render-to surface"),_T("clMemMan Debug warning"),MB_OK);
		#endif
		m_RenderSurfaceList[i].ReleaseData();
	}
	m_RenderSurfaceList.clear();

	for (i = 0; i<m_TextureTargetList.size();++i) {
		#ifdef _DEBUG
			if (m_TextureTargetList[i].lock) MessageBox(NULL,_T("unfreed target texture"),_T("clMemMan Debug warning"),MB_OK);
		#endif
		m_TextureTargetList[i].ReleaseData();
	}
	m_TextureTargetList.clear();

	for (i = 0; i<m_DSSurfList.size();++i) {
		#ifdef _DEBUG
			if (m_DSSurfList[i].lock) MessageBox(NULL,_T("unfreed depth stencil surface"),_T("clMemMan Debug warning"),MB_OK);
		#endif
		m_DSSurfList[i].ReleaseData();
	}
	m_DSSurfList.clear();

	for (i = 0; i<m_SysTextureList.size();++i) {
		#ifdef _DEBUG
			if (m_SysTextureList[i].lock) MessageBox(NULL,_T("unfreed system texture"),_T("clMemMan Debug warning"),MB_OK);
		#endif
		m_SysTextureList[i].ReleaseData();
	}
	m_SysTextureList.clear();
}

// ***********************************
// Get/Release Routines
// ***********************************
int clMemMan::getDSSurface(clMemDescr memDesc, LPDIRECT3DSURFACE9 &pDSSurface){
	int ID = findFirstDSSurface(memDesc);
	if (ID < 0) ID = genNewDSSurface(memDesc);

	pDSSurface = m_DSSurfList[ID].pDepthStencilSurface;
	m_DSSurfList[ID].lock = true;
	return ID;
}

void clMemMan::releaseDSSurface(int ID) {
	m_DSSurfList[ID].lock = false;
}

int clMemMan::getRenderSurface(clMemDescr memDesc, LPD3DXRENDERTOSURFACE &pRenderToSurface){
	int ID = findFirstRenderSurface(memDesc);
	if (ID < 0) ID = genNewRenderSurface(memDesc);

	pRenderToSurface			 = m_RenderSurfaceList[ID].pRenderToSurface;
	m_RenderSurfaceList[ID].lock = true;
	return ID;
}

void clMemMan::releaseRenderSurface(int ID) {
	m_RenderSurfaceList[ID].lock = false;
}

int clMemMan::getTextureTarget(clMemDescr memDesc, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface) {
	int ID = findFirstTextureTarget(memDesc);
	if (ID < 0) ID = genNewTextureTarget(memDesc);

	pTexture			   = m_TextureTargetList[ID].pTexture;
	pTextureSurface		   = m_TextureTargetList[ID].pTextureSurface;
	m_TextureTargetList[ID].lock = true;
	return ID;
}

void clMemMan::releaseTextureTarget(int ID) {
	m_TextureTargetList[ID].lock = false;
}

void clMemMan::releaseTextureTarget(int ID, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface) {
	m_TextureTargetList[ID].pTexture		  = pTexture;
	m_TextureTargetList[ID].pTextureSurface = pTextureSurface;
	m_TextureTargetList[ID].lock			  = false;
}

int clMemMan::getSysmemTexture(clMemDescr memDesc, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface) {
	int ID = findFirstSysmemTexture(memDesc);
	if (ID < 0) ID = genNewSysmemTexture(memDesc);

	pTexture			      = m_SysTextureList[ID].pTexture;
	pTextureSurface		      = m_SysTextureList[ID].pTextureSurface;
	m_SysTextureList[ID].lock = true;
	return ID;
}

void clMemMan::releaseSysmemTexture(int ID) {
	m_SysTextureList[ID].lock = false;
}

// ***********************************
// Search Routines
// ***********************************
int clMemMan::findFirstDSSurface(clMemDescr memDesc) {
	unsigned int i;

	for (i = 0; i<m_DSSurfList.size();++i)
		if ( m_DSSurfList[i].desc == memDesc && !m_DSSurfList[i].lock) break;

	#ifdef _DEBUG
		if (i < m_DSSurfList.size()) 
			return i; 
		else {
			OutputDebugString(_T("clMemMan: (INFO) :DSSurface created on the fly, on some systems that may reduce performance ("));
			DirectXUtils::OutputDebugInt(memDesc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(memDesc.m_iHeight);
			OutputDebugString(_T(")\n"));
			return -1;
		}
	#else
		return (i < m_DSSurfList.size()) ? i : -1;
	#endif
}

int clMemMan::findFirstRenderSurface(clMemDescr memDesc) {
	unsigned int i;

	for (i = 0; i<m_RenderSurfaceList.size();++i)
		if ( m_RenderSurfaceList[i].desc == memDesc &&
			!m_RenderSurfaceList[i].lock) break;

	#ifdef _DEBUG
		if (i < m_RenderSurfaceList.size()) 
			return i; 
		else {
			OutputDebugString(_T("clMemMan: (INFO) :RenderSurface created on the fly, on some systems that may reduce performance ("));
			DirectXUtils::OutputDebugInt(memDesc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(memDesc.m_iHeight);
			OutputDebugString(_T(")\n"));
			return -1;
		}
	#else
		return (i < m_RenderSurfaceList.size()) ? i : -1;
	#endif
}

int clMemMan::findFirstTextureTarget(clMemDescr memDesc) {
	unsigned int i;

	for (i = 0; i<m_TextureTargetList.size();++i)
		if ( m_TextureTargetList[i].desc == memDesc &&
			!m_TextureTargetList[i].lock) break;

	#ifdef _DEBUG
		if (i < m_TextureTargetList.size()) 
			return i; 
		else {
			OutputDebugString(_T("clMemMan: (INFO) :TextureTarget created on the fly, on some systems that may reduce performance ("));
			DirectXUtils::OutputDebugInt(memDesc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(memDesc.m_iHeight);
			OutputDebugString(_T(")\n"));
			return -1;
		}
	#else
		return (i < m_TextureTargetList.size()) ? i : -1;
	#endif
}


int clMemMan::findFirstSysmemTexture(clMemDescr memDesc) {
	unsigned int i;

	for (i = 0; i<m_SysTextureList.size();++i)
		if ( m_SysTextureList[i].desc == memDesc &&
			!m_SysTextureList[i].lock) break;

	return (i < m_SysTextureList.size()) ? i : -1;
}

// ***********************************
// Generate Routines
// ***********************************
int clMemMan::genNewDSSurface(clMemDescr memDesc) {
	int index = preOrderDSSurface(memDesc);
	m_DSSurfList[index].CreateData(m_pd3dDevice);
	return index;
}

int clMemMan::genNewRenderSurface(clMemDescr memDesc) {
	int index = preOrderRenderSurface(memDesc);
	m_RenderSurfaceList[index].CreateData(m_pd3dDevice);
	return index;
}

int clMemMan::genNewTextureTarget(clMemDescr memDesc) {
	int index = preOrderTextureTarget(memDesc);
	m_TextureTargetList[index].CreateData(m_pd3dDevice);
	return index;
}

int clMemMan::genNewSysmemTexture(clMemDescr memDesc) {
	int index = preOrderSysmemTexture(memDesc);
	m_SysTextureList[index].CreateData(m_pd3dDevice);
	return index;
}

// ***********************************
// Pre-Order Routines
// ***********************************
void clMemMan::preOrderDSSurface(clMemDescr memDesc, int iCount) {
	for(int i = 0;i<iCount;i++) preOrderDSSurface(memDesc);
}

void clMemMan::preOrderRenderSurface(clMemDescr memDesc, int iCount) {
	for(int i = 0;i<iCount;i++) preOrderRenderSurface(memDesc);
}

void clMemMan::preOrderTextureTarget(clMemDescr memDesc, int iCount) {
	for(int i = 0;i<iCount;i++) preOrderTextureTarget(memDesc);
}

void clMemMan::preOrderSysmemTexture(clMemDescr memDesc, int iCount) {
	for(int i = 0;i<iCount;i++) preOrderSysmemTexture(memDesc);
}

int clMemMan::preOrderDSSurface(clMemDescr memDesc) {
	DSSurfVecElem newElem(memDesc);
	m_DSSurfList.push_back(newElem);
	return int(m_DSSurfList.size())-1;
}

int clMemMan::preOrderRenderSurface(clMemDescr memDesc) {
	RenderSurfVecElem newElem(memDesc);
	m_RenderSurfaceList.push_back(newElem);
	return int(m_RenderSurfaceList.size())-1;
}

int clMemMan::preOrderTextureTarget(clMemDescr memDesc) {
	TexTargVecElem newElem(memDesc);
	m_TextureTargetList.push_back(newElem);
	return int(m_TextureTargetList.size())-1;
}

int clMemMan::preOrderSysmemTexture(clMemDescr memDesc) {
	SysmemTexVecElem newElem(memDesc);
	m_SysTextureList.push_back(newElem);
	return int(m_SysTextureList.size())-1;
}

void clMemMan::sortOrders(bool bAlternate) {
	TexTargVecElem tmp;

	unsigned int iFirstIndex = (unsigned int)m_TextureTargetList.size();
	while (m_TextureTargetList[iFirstIndex-1].creationPending() && iFirstIndex > 0) iFirstIndex--;

	// sort lists by stride for the NVIDIA (bAlternate=false), mix them up for the ATI  (bAlternate=true)
	if (bAlternate) {
		for (unsigned int i = iFirstIndex; i<m_TextureTargetList.size()-1;++i) {		
			for (unsigned int j = i; j<m_TextureTargetList.size();++j) {
				if ((i%2==0 && m_TextureTargetList[i].desc.m_iWidth < m_TextureTargetList[j].desc.m_iWidth) ||
					(i%2!=0 && m_TextureTargetList[i].desc.m_iWidth > m_TextureTargetList[j].desc.m_iWidth) ) {
					tmp = m_TextureTargetList[i];
					m_TextureTargetList[i] = m_TextureTargetList[j];
					m_TextureTargetList[j] = tmp;
				}
			}
		}
	} else {
		for (unsigned int i = iFirstIndex; i<m_TextureTargetList.size()-1;++i) {		
			for (unsigned int j = i; j<m_TextureTargetList.size();++j) {
				if (m_TextureTargetList[i].desc.m_iWidth < m_TextureTargetList[j].desc.m_iWidth) {
					tmp = m_TextureTargetList[i];
					m_TextureTargetList[i] = m_TextureTargetList[j];
					m_TextureTargetList[j] = tmp;
				}
			}
		}
	}
}

void clMemMan::createOrders(bool bAlternate) {
	unsigned int i;
	
	sortOrders(bAlternate);

	// create orders
	for (i = 0; i<m_DSSurfList.size();++i)			if (m_DSSurfList[i].creationPending())		  m_DSSurfList[i].CreateData(m_pd3dDevice);
	for (i = 0; i<m_RenderSurfaceList.size();++i)	if (m_RenderSurfaceList[i].creationPending()) m_RenderSurfaceList[i].CreateData(m_pd3dDevice);
	for (i = 0; i<m_TextureTargetList.size();++i)	if (m_TextureTargetList[i].creationPending()) m_TextureTargetList[i].CreateData(m_pd3dDevice);
	for (i = 0; i<m_SysTextureList.size();++i)		if (m_SysTextureList[i].creationPending())	  m_SysTextureList[i].CreateData(m_pd3dDevice);
}
