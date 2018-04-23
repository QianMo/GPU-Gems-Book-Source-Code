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

#pragma once

// The debugger can't handle symbols more than 255 characters long.
// STL often creates symbols longer than that.
// When symbols are longer than 255 characters, the warning is disabled.
#pragma warning(disable:4786)

// if verbose is defined messages are generated on every texture generation
//#define VERBOSE 1

#include <tools/directXUtils.h>
#include <vector>

class clMemDescr {
	public :
		int m_iWidth;
		int m_iHeight;
		D3DFORMAT m_d3dFormat;
		DWORD m_dUsage;

		clMemDescr() : m_iWidth(0), m_iHeight(0), m_d3dFormat(D3DFMT_UNKNOWN), m_dUsage(NULL) {};
		clMemDescr(int iWidth, int iHeight,	D3DFORMAT d3dFormat, DWORD dUsage) : m_iWidth(iWidth), m_iHeight(iHeight), m_d3dFormat(d3dFormat), m_dUsage(dUsage) {}
		clMemDescr(const clMemDescr* other) : m_iWidth(other->m_iWidth), m_iHeight(other->m_iHeight), m_d3dFormat(other->m_d3dFormat), m_dUsage(other->m_dUsage) {}
		clMemDescr(const PDIRECT3DTEXTURE9 texture) {
			D3DSURFACE_DESC pDesc;
			texture->GetLevelDesc(0,&pDesc);
			m_iWidth	= pDesc.Width;
			m_iHeight	= pDesc.Height;
			m_d3dFormat	= pDesc.Format;
			m_dUsage	= pDesc.Usage;
		};

		bool operator == ( const clMemDescr& other ) const {
			return (other.m_iWidth==m_iWidth && 
				    other.m_iHeight==m_iHeight && 
					other.m_d3dFormat==m_d3dFormat &&
					other.m_dUsage == m_dUsage); 
		}
};

class VecElem {
public :
	clMemDescr desc;
	bool lock;	
};

class DSSurfVecElem : public VecElem {
public:
	DSSurfVecElem()						{ pDepthStencilSurface = NULL; lock=false; }
	DSSurfVecElem(clMemDescr d)			{ pDepthStencilSurface = NULL; lock=false; desc=d;}
	DSSurfVecElem(clMemDescr d, bool l)	{ pDepthStencilSurface = NULL; lock=l; desc=d;}
	DSSurfVecElem(const DSSurfVecElem &other) { desc = other.desc; lock=other.lock; pDepthStencilSurface=other.pDepthStencilSurface;}
	void CreateData(LPDIRECT3DDEVICE9 pd3dDevice) {
		SAFE_RELEASE(pDepthStencilSurface);
		pd3dDevice->CreateDepthStencilSurface(desc.m_iWidth,desc.m_iHeight,desc.m_d3dFormat,D3DMULTISAMPLE_NONE,0,TRUE,&pDepthStencilSurface,NULL);
		#ifdef VERBOSE
			OutputDebugString(_T("Created DepthStencilSurface "));
			DirectXUtils::OutputDebugInt(desc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(desc.m_iHeight);
			OutputDebugString(_T("\n"));
		#endif
	}
	void ReleaseData() {
		SAFE_RELEASE(pDepthStencilSurface);
	}
	bool creationPending() {return pDepthStencilSurface == NULL;}

	LPDIRECT3DSURFACE9 pDepthStencilSurface;
};


class RenderSurfVecElem : public VecElem {
public:
	RenderSurfVecElem()						{ pRenderToSurface=NULL; lock=false; }
	RenderSurfVecElem(clMemDescr d)			{ pRenderToSurface=NULL; lock=false; desc=d; }
	RenderSurfVecElem(clMemDescr d, bool l)	{ pRenderToSurface=NULL; lock=l; desc=d; }
	RenderSurfVecElem(const RenderSurfVecElem &other) { desc=other.desc; lock=other.lock; pRenderToSurface=other.pRenderToSurface;}
	void CreateData(LPDIRECT3DDEVICE9 pd3dDevice) {
		SAFE_RELEASE(pRenderToSurface);
		D3DXCreateRenderToSurface(pd3dDevice,desc.m_iWidth,desc.m_iHeight,desc.m_d3dFormat,FALSE,D3DFMT_UNKNOWN,&pRenderToSurface );
		#ifdef VERBOSE
			OutputDebugString(_T("Created RenderToSurface "));
			DirectXUtils::OutputDebugInt(desc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(desc.m_iHeight);
			OutputDebugString(_T("\n"));
		#endif
	}
	void ReleaseData() {
		SAFE_RELEASE(pRenderToSurface);
	}
	bool creationPending() {return pRenderToSurface == NULL;}

	LPD3DXRENDERTOSURFACE pRenderToSurface;
};

class TexTargVecElem : public VecElem {
public:
	TexTargVecElem()						{ pTexture=NULL; pTextureSurface=NULL; lock=false; }
	TexTargVecElem(clMemDescr d)			{ pTexture=NULL; pTextureSurface=NULL; lock=false; desc=d; }
	TexTargVecElem(clMemDescr d, bool l)	{ pTexture=NULL; pTextureSurface=NULL; lock=l; desc=d; }
	TexTargVecElem(const TexTargVecElem &other) { desc=other.desc; lock=other.lock; pTexture=other.pTexture; pTextureSurface=other.pTextureSurface; }
	void CreateData(LPDIRECT3DDEVICE9 pd3dDevice) {
		SAFE_RELEASE(pTexture);
		SAFE_RELEASE(pTextureSurface);

		pd3dDevice->CreateTexture(desc.m_iWidth, desc.m_iHeight, 1, desc.m_dUsage, desc.m_d3dFormat, D3DPOOL_DEFAULT, &pTexture, 0);
		pTexture->GetSurfaceLevel( 0, &pTextureSurface );

		#ifdef VERBOSE
			if (desc.m_dUsage == D3DUSAGE_RENDERTARGET)
				OutputDebugString(_T("Created RenderTexture "));
			else
				OutputDebugString(_T("Created Texture "));

			DirectXUtils::OutputDebugInt(desc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(desc.m_iHeight);
			OutputDebugString(_T("\n"));
		#endif
	}
	void ReleaseData() {
		SAFE_RELEASE(pTexture);
		SAFE_RELEASE(pTextureSurface);
	}
	bool creationPending() {return pTexture == NULL;}

	PDIRECT3DTEXTURE9 pTexture;
	LPDIRECT3DSURFACE9 pTextureSurface;
};

class SysmemTexVecElem : public VecElem {
public:
	SysmemTexVecElem()						{ pTexture=NULL; pTextureSurface=NULL; lock=false; }
	SysmemTexVecElem(clMemDescr d)			{ pTexture=NULL; pTextureSurface=NULL; lock=false; desc=d; }
	SysmemTexVecElem(clMemDescr d, bool l)	{ pTexture=NULL; pTextureSurface=NULL; lock=l; desc=d; }
	SysmemTexVecElem(const TexTargVecElem &other) { desc=other.desc; lock=other.lock; pTexture=other.pTexture; pTextureSurface=other.pTextureSurface; }
	void CreateData(LPDIRECT3DDEVICE9 pd3dDevice) {
		SAFE_RELEASE(pTexture);
		SAFE_RELEASE(pTextureSurface);
		pd3dDevice->CreateTexture(desc.m_iWidth, desc.m_iHeight, 1, 0, desc.m_d3dFormat, D3DPOOL_SYSTEMMEM, &pTexture,0);
		pTexture->GetSurfaceLevel(0, &pTextureSurface);
		#ifdef VERBOSE
			OutputDebugString(_T("Created Sysmem texture "));
			DirectXUtils::OutputDebugInt(desc.m_iWidth);
			OutputDebugString(_T(" x "));
			DirectXUtils::OutputDebugInt(desc.m_iHeight);
			OutputDebugString(_T("\n"));
		#endif
	}
	void ReleaseData() {
		SAFE_RELEASE(pTexture);
		SAFE_RELEASE(pTextureSurface);
	}
	bool creationPending() {return pTexture == NULL;}

	PDIRECT3DTEXTURE9 pTexture;
	LPDIRECT3DSURFACE9 pTextureSurface;
};

typedef std::vector<RenderSurfVecElem> RENDERSURFVEC;
typedef std::vector<TexTargVecElem> TEXTARGVEC;
typedef std::vector<DSSurfVecElem> DSSURFVEC;
typedef std::vector<SysmemTexVecElem> SYSTEXVEC;

class clMemMan {
	public:
		LPDIRECT3DDEVICE9 m_pd3dDevice;

		clMemMan(LPDIRECT3DDEVICE9& pDevice);
		virtual ~clMemMan(void);

		int getDSSurface(clMemDescr memDesc, LPDIRECT3DSURFACE9 &pDSSurface);
		void releaseDSSurface(int ID);

		int getRenderSurface(clMemDescr memDesc, LPD3DXRENDERTOSURFACE &pRenderToSurface); 
		void releaseRenderSurface(int ID);

		int getTextureTarget(clMemDescr memDesc, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface); 
		void releaseTextureTarget(int ID);
		void releaseTextureTarget(int ID, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface);

		int getSysmemTexture(clMemDescr memDesc, PDIRECT3DTEXTURE9 &pTexture, LPDIRECT3DSURFACE9 &pTextureSurface);
		void releaseSysmemTexture(int ID);


		int preOrderDSSurface(clMemDescr memDesc);
		void preOrderDSSurface(clMemDescr memDesc, int iCount);
		int preOrderRenderSurface(clMemDescr memDesc);
		void preOrderRenderSurface(clMemDescr memDesc, int iCount);
		int preOrderTextureTarget(clMemDescr memDesc);
		void preOrderTextureTarget(clMemDescr memDesc, int iCount);
		int preOrderSysmemTexture(clMemDescr memDesc);
		void preOrderSysmemTexture(clMemDescr memDesc, int iCount);
		void createOrders(bool bAlternate);

	protected:
		RENDERSURFVEC m_RenderSurfaceList;
		TEXTARGVEC	  m_TextureTargetList;
		DSSURFVEC	  m_DSSurfList;
		SYSTEXVEC	  m_SysTextureList;

		void sortOrders(bool bAlternate);

		int findFirstDSSurface(clMemDescr memDesc);
		int genNewDSSurface(clMemDescr memDesc);
		int findFirstRenderSurface(clMemDescr memDesc);
		int genNewRenderSurface(clMemDescr memDesc);
		int findFirstTextureTarget(clMemDescr memDesc);
		int genNewTextureTarget(clMemDescr memDesc);
		int findFirstSysmemTexture(clMemDescr memDesc);
		int genNewSysmemTexture(clMemDescr memDesc);
};