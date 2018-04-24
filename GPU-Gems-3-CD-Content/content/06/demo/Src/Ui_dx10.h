#ifndef __UI_DX9_H
#define __UI_DX9_H

#include "Cfg.h"
#include "Ui.h"

class UiDx10 : public Ui
{
public:
	void OnRender(D3DXMATRIX const& view, D3DXMATRIX const& proj, D3DXVECTOR3 const& eyePt, float fElapsedTime);

	HRESULT OnD3D10CreateDevice(ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc);
	HRESULT OnD3D10ResizedSwapChain(ID3D10Device* pd3dDevice,const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc);
	void OnD3D10ReleasingSwapChain();
	void OnD3D10DestroyDevice();
};

#endif