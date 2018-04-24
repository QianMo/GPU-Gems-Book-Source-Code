#ifndef __UI_DX9_H
#define __UI_DX9_H

#include "Cfg.h"
#include "Ui.h"

class UiDx9 : public Ui
{
public:
	void OnRender(D3DXMATRIX const& view, D3DXMATRIX const& proj, D3DXVECTOR3 const& eyePt, float fElapsedTime);

	HRESULT OnCreateDevice(IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc);
	HRESULT OnResetDevice(const D3DSURFACE_DESC* pBackBufferSurfaceDesc);
	void OnLostDevice();
	void OnDestroyDevice();
};

#endif