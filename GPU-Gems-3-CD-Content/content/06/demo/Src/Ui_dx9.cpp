#include "DXUT.h"

#include "Ui_dx9.h"
#include "Platform.h"

void UiDx9::OnRender(D3DXMATRIX const& view, D3DXMATRIX const& proj, D3DXVECTOR3 const& eyePt, float fElapsedTime)
{
	HRESULT hr;

    // Render the light arrow so the user can visually see the light dir
    D3DXCOLOR arrowColor = D3DXCOLOR(1,1,0,1);
    V(mWindControl.OnRender9(arrowColor, &view, &proj, &eyePt));

	mHUD.OnRender(fElapsedTime); 
    mSampleUI.OnRender(fElapsedTime);
	mSimulationUI.OnRender(fElapsedTime);
}

HRESULT UiDx9::OnCreateDevice(IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc)
{
	HRESULT hr;
	V_RETURN(mDialogResourceManager.OnD3D9CreateDevice(pd3dDevice));
	V_RETURN(CDXUTDirectionWidget::StaticOnD3D9CreateDevice(pd3dDevice));
	return S_OK;
}

HRESULT UiDx9::OnResetDevice(const D3DSURFACE_DESC* pBackBufferSurfaceDesc)
{
	HRESULT hr;
    V_RETURN(mDialogResourceManager.OnD3D9ResetDevice());
	mWindControl.OnD3D9ResetDevice(pBackBufferSurfaceDesc);

	mHUD.SetLocation(pBackBufferSurfaceDesc->Width-170, 0);
    mHUD.SetSize(170, 170);
    mSampleUI.SetLocation(0, 84);
    mSampleUI.SetSize(170, 170);
    mSimulationUI.SetLocation(pBackBufferSurfaceDesc->Width-170, 0);
    mSimulationUI.SetSize(170, 300);
	return S_OK;
}

void UiDx9::OnLostDevice()
{
    mDialogResourceManager.OnD3D9LostDevice();
    CDXUTDirectionWidget::StaticOnD3D9LostDevice();
}

void UiDx9::OnDestroyDevice()
{
    CDXUTDirectionWidget::StaticOnD3D9DestroyDevice();
    mDialogResourceManager.OnD3D9DestroyDevice();
}