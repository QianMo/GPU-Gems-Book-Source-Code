#include "DXUT.h"

#include "Ui_dx10.h"
#include "Platform.h"

void UiDx10::OnRender(D3DXMATRIX const& view, D3DXMATRIX const& proj, D3DXVECTOR3 const& eyePt, float fElapsedTime)
{
	HRESULT hr;

    // Render arrow to visually depict the wind direction
    D3DXCOLOR arrowColor = D3DXCOLOR(1,1,0,1);
    V(mWindControl.OnRender10(arrowColor, &view, &proj, &eyePt));

	mHUD.OnRender(fElapsedTime); 
    mSampleUI.OnRender(fElapsedTime);
	mSimulationUI.OnRender(fElapsedTime);
}

HRESULT UiDx10::OnD3D10CreateDevice(ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc)
{
	HRESULT hr;
	V_RETURN(mDialogResourceManager.OnD3D10CreateDevice(pd3dDevice));
    V_RETURN(CDXUTDirectionWidget::StaticOnD3D10CreateDevice(pd3dDevice));
	return S_OK;
}

HRESULT UiDx10::OnD3D10ResizedSwapChain(ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc)
{
	HRESULT hr;
	V_RETURN(mDialogResourceManager.OnD3D10ResizedSwapChain(pd3dDevice, pBackBufferSurfaceDesc));

	mHUD.SetLocation(pBackBufferSurfaceDesc->Width-170, 0);
    mHUD.SetSize(170, 170);
    mSampleUI.SetLocation(0, 84);
    mSampleUI.SetSize(170, 170);
    mSimulationUI.SetLocation(pBackBufferSurfaceDesc->Width-170, 0);
    mSimulationUI.SetSize(170, 300);

	return S_OK;
}

void UiDx10::OnD3D10ReleasingSwapChain()
{
	mDialogResourceManager.OnD3D10ReleasingSwapChain();
}

void UiDx10::OnD3D10DestroyDevice()
{
	mDialogResourceManager.OnD3D10DestroyDevice();
    CDXUTDirectionWidget::StaticOnD3D10DestroyDevice();
}