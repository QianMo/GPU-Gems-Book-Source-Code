#include "../Framework/Common.h"
#include "Application_D3D10.h"
#include "ShadowMap_D3D10.h"

ShadowMap_D3D10::ShadowMap_D3D10()
{
  m_pTexture = NULL;
  m_pDSV = NULL;
  m_pSRV = NULL;
  m_iArraySize = 0;
}

ShadowMap_D3D10::~ShadowMap_D3D10()
{
  Destroy();
}

bool ShadowMap_D3D10::Create(int iSize)
{
  HRESULT hr;

  // Create render target texture
  //
  D3D10_TEXTURE2D_DESC DescTex = {0};
  DescTex.Width = iSize;
  DescTex.Height = iSize;
  DescTex.MipLevels = 1;
  DescTex.Format = DXGI_FORMAT_R32_TYPELESS;
  DescTex.SampleDesc.Count = 1;
  DescTex.SampleDesc.Quality = 0;
  DescTex.Usage = D3D10_USAGE_DEFAULT;
  DescTex.BindFlags = D3D10_BIND_DEPTH_STENCIL | D3D10_BIND_SHADER_RESOURCE;
  DescTex.CPUAccessFlags = 0;
  DescTex.MiscFlags = 0;
  DescTex.ArraySize = 1;
  hr = GetApp()->GetDevice()->CreateTexture2D(&DescTex, NULL, &m_pTexture);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map texture failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create depth stencil view
  //
  D3D10_DEPTH_STENCIL_VIEW_DESC DescDSV = {};
  DescDSV.Format = DXGI_FORMAT_D32_FLOAT;
  DescDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
  DescDSV.Texture2D.MipSlice = 0;
  hr = GetApp()->GetDevice()->CreateDepthStencilView(m_pTexture, &DescDSV, &m_pDSV);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map depth stencil view failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create shader resource view
  //
  D3D10_SHADER_RESOURCE_VIEW_DESC DescSRV = {};
  DescSRV.Format = DXGI_FORMAT_R32_FLOAT;
  DescSRV.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
  DescSRV.Texture2D.MipLevels = 1;
  DescSRV.Texture2D.MostDetailedMip = 0;
  hr = GetApp()->GetDevice()->CreateShaderResourceView(m_pTexture, &DescSRV, &m_pSRV );
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map resource view failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  m_iSize = iSize;
  m_iArraySize = 1;
  _snprintf(m_strInfo, 1024, "R32, %i²", m_iSize);
  return true;
}

bool ShadowMap_D3D10::CreateAsTextureArray(int iSize, int iArraySize)
{
  HRESULT hr;

  // Create render target texture array
  //
  D3D10_TEXTURE2D_DESC DescTex = {0};
  DescTex.Width = iSize;
  DescTex.Height = iSize;
  DescTex.MipLevels = 1;
  DescTex.Format = DXGI_FORMAT_R32_TYPELESS;
  DescTex.SampleDesc.Count = 1;
  DescTex.SampleDesc.Quality = 0;
  DescTex.Usage = D3D10_USAGE_DEFAULT;
  DescTex.BindFlags = D3D10_BIND_DEPTH_STENCIL | D3D10_BIND_SHADER_RESOURCE;
  DescTex.CPUAccessFlags = 0;
  DescTex.MiscFlags = 0;
  DescTex.ArraySize = iArraySize;
  hr = GetApp()->GetDevice()->CreateTexture2D(&DescTex, NULL, &m_pTexture);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map texture array failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create depth stencil view
  //
  D3D10_DEPTH_STENCIL_VIEW_DESC DescDSV = {};
  DescDSV.Format = DXGI_FORMAT_D32_FLOAT;
  DescDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
  DescDSV.Texture2DArray.FirstArraySlice = 0;
  DescDSV.Texture2DArray.ArraySize = iArraySize;
  DescDSV.Texture2DArray.MipSlice = 0;
  hr = GetApp()->GetDevice()->CreateDepthStencilView(m_pTexture, &DescDSV, &m_pDSV);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map depth stencil view failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create shader resource view
  //
  D3D10_SHADER_RESOURCE_VIEW_DESC DescSRV = {};
  DescSRV.Format = DXGI_FORMAT_R32_FLOAT;
  DescSRV.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
  DescSRV.Texture2DArray.ArraySize = iArraySize;
  DescSRV.Texture2DArray.MipLevels = 1;
  DescSRV.Texture2DArray.FirstArraySlice = 0;
  DescSRV.Texture2DArray.MostDetailedMip = 0;
  hr = GetApp()->GetDevice()->CreateShaderResourceView(m_pTexture, &DescSRV, &m_pSRV );
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map resource view failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  m_iSize = iSize;
  m_iArraySize = iArraySize;
  _snprintf(m_strInfo, 1024, "R32, %i²x%i", m_iSize, m_iArraySize);
  return true;
}

bool ShadowMap_D3D10::CreateAsTextureCube(int iSize)
{
  HRESULT hr;

  // Create render target texture array
  //
  D3D10_TEXTURE2D_DESC DescTex = {0};
  DescTex.Width = iSize;
  DescTex.Height = iSize;
  DescTex.MipLevels = 1;
  DescTex.Format = DXGI_FORMAT_R32_TYPELESS;
  DescTex.SampleDesc.Count = 1;
  DescTex.SampleDesc.Quality = 0;
  DescTex.Usage = D3D10_USAGE_DEFAULT;
  DescTex.BindFlags = D3D10_BIND_DEPTH_STENCIL | D3D10_BIND_SHADER_RESOURCE;
  DescTex.CPUAccessFlags = 0;
  DescTex.MiscFlags = D3D10_RESOURCE_MISC_TEXTURECUBE;
  DescTex.ArraySize = 6;
  hr = GetApp()->GetDevice()->CreateTexture2D(&DescTex, NULL, &m_pTexture);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map texture array failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create depth stencil view
  //
  D3D10_DEPTH_STENCIL_VIEW_DESC DescDSV = {};
  DescDSV.Format = DXGI_FORMAT_D32_FLOAT;
  DescDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DARRAY;
  DescDSV.Texture2DArray.FirstArraySlice = 0;
  DescDSV.Texture2DArray.ArraySize = 6;
  DescDSV.Texture2DArray.MipSlice = 0;
  hr = GetApp()->GetDevice()->CreateDepthStencilView(m_pTexture, &DescDSV, &m_pDSV);
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map depth stencil view failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create shader resource view
  //
  D3D10_SHADER_RESOURCE_VIEW_DESC DescSRV = {};
  DescSRV.Format = DXGI_FORMAT_R32_FLOAT;
  DescSRV.ViewDimension = D3D10_SRV_DIMENSION_TEXTURECUBE;
  DescSRV.TextureCube.MipLevels = 1;
  DescSRV.TextureCube.MostDetailedMip = 0;
  hr = GetApp()->GetDevice()->CreateShaderResourceView(m_pTexture, &DescSRV, &m_pSRV );
  if(FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating shadow map resource view failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  m_iSize = iSize;
  m_iArraySize = 6;
  _snprintf(m_strInfo, 1024, "R32, %i²x%i", m_iSize, m_iArraySize);
  return true;
}


void ShadowMap_D3D10::Destroy(void)
{
  if(m_pDSV != NULL)
  {
    m_pDSV->Release();
    m_pDSV = NULL;
  }

  if(m_pSRV != NULL)
  {
    m_pSRV->Release();
    m_pSRV = NULL;
  }

  if(m_pTexture != NULL)
  {
    m_pTexture->Release();
    m_pTexture = NULL;
  }
}

void ShadowMap_D3D10::EnableRendering(void)
{
  // Enable rendering to shadow map
  GetApp()->GetDevice()->OMSetRenderTargets(0, NULL, m_pDSV);

  // Setup view port
  D3D10_VIEWPORT vp;
  vp.Width = m_iSize;
  vp.Height = m_iSize;
  vp.MinDepth = 0.0f;
  vp.MaxDepth = 1.0f;
  vp.TopLeftX = 0;
  vp.TopLeftY = 0;
  GetApp()->GetDevice()->RSSetViewports(1, &vp);
}

void ShadowMap_D3D10::DisableRendering(void)
{
  // Disable rendering to shadow map
  GetApp()->SetDefaultRenderTarget();
}

int ShadowMap_D3D10::GetMemoryInMB(void)
{
  return m_iArraySize * m_iSize * m_iSize * 8 / 1048576;
}