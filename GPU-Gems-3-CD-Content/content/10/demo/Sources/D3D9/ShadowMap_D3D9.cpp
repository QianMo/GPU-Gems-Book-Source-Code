#include "../Framework/Common.h"
#include "Application_D3D9.h"
#include "ShadowMap_D3D9.h"

ShadowMap_D3D9::ShadowMap_D3D9()
{
  m_pTexture = NULL;
  m_pSurface = NULL;
  m_pDSSurface = NULL;
  m_pOldDSSurface = NULL;
  m_pOldRenderTarget = NULL;
  m_iBytesPerTexel = 0;
}

ShadowMap_D3D9::~ShadowMap_D3D9()
{
  Destroy();
}

bool ShadowMap_D3D9::Create(int iSize)
{
  HRESULT hr;

  // Create a renderable texture
  //
  D3DFORMAT ColorFormat[5] = { D3DFMT_R32F, D3DFMT_R16F, D3DFMT_L16,
                               D3DFMT_A2R10G10B10, D3DFMT_A8R8G8B8};
  char *strFormat[5] = {"D3DFMT_R32F", "D3DFMT_R16F", "D3DFMT_L16", "D3DFMT_A2R10G10B10", "D3DFMT_A8R8G8B8"};
  int BytesPerTexel[5] = {32/8, 16/8, 16/8, 32/8, 32/8};
  int iFormat;
  for(iFormat = 0; iFormat < 5; iFormat++)
  {
    hr = GetApp()->GetDevice()->CreateTexture(iSize, iSize, 1, D3DUSAGE_RENDERTARGET, ColorFormat[iFormat], D3DPOOL_DEFAULT, &m_pTexture, NULL);
    m_iBytesPerTexel = BytesPerTexel[iFormat];
    if(SUCCEEDED(hr)) break;
  }
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("Creating render texture failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Get its surface (used as render target)
  //
  hr = m_pTexture->GetSurfaceLevel(0, &m_pSurface);
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("GetSurfaceLevel failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create a depth stencil surface
  //
  hr = GetApp()->GetDevice()->CreateDepthStencilSurface(iSize, iSize, D3DFMT_D24S8, D3DMULTISAMPLE_NONE, 0, TRUE, &m_pDSSurface, NULL);
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("CreateDepthStencilSurface failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create a new viewport
  //
  m_Viewport.X = 0;
  m_Viewport.Y = 0;
  m_Viewport.Width  = iSize;
  m_Viewport.Height = iSize;
  m_Viewport.MinZ = 0.0f;
  m_Viewport.MaxZ = 1.0f;

  m_iSize = iSize;
  if(iFormat == 4 || iFormat == 5)
    _snprintf(m_strInfo, 1024, "%s (only using R channel), %i²", strFormat[iFormat], m_iSize);
  else
    _snprintf(m_strInfo, 1024, "%s, %i²", strFormat[iFormat], m_iSize);
  return true;
}

void ShadowMap_D3D9::Destroy(void)
{
  // Release everything
  //
  if (m_pDSSurface != NULL)
  {
    m_pDSSurface->Release();
    m_pDSSurface = NULL;
  }

  if (m_pSurface != NULL)
  {
    m_pSurface->Release();
    m_pSurface = NULL;
  }

  if (m_pTexture != NULL)
  {
    m_pTexture->Release();
    m_pTexture = NULL;
  }

  // Release temp vars just to be sure
  // nothing is left hanging around
  //
  if (m_pOldDSSurface != NULL)
  {
    m_pOldDSSurface->Release();
    m_pOldDSSurface = NULL;
  }

  if (m_pOldRenderTarget != NULL)
  {
    m_pOldRenderTarget->Release();
    m_pOldRenderTarget = NULL;
  }
}

void ShadowMap_D3D9::EnableRendering(void)
{
  // Store original values
  //
  GetApp()->GetDevice()->GetViewport(&m_OldViewport);
  GetApp()->GetDevice()->GetRenderTarget(0, &m_pOldRenderTarget);
  GetApp()->GetDevice()->GetDepthStencilSurface(&m_pOldDSSurface);

  // Set new values
  //
  GetApp()->GetDevice()->SetViewport(&m_Viewport);
  GetApp()->GetDevice()->SetRenderTarget(0, m_pSurface);
  GetApp()->GetDevice()->SetDepthStencilSurface(m_pDSSurface);
}

void ShadowMap_D3D9::DisableRendering(void)
{
  // Restore old depth stencil
  //
  GetApp()->GetDevice()->SetDepthStencilSurface(m_pOldDSSurface);

  // releasing is necessary due to reference counting
  if (m_pOldDSSurface != NULL)
  {
    m_pOldDSSurface->Release();
    m_pOldDSSurface = NULL;
  }

  // Restore old render target
  //
  GetApp()->GetDevice()->SetRenderTarget(0, m_pOldRenderTarget);

  // releasing is necessary due to reference counting
  if (m_pOldRenderTarget != NULL)
  {
    m_pOldRenderTarget->Release();
    m_pOldRenderTarget = NULL;
  }

  // Restore old viewport
  //
  GetApp()->GetDevice()->SetViewport(&m_OldViewport);
}

LPDIRECT3DTEXTURE9 ShadowMap_D3D9::GetColorTexture(void)
{
  return m_pTexture;
}

int ShadowMap_D3D9::GetMemoryInMB(void)
{
  return m_iSize * m_iSize * m_iBytesPerTexel / 1048576;
}