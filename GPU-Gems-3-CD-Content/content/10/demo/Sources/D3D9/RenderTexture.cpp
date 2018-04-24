#include "../Framework/Common.h"
#include "Application_D3D9.h"
#include "RenderTexture.h"

RenderTexture::RenderTexture()
{
  m_pTexture = NULL;
  m_pSurface = NULL;
  m_pDSSurface = NULL;
  m_pOldDSSurface = NULL;
  m_pOldRenderTarget = NULL;
}

RenderTexture::~RenderTexture()
{
  Destroy();
}

bool RenderTexture::Create(const CreationParams &cp)
{
  HRESULT hr;

  // Create a renderable texture
  //
  hr = GetApp()->GetDevice()->CreateTexture(cp.iWidth, cp.iHeight, 1, D3DUSAGE_RENDERTARGET, cp.ColorFormat, D3DPOOL_DEFAULT, &m_pTexture, NULL);
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
  hr = GetApp()->GetDevice()->CreateDepthStencilSurface(cp.iWidth, cp.iHeight, cp.DepthFormat, D3DMULTISAMPLE_NONE, 0, TRUE, &m_pDSSurface, NULL);
  if (FAILED(hr))
  {
    MessageBox(NULL, TEXT("CreateDepthStencilSurface failed!"), TEXT("Error!"), MB_OK);
    return false;
  }

  // Create a new viewport
  //
  m_Viewport.X = 0;
  m_Viewport.Y = 0;
  m_Viewport.Width  = cp.iWidth;
  m_Viewport.Height = cp.iHeight;
  m_Viewport.MinZ = 0.0f;
  m_Viewport.MaxZ = 1.0f;

  // store creation params
  m_Params = cp;
  return true;
}

void RenderTexture::Destroy(void)
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

void RenderTexture::EnableRendering(void)
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

void RenderTexture::DisableRendering(void)
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

LPDIRECT3DTEXTURE9 RenderTexture::GetColorTexture(void)
{
  return m_pTexture;
}

