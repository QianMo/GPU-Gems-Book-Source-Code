///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : RenderTarget.cpp
//  Desc : Render targets class helper
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "RenderTarget.h"
#include "D3dApp.h"

int CRenderTarget::Create(int iWidth, int iHeight, int iFormat, int iHasMipMaps, int iAASamples)
{
  if(!iWidth || !iHeight || !iFormat)
  {
    return APP_ERR_INVALIDPARAM;
  }
  
  m_iWidth=iWidth;
  m_iHeight=iHeight;
  m_iFormat=iFormat;
  
  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();    

  if(!iAASamples)
  {
    if(FAILED(plD3DDevice->CreateTexture(iWidth, iHeight, 1,
      D3DUSAGE_RENDERTARGET|((iHasMipMaps)?D3DUSAGE_AUTOGENMIPMAP:0), (D3DFORMAT) iFormat,
      D3DPOOL_DEFAULT, &m_plD3Tex, 0)))
    {
      OutputMsg("Error", "Creating render target");
      return APP_ERR_INITFAIL;
    }

    if(FAILED(m_plD3Tex->GetSurfaceLevel(0, &m_plD3Surf)))
    {
      OutputMsg("Error", "Getting render target surface");
      return APP_ERR_INITFAIL;
    }
  }
  else
  {
    if(FAILED(plD3DDevice->CreateRenderTarget(iWidth, iHeight, (D3DFORMAT) iFormat, (D3DMULTISAMPLE_TYPE)iAASamples, 0, 0, &m_plD3Surf, 0)))
    {
      OutputMsg("Error", "Creating render target surface");
      return APP_ERR_INITFAIL;
    }
  }

  return APP_OK;
}

void CRenderTarget::Release()
{
  SAFE_RELEASE(m_plD3Tex)
  SAFE_RELEASE(m_plD3Surf)
  m_iWidth=0;
  m_iHeight=0;
  m_iFormat=D3DFMT_A8R8G8B8;
}

int CRenderTarget::GenerateMipMaps()
{  
  m_plD3Tex->SetAutoGenFilterType(D3DTEXF_LINEAR);
  m_plD3Tex->GenerateMipSubLevels();
  return APP_OK;
}
