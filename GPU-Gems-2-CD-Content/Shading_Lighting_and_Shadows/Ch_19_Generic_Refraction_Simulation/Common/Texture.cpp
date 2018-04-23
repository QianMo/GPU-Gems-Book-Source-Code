///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Texture.cpp
//  Desc : Texture class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Texture.h"
#include "D3DApp.h"
#include <stdio.h>

int CTexture:: Create(const char *pFileName, bool bCompress)
{
  if(!pFileName || strlen(pFileName)<=2)
  {
    return APP_ERR_INVALIDPARAM;
  }

  // Make sure all data released, just in case client misuses
  Release();

  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();

  // Load texture
  sprintf(m_pFileName,"%s%s", APP_DATAPATH_TEXTURES , pFileName);
  if(FAILED(D3DXCreateTextureFromFileEx(plD3DDevice,
                                        m_pFileName,
                                        0,0,              // texture size
                                        0,                // mip levels
                                        0,                // texture usage default
                                        D3DFMT_FROM_FILE, //(bCompress)?D3DFMT_V8U8:D3DFMT_A8B8G8R8,  // internal format
                                        D3DPOOL_MANAGED,  // mem pool
                                        D3DX_FILTER_BOX|D3DX_FILTER_DITHER,  // filter
                                        D3DX_FILTER_BOX|D3DX_FILTER_DITHER,  // mip filter
                                        0,                // color key
                                        0,
                                        0, 
                                        &m_plD3DTexture)))
  {
    OutputMsg("Error", "Loading texture %s", m_pFileName);
    return APP_ERR_READFAIL;
  }

  D3DSURFACE_DESC pSurfaceDest;
  m_plD3DTexture->GetLevelDesc(0, &pSurfaceDest);

  m_iWidth=pSurfaceDest.Width;
  m_iHeight=pSurfaceDest.Height;
  return APP_OK;
}

void CTexture:: Release()
{
  m_iWidth=0;
  m_iHeight=0;
  m_iBps=0;
  memset(m_pFileName,0,256);  
  SAFE_RELEASE(m_plD3DTexture);
}

