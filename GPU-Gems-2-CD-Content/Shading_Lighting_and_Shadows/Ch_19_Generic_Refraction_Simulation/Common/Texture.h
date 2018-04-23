///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Texture.h
//  Desc : Texture class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CTexture
{
public:
  CTexture():m_iWidth(0), m_iHeight(0), m_iBps(0),m_plD3DTexture(0)
  {    
    memset(m_pFileName,0,256);  
  };
  ~CTexture()
  {
    Release();
  };

  // create texture from file 
  int Create(const char *pFileName, bool bCompress);
  // release texture
  void Release();
  
  // Get texture interface
  const IDirect3DTexture9 *GetTexture() const
  {
    return  m_plD3DTexture;
  }

  // Get texture properties
  void GetTextureProperties(int &iWidth, int &iHeight, int &iBps) const
  {
    iWidth=m_iWidth; iHeight=m_iHeight; iBps=m_iBps;
  }

  // Get texture name
  const char *GetTextureFileName() const
  {
    return m_pFileName;
  }

private:
  // Texture properties
  int   m_iWidth, m_iHeight, m_iBps;
  char  m_pFileName[256];

  // Texture interface
  IDirect3DTexture9 *m_plD3DTexture;
};
