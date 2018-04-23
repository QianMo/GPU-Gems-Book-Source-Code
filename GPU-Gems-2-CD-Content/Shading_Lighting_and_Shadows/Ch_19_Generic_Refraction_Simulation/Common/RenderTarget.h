///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : RenderTarget.h
//  Desc : Render targets class helper
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CRenderTarget
{
public:
  CRenderTarget(): m_plD3Tex(0), m_plD3Surf(0), m_iWidth(0), m_iHeight(0), m_iFormat(D3DFMT_A8R8G8B8)
  {
  }

  ~CRenderTarget()
  {
    Release();
  }

  // Create a rendertarget
  int Create(int iWidth, int iHeight, int iFormat=D3DFMT_A8R8G8B8, int iHasMipMaps=0, int iAASamples=0);
  // Free rendertarget resources
  void Release();
  // Create texture mipmaps
  int GenerateMipMaps();

  // Accessors  

  void GetProperties(int &iWidth, int &iHeight, int &iFormat) const
  {
    iWidth=m_iWidth;
    iHeight=m_iHeight;
    iFormat=m_iFormat;
  }

  int GetWidth() const
  {
    return m_iWidth;
  }

  int GetHeight() const
  {
    return m_iHeight;
  }

  int GetFormat() const
  {
    return m_iFormat;
  }

  const IDirect3DTexture9 *GetTexture() const
  {
    return m_plD3Tex;
  }

  const IDirect3DSurface9 *GetSurface() const
  {
    return m_plD3Surf;
  }

  // Manipulators

  IDirect3DTexture9 *GetTexture() 
  {
    return m_plD3Tex;
  }

  IDirect3DSurface9 *GetSurface() 
  {
    return m_plD3Surf;
  }

private:
  int m_iWidth, m_iHeight, m_iFormat;
  IDirect3DTexture9 *m_plD3Tex;
  IDirect3DSurface9 *m_plD3Surf;
};
