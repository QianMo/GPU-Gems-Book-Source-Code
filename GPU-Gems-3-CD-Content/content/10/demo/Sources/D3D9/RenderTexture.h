#pragma once

// This is a simple wrapper class for creating
// and using a renderable texture.
//
class RenderTexture
{
public:
  RenderTexture();
  ~RenderTexture();

  struct CreationParams
  {
    int iWidth;
    int iHeight;  
    bool bColorTexture;
    D3DFORMAT ColorFormat;
    D3DFORMAT DepthFormat;
  };

  bool Create(const CreationParams &cp);
  void Destroy(void);

  // start rendering to texture
  void EnableRendering(void);
  // stop rendering to texture
  void DisableRendering(void);

  inline const CreationParams &GetParams(void) { return m_Params; }
  inline D3DVIEWPORT9 &GetViewport(void) { return m_Viewport; }
  
  LPDIRECT3DTEXTURE9 GetColorTexture(void);

private:
  CreationParams m_Params;

  LPDIRECT3DTEXTURE9 m_pTexture;
  LPDIRECT3DSURFACE9 m_pSurface;
  LPDIRECT3DSURFACE9 m_pDSSurface;
  D3DVIEWPORT9 m_Viewport;

  // temp
  LPDIRECT3DSURFACE9 m_pOldDSSurface;
  LPDIRECT3DSURFACE9 m_pOldRenderTarget;
  D3DVIEWPORT9 m_OldViewport;
};
