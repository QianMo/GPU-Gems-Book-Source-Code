#pragma once

#include "../Framework/ShadowMap.h"

// This is a simple wrapper class for creating
// and using renderable shadow map textures.
//
class ShadowMap_D3D9 : public ShadowMap
{
public:
  ShadowMap_D3D9();
  ~ShadowMap_D3D9();

  bool Create(int iSize);
  void Destroy(void);

  // start rendering to texture
  void EnableRendering(void);
  // stop rendering to texture
  void DisableRendering(void);

  int GetMemoryInMB(void);

  inline D3DVIEWPORT9 &GetViewport(void) { return m_Viewport; }
  
  LPDIRECT3DTEXTURE9 GetColorTexture(void);

private:
  int m_iBytesPerTexel;

  LPDIRECT3DTEXTURE9 m_pTexture;
  LPDIRECT3DSURFACE9 m_pSurface;
  LPDIRECT3DSURFACE9 m_pDSSurface;
  D3DVIEWPORT9 m_Viewport;

  // temp
  LPDIRECT3DSURFACE9 m_pOldDSSurface;
  LPDIRECT3DSURFACE9 m_pOldRenderTarget;
  D3DVIEWPORT9 m_OldViewport;
};
