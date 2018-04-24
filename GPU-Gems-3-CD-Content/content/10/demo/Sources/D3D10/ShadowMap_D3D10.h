#pragma once

#include "../Framework/ShadowMap.h"

// This is a simple wrapper class for creating
// and using renderable shadow map textures.
//
class ShadowMap_D3D10 : public ShadowMap
{
public:
  ShadowMap_D3D10();
  ~ShadowMap_D3D10();

  bool Create(int iSize);
  bool CreateAsTextureArray(int iSize, int iArraySize);
  bool CreateAsTextureCube(int iSize);
  void Destroy(void);

  // start rendering to texture
  void EnableRendering(void);
  // stop rendering to texture
  void DisableRendering(void);

  int GetMemoryInMB(void);

public:
  int m_iArraySize;

  ID3D10Texture2D *m_pTexture;
  ID3D10DepthStencilView *m_pDSV;
  ID3D10ShaderResourceView *m_pSRV;
};