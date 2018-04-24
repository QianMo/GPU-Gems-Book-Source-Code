#pragma once

#include "../Framework/ShadowMap.h"

// This is a simple wrapper class for creating
// and using renderable shadow map textures.
//
class ShadowMap_OGL : public ShadowMap
{
public:
  ShadowMap_OGL();
  ~ShadowMap_OGL();

  bool Create(int &iSize);
  bool CreateAsTextureArray(int iSize, int iArraySize);
  void Destroy(void);

  // start rendering to texture
  void EnableRendering(void);
  // stop rendering to texture
  void DisableRendering(void);

  void Bind(void);
  void Unbind(void);

  int GetMemoryInMB(void);

public:
  int m_iArraySize;

  GLuint m_iTexture;

  // frame buffer object
  GLuint m_iFrameBuffer;
};