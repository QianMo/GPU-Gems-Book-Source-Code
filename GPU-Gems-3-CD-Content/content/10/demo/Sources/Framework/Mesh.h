#pragma once

#include "BoundingBox.h"

// Base class for meshes
//
// Mesh_D3D9 and Mesh_D3D10 will derive from this
class Mesh
{
public:
  Mesh();
  virtual ~Mesh();

  // API specific functions
  virtual bool CreateBuffers(void) = 0;
  virtual void Draw(void) = 0;

  // generic functions
  bool LoadFromLWO(const char *strFile);
  void CalculateOOBB(void);

public:
  unsigned int m_iNumVertices;
  unsigned int m_iNumTris;
  unsigned int m_iVertexSize;
  unsigned short *m_pIndices;
  float *m_pVertices;

  BoundingBox m_OOBB;
};
