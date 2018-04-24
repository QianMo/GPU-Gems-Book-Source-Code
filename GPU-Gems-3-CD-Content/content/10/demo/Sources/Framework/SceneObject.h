#pragma once

#include "BoundingBox.h"

class Mesh;

class SceneObject
{
public:
  SceneObject();
  ~SceneObject();

  // computes world space bounding box (8 corner points)
  void CalculateAABB(void);

  void SetMesh(Mesh *pMesh);

public:
  // special flag for rendering terrain (does not cast a shadow)
  bool m_bOnlyReceiveShadows;

  // used in geometry shader (D3D10)
  int m_iFirstSplit;
  int m_iLastSplit;

  // mesh data
  Mesh *m_pMesh;

  // world space transformation matrix
  Matrix m_mWorld;

  // object space bounding box
  BoundingBox m_OOBB;
  // world space bounding box (use CalculateAABB to update)
  BoundingBox m_AABB;
};
