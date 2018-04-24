#include "Common.h"
#include "SceneObject.h"
#include "Mesh.h"

SceneObject::SceneObject()
{
  m_mWorld.SetIdentity();
  m_iFirstSplit = INT_MAX;
  m_iLastSplit = INT_MIN;
  m_bOnlyReceiveShadows = false;
}

SceneObject::~SceneObject()
{
}

// computes world space bounding box (8 corner points)
void SceneObject::CalculateAABB(void)
{
  // transform OOBB points to world space
  Vector4 vTransformed[8];
  for(int i=0;i<8;i++)
  {
    vTransformed[i] = Transform(m_OOBB.m_pPoints[i], m_mWorld);
  }
  // set new AABB
  m_AABB.Set(vTransformed, 8, sizeof(Vector4));
}

void SceneObject::SetMesh(Mesh *pMesh)
{
  m_pMesh = pMesh;
  m_OOBB = pMesh->m_OOBB;
}
