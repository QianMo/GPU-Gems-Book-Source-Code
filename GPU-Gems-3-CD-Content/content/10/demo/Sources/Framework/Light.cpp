#include "Common.h"
#include "Light.h"
#include "SceneObject.h"
#include "IntersectionTests.h"
#include "Application.h"
#include "DemoSetup.h"

Light::Light()
{
  m_Type = TYPE_ORTHOGRAPHIC;
  m_vUpVector = Vector3(0, 1, 0);
  m_fAspectRatio = 1.0f;
  m_vSource = Vector3(0, 200, 0);
  m_vTarget = Vector3(0, 0, 0);
  m_fNear = 50.0f;
  m_fFar = 400.0f;
  m_fFOV = DegreeToRadian(90.0f);
  m_vLightDiffuse = Vector3(0.7f,0.7f,0.7f);
  m_vLightAmbient = Vector3(0.25f,0.25f,0.25f);

  ZeroMemory(&m_ControlState, sizeof(ControlState));
  m_ControlState.m_vRotation = Vector3(-DegreeToRadian(130.0f), -DegreeToRadian(35.0f),0);
}

// processes light controls
void Light::DoControls(void)
{
  float fDeltaTime = DeltaTimeUpdate(m_ControlState.m_fLastUpdate);

  // Rotate light
  //
  if(GetKeyDown(VK_LEFT)) m_ControlState.m_vRotation.x += 0.02f * fDeltaTime;
  else if(GetKeyDown(VK_RIGHT)) m_ControlState.m_vRotation.x -= 0.02f * fDeltaTime;
  if(GetKeyDown(VK_UP)) m_ControlState.m_vRotation.y += 0.01f * fDeltaTime;
  else if(GetKeyDown(VK_DOWN)) m_ControlState.m_vRotation.y -= 0.01f * fDeltaTime;

  m_ControlState.m_vRotation.y = Clamp(m_ControlState.m_vRotation.y, DegreeToRadian(-89.9f), DegreeToRadian(0.0f));
  float ch = cosf(m_ControlState.m_vRotation.x);
  float sh = sinf(m_ControlState.m_vRotation.x);
  float cp = cosf(m_ControlState.m_vRotation.y);
  float sp = sinf(m_ControlState.m_vRotation.y);
  Vector3 vDist = m_vTarget - m_vSource;
  m_vSource = m_vTarget + Vector3(sh*cp, -sp, cp*ch) * vDist.Length();

  // Switch light type
  //
  if(GetKeyDown('T'))
  {
    if(!m_ControlState.m_bSwitchingType)
    {
      m_Type = (m_Type == Light::TYPE_ORTHOGRAPHIC) ? Light::TYPE_PERSPECTIVE : Light::TYPE_ORTHOGRAPHIC;
      m_ControlState.m_bSwitchingType = true;
    }
  }
  else
  {
    m_ControlState.m_bSwitchingType = false;
  }

  CalculateMatrices();
}

// calculates default light matrices
void Light::CalculateMatrices(void)
{
  // view matrix
  m_mView = MatrixLookAtLH(m_vSource, m_vTarget, m_vUpVector);

  // projection matrix
  if(m_Type == TYPE_PERSPECTIVE)
  {
    m_mProj = MatrixPerspectiveFovLH(m_fFOV, m_fAspectRatio, m_fNear, m_fFar);
  }
  else
  {
    // this is just a funny way to calculate a size for the light using FOV
    float fFarPlaneSize = 2 * tanf(m_fFOV * 0.5f) * m_fFar;
    m_mProj = MatrixOrthoLH(fFarPlaneSize * m_fAspectRatio, fFarPlaneSize, m_fNear, m_fFar);
  }
}

// finds scene objects that overlap given frustum from light's view
std::vector<SceneObject *> Light::FindCasters(const Frustum &frustum)
{
  Vector3 vDir = Normalize(m_vTarget - m_vSource);

  std::vector<SceneObject *> casters;
  casters.reserve(g_SceneObjects.size());
  for(unsigned int i=0; i<g_SceneObjects.size(); i++)
  {
    SceneObject *pObject = g_SceneObjects[i];
    if(pObject->m_bOnlyReceiveShadows) continue;

    // do intersection test
    // orthogonal light
    if(m_Type == TYPE_ORTHOGRAPHIC)
    {
      // use sweep intersection
      if(g_iVisibilityTest == VISTEST_ACCURATE) {
        // test accurately
        if(!SweepIntersectionTest(pObject->m_AABB, frustum, vDir)) continue;
      } else if(g_iVisibilityTest == VISTEST_CHEAP) {
        // test only with AABB of frustum
        if(!SweepIntersectionTest(pObject->m_AABB, frustum.m_AABB, vDir)) continue;
      }
    }
    // perspective light
    else if(m_Type == TYPE_PERSPECTIVE)
    {
      // the same kind of sweep intersection doesn't really work here, but we can
      // approximate it by using the direction to center of AABB as the sweep direction
      // (note that sometimes this will fail)
      Vector3 vDirToCenter = Normalize(((pObject->m_AABB.m_vMax + pObject->m_AABB.m_vMin) * 0.5f) - m_vSource);
      if(g_iVisibilityTest == VISTEST_ACCURATE)
      {
        // test accurately
        if(!SweepIntersectionTest(pObject->m_AABB, frustum, vDirToCenter)) continue;
      } else if(g_iVisibilityTest == VISTEST_CHEAP) {
        // test only with AABB of frustum
        if(!SweepIntersectionTest(pObject->m_AABB, frustum.m_AABB, vDirToCenter)) continue;
      }
    }

    casters.push_back(pObject);
  }
  return casters;
}


// build a matrix for cropping light's projection
// given vectors are in light's clip space
inline Matrix Light::BuildCropMatrix(const Vector3 &vMin, const Vector3 &vMax)
{
  float fScaleX, fScaleY, fScaleZ;
  float fOffsetX, fOffsetY, fOffsetZ;

  fScaleX = 2.0f / (vMax.x - vMin.x);
  fScaleY = 2.0f / (vMax.y - vMin.y);

  fOffsetX = -0.5f * (vMax.x + vMin.x) * fScaleX;
  fOffsetY = -0.5f * (vMax.y + vMin.y) * fScaleY;

  fScaleZ = 1.0f / (vMax.z - vMin.z);
  fOffsetZ = -vMin.z * fScaleZ;

  // crop volume matrix
  return Matrix(   fScaleX,     0.0f,     0.0f,   0.0f,
                          0.0f,  fScaleY,     0.0f,   0.0f,
                          0.0f,     0.0f,  fScaleZ,   0.0f,
                      fOffsetX, fOffsetY, fOffsetZ,   1.0f  );
}

// helper function for computing AABB in clip space
inline BoundingBox CreateClipSpaceAABB(const BoundingBox &bb, const Matrix &mViewProj)
{
  Vector4 vTransformed[8];
  // for each point
  for(int i=0;i<8;i++)
  {
    // transform to projection space
    vTransformed[i] = Transform(bb.m_pPoints[i], mViewProj);

    // compute clip-space coordinates
    vTransformed[i].x /= vTransformed[i].w;
    vTransformed[i].y /= vTransformed[i].w;
    vTransformed[i].z /= vTransformed[i].w;
  }

  return BoundingBox(vTransformed, 8, sizeof(Vector4));
}

// crops the light volume on given frustum (scene-independent projection)
Matrix Light::CalculateCropMatrix(const Frustum &frustum)
{
  Matrix mViewProj = m_mView * m_mProj;

  BoundingBox cropBB;

  // find boundaries in light’s clip space
  cropBB = CreateClipSpaceAABB(frustum.m_AABB, mViewProj);

  // use default near plane
  cropBB.m_vMin.z = 0.0f;

  // finally, create matrix
  return BuildCropMatrix(cropBB.m_vMin, cropBB.m_vMax);
}

// crops the light volume on given objects, constrained by given frustum
Matrix Light::CalculateCropMatrix(const std::vector<SceneObject *> &casters, const std::vector<SceneObject *> &receivers, const Frustum &frustum)
{
  if(!g_bUseSceneDependentProjection) return CalculateCropMatrix(frustum);

  Matrix mViewProj = m_mView * m_mProj;

  // bounding box limits
  BoundingBox receiversBB, splitBB, castersBB;

  // for each caster
  // find boundaries in light’s clip space
  for(unsigned int i = 0; i < casters.size(); i++)
    castersBB.Union(CreateClipSpaceAABB(casters[i]->m_AABB, mViewProj));

  // for each receiver
  // find boundaries in light’s clip space
  for(unsigned int i = 0; i < receivers.size(); i++)
  {
    receiversBB.Union(CreateClipSpaceAABB(receivers[i]->m_AABB, mViewProj));
  }

  // find frustum boundaries in light’s clip space
  splitBB = CreateClipSpaceAABB(frustum.m_AABB, mViewProj);

  // next we will merge the bounding boxes
  //
  BoundingBox cropBB;
  cropBB.m_vMin.x = Max(Max(castersBB.m_vMin.x, receiversBB.m_vMin.x), splitBB.m_vMin.x);
  cropBB.m_vMax.x = Min(Min(castersBB.m_vMax.x, receiversBB.m_vMax.x), splitBB.m_vMax.x);
  cropBB.m_vMin.y = Max(Max(castersBB.m_vMin.y, receiversBB.m_vMin.y), splitBB.m_vMin.y);
  cropBB.m_vMax.y = Min(Min(castersBB.m_vMax.y, receiversBB.m_vMax.y), splitBB.m_vMax.y);
  cropBB.m_vMin.z = castersBB.m_vMin.z;
  cropBB.m_vMax.z = Min(receiversBB.m_vMax.z, splitBB.m_vMax.z);

  // when there are no casters, the merged
  // bounding box will be infinitely small
  if(casters.size() == 0)
  {
    // it will cause artifacts when rendering receivers,
    // so just use the frustum bounding box instead
    cropBB.m_vMin = splitBB.m_vMin;
    cropBB.m_vMax = splitBB.m_vMax;
  }

  // finally, create matrix
  return BuildCropMatrix(cropBB.m_vMin, cropBB.m_vMax);
}


// returns direction of light
Vector3 Light::GetDir(void)
{
  return Normalize(m_vTarget - m_vSource);
}
