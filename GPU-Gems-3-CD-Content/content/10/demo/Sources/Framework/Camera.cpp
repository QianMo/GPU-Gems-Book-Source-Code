#include "Common.h"
#include "Camera.h"
#include "SceneObject.h"
#include "Application.h"
#include "IntersectionTests.h"
#include "DemoSetup.h"

Camera::Camera()
{
  m_vSource = Vector3(0,20,-110);
  m_vTarget = Vector3(0,20,-109);
  m_fNearMin = 1.0f;
  m_fFarMax = 400.0f;
  m_fNear = 1.0f;
  m_fFar = 400.0f;
  m_fFOV = DegreeToRadian(45.0f);
  m_vUpVector = Vector3(0, 1, 0);
  ZeroMemory(&m_ControlState, sizeof(ControlState));
}

// finds scene objects inside the camera frustum
std::vector<SceneObject *> Camera::FindReceivers(void)
{
  Frustum cameraFrustum = CalculateFrustum(m_fNearMin, m_fFarMax);
  cameraFrustum.CalculateAABB();

  std::vector<SceneObject *> receivers;
  receivers.reserve(g_SceneObjects.size());
  for(unsigned int i=0; i<g_SceneObjects.size(); i++)
  {
    SceneObject *pObject = g_SceneObjects[i];

    // intersection test
    if(g_iVisibilityTest == VISTEST_ACCURATE) {
      // test accurately
      if(!IntersectionTest(pObject->m_AABB, cameraFrustum)) continue;
    } else if(g_iVisibilityTest == VISTEST_CHEAP) {
      // test only with AABB of frustum
      if(!IntersectionTest(pObject->m_AABB, cameraFrustum.m_AABB)) continue;
    }
    
    receivers.push_back(pObject);
  }
  return receivers;
}

// calculates split plane distances in view space
void Camera::CalculateSplitPositions(float *pDistances)
{
  // Practical split scheme:
  //
  // CLi = n*(f/n)^(i/numsplits)
  // CUi = n + (f-n)*(i/numsplits)
  // Ci = CLi*(lambda) + CUi*(1-lambda)
  //
  // lambda scales between logarithmic and uniform
  //
 
  for(int i = 0; i < g_iNumSplits; i++)
  {
    float fIDM = i / (float)g_iNumSplits;
    float fLog = m_fNear * powf(m_fFar/m_fNear, fIDM);
    float fUniform = m_fNear + (m_fFar - m_fNear) * fIDM;
    pDistances[i] = fLog * g_fSplitSchemeWeight + fUniform * (1 - g_fSplitSchemeWeight);
  }

  // make sure border values are accurate
  pDistances[0] = m_fNear;
  pDistances[g_iNumSplits] = m_fFar;
}

// computes a frustum with given far and near planes
Frustum Camera::CalculateFrustum(float fNear, float fFar)
{
  Vector3 vZ = Normalize(m_vTarget - m_vSource);
  Vector3 vX = Normalize(Cross(m_vUpVector, vZ));
  Vector3 vY = Normalize(Cross(vZ, vX));

  float fAspect = GetAppBase()->GetAspectRatio();

  float fNearPlaneHalfHeight = tanf(m_fFOV * 0.5f) * fNear;
  float fNearPlaneHalfWidth = fNearPlaneHalfHeight * fAspect;

  float fFarPlaneHalfHeight = tanf(m_fFOV * 0.5f) * fFar;
  float fFarPlaneHalfWidth = fFarPlaneHalfHeight * fAspect;

  Vector3 vNearPlaneCenter = m_vSource + vZ * fNear;
  Vector3 vFarPlaneCenter = m_vSource + vZ * fFar;

  Frustum frustum;
  frustum.m_pPoints[0] = Vector3(vNearPlaneCenter - vX*fNearPlaneHalfWidth - vY*fNearPlaneHalfHeight);
  frustum.m_pPoints[1] = Vector3(vNearPlaneCenter - vX*fNearPlaneHalfWidth + vY*fNearPlaneHalfHeight);
  frustum.m_pPoints[2] = Vector3(vNearPlaneCenter + vX*fNearPlaneHalfWidth + vY*fNearPlaneHalfHeight);
  frustum.m_pPoints[3] = Vector3(vNearPlaneCenter + vX*fNearPlaneHalfWidth - vY*fNearPlaneHalfHeight);

  frustum.m_pPoints[4] = Vector3(vFarPlaneCenter - vX*fFarPlaneHalfWidth - vY*fFarPlaneHalfHeight);
  frustum.m_pPoints[5] = Vector3(vFarPlaneCenter - vX*fFarPlaneHalfWidth + vY*fFarPlaneHalfHeight);
  frustum.m_pPoints[6] = Vector3(vFarPlaneCenter + vX*fFarPlaneHalfWidth + vY*fFarPlaneHalfHeight);
  frustum.m_pPoints[7] = Vector3(vFarPlaneCenter + vX*fFarPlaneHalfWidth - vY*fFarPlaneHalfHeight);

  // update frustum AABB
  frustum.CalculateAABB();

  return frustum;
}

// adjust the camera planes to contain the visible scene as tightly as possible
void Camera::AdjustPlanes(const std::vector<SceneObject *> &VisibleObjects)
{
  if(VisibleObjects.size() == 0) return;

  // find the nearest and farthest points of given
  // scene objects in camera's view space
  //
  float fMaxZ = 0;
  float fMinZ = FLT_MAX;

  Vector3 vDir = Normalize(m_vTarget - m_vSource);

  // for each object
  for(unsigned int i = 0; i < VisibleObjects.size(); i++)
  {
    SceneObject *pObject = VisibleObjects[i];

    // for each point in AABB
    for(int j = 0; j < 8; j++)
    {
      // calculate z-coordinate distance to near plane of camera
      Vector3 vPointToCam = pObject->m_AABB.m_pPoints[j] - m_vSource;
      float fZ = Dot(vPointToCam, vDir);

      // find boundary values
      if(fZ > fMaxZ) fMaxZ = fZ;
      if(fZ < fMinZ) fMinZ = fZ;
    }
  }

  // use smallest distance as new near plane
  // and make sure it is not too small
  m_fNear = Max(fMinZ, m_fNearMin);
  // use largest distance as new far plane
  // and make sure it is larger than nearPlane
  m_fFar = Max(fMaxZ, m_fNear + 1.0f);
  // update matrices
  CalculateMatrices();
}

// processes camera controls
void Camera::DoControls(void)
{
  float fDeltaTime = DeltaTimeUpdate(m_ControlState.fLastUpdate);

  Vector3 vZ = Normalize(m_vTarget - m_vSource);
  Vector3 vX = Normalize(Cross(m_vUpVector, vZ));
  Vector3 vY = Normalize(Cross(vZ, vX));

  // Rotating with mouse left
  //
  //
  if(GetMouseDown(VK_LBUTTON))
  {

    if(!m_ControlState.bRotating)
    {
      GetCursorPos(&m_ControlState.pntMouse);
      m_ControlState.bZooming = false;
      m_ControlState.bStrafing = false;
      m_ControlState.bRotating = true;
    }

    POINT pntMouseCurrent;
    GetCursorPos(&pntMouseCurrent);

    Vector3 vTargetToSource = m_vTarget - m_vSource;
    float fLength = vTargetToSource.Length();

    m_ControlState.fRotX += 0.005f * (float)(pntMouseCurrent.x - m_ControlState.pntMouse.x);
    m_ControlState.fRotY += 0.005f * (float)(pntMouseCurrent.y - m_ControlState.pntMouse.y);

    m_ControlState.fRotY = Clamp(m_ControlState.fRotY, DegreeToRadian(-89.9f), DegreeToRadian(89.9f));

    Matrix mRotation;
    mRotation.SetIdentity();
    mRotation.SetRotation(Vector3(m_ControlState.fRotX, m_ControlState.fRotY,0));
    m_vTarget.x = mRotation._31 * fLength + m_vSource.x;
    m_vTarget.y = mRotation._32 * fLength + m_vSource.y;
    m_vTarget.z = mRotation._33 * fLength + m_vSource.z;

    m_ControlState.pntMouse = pntMouseCurrent;

  // Strafing with mouse middle
  //
  //
  } else if(GetMouseDown(VK_MBUTTON)) {

    if(!m_ControlState.bStrafing)
    {
      GetCursorPos(&m_ControlState.pntMouse);
      m_ControlState.bZooming = false;
      m_ControlState.bRotating = false;
      m_ControlState.bStrafing = true;
    }

    POINT pntMouseCurrent;
    GetCursorPos(&pntMouseCurrent);

    m_vSource -= vX * 0.15f * (float)(pntMouseCurrent.x - m_ControlState.pntMouse.x);
    m_vTarget -= vX * 0.15f * (float)(pntMouseCurrent.x - m_ControlState.pntMouse.x);

    m_vSource += vY * 0.15f * (float)(pntMouseCurrent.y - m_ControlState.pntMouse.y);
    m_vTarget += vY * 0.15f * (float)(pntMouseCurrent.y - m_ControlState.pntMouse.y);

    m_ControlState.pntMouse = pntMouseCurrent;


  // Zooming with mouse right
  //
  //
  } else if(GetMouseDown(VK_RBUTTON)) {

    if(!m_ControlState.bZooming)
    {
      GetCursorPos(&m_ControlState.pntMouse);
      m_ControlState.bZooming = true;
      m_ControlState.bRotating = false;
      m_ControlState.bStrafing = false;
    }

    POINT pntMouseCurrent;
    GetCursorPos(&pntMouseCurrent);

    m_vSource += vZ * 0.5f * (float)(pntMouseCurrent.y - m_ControlState.pntMouse.y);
    m_vTarget += vZ * 0.5f * (float)(pntMouseCurrent.y - m_ControlState.pntMouse.y);

    m_ControlState.pntMouse = pntMouseCurrent;

  // Mouse is idle
  //
  } else {
    m_ControlState.bZooming = false;
    m_ControlState.bRotating = false;
    m_ControlState.bStrafing = false;
  }


  // Move forward/backward
  //
  if(GetKeyDown('W'))
  {
    m_vSource += vZ * fDeltaTime;
    m_vTarget += vZ * fDeltaTime;
  }
  else if(GetKeyDown('S'))
  {
    m_vSource -= vZ * fDeltaTime;
    m_vTarget -= vZ * fDeltaTime;
  }

  // Strafing
  //
  if(GetKeyDown('D'))
  {
    m_vSource += vX * fDeltaTime;
    m_vTarget += vX * fDeltaTime;
  }
  else if(GetKeyDown('A'))
  {
    m_vSource -= vX * fDeltaTime;
    m_vTarget -= vX * fDeltaTime;
  }

  // Up/down (many control preferences here :)
  //
  if(GetKeyDown('Q') || GetKeyDown(VK_SPACE))
  {
    m_vSource += vY * fDeltaTime;
    m_vTarget += vY * fDeltaTime;
  }
  else if(GetKeyDown('E') || GetKeyDown('C') || GetKeyDown(VK_CONTROL))
  {
    m_vSource -= vY * fDeltaTime;
    m_vTarget -= vY * fDeltaTime;
  }

  CalculateMatrices();
}

// updates matrices from current settings
void Camera::CalculateMatrices(void)
{
  // view matrix
  m_mView = MatrixLookAtLH(m_vSource, m_vTarget, m_vUpVector);

  // projection matrix
  m_mProj = MatrixPerspectiveFovLH(m_fFOV, GetAppBase()->GetAspectRatio(), m_fNear, m_fFar);
}
