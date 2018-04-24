#pragma once

#include "Frustum.h"

class SceneObject;

class Camera
{
public:
  Camera();
  
  // finds scene objects inside the camera frustum
  std::vector<SceneObject *> FindReceivers(void);

  // calculates split plane distances in view space
  void CalculateSplitPositions(float *pDistances);

  // calculates a frustum with given far and near planes
  Frustum CalculateFrustum(float fNear, float fFar);

  // adjust the camera planes to contain the visible scene as tightly as possible
  void AdjustPlanes(const std::vector<SceneObject *> &VisibleObjects);

  // processes camera controls
  void DoControls(void);

  // camera settings
  Vector3 m_vSource;
  Vector3 m_vTarget;
  float m_fNear;
  float m_fFar;
  float m_fFOV;
  Vector3 m_vUpVector;

  // used when adjusting camera planes
  float m_fFarMax;
  float m_fNearMin;

  // matrices, updated with CalculateMatrices()
  Matrix m_mView;
  Matrix m_mProj;

  // updates view and projection matrices from current settings
  void CalculateMatrices(void);

private:

  // camera control states
  struct ControlState
  {
    bool bZooming;
    bool bStrafing;
    bool bRotating;
    POINT pntMouse;
    double fLastUpdate;
    float fRotX;
    float fRotY;
  };

  ControlState m_ControlState;
};
