#pragma once

#include "Frustum.h"

class SceneObject;

class Light
{
public:
  Light();

  // finds scene objects that overlap given frustum from light's view
  std::vector<SceneObject *> FindCasters(const Frustum &frustum);

  // processes light controls
  void DoControls(void);

  // calculates light matrix to contain given visible objects
  void CalculateMatrices(void);

  // crops the light volume on given frustum (scene-independent projection)
  Matrix CalculateCropMatrix(const Frustum &frustum);

  // crops the light volume on given objects, constrained by given frustum
  Matrix CalculateCropMatrix(const std::vector<SceneObject *> &casters, const std::vector<SceneObject *> &receivers, const Frustum &frustum);

  // retuns direction of light
  Vector3 GetDir(void);

private:
  
  // build a matrix for cropping light's projection
  inline Matrix BuildCropMatrix(const Vector3 &vMin, const Vector3 &vMax);

public:

  enum LightType
  {
    TYPE_ORTHOGRAPHIC, // eg. directional light
    TYPE_PERSPECTIVE   // eg. spot light
  };

  // type of light
  LightType m_Type;

  // matrices, updated with CalculateMatrices()
  Matrix m_mView;
  Matrix m_mProj;

  // settings
  Vector3 m_vSource;
  Vector3 m_vTarget;
  float m_fNear;
  float m_fFar;
  float m_fFOV;
  Vector3 m_vUpVector;
  float m_fAspectRatio;
  Vector3 m_vLightDiffuse;
  Vector3 m_vLightAmbient;

private:

  struct ControlState
  {
    Vector3 m_vRotation;
    bool m_bSwitchingType;
    double m_fLastUpdate;
  };

  ControlState m_ControlState;
};
