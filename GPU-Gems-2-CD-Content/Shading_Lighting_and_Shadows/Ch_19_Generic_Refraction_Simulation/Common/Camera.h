///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Camera.h
//  Desc : Generic camera class
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class  CCamera
{
public:
  CCamera():m_pPosition(0,0,0) 
  {
    m_fFov=0;
    m_fAspect=0;
    m_fNearPlane=0;
    m_fFarPlane=0;
    m_iWidth=0;
    m_iHeight=0;
    m_fPitch=0;
    m_fHeading=0; 
    m_fRoll=0;
  }

  ~CCamera()
  {
    Release();
  }

  // Create camera 
  int Create(float fFov, int iScrx, int iScry, float fNearPlane, float fFarPlane);
  // Release 
  void Release();  
  // Make cam look at pivot
  void LookAt(const CVector3f &pEye, const CVector3f &pPivot, const CVector3f &pUp);
  // Slide camera
  void Slide(float fDeltaU, float fDeltaV, float fDeltaN);  
  // Set camera shape
  void SetShape(float fFov, int iScrx, int iScry, float fNearPlane=0.0f, float fFarPlane=1.0f);

  // Set camera eye position
  void SetPosition(const CVector3f &pEye)
  {
    m_pPosition=pEye;
  }

  // Set camera direction
  void SetDirection(const CVector3f &pDirection)
  {
    m_pDirection=pDirection;
  }

  // Get camera position
  const CVector3f &GetPosition() const
  {
    return  m_pPosition;
  }

  // Get camera direction
  const CVector3f &GetDirection() const
  {
    return  m_pDirection;
  }

  // Get euler angles
  float GetHeading() const
  {
    return  m_fHeading;
  }

  float GetPitch() const
  {
    return  m_fPitch;
  }

  float GetRoll() const
  {
    return m_fRoll;
  }

  // Set euler angles
  void SetHeading(float fHeading)
  {
    m_fHeading=fHeading;
  }

  void SetPitch(float fPitch)
  {
    m_fPitch=fPitch;
  }

  void SetRoll(float fRoll)
  {
    m_fRoll=fRoll;
  }

  // Get current camera view matrix
  const CMatrix44f &GetViewMatrix() const
  {
    return m_pView; 
  }

  // Get current projection matrix
  const CMatrix44f &GetProjectionMatrix() const
  {
    return m_pProjection;
  }

  // Returns camera properties
  void GetCameraProperties(float &fFov, float &fAspect, float &fNearPlane, float &fFarPlane) const
  {
    fFov=m_fFov;
    fAspect=m_fAspect;
    fNearPlane=m_fNearPlane;
    fFarPlane=m_fFarPlane;
  }

  // Returns camera properties
  void GetCameraSize(int &iWidth, int &iHeight) const
  {
    iWidth=m_iWidth;
    iHeight=m_iHeight;
  }

  // Returns direction vectors
  void GetDirectionVectors(CVector3f &pU, CVector3f &pV, CVector3f &pN) const
  {
    pU=m_pU;
    pV=m_pV;
    pN=m_pN;
  }

private:

  // Camera properties
  int m_iWidth, m_iHeight;
  float m_fFov, m_fAspect, m_fNearPlane, m_fFarPlane, m_fPitch, m_fHeading, m_fRoll;
  CVector3f m_pPosition, m_pDirection, m_pU, m_pV, m_pN;  
  CMatrix44f m_pView, m_pProjection;

  // Setup camera matrix
  void SetMatrix();
  // Set camera perspective
  void SetPerspective();
};