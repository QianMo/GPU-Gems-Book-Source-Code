///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Camera.cpp
//  Desc : Generic camera class
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Camera.h"

int CCamera::Create(float fFov, int iScrx, int iScry, float fNearPlane, float fFarPlane)
{
  SetShape(fFov, iScrx, iScry, fNearPlane, fFarPlane);
  return APP_OK;
}

void CCamera::Release()
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

void CCamera::LookAt(const CVector3f &pEye, const CVector3f &pPivot, const CVector3f &pUp)
{  
  // Compute n
  m_pN.Set(pEye.m_fX-pPivot.m_fX, pEye.m_fY-pPivot.m_fY, pEye.m_fZ-pPivot.m_fZ);
  // Compute u 
  m_pU=pUp.Cross(m_pN);
  // Compute v 
  m_pV=m_pN.Cross(m_pU);

  // Normalize vectors
  m_pU.Normalize(); // x axis
  m_pV.Normalize(); // y axis
  m_pN.Normalize(); // z axiz

  m_pPosition=pEye;

  // Update camera matrix
  SetMatrix();  
}

void CCamera::Slide(float fDeltaU, float fDeltaV, float fDeltaN)
{
  // Slide in cam axis
  m_pPosition.m_fX+=fDeltaU*m_pU.m_fX+fDeltaV*m_pV.m_fX+fDeltaN*m_pN.m_fX;
  m_pPosition.m_fY+=fDeltaU*m_pU.m_fY+fDeltaV*m_pV.m_fY+fDeltaN*m_pN.m_fY;
  m_pPosition.m_fZ+=fDeltaU*m_pU.m_fZ+fDeltaV*m_pV.m_fZ+fDeltaN*m_pN.m_fZ;

  // Update camera matrix
  SetMatrix();    
}

void CCamera::SetShape(float fFov, int iScrx, int iScry, float fNearPlane, float fFarPlane)
{
  // Save values
  m_iWidth=iScrx;
  m_iHeight=iScry;
  m_fFov=fFov; 
  m_fNearPlane=fNearPlane;
  m_fFarPlane=fFarPlane;

  // Compute aspect ratio
  m_fAspect=((float) m_iWidth/(float) m_iHeight);
  
  // Update projection matrix
  SetPerspective();
}

void CCamera::SetMatrix()
{    
  CVector3f pEyeDotDir(-m_pPosition.Dot(m_pU), -m_pPosition.Dot(m_pV),-m_pPosition.Dot(m_pN));

  // Compute camera matrix
  m_pView.Set(m_pU.m_fX      , m_pV.m_fX      , m_pN.m_fX      , 0,
              m_pU.m_fY      , m_pV.m_fY      , m_pN.m_fY      , 0,
              m_pU.m_fZ      , m_pV.m_fZ      , m_pN.m_fZ      , 0,
              pEyeDotDir.m_fX, pEyeDotDir.m_fY, pEyeDotDir.m_fZ, 1);
}

void CCamera:: SetPerspective()
{
  m_pProjection.Identity();
  m_pProjection.PerspectiveFovRH(DEGTORAD(m_fFov), m_fAspect, m_fNearPlane, m_fFarPlane);    
}
