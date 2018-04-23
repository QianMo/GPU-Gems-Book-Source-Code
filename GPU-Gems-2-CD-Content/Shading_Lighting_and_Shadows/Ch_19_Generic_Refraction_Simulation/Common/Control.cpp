///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Control.cpp
//  Desc : Hardcoded camera controler, for debugging.
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Control.h"
#include "Win32App.h"
#include "Camera.h"

void CCameraControl:: Release()
{
  m_pCamera=0;
  m_pInput=0;
}

int CCameraControl:: Update(float fTimeSpan)
{
  if(!m_pCamera || !m_pInput)
  {
    return APP_ERR_INVALIDPARAM;
  }

  float fDirection=0, fSlide=0, fTimeScale=fTimeSpan; // rescale time

  // get mouse relative position
  m_pInput->UpdateMouseInput(1);
  CVector3f *pCoordinates=m_pInput->GetMouseRelativeCoordinates();

  if(m_pInput->GetKeyPressed(VK_SPACE)) 
  {
    fTimeScale*=10.0f;
  }

  // get camera properties
  float fHeading=m_pCamera->GetHeading(), fPitch=m_pCamera->GetPitch();
  CVector3f pPosition=m_pCamera->GetPosition();
  CVector3f pDirection=m_pCamera->GetDirection();

  if(m_pInput->GetKeyPressed('W')) 
  { 
    fDirection=+fTimeScale; 
  }

  if(m_pInput->GetKeyPressed('S')) 
  { 
    fDirection=-fTimeScale; 
  }

  if(m_pInput->GetKeyPressed('A')) 
  { 
    fSlide=-fTimeScale;  
  }

  if(m_pInput->GetKeyPressed('D')) 
  {
    fSlide=fTimeScale; 
  }

  fHeading+=(pCoordinates->m_fX);    
  fPitch+=(pCoordinates->m_fY);  

  // clamp pitching
  if(fPitch>89)  
  {
    fPitch=89;
  }

  if(fPitch<-89) 
  {
    fPitch=-89;
  }
  
  // compute camera front vector 
  pDirection.m_fX=sinf(DEGTORAD(-fHeading))*cosf(DEGTORAD(fPitch));
  pDirection.m_fY=cosf(DEGTORAD(-fHeading))*cosf(DEGTORAD(fPitch));
  pDirection.m_fZ=sinf(DEGTORAD(fPitch));

  // update camera position 
  pPosition+=pDirection*fDirection;

  // update camera data
  m_pCamera->SetPosition(pPosition);
  m_pCamera->SetDirection(pDirection);
  m_pCamera->SetHeading(fHeading);
  m_pCamera->SetPitch(fPitch);

  // make camera look at pivot
  m_pCamera->LookAt(pPosition, pPosition+pDirection, CVector3f(0.0f,0.0f,1.0f));  

  // apply "slidding"
  if(fSlide)
  {
    m_pCamera->Slide(fSlide, 0, 0);
  }
  return APP_OK;
}