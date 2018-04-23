///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Control.h
//  Desc : Hardcoded camera controler, for debugging
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CCamera;
class CWin32Input;

class CCameraControl 
{
public:
  CCameraControl()
  {
    m_pCamera=0;
    m_pInput=0;
    m_fVelocity=0;
    m_fAceleration=0;
    m_bMoved=1;
  }
  ~CCameraControl()
  {
    Release();
  }

  // Set controler input  
  void SetInput(CWin32Input &pInput)
  {
    m_pInput=&pInput;
  }
  // Set camera
  void SetCamera(CCamera &pCam)
  {
    m_pCamera=&pCam;
  }

  // Release resources
  void Release();
  // Update controler
  int Update(float fTimeSpan);

private:
  CCamera     *m_pCamera;
  CWin32Input *m_pInput;
  float        m_fVelocity,
               m_fAceleration;
  bool         m_bMoved;
};
