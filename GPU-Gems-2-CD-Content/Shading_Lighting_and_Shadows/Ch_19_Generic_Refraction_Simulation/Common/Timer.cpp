///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Timer.cpp
//  Desc : Generic timer class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Timer.h"
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")

// create timer
int CTimer:: Create()
{
  // im assuming performance timer is always available
  QueryPerformanceFrequency((LARGE_INTEGER *) &m_nFrequency);
  // get start time
  QueryPerformanceCounter((LARGE_INTEGER *) &m_nStart);

  // compute timer resolution
  m_fResolution =  1.0f/((float)m_nFrequency);
  
  //m_fResolution=1.0f/1000.0f;
  //m_nStart=timeGetTime();

  // elapsed time
  m_nRelativeStart = m_nStart;
  m_nLastTime= m_nStart;

  return APP_OK;
}

// reset timer
void CTimer:: Reset()
{
  m_nLastTime=0;
  m_nCurrTime=0;
  m_fCurrentTime=0;
  m_fRelativeTime=0;

  QueryPerformanceCounter((LARGE_INTEGER *) &m_nStart);
  //m_nStart=timeGetTime();
  m_nRelativeStart=m_nStart;
  m_nLastTime=m_nStart;  
}

// get current time 
float CTimer:: GetCurrTime()
{
  __int64 nTime;
  // get current time
  QueryPerformanceCounter((LARGE_INTEGER *) &nTime);
  //nTime=timeGetTime();
  m_fCurrentTime=((float)(nTime-((float)m_nStart))* m_fResolution);
  
  return m_fCurrentTime;
}

// get current time 
float CTimer:: GetRelativeTime()
{
  __int64 nTime;

  // get current time
  QueryPerformanceCounter((LARGE_INTEGER *) &nTime);
 // nTime=timeGetTime();
  m_fRelativeTime=((float)(nTime-((float)m_nRelativeStart))* m_fResolution);
  m_nRelativeStart=nTime;
  
  return m_fRelativeTime;
}
