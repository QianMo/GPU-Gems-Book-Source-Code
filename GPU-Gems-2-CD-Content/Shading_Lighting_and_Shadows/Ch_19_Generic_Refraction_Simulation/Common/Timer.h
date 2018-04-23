///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Timer.h
//  Desc : Generic timer class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Common.h"

class CTimer
{
public:  
  CTimer():m_nFrequency(0), m_nStart(0), m_nRelativeStart(0), m_fResolution(0),
      m_nLastTime(0), m_nCurrTime(0), m_fCurrentTime(0),m_fRelativeTime(0)
  {
  };

  // Create/Reset timer
  int Create();
  void Reset();

  // Get current time 
  float GetRelativeTime();
  float GetCurrTime();
  
  // Get timer frequency
  __int64 GetFrequency()
  {
    return m_nFrequency; 
  };

   
private:
  __int64  m_nFrequency,         // frequency
           m_nStart,             // performance timer start 
           m_nRelativeStart,     // performance relative timer start
           m_nLastTime,          // previous timer value
           m_nCurrTime;          // current time

  float    m_fResolution,        // resolution
           m_fCurrentTime,       // current time in seconds
           m_fRelativeTime;      // relative time in seconds
};
