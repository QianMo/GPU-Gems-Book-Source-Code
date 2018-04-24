#pragma once

#include <windows.h>
#include <stdio.h>
#include <vector>
#include <set>

#include <math.h>
#include <float.h>

#include "Math.h"

// a simple function to convert degrees to radians
#define DegreeToRadian(fDegrees) ((3.14159265f/180.0f)*fDegrees)

// swap variables of any type
template<typename Type>
inline void Swap(Type &A, Type &B)
{
  Type C=A;
  A=B;
  B=C;
}

// clamp variables of any type
template<typename Type>
inline Type Clamp(const Type &A, const Type &Min, const Type &Max)
{
  if (A < Min) return Min;
  if (A > Max) return Max;
  return A;
}

// return smaller of the given variables
template<typename Type>
inline Type Min(const Type &A, const Type &B)
{
  if (A < B) return A;
  return B;
}

// return larger of the given variables
template<typename Type>
inline Type Max(const Type &A, const Type &B)
{
  if (A > B) return A;
  return B;
}

// time in seconds
inline double GetAccurateTime(void)
{
  __int64 iCurrentTime = 0;
  __int64 iFrequency = 1;
  QueryPerformanceFrequency((LARGE_INTEGER*)&iFrequency);
  QueryPerformanceCounter((LARGE_INTEGER*)&iCurrentTime);
  return (double)iCurrentTime / (double)iFrequency;
}

inline float DeltaTimeUpdate(double &fLastUpdate)
{
  double fTimeNow = GetAccurateTime();
  float fDeltaTime = 40.0f * Clamp((float)(fTimeNow - fLastUpdate), 0.0f, 1.0f);
  fLastUpdate = fTimeNow;
  return fDeltaTime;
}
