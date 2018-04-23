/* 
  Copyright (C) 2003, Kevin Moule (krmoule@cgl.uwaterloo.ca)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>

#ifdef WIN32
#include <windows.h>
#endif /* WIN32 */

#include "timer.h"

#ifdef WIN32
//-------------------------------------------------------------------
// mtimer_t::mtimer_t
//-------------------------------------------------------------------
mtimer_t::mtimer_t(void)
  {
  t.QuadPart = 0;
  }

//-------------------------------------------------------------------
// mtimer_t::~mtimer_t
//-------------------------------------------------------------------
mtimer_t::~mtimer_t(void)
  {
  }

//-------------------------------------------------------------------
// mtimer_t::zero
//-------------------------------------------------------------------
mtimer_t mtimer_t::zero(void)
  {
  mtimer_t ret;
  ret.t.QuadPart = 0;
  return ret;
  }

//-------------------------------------------------------------------
// mtimer_t::now
//-------------------------------------------------------------------
mtimer_t mtimer_t::now(void)
  {
  mtimer_t ret;
  QueryPerformanceCounter(&ret.t);
  return ret;
  }

//-------------------------------------------------------------------
// mtimer_t::value
//-------------------------------------------------------------------
float mtimer_t::value(void) const
  {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return (float)(t.QuadPart)/(freq.QuadPart/1000);
  }

//-------------------------------------------------------------------
// operator+
//-------------------------------------------------------------------
mtimer_t operator+(const mtimer_t& a, const mtimer_t& b)
  {
  mtimer_t ret;
  ret.t.QuadPart = a.t.QuadPart + b.t.QuadPart;
  return ret;
  }

//-------------------------------------------------------------------
// operator-
//-------------------------------------------------------------------
mtimer_t operator-(const mtimer_t& a, const mtimer_t& b)
  {
  mtimer_t ret;
  ret.t.QuadPart = a.t.QuadPart - b.t.QuadPart;
  return ret;
  }

//-------------------------------------------------------------------
// operator<<(std::ostream, mtimer_t
//-------------------------------------------------------------------
std::ostream& operator<<(std::ostream& s, const mtimer_t& t)
  {
  s << t.value();
  return s;
  }

#else

//-------------------------------------------------------------------
// mtimer_t::mtimer_t
//-------------------------------------------------------------------
mtimer_t::mtimer_t(void)
  {
  t.tv_sec = 0;
  t.tv_usec = 0;
  }

//-------------------------------------------------------------------
// mtimer_t::~mtimer_t
//-------------------------------------------------------------------
mtimer_t::~mtimer_t(void)
  {
  }

//-------------------------------------------------------------------
// mtimer_t::zero
//-------------------------------------------------------------------
mtimer_t mtimer_t::zero(void)
  {
  mtimer_t ret;
  ret.t.tv_sec = 0;
  ret.t.tv_usec = 0;
  return ret;
  }

//-------------------------------------------------------------------
// mtimer_t::now
//-------------------------------------------------------------------
mtimer_t mtimer_t::now(void)
  {
  mtimer_t ret;
  gettimeofday(&ret.t, NULL);
  return ret;
  }

//-------------------------------------------------------------------
// mtimer_t::value
//-------------------------------------------------------------------
float mtimer_t::value(void) const
  {
  float sec = (float)t.tv_sec*1000;
  float msec = (float)t.tv_usec/1000;
  return (sec + msec);
  }

//-------------------------------------------------------------------
// operator+
//-------------------------------------------------------------------
mtimer_t operator+(const mtimer_t& a, const mtimer_t& b)
  {
  mtimer_t ret;

  ret.t.tv_sec = a.t.tv_sec + b.t.tv_sec;
  ret.t.tv_usec = a.t.tv_usec + b.t.tv_usec;

  // adjust microsecond if the value is out of range
  if (ret.t.tv_usec < 0)
    {
    ret.t.tv_sec -= 1;
    ret.t.tv_usec += 1000000;
    }

  // adjust microsecond if the value is out of range
  if (ret.t.tv_usec > 1000000)
    {
    ret.t.tv_sec += 1;
    ret.t.tv_usec -= 1000000;
    }

  return ret;
  }

//-------------------------------------------------------------------
// operator-
//-------------------------------------------------------------------
mtimer_t operator-(const mtimer_t& a, const mtimer_t& b)
  {
  mtimer_t ret;
  
  ret.t.tv_sec = a.t.tv_sec - b.t.tv_sec;
  ret.t.tv_usec = a.t.tv_usec - b.t.tv_usec;

  // adjust microsecond if the value is out of range
  if (ret.t.tv_usec < 0)
    {
    ret.t.tv_sec -= 1;
    ret.t.tv_usec += 1000000;
    }

  // adjust microsecond if the value is out of range
  if (ret.t.tv_usec > 1000000)
    {
    ret.t.tv_sec += 1;
    ret.t.tv_usec -= 1000000;
    }

  return ret;
  }

//-------------------------------------------------------------------
// operator<<(std::ostream, mtimer_t)
//-------------------------------------------------------------------
std::ostream& operator<<(std::ostream& s, const mtimer_t& t)
  {
  float sec = (float)t.t.tv_sec*1000;
  float msec = (float)t.t.tv_usec/1000;
  s << (sec + msec);
  return s;
  }

#endif /* WIN32 */
