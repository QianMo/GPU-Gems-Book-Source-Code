// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
// 
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
// 
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
// 
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
// 
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif /* WIN32 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <GL/gl.h>
#include "Camera.hpp"

using namespace SH;

Camera::Camera()
{
  proj = perspective(45, 1, 0.1, 100);
}

void Camera::glModelView()
{
  float values[16];
  for (int i = 0; i < 16; i++) trans[i%4](i/4).getValues(&values[i]);
  glMultMatrixf(values);
  for (int i = 0; i < 16; i++) rots[i%4](i/4).getValues(&values[i]);
  glMultMatrixf(values);
}

void Camera::glProjection(float aspect)
{
  proj = perspective(45, aspect, 0.1, 100);
  float values[16];
  for (int i = 0; i < 16; i++) proj[i%4](i/4).getValues(&values[i]);
  glMultMatrixf(values);
}

ShMatrix4x4f Camera::shModelView()
{
  return (trans | rots);
}

ShMatrix4x4f Camera::shInverseModelView()
{
  ShMatrix4x4f invtrans; 
  invtrans[0](3) = -trans[0](3);
  invtrans[1](3) = -trans[1](3);
  invtrans[2](3) = -trans[2](3);

  return transpose(rots) | invtrans;
}

ShMatrix4x4f Camera::shModelViewProjection(ShMatrix4x4f viewport)
{
  return (viewport | (proj | (trans | rots)));
}

void Camera::move(float x, float y, float z)
{
  ShMatrix4x4f m;
  m[0](3) = x;
  m[1](3) = y;
  m[2](3) = z;

  trans = (m | trans);
}

void Camera::rotate(float a, float x, float y, float z)
{
  float cosa = cosf((M_PI/180)*a);
  float sina = sinf((M_PI/180)*a);
  ShMatrix4x4f m;
    
  m[0](0) = x*x*(1-cosa) +   cosa;
  m[0](1) = x*y*(1-cosa) - z*sina;
  m[0](2) = x*z*(1-cosa) + y*sina;
      
  m[1](0) = y*x*(1-cosa) + z*sina;
  m[1](1) = y*y*(1-cosa) +   cosa;
  m[1](2) = y*z*(1-cosa) - x*sina;
      
  m[2](0) = z*x*(1-cosa) - y*sina;
  m[2](1) = z*y*(1-cosa) + x*sina;
  m[2](2) = z*z*(1-cosa) +   cosa;

  rots = (m | rots);
}

//-------------------------------------------------------------------
// perspective 
//-------------------------------------------------------------------
ShMatrix4x4f Camera::perspective(float fov, float aspect, float znear, float zfar)
{
  float zmin = znear;
  float zmax = zfar;
  float ymax = zmin*tan(fov*(M_PI/360));
  float ymin = -ymax;
  float xmin = ymin*aspect;
  float xmax = ymax*aspect;

  ShMatrix4x4f ret;

  ret[0](0) = 2.0*zmin/(xmax-xmin);
  ret[0](1) = 0.0;
  ret[0](2) = 0.0;
  ret[0](3) = 0.0;

  ret[1](0) = 0.0;
  ret[1](1) = 2.0*zmin/(ymax-ymin);
  ret[1](2) = 0.0;
  ret[1](3) = 0.0;

  ret[2](0) = 0.0;
  ret[2](1) = 0.0;
  ret[2](2) = -(zmax+zmin)/(zmax-zmin);
  ret[2](3) = -2.0*zmax*zmin/(zmax-zmin);

  ret[3](0) = 0.0;
  ret[3](1) = 0.0;
  ret[3](2) = -1.0;
  ret[3](3) = 0.0;

  return ret;
}
