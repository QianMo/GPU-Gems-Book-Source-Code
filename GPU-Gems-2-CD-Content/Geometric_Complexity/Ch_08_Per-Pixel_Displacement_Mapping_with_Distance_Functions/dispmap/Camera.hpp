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
#ifndef DISPMAP_CAMERA_HPP
#define DISPMAP_CAMERA_HPP

#include <sh/sh.hpp>

class Camera {
public:
  Camera();

  void move(float x, float y, float z);
  void rotate(float a, float x, float y, float z);
  
  void glModelView();
  void glProjection(float aspect);

  SH::ShMatrix4x4f shModelView();
  SH::ShMatrix4x4f shInverseModelView();
  SH::ShMatrix4x4f shModelViewProjection(SH::ShMatrix4x4f viewport);

  void resetRotation() {
    rots = SH::ShMatrix4x4f();
  }

private:
  SH::ShMatrix4x4f perspective(float fov, float aspect, float znear, float zfar);

  SH::ShMatrix4x4f proj;
  SH::ShMatrix4x4f rots;
  SH::ShMatrix4x4f trans;
};

#endif
