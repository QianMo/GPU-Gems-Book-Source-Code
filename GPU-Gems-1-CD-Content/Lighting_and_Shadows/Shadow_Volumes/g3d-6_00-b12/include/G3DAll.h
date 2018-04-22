/** 
  @file G3DAll.h
 
  Includes all graphics3D/GLG3D files and, uses the G3D namespace, and
  under MSVC automatically adds the required files to the library list.
  
  This requires OpenGL and SDL headers.  If you don't want all of this,
  #include <graphics3d.h> separately.

  @maintainer Morgan McGuire, matrix@graphics3d.com
 
  @created 2002-01-01
  @edited  2002-12-13

 Copyright 2000-2003, Morgan McGuire.
 All rights reserved.
 */

#ifndef G3D_G3DALL_H
#define G3D_G3DALL_H

#include "graphics3D.h"
#include "GLG3D.h"

using namespace G3D;
using G3D::Texture;
using G3D::TextureRef;
using G3D::RenderDevice;
using G3D::AMPM;

#endif
