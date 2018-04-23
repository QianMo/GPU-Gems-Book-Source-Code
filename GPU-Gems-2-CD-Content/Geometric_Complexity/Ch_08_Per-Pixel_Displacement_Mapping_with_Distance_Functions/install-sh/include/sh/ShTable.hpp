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
#ifndef SHTABLE_HPP
#define SHTABLE_HPP

#include "ShBaseTexture.hpp"

namespace SH {

struct
ShTableTraits : public ShTextureTraits {
  ShTableTraits()
    : ShTextureTraits(1, SH_FILTER_NONE, SH_WRAP_CLAMP_TO_EDGE, SH_CLAMPED)
  {
  }
};

template<typename T> class ShTableRect;

template<typename T>
class ShTable1D
  : public ShBaseTexture1D<T> {
public:
  ShTable1D()
    : ShBaseTexture1D<T>(ShTableTraits())
  {}
  ShTable1D(int width)
    : ShBaseTexture1D<T>(width, ShTableTraits())
  {}
	typedef ShTableRect<T> rectangular_type;
	typedef ShBaseTexture1D<T> base_type;
  typedef T return_type;
};

template<typename T>
class ShTable2D
  : public ShBaseTexture2D<T> {
public:
  ShTable2D()
    : ShBaseTexture2D<T>(ShTableTraits())
  {}
  ShTable2D(int width, int height)
    : ShBaseTexture2D<T>(width, height, ShTableTraits())
  {}
	typedef ShTableRect<T> rectangular_type;
	typedef ShBaseTexture2D<T> base_type;
  typedef T return_type;
};

template<typename T>
class ShTableRect
  : public ShBaseTextureRect<T> {
public:
  ShTableRect()
    : ShBaseTextureRect<T>(ShTableTraits())
  {}
  ShTableRect(int width, int height)
    : ShBaseTextureRect<T>(width, height, ShTableTraits())
  {}
	typedef ShTableRect<T> rectangular_type;
	typedef ShBaseTextureRect<T> base_type;
  typedef T return_type;
};

template<typename T>
class ShTable3D
  : public ShBaseTexture3D<T> {
public:
  ShTable3D()
    : ShBaseTexture3D<T>(ShTableTraits())
  {}
  ShTable3D(int width, int height, int depth)
    : ShBaseTexture3D<T>(width, height, depth, ShTableTraits())
  {}
	typedef ShTableRect<T> rectangular_type;
	typedef ShBaseTexture3D<T> base_type;
  typedef T return_type;
};

template<typename T>
class ShTableCube
  : public ShBaseTextureCube<T> {
public:
  ShTableCube()
    : ShBaseTextureCube<T>(ShTableTraits())
  {}
  ShTableCube(int width, int height)
    : ShBaseTextureCube<T>(width, height, ShTableTraits())
  {}
	typedef ShTableRect<T> rectangular_type;
	typedef ShBaseTextureCube<T> base_type;
  typedef T return_type;
};

}

#endif
