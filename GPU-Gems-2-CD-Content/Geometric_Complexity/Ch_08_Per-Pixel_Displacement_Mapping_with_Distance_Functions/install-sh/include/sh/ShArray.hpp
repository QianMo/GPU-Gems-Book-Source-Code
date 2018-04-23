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
#ifndef SHARRAY_HPP
#define SHARRAY_HPP

#include "ShBaseTexture.hpp"

namespace SH {

/** Default traits for ShArray.
 * An array is a texture that does not support filtering or interpolation.
 */
struct
ShArrayTraits : public ShTextureTraits {
  ShArrayTraits()
    : ShTextureTraits(0, SH_FILTER_NONE, SH_WRAP_CLAMP_TO_EDGE, SH_CLAMPED)
  {
  }
};

template<typename T> class ShArrayRect;

/** One-dimensional array.
 */
template<typename T>
class ShArray1D
  : public ShBaseTexture1D<T> {
public:
  ShArray1D()
    : ShBaseTexture1D<T>(ShArrayTraits())
  {}
  ShArray1D(int width)
    : ShBaseTexture1D<T>(width, ShArrayTraits())
  {}
	typedef ShArrayRect<T> rectangular_type;
	typedef ShBaseTexture1D<T> base_type;
  typedef T return_type;
};

/** Two-dimensional square power-of-two array.
 */
template<typename T>
class ShArray2D
  : public ShBaseTexture2D<T> {
public:
  ShArray2D()
    : ShBaseTexture2D<T>(ShArrayTraits())
  {}
  ShArray2D(int width, int height)
    : ShBaseTexture2D<T>(width, height, ShArrayTraits())
  {}
	typedef ShArrayRect<T> rectangular_type;
	typedef ShBaseTexture2D<T> base_type;
  typedef T return_type;
};

/** Two-dimensional non-square array.
 */
template<typename T>
class ShArrayRect
  : public ShBaseTextureRect<T> {
public:
  ShArrayRect()
    : ShBaseTextureRect<T>(ShArrayTraits())
  {}
  ShArrayRect(int width, int height)
    : ShBaseTextureRect<T>(width, height, ShArrayTraits())
  {}
	typedef ShArrayRect<T> rectangular_type;
	typedef ShBaseTextureRect<T> base_type;
  typedef T return_type;
};

/** Three-dimensional array.
 */
template<typename T>
class ShArray3D
  : public ShBaseTexture3D<T> {
public:
  ShArray3D()
    : ShBaseTexture3D<T>(ShArrayTraits())
  {}
  ShArray3D(int width, int height, int depth)
    : ShBaseTexture3D<T>(width, height, depth, ShArrayTraits())
  {}
	typedef ShArrayRect<T> rectangular_type;
	typedef ShBaseTexture3D<T> base_type;
  typedef T return_type;
};

/** Cube array.
 * A cube array is indexed by a 3D vector, and has six square power-of-two
 * faces.   The texel indexed depends only on the direction of the vector.
 */
template<typename T>
class ShArrayCube
  : public ShBaseTextureCube<T> {
public:
  ShArrayCube()
    : ShBaseTextureCube<T>(ShArrayTraits())
  {}
  ShArrayCube(int width, int height)
    : ShBaseTextureCube<T>(width, height, ShArrayTraits())
  {}
	typedef ShArrayRect<T> rectangular_type;
	typedef ShBaseTextureCube<T> base_type;
  typedef T return_type;
};

}

#endif
