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
#ifndef SHTEXTURE_HPP
#define SHTEXTURE_HPP

#include "ShBaseTexture.hpp"

namespace SH {

/** Trait class to represent texture filtering modes.
 */
struct 
ShFilteredTextureTraits : public ShTextureTraits {
  ShFilteredTextureTraits()
    : ShTextureTraits(1, SH_FILTER_MIPMAP, SH_WRAP_CLAMP_TO_EDGE, SH_CLAMPED)
  {
  }
};

template<typename T> class ShTextureRect;

/** One-dimensional power-of-two texture type.
 */
template<typename T>
class ShTexture1D
  : public ShBaseTexture1D<T> {
public:
  ShTexture1D()
    : ShBaseTexture1D<T>(ShFilteredTextureTraits())
  {}
  ShTexture1D(int width)
    : ShBaseTexture1D<T>(width, ShFilteredTextureTraits())
  {}
	typedef ShTextureRect<T> rectangular_type;
	typedef ShBaseTexture1D<T> base_type;
  typedef T return_type;
};

/** Two-dimensional square power-of-two texture type.
 */
template<typename T>
class ShTexture2D
  : public ShBaseTexture2D<T> {
public:
  ShTexture2D()
    : ShBaseTexture2D<T>(ShFilteredTextureTraits())
  {}
  ShTexture2D(int width, int height)
    : ShBaseTexture2D<T>(width, height, ShFilteredTextureTraits())
  {}
//    ShTexture2D(int width, int height, ShTextureTraits& traits)
//     : ShBaseTexture2D<T>(width, height, traits)
//   {}
	typedef ShTextureRect<T> rectangular_type;
	typedef ShBaseTexture2D<T> base_type;
  typedef T return_type;
};

/** Two-dimensional rectangular texture type.
 */
template<typename T>
class ShTextureRect
  : public ShBaseTextureRect<T> {
public:
  ShTextureRect()
    : ShBaseTextureRect<T>(ShFilteredTextureTraits())
  {}
  ShTextureRect(int width, int height)
    : ShBaseTextureRect<T>(width, height, ShFilteredTextureTraits())
  {}
//    ShTextureRect(int width, int height, ShTextureTraits& traits)
//     : ShBaseTextureRect<T>(width, height, traits)
//   {}
	typedef ShTextureRect<T> rectangular_type;
	typedef ShBaseTextureRect<T> base_type;
  typedef T return_type;
};

/** Three-dimensional texture type.
 */
template<typename T>
class ShTexture3D
  : public ShBaseTexture3D<T> {
public:
  ShTexture3D()
    : ShBaseTexture3D<T>(ShFilteredTextureTraits())
  {}
  ShTexture3D(int width, int height, int depth)
    : ShBaseTexture3D<T>(width, height, depth, ShFilteredTextureTraits())
  {}
	typedef ShTextureRect<T> rectangular_type;
	typedef ShBaseTexture3D<T> base_type;
  typedef T return_type;
};

/** Cubic texture type.
 * Cubic textures have six faces and are indexed by a three-dimensional
 * vector.   This vector does NOT have to be unit length.   The texel
 * indexed will depend only on the direction of the vector.
 */
template<typename T>
class ShTextureCube
  : public ShBaseTextureCube<T> {
public:
  ShTextureCube()
    : ShBaseTextureCube<T>(ShFilteredTextureTraits())
  {}
  ShTextureCube(int width, int height)
    : ShBaseTextureCube<T>(width, height, ShFilteredTextureTraits())
  {}
	typedef ShTextureRect<T> rectangular_type;
	typedef ShBaseTextureCube<T> base_type;
  typedef T return_type;
};

}

#endif
