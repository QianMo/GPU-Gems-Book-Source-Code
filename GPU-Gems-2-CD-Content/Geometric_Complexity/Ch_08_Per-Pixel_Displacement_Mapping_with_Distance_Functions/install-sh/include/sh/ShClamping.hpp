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
#ifndef SHCLAMPING_HPP
#define SHCLAMPING_HPP

namespace SH {

/** Set Clamp trait in Texture type.
 * Use this template to indicate that a texture should be set up with
 * the clamp trait enabled.   This version can be
 * used with any dimension of texture.
 */
template<typename T>
class ShClamped : public T {
public:
  ShClamped()
    : T()
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_CLAMPED);
  }
  ShClamped(int width)
    : T(width)
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_CLAMPED);
  }
  ShClamped(int width, int height)
    : T(width, height)
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_CLAMPED);
  }
  ShClamped(int width, int height, int depth)
    : T(width, height, depth)
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_CLAMPED);
  }

  typedef ShClamped<typename T::rectangular_type> rectangular_type;
  typedef typename T::base_type base_type;
  typedef typename T::return_type return_type;
};

/** Reset Clamp trait in Texture type.
 * Use this template to indicate that a texture should be set up without
 * the clamp trait enabled.   This version can be
 * used with any dimension of texture.
 */
template<typename T>
class ShUnclamped : public T {
public:
  ShUnclamped()
    : T()
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_UNCLAMPED);
  }
  ShUnclamped(int width)
    : T(width)
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_UNCLAMPED);
  }
  ShUnclamped(int width, int height)
    : T(width, height)
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_UNCLAMPED);
  }
  ShUnclamped(int width, int height, int depth)
    : T(width, height, depth)
  {
    this->m_node->traits().clamping(ShTextureTraits::SH_UNCLAMPED);
  }

  typedef ShUnclamped<typename T::rectangular_type> rectangular_type;
  typedef typename T::base_type base_type;
  typedef typename T::return_type return_type;
};

}

#endif
