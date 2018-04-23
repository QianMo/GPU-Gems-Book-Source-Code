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
#ifndef SHWRAP_HPP
#define SHWRAP_HPP

namespace SH {

/** Set WrapClamp trait in Texture type.
 * Use this template to indicate that a texture should be set up with 
 * the wrap/clamp boundary treatment enabled.   This version can be
 * used with any dimension of texture.
 */
template<typename T>
class ShWrapClamp : public T {
public:
  ShWrapClamp()
    : T()
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP);
  }
  ShWrapClamp(int width)
    : T(width)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP);
  }
  ShWrapClamp(int width, int height)
    : T(width, height)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP);
  }
  ShWrapClamp(int width, int height, int depth)
    : T(width, height, depth)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP);
  }

  typedef ShWrapClamp<typename T::rectangular_type> rectangular_type;
  typedef typename T::base_type base_type;
  typedef typename T::return_type return_type;
};

/** Set WrapClampToEdge trait in Texture type.
 * Use this template to indicate that a texture should be set up with 
 * the wrap/clamp-to-boundary trait enabled.   This version can be
 * used with any dimension of texture.
 */
template<typename T>
class ShWrapClampToEdge : public T {
public:
  ShWrapClampToEdge()
    : T()
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP_TO_EDGE);
  }
  ShWrapClampToEdge(int width)
    : T(width)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP_TO_EDGE);
  }
  ShWrapClampToEdge(int width, int height)
    : T(width, height)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP_TO_EDGE);
  }
  ShWrapClampToEdge(int width, int height, int depth)
    : T(width, height, depth)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_CLAMP_TO_EDGE);
  }

  typedef ShWrapClampToEdge<typename T::rectangular_type> rectangular_type;
  typedef typename T::base_type base_type;
  typedef typename T::return_type return_type;
};

/** Set WrapRepeat trait in Texture type.
 * Use this template to indicate that a texture should be set up with 
 * the wrap-repeat boundary trait enabled.   This version can be
 * used with any dimension of texture.
 */
template<typename T>
class ShWrapRepeat : public T {
public:
  ShWrapRepeat()
    : T()
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_REPEAT);
  }
  ShWrapRepeat(int width)
    : T(width)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_REPEAT);
  }
  ShWrapRepeat(int width, int height)
    : T(width, height)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_REPEAT);
  }
  ShWrapRepeat(int width, int height, int depth)
    : T(width, height, depth)
  {
    this->m_node->traits().wrapping(ShTextureTraits::SH_WRAP_REPEAT);
  }

  typedef ShWrapRepeat<typename T::rectangular_type> rectangular_type;
  typedef typename T::base_type base_type;
  typedef typename T::return_type return_type;
};

}

#endif
