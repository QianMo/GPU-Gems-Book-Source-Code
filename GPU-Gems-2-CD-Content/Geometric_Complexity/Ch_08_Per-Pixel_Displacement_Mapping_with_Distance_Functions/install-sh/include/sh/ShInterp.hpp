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
#ifndef SHINTERP_HPP
#define SHINTERP_HPP

namespace SH {

/** Set Interpolation level in Texture type.
 * Use this template to indicate that a texture should be interpolated
 * to a particular level L.
 * For example, ShInterp<0, T> implies a nearest-neighbour lookup,
 * whereas ShInterp<1, T> implies linear interpolation.
 */
template<int L, typename T>
class ShInterp : public T {
public:
  static int level() {
    if (L >= 2) return 3; else return L;
  }
  
  ShInterp()
    : T()
  {
    this->m_node->traits().interpolation(level());
  }
  ShInterp(int width)
    : T(width)
  {
    this->m_node->traits().interpolation(level());
  }
  ShInterp(int width, int height)
    : T(width, height)
  {
    this->m_node->traits().interpolation(level());
  }
  ShInterp(int width, int height, int depth)
    : T(width, height, depth)
  {
    this->m_node->traits().interpolation(level());
  }

  typedef typename T::return_type return_type;
  
};

}

#endif
