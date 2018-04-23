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
#ifndef SHLIBDERIVIMPL_HPP
#define SHLIBDERIVIMPL_HPP

#include "ShLibDeriv.hpp"

namespace SH {

template<int N, typename T>
inline
ShGeneric<N, T> dx(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shDX(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> dy(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shDY(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> fwidth(const ShGeneric<N, T>& var)
{
  return max(abs(dx(var)), abs(dy(var)));
}

template<typename T>
inline
ShGeneric<2, T> gradient(const ShGeneric<1, T>& var)
{
  return ShAttrib2f(dx(var), dy(var));
}

template<int N, typename T>
inline
ShMatrix<2, N, SH_TEMP, T> jacobian(const ShGeneric<N, T>& var)
{
  ShMatrix<2, N, SH_TEMP, T> ret;
  ret[0] = dx(var);
  ret[1] = dy(var);
  return ret;
}



}

#endif
