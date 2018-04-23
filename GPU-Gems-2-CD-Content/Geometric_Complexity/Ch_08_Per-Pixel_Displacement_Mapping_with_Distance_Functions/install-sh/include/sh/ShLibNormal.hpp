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
#ifndef SHLIBNORMAL_HPP
#define SHLIBNORMAL_HPP

#include "ShNormal.hpp"
#include "ShVector.hpp"
#include "ShLib.hpp"

namespace SH {

SH_SHLIB_USUAL_NON_UNIT_OPS_RETTYPE(ShNormal, ShNormal);

template<int N, ShBindingType B1, typename T, bool S1>
  ShNormal<N, SH_TEMP, T, false>
  abs(const ShNormal<N, B1, T, S1>& var)
  {
    ShGeneric<N, T> t = abs(static_cast< ShGeneric<N, T> >(var));
    ShNormal<N, SH_TEMP, T, false> vec(t.node(), t.swizzle(), t.neg());
    return vec;
  }

template<int N, ShBindingType B1, typename T, bool S1>
  ShNormal<N, SH_TEMP, T, false>
  normalize(const ShNormal<N, B1, T, S1>& var)
  {
    ShGeneric<N, T> t = normalize(static_cast< ShGeneric<N, T> >(var));
    ShNormal<N, SH_TEMP, T, false> vec(t.node(), t.swizzle(), t.neg());
    return vec;
  }

SH_SHLIB_USUAL_SUBTRACT(ShNormal);

SH_SHLIB_LEFT_MATRIX_OPERATION(ShNormal, operator|, M);

template<int N, ShBindingType B1, ShBindingType B2, typename T, bool S1, bool S2>
ShGeneric<1, T> operator|(const ShNormal<N, B1, T, S1>& a,
                            const ShNormal<N, B2, T, S2>& b)
{
  return dot(a, b);
}

template<int N, ShBindingType B1, ShBindingType B2, typename T, bool S1, bool S2>
ShGeneric<1, T> operator|(const ShVector<N, B1, T, S1>& a,
                            const ShNormal<N, B2, T, S2>& b)
{
  return dot(a, b);
}

template<int N, ShBindingType B1, ShBindingType B2, typename T, bool S1, bool S2>
ShGeneric<1, T> operator|(const ShNormal<N, B1, T, S1>& a,
                            const ShVector<N, B2, T, S2>& b)
{
  return dot(a, b);
}

template<ShBindingType B1, ShBindingType B2, typename T, bool S1>
ShNormal<3, SH_TEMP, T, false> operator|(const ShMatrix<4, 4, B1, T>& m,
                                             const ShNormal<3, B2, T, S1>& v)
{
  ShNormal<3, SH_TEMP, T, false> t;
  for (int i = 0; i < 3; i++) {
    t(i) = dot(m[i](0,1,2), v);
  }
  return t;
}

template<ShBindingType B1, ShBindingType B2, typename T, bool S1>
ShNormal<2, SH_TEMP, T, false> operator|(const ShMatrix<3, 3, B1, T>& m,
                                             const ShNormal<2, B2, T, S1>& v)
{
  ShNormal<2, SH_TEMP, T, false> t;
  for (int i = 0; i < 2; i++) {
    t(i) = dot(m[i](0,1), v);
  }
  return t;
}

template<ShBindingType B1, ShBindingType B2, typename T, bool S1>
ShNormal<1, SH_TEMP, T, false> operator|(const ShMatrix<2, 2, B1, T>& m,
                                         const ShNormal<1, B2, T, S1>& v)
{
  ShNormal<1, SH_TEMP, T, false> t;
  for (int i = 0; i < 1; i++) {
    t(i) = dot(m[i](0), v);
  }
  return t;
}


}

#endif
