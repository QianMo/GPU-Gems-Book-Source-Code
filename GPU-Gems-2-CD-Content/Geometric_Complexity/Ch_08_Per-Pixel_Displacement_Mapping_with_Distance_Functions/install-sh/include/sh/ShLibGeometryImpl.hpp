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
#ifndef SHLIBGEOMETRYIMPL_HPP
#define SHLIBGEOMETRYIMPL_HPP

#include "ShLibClamp.hpp"
#include "ShAttrib.hpp"
#include "ShInstructions.hpp"

namespace SH {

template<typename T1, typename T2>
inline
ShGeneric<3, CT1T2> cross(const ShGeneric<3, T1>& left, const ShGeneric<3, T2>& right)
{
  ShAttrib<3, SH_TEMP, CT1T2> t;
  shXPD(t, left, right);
  return t;
}

template<typename T1, typename T2>
inline
ShGeneric<3, CT1T2> operator^(const ShGeneric<3, T1>& left, const ShGeneric<3, T2>& right)
{
  return cross(left, right);
}

template<int N, typename T>
inline
ShGeneric<N, T> normalize(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shNORM(t, var);
  return t;
}

template<int N, typename T1, typename T2>
inline
ShGeneric<1, CT1T2> dot(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shDOT(t, left, right);
  return t;
}

template<int N, typename T1, typename T2>
inline
ShGeneric<1,  CT1T2> operator|(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  return dot(left, right);
}
SH_SHLIB_CONST_N_OP_RETSIZE_BOTH(dot, 1);

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> reflect(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b)
{
  ShGeneric<N, T2> bn = normalize(b);
  return 2 * dot(a, b) * b - a;
}

template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3> refract(const ShGeneric<N, T1>& v, const ShGeneric<N, T2>& n,
                        const ShGeneric<1, T3>& theta)
{
  ShGeneric<N, T1> vn = normalize(v);
  ShGeneric<N, T2> nn = normalize(n);
  ShGeneric<1, CT1T2T3> c = (vn|nn);
  ShGeneric<1, CT1T2T3> k = c*c - ShDataTypeConstant<CT1T2T3, SH_HOST>::One;
  k = ShDataTypeConstant<CT1T2T3, SH_HOST>::One + theta*theta*k;
  k = clamp(k, ShDataTypeConstant<CT1T2T3, SH_HOST>::Zero, ShDataTypeConstant<CT1T2T3, SH_HOST>::One); 
  ShGeneric<1, CT1T2T3> a = theta;
  ShGeneric<1, CT1T2T3> b = theta*c + sqrt(k);
  return (a*vn + b*nn);
}

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> faceforward(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b)
{
  return (2 * (dot(a, b) > 0) - 1) * b;
}

template<typename T1, typename T2, typename T3>
inline
ShGeneric<4, CT1T2T3> lit(const ShGeneric<1, T1>& a,
                          const ShGeneric<1, T2>& b,
                          const ShGeneric<1, T3>& c)
{
  ShAttrib<4, SH_TEMP, CT1T2T3> r;
  r(0,3) = ShAttrib<2, SH_CONST, CT1T2T3>(1, 1);
  r(1) = pos(a);
  r(2) = (a < 0 && b < 0) * pow(b, c);
  return r;
}

template<int N, typename T>
ShGeneric<1, T> distance(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b)
{
  return length(a-b);
}

template<int N, typename T>
ShGeneric<1, T> distance_1(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b)
{
  return length_1(a-b);
}

template<int N, typename T>
ShGeneric<1, T> distance_inf(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b)
{
  return length_inf(a-b);
}

template<int N, typename T>
ShGeneric<1, T> length(const ShGeneric<N, T>& a)
{
  return sqrt(dot(a, a));
}

template<int N, typename T>
ShGeneric<1, T> length_1(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b)
{
  return sum(abs(a));
}

template<int N, typename T>
ShGeneric<1, T> length_inf(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b)
{
  return max(abs(a));
}


}

#endif
