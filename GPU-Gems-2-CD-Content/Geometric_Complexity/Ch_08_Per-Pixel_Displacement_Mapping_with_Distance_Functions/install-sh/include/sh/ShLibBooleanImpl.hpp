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
#ifndef SHLIBBOOLEANIMPL_HPP
#define SHLIBBOOLEANIMPL_HPP

#include "ShLibBoolean.hpp"
#include "ShInstructions.hpp"
#include "ShAttrib.hpp"

namespace SH {

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator<(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSLT(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator<(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSLT(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator<(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSLT(t, left, right);
  return t;
}
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator<(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shSLT(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator<);
SH_SHLIB_CONST_N_OP_BOTH(operator<);

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator<=(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSLE(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator<=(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSLE(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator<=(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSLE(t, left, right);
  return t;
}
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator<=(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shSLE(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator<=);
SH_SHLIB_CONST_N_OP_BOTH(operator<=);

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator>(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSGT(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator>(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSGT(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator>(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSGT(t, left, right);
  return t;
}
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator>(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shSGT(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator>);
SH_SHLIB_CONST_N_OP_BOTH(operator>);

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator>=(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSGE(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator>=(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSGE(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator>=(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSGE(t, left, right);
  return t;
}
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator>=(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shSGE(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator>=);
SH_SHLIB_CONST_N_OP_BOTH(operator>=);

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator==(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSEQ(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator==(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSEQ(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator==(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSEQ(t, left, right);
  return t;
}
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator==(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shSEQ(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator==);
SH_SHLIB_CONST_N_OP_BOTH(operator==);

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator!=(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSNE(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator!=(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSNE(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator!=(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shSNE(t, left, right);
  return t;
}
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator!=(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shSNE(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator!=);
SH_SHLIB_CONST_N_OP_BOTH(operator!=);

template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3> cond(const ShGeneric<N, T1>& condition, const ShGeneric<N, T2>& left,
                     const ShGeneric<N, T3>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shCOND(t, condition, left, right);
  return t;
}
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3> cond(const ShGeneric<1, T1>& condition, const ShGeneric<N, T2>& left,
                     const ShGeneric<N, T3>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shCOND(t, condition, left, right);
  return t;
}
template<typename T1, typename T2, typename T3>
ShGeneric<1, CT1T2T3> cond(const ShGeneric<1, T1>& condition, const ShGeneric<1, T2>& left,
                     const ShGeneric<1, T3>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2T3> t;
  shCOND(t, condition, left, right);
  return t;
}

// TODO

template<int N, typename T>
ShGeneric<N, T> operator!(const ShGeneric<N, T>& a)
{
  return 1.0f - (a > 0.0f);
}


template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator&&(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b)
{
  return min(a,b);
}

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator||(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b)
{
  return max(a,b);
}

template<int N, typename T>
ShGeneric<1, T> any(const ShGeneric<N, T>& a)
{
  ShAttrib<1, SH_TEMP, T> t = a(0);
  for (int i = 1; i < N; i++) {
    t = t || a(i);
  }
  return t;
}

template<int N, typename T>
ShGeneric<1, T> all(const ShGeneric<N, T>& a)
{
  ShAttrib<1, SH_TEMP, T> t = a(0);
  for (int i = 1; i < N; i++) {
    t = t && a(i);
  }
  return t;
}

}

#endif
