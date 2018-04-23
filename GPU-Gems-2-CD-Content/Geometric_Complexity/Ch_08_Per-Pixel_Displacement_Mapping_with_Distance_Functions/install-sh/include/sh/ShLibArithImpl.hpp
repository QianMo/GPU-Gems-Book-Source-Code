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
#ifndef SHLIBARITHIMPL_HPP
#define SHLIBARITHIMPL_HPP

#include "ShLibArith.hpp"
#include "ShInstructions.hpp"

namespace SH {

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator+(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shADD(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator+(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shADD(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator+(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shADD(t, left, right);
  return t;
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2>
operator+(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shADD(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator+);
SH_SHLIB_CONST_N_OP_BOTH(operator+);

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator-(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shADD(t, left, -right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator-(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shADD(t, left, -right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator-(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shADD(t, left, -right);
  return t;
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2>
operator-(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shADD(t, left, -right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator-);
SH_SHLIB_CONST_N_OP_BOTH(operator-);

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator*(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMUL(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator*(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMUL(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator*(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMUL(t, left, right);
  return t;
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2>
operator*(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shMUL(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator*);
SH_SHLIB_CONST_N_OP_BOTH(operator*);

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator/(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shDIV(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator/(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shDIV(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2>
operator/(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shDIV(t, left, right);
  return t;
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2>
operator/(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shDIV(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(operator/);
SH_SHLIB_CONST_N_OP_LEFT(operator/);

template<int N, typename T>
inline
ShGeneric<N, T> exp(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shEXP(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> exp2(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shEXP2(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> exp10(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shEXP10(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> expm1(const ShGeneric<N, T>& var)
{
  return exp(var - 1.0);
}

template<int N, typename T>
inline
ShGeneric<N, T> log(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shLOG(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> log2(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shLOG2(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> log10(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shLOG10(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> logp1(const ShGeneric<N, T>& var)
{
  return log(var + 1.0);
}

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> pow(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shPOW(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> pow(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shPOW(t, left, right);
  return t;
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2> pow(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shPOW(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(pow);
SH_SHLIB_CONST_N_OP_RIGHT(pow);

template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> mad(const ShGeneric<N, T1>& m1, const ShGeneric<N, T2>& m2, 
                    const ShGeneric<N, T3>& a)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shMAD(t, m1, m2, a);
  return t;
}
template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> mad(const ShGeneric<N, T1>& m1, const ShGeneric<1, T2>& m2, 
                    const ShGeneric<N, T3>& a)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shMAD(t, m1, m2, a);
  return t;
}
template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> mad(const ShGeneric<1, T1>& m1, const ShGeneric<N, T2>& m2, 
                    const ShGeneric<N, T3>& a)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shMAD(t, m1, m2, a);
  return t;
}
template<typename T1, typename T2, typename T3>
inline
ShGeneric<1, CT1T2T3> mad(const ShGeneric<1, T1>& m1, const ShGeneric<1, T2>& m2, 
                    const ShGeneric<1, T3>& a)
{
  ShAttrib<1, SH_TEMP, CT1T2T3> t;
  shMAD(t, m1, m2, a);
  return t;
}

//template<int N, typename T> 
//inline
//ShGeneric<N, T> mad(T m1, const ShGeneric<N, T>& m2, const ShGeneric<N, T>& a)
//{
//  ShAttrib<N, SH_TEMP, T> t;
//  shMAD(t, ShAttrib<1, SH_CONST, T>(m1), m2, a);
//  return t;
//}
//template<int N, typename T>
//inline
//ShGeneric<N, T> mad(const ShGeneric<N, T>& m1, T m2, const ShGeneric<N, T>& a)
//{
//  ShAttrib<N, SH_TEMP, T> t;
//  shMAD(t, m1, ShAttrib<1, SH_CONST, T>(m2), a);
//  return t;
//}

template<int N, typename T1, typename T2> 
inline
ShGeneric<N, CT1T2> mad(double m1, const ShGeneric<N, T1>& m2, const ShGeneric<N, T2>& a)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMAD(t, ShAttrib<1, SH_CONST, CT1T2>(m1), m2, a);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> mad(const ShGeneric<N, T1>& m1, double m2, const ShGeneric<N, T2>& a)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMAD(t, m1, ShAttrib<1, SH_CONST, CT1T2>(m2), a);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> rcp(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shRCP(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> sqrt(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shSQRT(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> rsqrt(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shRSQ(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> cbrt(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shCBRT(t, var);
  return t;
}

template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> lerp(const ShGeneric<N, T1>& f, const ShGeneric<N, T2>& a, 
                     const ShGeneric<N, T3>& b)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shLRP(t, f, a, b);
  return t;
}

template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> lerp(const ShGeneric<1, T1>& f, const ShGeneric<N, T2>& a, 
                     const ShGeneric<N, T3>& b)
{
  ShAttrib<N, SH_TEMP, CT1T2T3> t;
  shLRP(t, f, a, b);
  return t;
}

template<typename T1, typename T2, typename T3>
inline
ShGeneric<1, CT1T2T3> lerp(const ShGeneric<1, T1>& f, const ShGeneric<1, T2>& a, 
                     const ShGeneric<1, T3>& b)
{
  ShAttrib<1, SH_TEMP, CT1T2T3> t;
  shLRP(t, f, a, b);
  return t;
}
//@todo type see explanation in LibArith.hpp file
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> lerp(double f, const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shLRP(t, f, a, b);
  return t;
}

template<int N, typename T>
inline
ShGeneric<1, T> sum(const ShGeneric<N, T>& var)
{
  ShAttrib<1, SH_TEMP, T> t;
  shCSUM(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<1, T> prod(const ShGeneric<N, T>& var)
{
  ShAttrib<1, SH_TEMP, T> t;
  shCMUL(t, var);
  return t;
}


}

#endif
