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
#ifndef SHLIBCLAMPIMPL_HPP
#define SHLIBCLAMPIMPL_HPP

#include "ShLibClamp.hpp"
#include "ShInstructions.hpp"
#include "ShAttrib.hpp"
#include "ShLibMiscImpl.hpp"

namespace SH {

template<int N, typename T>
inline
ShGeneric<N, T> abs(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shABS(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> ceil(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shCEIL(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> floor(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shFLR(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> round(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shRND(t, var);
  return t;
}

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> mod(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMOD(t, left, right);
  return t;
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> mod(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMOD(t, left, right);
  return t;
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2> mod(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  ShAttrib<1, SH_TEMP, CT1T2> t;
  shMOD(t, left, right);
  return t;
}

template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> operator%(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  return mod(left, right);
}
template<int N, typename T1, typename T2>
inline
ShGeneric<N, CT1T2> operator%(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right)
{
  return mod(left, right);
}
template<typename T1, typename T2>
inline
ShGeneric<1, CT1T2> operator%(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right)
{
  return mod(left, right);
}
SH_SHLIB_CONST_SCALAR_OP(mod);
SH_SHLIB_CONST_N_OP_LEFT(mod);
SH_SHLIB_CONST_SCALAR_OP(operator%);
SH_SHLIB_CONST_N_OP_LEFT(operator%);

template<int N, typename T>
inline
ShGeneric<N, T> frac(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shFRAC(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> pos(const ShGeneric<N, T>& var)
{
  return max(var, fillcast<N>(0.0f));
}

template<int N, typename T1, typename T2>
inline
ShGeneric<N,  CT1T2> max(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMAX(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(max);

template<int N, typename T1, typename T2>
inline
ShGeneric<N,  CT1T2> min(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right)
{
  ShAttrib<N, SH_TEMP, CT1T2> t;
  shMIN(t, left, right);
  return t;
}
SH_SHLIB_CONST_SCALAR_OP(min);

template<int N, typename T>
ShGeneric<1, T> max(const ShGeneric<N, T>& a)
{
  int lhswz[N/2 + N%2];
  for (int i = 0; i < N/2 + N%2; i++) {
    lhswz[i] = i;
  }
  int rhswz[N/2];
  for (int i = 0; i < N/2; i++) {
    rhswz[i] = i + N/2 + N%2;
  }

  return max(max(a.template swiz<N/2 + N%2>(lhswz)), max(a.template swiz<N/2>(rhswz)));
}

template<typename T>
ShGeneric<1, T> max(const ShGeneric<1, T>& a)
{
  return a;
}

template<int N, typename T>
ShGeneric<1, T> min(const ShGeneric<N, T>& a)
{
  int lhswz[N/2 + N%2];
  for (int i = 0; i < N/2 + N%2; i++) {
    lhswz[i] = i;
  }
  int rhswz[N/2];
  for (int i = 0; i < N/2; i++) {
    rhswz[i] = i + N/2 + N%2;
  }

  return min(min(a.template swiz<N/2 + N%2>(lhswz)), min(a.template swiz<N/2>(rhswz)));
}

template<typename T>
ShGeneric<1, T> min(const ShGeneric<1, T>& a)
{
  return a;
}

template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> clamp(const ShGeneric<N, T1>& a,
                      const ShGeneric<N, T2>& b, const ShGeneric<N, T3>& c)
{
  return min(max(a, b), c);
}
template<int N, typename T1, typename T2, typename T3>
inline
ShGeneric<N, CT1T2T3> clamp(const ShGeneric<N, T1>& a,
                      const ShGeneric<1, T2>& b, const ShGeneric<1, T3>& c)
{
  return min(max(a, fillcast<N>(b)), fillcast<N>(c));
}

template<typename T1, typename T2, typename T3>
inline
ShGeneric<1, CT1T2T3> clamp(const ShGeneric<1, T1>& a,
                      const ShGeneric<1, T2>& b, const ShGeneric<1, T3>& c)
{
  return min(max(a, b), c);
}
SH_SHLIB_CONST_TRINARY_OP_011(clamp);

template<int N, typename T>
inline
ShGeneric<N, T> sat(const ShGeneric<N, T>& var)
{
  return min(var, fillcast<N>(ShConstAttrib1f(1.0)));
}

template<int N, typename T>
inline
ShGeneric<N, T> sign(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shSGN(t, var);
  return t;
}

}

#endif
