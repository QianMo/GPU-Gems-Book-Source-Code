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
#ifndef SHLIBTRIGIMPL_HPP
#define SHLIBTRIGIMPL_HPP

#include "ShLibTrig.hpp"
#include "ShAttrib.hpp"
#include "ShInstructions.hpp"

namespace SH {

template<int N, typename T>
inline
ShGeneric<N, T> acos(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shACOS(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> asin(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shASIN(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> atan(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shATAN(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> atan2(const ShGeneric<N, T>& y, const ShGeneric<N, T>& x)
{
  ShAttrib<N, SH_TEMP, T> t;
  shATAN2(t, y, x);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> cos(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shCOS(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> sin(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shSIN(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, T> tan(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, T> t;
  shTAN(t, var);
  return t;
}

}

#endif
