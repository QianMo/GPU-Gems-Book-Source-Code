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
#ifndef SHLIBINTERVALIMPL_HPP
#define SHLIBINTERVALIMPL_HPP

#include "ShLibInterval.hpp"

namespace SH {

template<int N, typename T>
inline
ShGeneric<N, SH_REGULARTYPE(T)> lo(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, SH_REGULARTYPE(T)> t;
  shLO(t, var);
  return t;
}

template<int N, typename T>
inline
ShGeneric<N, SH_REGULARTYPE(T)> hi(const ShGeneric<N, T>& var)
{
  ShAttrib<N, SH_TEMP, SH_REGULARTYPE(T)> t;
  shHI(t, var);
  return t;
}

}

#endif
