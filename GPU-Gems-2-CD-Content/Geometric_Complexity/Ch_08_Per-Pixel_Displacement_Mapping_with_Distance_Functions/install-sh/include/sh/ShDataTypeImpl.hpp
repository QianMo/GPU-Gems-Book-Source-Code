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
#ifndef SHDATATYPEIMPL_HPP
#define SHDATATYPEIMPL_HPP

#include <climits>
#include <cmath>
#include <algorithm>
#include "ShDataType.hpp"

namespace SH {

/** Returns the boolean cond in the requested data type */
template<typename T, ShDataType DT>
inline
typename ShDataTypeCppType<T, DT>::type shDataTypeCond(bool cond) 
{
  return cond ? ShDataTypeConstant<T, DT>::One : ShDataTypeConstant<T, DT>::Zero;
}

/** Returns a whether the two values are exactly the same.
 * This is is useful for the range types.
 * @{
 */
template<typename T>
inline
bool shDataTypeEqual(const T &a, 
                     const T &b) 
{
  return a == b;
}

template<typename T>
inline
bool shDataTypeEqual(const ShInterval<T> &a, const ShInterval<T> &b)
{
  return (a.lo() == b.lo()) && (a.hi() == b.hi()); 
}
// @}

/** Returns whether the value is always greater than zero (i.e. true) 
 * @{
 */
template<typename T>
inline
bool shDataTypeIsPositive(const T &a)
{
  return a > 0; 
}

template<typename T>
inline
bool shDataTypeIsPositive(const ShInterval<T> &a)
{
  return (a.lo() > 0); 
}

//@}

/** Casts one data type to another data type 
 * All the built-in types can use C++ casts
 * for all the casts required by Sh internally.
 */
template<typename T1, ShDataType DT1, typename T2, ShDataType DT2>
void shDataTypeCast(typename ShDataTypeCppType<T1, DT1>::type &dest,
                    const typename ShDataTypeCppType<T2, DT2>::type &src)
{
  typedef typename ShDataTypeCppType<T1, DT1>::type desttype; 
  dest = (desttype)(src);
}


}

#endif
