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
#ifndef SHLIBINTERVAL_HPP
#define SHLIBINTERVAL_HPP

#include "ShGeneric.hpp"
#include "ShInterval.hpp"
#include "ShLib.hpp"

#define SH_REGULARTYPE(T) typename ShStorageTypeInfo<T>::RegularType
#define SH_IA_TYPE(T) typename ShStorageTypeInfo<T>::IntervalType
namespace SH {

/** \defgroup lib_interval Interval Arithmetic  
 * @ingroup library
 * Operations related to derivatives of values.
 * @{
 */

/** lower bound 
 */
template<int N, typename T>
ShGeneric<N, SH_REGULARTYPE(T)> lo(const ShGeneric<N, T>& var);

 /** upper bound
 */
template<int N, typename T>
ShGeneric<N, SH_REGULARTYPE(T)> hi(const ShGeneric<N, T>& var);

/*@}*/

}

#include "ShLibIntervalImpl.hpp"

#undef SH_REGULARTYPE
#undef SH_IA_TYPE

#endif
