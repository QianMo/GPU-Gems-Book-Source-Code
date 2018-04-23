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
#ifndef SHLIBCLAMP_HPP
#define SHLIBCLAMP_HPP

#include "ShGeneric.hpp"
#include "ShLib.hpp"

#ifndef WIN32
namespace SH {


/** \defgroup lib_clamping Clamping
 * @ingroup library
 * Various operations that reduce information in values somehow,
 * e.g. clamping them to ranges, throwing away their sign, or
 * selecting one out of two tuples.
 * @{
 */

/** Absolute value.
 * Returns the magnitude.
 * Operates componentwise on tuples.
 */
template<int N, typename T>
ShGeneric<N, T> abs(const ShGeneric<N, T>& var);

/** Ceiling.
 * Returns the least integer >= argument. 
 * Operates componentwise on tuples.
 */
template<int N, typename T>
ShGeneric<N, T> ceil(const ShGeneric<N, T>& var);

/** Floor.
 * Returns the greatest integer <= argument.
 * Operates componentwise on tuples.
 */
template<int N, typename T>
ShGeneric<N, T> floor(const ShGeneric<N, T>& var);

/** Round.
 * Returns the nearest integer to the argument.
 * Operates componentwise on tuples.
 */
template<int N, typename T>
ShGeneric<N, T> round(const ShGeneric<N, T>& var);

/** Float modulus. 
 * The result is always positive.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
mod(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
mod(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<typename T1, typename T2>
ShGeneric<1, CT1T2>
mod(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
operator%(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
operator%(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<typename T1, typename T2>
ShGeneric<1, CT1T2>
operator%(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);

SH_SHLIB_CONST_SCALAR_OP_DECL(mod);
SH_SHLIB_CONST_N_OP_LEFT_DECL(mod);
SH_SHLIB_CONST_SCALAR_OP_DECL(operator%);
SH_SHLIB_CONST_N_OP_LEFT_DECL(operator%);

/** Fractional part.
 * Discards the integer part of each componenent in var.
 */
template<int N, typename T>
ShGeneric<N, T> frac(const ShGeneric<N, T>& var);

/** Take positive part. 
 * Clamps a value to zero if it is negative.   
 * This is useful to wrap dot products in lighting models.
 */
template<int N, typename T>
ShGeneric<N, T> pos(const ShGeneric<N, T>& x);

/** Maximum.
 * Creates a tuple of componentwise maximums of a pair of input tuples.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
max(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);

SH_SHLIB_CONST_SCALAR_OP_DECL(max);

/** Minimum.
 * Creates a tuple of componentwise minimums of a pair of input tuples.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
min(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);

SH_SHLIB_CONST_SCALAR_OP_DECL(min);

/** Maximum of all components
 */
template<int N, typename T>
ShGeneric<1, T> max(const ShGeneric<N, T>& a);

/** Minimum of all components
 */
template<int N, typename T>
ShGeneric<1, T> min(const ShGeneric<N, T>& a);

/** Componentwise clamping.
 * Clamps a between b and c.
 */
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3> 
clamp(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b, const ShGeneric<N, T3>& c);
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3> 
clamp(const ShGeneric<N, T1>& a, const ShGeneric<1, T2>& b, const ShGeneric<1, T3>& c);
template<typename T1, typename T2, typename T3>
ShGeneric<1, CT1T2T3> 
clamp(const ShGeneric<1, T1>& a, const ShGeneric<1, T2>& b, const ShGeneric<1, T3>& c);

SH_SHLIB_CONST_TRINARY_OP_011_DECL(clamp);

/** Componentwise saturation.
 * Equivalent to componentwise minimum with 1.
 */
template<int N, typename T>
ShGeneric<N, T> sat(const ShGeneric<N, T>& a);

/** Componentwise sign.
 * Returns -1.0 if argument is less than 0.0, 1.0 if argument is greater
 * than 0.0, 0.0 otherwise.
 * Operates componentwise on tuples.
 */
template<int N, typename T>
ShGeneric<N, T> sign(const ShGeneric<N, T>& var);

/*@}*/

}
#endif

#include "ShLibClampImpl.hpp"

#endif
