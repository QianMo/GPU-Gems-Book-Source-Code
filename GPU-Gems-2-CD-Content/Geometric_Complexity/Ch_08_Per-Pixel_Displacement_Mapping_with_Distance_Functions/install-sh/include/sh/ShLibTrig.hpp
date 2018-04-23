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
#ifndef SHLIBTRIG_HPP
#define SHLIBTRIG_HPP

#include "ShGeneric.hpp"
#include "ShLib.hpp"


#ifndef WIN32
namespace SH {

/** \defgroup lib_trig Trigonometric functions
 * @ingroup library
 * @todo tan, atan, atan2, hyperbolic functions, secant, cosecant, etc.
 * @{
 */

/** Arccosine. 
 * Operates componentwise on tuples.
 * A value of x in [-1, 1] gives a result in [0, pi].
 * Input values outside the range [-1,1] will give undefined results.
 */
template<int N, typename T>
ShGeneric<N, T> acos(const ShGeneric<N, T>& var);

/** Arcsine. 
 * Operates componentwise on tuples.
 * A value of x in [-1, 1] gives a result in [-pi/2, pi/2].
 * Input values outside the range [-1,1] will give undefined results.
 */
template<int N, typename T>
ShGeneric<N, T> asin(const ShGeneric<N, T>& var);

/** Arctangent. 
 * Operates componentwise on tuples.
 * Gives a result in [-pi/2, pi/2].
 */
template<int N, typename T>
ShGeneric<N, T> atan(const ShGeneric<N, T>& var);

/** Arctangent of two variables. 
 * Operates componentwise on tuples of y/x.
 * Gives a result in [-pi/2, pi/2].
 */
template<int N, typename T>
ShGeneric<N, T> atan2(const ShGeneric<N, T>& y, const ShGeneric<N, T>& x);

/** Cosine.
 * Operates componentwise on tuples.
 * Returns the cosine of x.   Any value of x gives a result
 * in the range [-1,1].
 */
template<int N, typename T>
ShGeneric<N, T> cos(const ShGeneric<N, T>& var);

/** Sine.
 * Operates componentwise on tuples.
 * Returns the sine of x.   Any value of x gives a result
 * in the range [-1,1].
 */
template<int N, typename T>
ShGeneric<N, T> sin(const ShGeneric<N, T>& var);

/** Tangent.
 * Operates componentwise on tuples.
 * Returns the tangent of x.   Equivalent to sin(x)/cos(x).
 */
template<int N, typename T>
ShGeneric<N, T> tan(const ShGeneric<N, T>& var);

/*@}*/

}
#endif

#include "ShLibTrigImpl.hpp"

#endif
