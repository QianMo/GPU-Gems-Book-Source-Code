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
#ifndef SHLIBGEOMETRY_HPP
#define SHLIBGEOMETRY_HPP

#include "ShGeneric.hpp"
#include "ShLib.hpp"

#ifndef WIN32
namespace SH {

/** \defgroup lib_geometry Geometric operations.
 * @ingroup library
 * @todo distance, sqdistance, length, rlength, rsqlength, sqlength, perp,
 * perpdot, projnorm should be implemented too.   Different distance functions
 * also possible: L1, L2, Linf.
 * @{
 */

/** Take the cross product of two 3-tuples.
 * Note that this operation is not limited to vectors.
 */
template<typename T1, typename T2>
ShGeneric<3, CT1T2> 
cross(const ShGeneric<3, T1>& left, const ShGeneric<3, T2>& right);

/// Equivalent to cross(left, right)
template<int N, typename T1, typename T2>
ShGeneric<3, CT1T2> 
operator^(const ShGeneric<3, T1>& left, const ShGeneric<3, T2>& right);

/** Normalize an n-tuple to unit length.
 * Divides an n-tuple by its Euclidean length.   
 */
template<int N, typename T>
ShGeneric<N, T> normalize(const ShGeneric<N, T>& var);

/** Compute reflection vector.
 * Reflect vector a about normal n.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> 
reflect(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b);

/** Compute refraction vector.
 * Refract vector a about normal b using relative index of refraction c.
 */
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3> 
refract(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b, const ShGeneric<1, T3>& c);

/** Make a vector face the same way as another
 * Negates b if it does not face the same way as a (i.e. the dot
 * product between a and b is negative).
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> 
faceforward(const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b);

/** Compute lighting coefficents.
 * returns a 4-vector consisting of:
 *  - 1
 *  - max(a,0)
 *  - b^c if (a > 0 and b < 0), 0 otherwise
 *  - 1
 */
template<typename T1, typename T2, typename T3>
ShGeneric<4, CT1T2T3> 
lit(const ShGeneric<1, T1>& a,
    const ShGeneric<1, T2>& b,
    const ShGeneric<1, T3>& c);

/** Inner (dot) product.
 */
template<int N, typename T1, typename T2>
ShGeneric<1, CT1T2> 
dot(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);

template<int N, typename T1, typename T2>
ShGeneric<1, CT1T2> 
operator|(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);

SH_SHLIB_CONST_N_OP_RETSIZE_BOTH_DECL(dot, 1);

/*@}*/

/** Euclidean Distance
 * Computes the Euclidean distance between the two points a and b.
 */
template<int N, typename T>
ShGeneric<1, T> distance(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b);

/** Manhattan Distance
 * Computes the Manhattan distance between the two points a and b.
 */
template<int N, typename T>
ShGeneric<1, T> distance_1(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b);

/** Inf Distance
 * Computes the Inf norm distance between the two points a and b.
 */
template<int N, typename T>
ShGeneric<1, T> distance_inf(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b);

/** Euclidean Length
 * Computes the euclidean distance between the zero vector and a.
 */
template<int N, typename T>
ShGeneric<1, T> length(const ShGeneric<N, T>& a);

}
#endif

#include "ShLibGeometryImpl.hpp"

#endif
