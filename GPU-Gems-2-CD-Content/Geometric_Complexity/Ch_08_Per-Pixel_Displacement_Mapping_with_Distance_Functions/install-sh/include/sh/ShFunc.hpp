// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Bryan Chan, Michael D. McCool
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
#ifndef SHUTIL_FUNC_HPP 
#define SHUTIL_FUNC_HPP 

#include "ShVariable.hpp"
#include "ShAttrib.hpp"

#ifndef WIN32
namespace ShUtil {

using namespace SH;

//TODO extend to vector versions

/** \file ShFunc.hpp
 * \brief Miscellaneous small Sh utility functions.  
 */


/** \brief Cubic interpolated step between 0 and 1. 
 * Returns 0 if x < a,
 * 1 if x > b, 
 * cubic interpolation between 0 and 1 otherwise
 */
template<int N, typename T>
ShGeneric<N, T> smoothstep(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b,
    const ShGeneric<N, T> x); 

/** \brief Euclidean distance between two points.
 */
template<int N, typename T>
ShGeneric<1, T> distance(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b); 

/** \brief L1 distance between two points
 * The L1 distance is a sum of the absolute component-wise differences.
 */
template<int N, typename T>
ShGeneric<1, T> lOneDistance(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b); 

/** \brief Linfinity distance between two points 
 * Linfinity distance is the maximum absolute component-wise difference.
 */
template<int N, typename T>
ShGeneric<1, T> lInfDistance(const ShGeneric<N, T>& a, const ShGeneric<N, T>& b); 

/** \brief Parallel linear congruential generator
 * 
 * This does not work very well right now.  Use hashmrg instead.
 *
 * \sa  template<int N, typename T> ShGeneric<N, T> hashmrg(const ShGeneric<N, T>& p)
 */
// TODO: may not work as intended on 24-bit floats
// since there may not be enough precision 
template<int N, typename T>
ShGeneric<N, T> hashlcg(const ShGeneric<N, T>& p); 

/** \brief MRG style pseudorandom vector generator
 *
 * Generates a random vector using a multiple-recursive generator style. (LCG on steroids)
 * Treat x,y,z,w as seeds a0, a1, a2, a3
 * and repeatedly apply an = b * (an-1, an-2, an-3, an-4), where b is a vector
 * Take as output (an, an-1, an-2, an3) after suitable number of iterations.
 *
 * This appears to reduce correlation in the output components when input components are 
 * similar, but the behaviour needs to be studied further.
 *
 * \sa template<int N, typename T> ShGeneric<N, T> hashlcg(const ShGeneric<N, T>& p)
 */
template<int N, typename T>
ShGeneric<N, T> hashmrg(const ShGeneric<N, T>& p); 


/** \brief Sorts components of an n-tuple
 * Uses an even-odd transposition sort to sort the components of an n-tuple.
 * (Ordered from smallest to largset)
 */
template<int N, ShBindingType Binding, typename T>
ShAttrib<N, Binding, T> evenOddSort(const ShAttrib<N, Binding, T>& v);

/** \brief Sorts groups of components v[i](j), 0 <= i < S
 * by the components in v[0](j) 0 <= j < N.
 * This also uses an even-odd transposition sort.
 */
template<int S, int N, ShBindingType Binding, typename T>
void groupEvenOddSort(ShAttrib<N, Binding, T> v[]);

/** \brief Sorts groups of components v[i](j), 0 <= i < S
 * by the components in v[0](j) 0 <= j < N.
 * This uses a bitonic sorting network 
 */
//template<int S, int N, ShBindingType Binding, typename T>
//void groupBitonicSort(ShAttrib<N, Binding, T> v[]);

/** \brief Given orthonormal basis b0, b1, b2 and vector v relative to coordinate space C,
 * does change of basis on v to the orthonormal basis b0, b1, b2
 */
template<typename T>
ShGeneric<3, T> changeBasis(const ShGeneric<3, T> &b0, 
    const ShGeneric<3, T> &b1, const ShGeneric<3, T> &b2, const ShGeneric<3, T> &v); 

}
#endif

#include "ShFuncImpl.hpp"

#endif // SHUTIL_FUNC_HPP 
