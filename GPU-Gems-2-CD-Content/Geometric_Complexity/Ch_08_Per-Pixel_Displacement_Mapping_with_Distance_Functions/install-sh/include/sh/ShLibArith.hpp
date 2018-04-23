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
#ifndef SHLIBARITH_HPP
#define SHLIBARITH_HPP

#include "ShGeneric.hpp"
#include "ShLib.hpp"

#ifndef WIN32
namespace SH {

/** \defgroup lib_arith Arithmetic operations
 * @ingroup library
 * @{
 */

/** Addition.
 * On tuples, this operator acts componentwise.
 * @todo scalar promotion.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator+(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator+(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator+(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator+(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);
SH_SHLIB_CONST_SCALAR_OP_DECL(operator+);
SH_SHLIB_CONST_N_OP_BOTH_DECL(operator+);

/** Subtraction.
 * On tuples, this operator acts componentwise.
 * @todo scalar promotion.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator-(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator-(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator-(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator-(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);
SH_SHLIB_CONST_SCALAR_OP_DECL(operator-);
SH_SHLIB_CONST_N_OP_BOTH_DECL(operator-);

/** Multiplication.
 * On tuples, this operator acts componentwise.
 * If a scalar is multiplied by a tuple, the scalar is promoted by
 * duplication to a tuple.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator*(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator*(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator*(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator*(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);
SH_SHLIB_CONST_SCALAR_OP_DECL(operator*);
SH_SHLIB_CONST_N_OP_BOTH_DECL(operator*);

/** Division.
 * On tuples, this operator acts componentwise.
 * If a tuple is divided by a scalar (or vice versa), the scalar is promoted by
 * duplication to a tuple.
 */
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator/(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator/(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2> operator/(const ShGeneric<1, T1>& left, const ShGeneric<N, T2>& right);
template<typename T1, typename T2>
ShGeneric<1, CT1T2> operator/(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);
SH_SHLIB_CONST_SCALAR_OP_DECL(operator/);
SH_SHLIB_CONST_N_OP_LEFT_DECL(operator/);


/** Natural Exponent.
 * Operates componentwise on tuples.
 * Returns the natural exponent of x.
 */
template<int N, typename T>
ShGeneric<N, T> exp(const ShGeneric<N, T>& var);

/** Exponent base 2.
 * Operates componentwise on tuples.
 * Returns the exponent of x using base 2.
 */
template<int N, typename T>
ShGeneric<N, T> exp2(const ShGeneric<N, T>& var);

/** Exponent base 10.
 * Operates componentwise on tuples.
 * Returns the exponent of x using base 10.
 */
template<int N, typename T>
ShGeneric<N, T> exp(const ShGeneric<N, T>& var);

/** Minus-one Exponent base 10.
 * Operates componentwise on tuples.
 * Returns the exponent of x - 1 using base 10.
 */
template<int N, typename T>
ShGeneric<N, T> expm1(const ShGeneric<N, T>& x);

/** Natural Logarithm.
 * Operates componentwise on tuples.
 * Returns the natural logarithm of x.
 */
template<int N, typename T>
ShGeneric<N, T> log(const ShGeneric<N, T>& var);

/** Logarithm base 2.
 * Operates componentwise on tuples.
 * Returns the logarithm of x using base 2.
 */
template<int N, typename T>
ShGeneric<N, T> log2(const ShGeneric<N, T>& var);

/** Logarithm base 10.
 * Operates componentwise on tuples.
 * Returns the logarithm of x using base 10.
 */
template<int N, typename T>
ShGeneric<N, T> log(const ShGeneric<N, T>& var);

/** Plus-One Logarithm base 10.
 * Operates componentwise on tuples.
 * Returns the logarithm of x + 1 using base 10.
 */
template<int N, typename T>
ShGeneric<N, T> logp1(const ShGeneric<N, T>& x);

/** Power.
 * Raise a tuple to a power.
 * @todo scalar promotion.
 */
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2>
pow(const ShGeneric<N, T1>& left, const ShGeneric<N, T2>& right);
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2>
pow(const ShGeneric<N, T1>& left, const ShGeneric<1, T2>& right);
template<typename T1, typename T2, typename T3>
ShGeneric<1, CT1T2> pow(const ShGeneric<1, T1>& left, const ShGeneric<1, T2>& right);

SH_SHLIB_CONST_SCALAR_OP_DECL(pow);
SH_SHLIB_CONST_N_OP_RIGHT_DECL(pow);

/** Multiply and add.
 * This is an intrinsic to access the assembly instruction of the same name.
 * Multiply-add is potentially cheaper than a separate multiply and
 * add.  Note: potentially.
 */
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3>
mad(const ShGeneric<N, T1>& m1, const ShGeneric<N, T2>& m2, 
                    const ShGeneric<N, T3>& a);
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3>
mad(const ShGeneric<N, T1>& m1, const ShGeneric<1, T2>& m2, 
                    const ShGeneric<N, T3>& a);
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3>
mad(const ShGeneric<1, T1>& m1, const ShGeneric<N, T2>& m2, 
                    const ShGeneric<N, T3>& a);
template<typename T1, typename T2, typename T3>
ShGeneric<1, CT1T2T3> mad(const ShGeneric<1, T1>& m1, const ShGeneric<1, T2>& m2, 
                    const ShGeneric<1, T3>& a);

//@todo type should not use double here, but overloading problems need to be
//resolved
//template<int N, typename T> 
//ShGeneric<N, T> 
//mad(T m1, const ShGeneric<N, T>& m2, const ShGeneric<N, T>& a);
//template<int N, typename T> 
//ShGeneric<N, T>
//mad(const ShGeneric<N, T>& m1, T m2, const ShGeneric<N, T>& a);

//@todo type not sure these are a good idea
template<int N, typename T1, typename T2> 
ShGeneric<N, CT1T2> 
mad(double m1, const ShGeneric<N, T1>& m2, const ShGeneric<N, T2>& a);
template<int N, typename T1, typename T2> 
ShGeneric<N, CT1T2>
mad(const ShGeneric<N, T1>& m1, double m2, const ShGeneric<N, T2>& a);

/* Reciprocal
 * One divided by the given value, for each component.
 */
template<int N, typename T>
ShGeneric<N, T> rcp(const ShGeneric<N, T>& var);

/* Square root.
 * The square root of each component of the input is evaluated.
 */
template<int N, typename T>
ShGeneric<N, T> sqrt(const ShGeneric<N, T>& var);

/* Reciprocal square root.
 * The inverse of the square root of each component of the input is evaluated.
 */
template<int N, typename T>
ShGeneric<N, T> rsqrt(const ShGeneric<N, T>& var);

/* Cube root.
 * The cube root of each component of the input is evaluated.
 */
template<int N, typename T>
ShGeneric<N, T> cbrt(const ShGeneric<N, T>& var);

/*@}*/

/** Linear interpolation.
 * Blend between two tuples.   The blend value can be a scalar
 * or a tuple.
 */
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3>
lerp(const ShGeneric<N, T1>& f, const ShGeneric<N, T2>& a, 
                     const ShGeneric<N, T3>& b);
template<int N, typename T1, typename T2, typename T3>
ShGeneric<N, CT1T2T3>
lerp(const ShGeneric<1, T1>& f, const ShGeneric<N, T2>& a, 
                     const ShGeneric<N, T3>& b);
template<typename T1, typename T2, typename T3>
ShGeneric<1, CT1T2T3> 
lerp(const ShGeneric<1, T1>& f, const ShGeneric<1, T2>& a, 
     const ShGeneric<1, T3>& b);

template<int N, typename T1, typename T2>
ShGeneric<N, CT1T2>
lerp(double f, const ShGeneric<N, T1>& a, const ShGeneric<N, T2>& b);


/* Sum of components.
 * Addition of all components into a single result.
 */
template<int N, typename T>
ShGeneric<1, T> sum(const ShGeneric<N, T>& var);

/* Product of components.
 * Multiplication of all components into a single result.
 */
template<int N, typename T>
ShGeneric<1, T> prod(const ShGeneric<N, T>& var);

}
#endif

#include "ShLibArithImpl.hpp"

#endif
