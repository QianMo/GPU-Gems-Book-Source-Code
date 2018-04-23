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
#ifndef SHINTERVAL_HPP
#define SHINTERVAL_HPP

#include <iostream>

namespace SH {

/*** Sh class for standard floating point interval arithmetic, without rounding
 *
 * Derived from NuS.hh and NuS.cc from the MetaMedia project.
 *
 * Currently does not handle rounding errors yet 
 */
template<typename T>
struct ShInterval {
  typedef T DataType;
  T m_lo;
  T m_hi;

  /** Constructs an interval with undefined value */
  ShInterval();

  /** Constructs an interval [value,value] */
  ShInterval(const T& value);

  /** Constructs an interval with the given bounds */
  ShInterval(const T& lo, const T& hi);

  template<typename T2>
  ShInterval(const ShInterval<T2> &other);

  /** accessor methods **/
  // @todo why are these even here if m_lo, m_hi are public?
  // @todo why are m_lo and m_hi public?
  T& lo();
  const T& lo() const;

  T& hi();
  const T& hi() const;

  /** Useful helpers **/ 
  /** Returns m_hi - m_lo **/
  const T& width() const;

  /** Returns (m_hi + m_lo) / 2 **/
  const T& centre() const;

  /** Returns width() / 2 **/
  const T& radius() const;

  /** Arithmetic operators **/
  ShInterval& operator=(const T &value);
  ShInterval& operator=(const ShInterval<T> &other);
  ShInterval& operator+=(const T &value);
  ShInterval& operator+=(const ShInterval<T> &other);
  ShInterval& operator-=(const T &value);
  ShInterval& operator-=(const ShInterval<T> &other);
  ShInterval& operator*=(const T &value);
  ShInterval& operator*=(const ShInterval<T> &other);
  ShInterval& operator/=(const T &value);
  ShInterval& operator/=(const ShInterval<T> &other);

  /** Float modulus - result is always positive 
   *@{*/
  ShInterval& operator%=(const T &value);
  ShInterval& operator%=(const ShInterval<T> &other);
  // @}

  /** Scalar arithmetic operators **/

  /** Negation **/
  ShInterval operator-() const;

  /** Output operator **/
  template<typename TT>
  friend std::ostream& operator<<(std::ostream& out, const ShInterval<TT> &value);


  /** Input operator (format matches output) **/
  template<typename TT>
  friend std::istream& operator>>(std::istream& out, ShInterval<TT> &value);

};

/** Arithmetic operators **/
// TODO fill in the remaining interval with scalar ops
template<typename T>
ShInterval<T> operator+(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator-(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator-(const T &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator*(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator/(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator%(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> cbrt(const ShInterval<T> &a);

template<typename T>
ShInterval<T> exp(const ShInterval<T> &a);

template<typename T>
ShInterval<T> exp2(const ShInterval<T> &a);

template<typename T>
ShInterval<T> exp10(const ShInterval<T> &a);

template<typename T>
ShInterval<T> log(const ShInterval<T> &a);

template<typename T>
ShInterval<T> log2(const ShInterval<T> &a);

template<typename T>
ShInterval<T> log10(const ShInterval<T> &a);

template<typename T>
ShInterval<T> frac(const ShInterval<T> &a);

template<typename T>
ShInterval<T> fmod(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> pow(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> rcp(const ShInterval<T> &a);

template<typename T>
ShInterval<T> rsq(const ShInterval<T> &a);

template<typename T>
ShInterval<T> sgn(const ShInterval<T> &a);

template<typename T>
ShInterval<T> sqrt(const ShInterval<T> &a);

/** Trig Operators */
template<typename T>
ShInterval<T> acos(const ShInterval<T> &a);

template<typename T>
ShInterval<T> asin(const ShInterval<T> &a);

template<typename T>
ShInterval<T> atan(const ShInterval<T> &a);

template<typename T>
ShInterval<T> atan2(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> cos(const ShInterval<T> &a); 
template<typename T>
ShInterval<T> sin(const ShInterval<T> &a);

template<typename T>
ShInterval<T> tan(const ShInterval<T> &a);

/** Comparison Operators **/
// @todo should think about how to represent tri-state logic values.
// For now output is interval (follows the t x t -> t convention of
// types for the standard operators)
template<typename T>
ShInterval<T> operator<(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator<=(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator>(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator>(const ShInterval<T> &a, const T& b); 

template<typename T>
ShInterval<T> operator>=(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator==(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> operator!=(const ShInterval<T> &a, const ShInterval<T> &b);

/// Returns true iff lo = a.lo and hi = a.hi 
template<typename T>
bool boundsEqual(const ShInterval<T> &a);


/** Clamping operators **/
template<typename T>
ShInterval<T> min(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> max(const ShInterval<T> &a, const ShInterval<T> &b);

template<typename T>
ShInterval<T> floor(const ShInterval<T> &a);

template<typename T>
ShInterval<T> ceil(const ShInterval<T> &a);

template<typename T>
ShInterval<T> rnd(const ShInterval<T> &a);

template<typename T>
ShInterval<T> abs(const ShInterval<T> &a);

/** Misc Operators **/
template<typename T>
ShInterval<T> cond(const ShInterval<T> &a, const ShInterval<T> &b, const ShInterval<T> &c);

template<typename T>
ShInterval<T> lerp(const ShInterval<T> &a, const ShInterval<T> &b, const ShInterval<T> &c);
}

#include "ShIntervalImpl.hpp"
  
#endif
