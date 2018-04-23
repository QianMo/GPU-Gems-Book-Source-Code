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
#ifndef SHINTERVALIMPL_HPP
#define SHINTERVALIMPL_HPP

#include <cmath>
#include "ShMath.hpp"
#include "ShInterval.hpp"

namespace SH {

// @todo replace uses of std::fabs 

/** Constructs an interval with undefined value */
template<typename T>
ShInterval<T>::ShInterval()
{}

/** Constructs an interval [value,value] */
template<typename T>
ShInterval<T>::ShInterval(const T& value)
  : m_lo(value), m_hi(value)
{}

/** Constructs an interval with the given bounds */
template<typename T>
ShInterval<T>::ShInterval(const T& lo, const T& hi)
  : m_lo(lo), m_hi(hi)
{}

template<typename T>
template<typename T2>
ShInterval<T>::ShInterval(const ShInterval<T2> &other)
  : m_lo((T)(other.lo())), m_hi((T)(other.hi()))
{}

/** accessor methods **/
template<typename T>
T& ShInterval<T>::lo()
{
  return m_lo;
}

template<typename T>
const T& ShInterval<T>::lo() const
{
  return m_lo;
}

template<typename T>
T& ShInterval<T>::hi()
{
  return m_hi;
}

template<typename T>
const T& ShInterval<T>::hi() const
{
  return m_hi;
}

/** Useful helpers **/
template<typename T>
const T& ShInterval<T>::width() const
{
  return m_hi - m_lo;
}

template<typename T>
const T& ShInterval<T>::centre() const
{
  return (m_hi + m_lo) / 2; 
}

template<typename T>
const T& ShInterval<T>::radius() const
{
  return width() / 2;
}

/** Arithmetic operators **/
template<typename T>
ShInterval<T>& ShInterval<T>::operator=(const T &value)
{
  m_lo = m_hi = value;
  return *this;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator=(const ShInterval<T> &other)
{
  m_lo = other.m_lo;
  m_hi = other.m_hi;
  return *this;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator+=(const T &value)
{
  m_lo += value;
  m_hi += value;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator+=(const ShInterval<T> &other)
{
  m_lo += other.m_lo;
  m_hi += other.m_hi;
  return *this;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator-=(const T &value)
{
  m_lo -= value;
  m_hi -= value;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator-=(const ShInterval<T> &other)
{
  m_lo -= other.m_hi;
  m_hi -= other.m_lo;
  return *this;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator*=(const T &value)
{
  m_lo = m_lo * value;
  m_hi = m_hi * value;
  if(value < 0) std::swap(m_lo, m_hi); 
  return *this;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator*=(const ShInterval<T> &other)
{
  T ll, lh, hl, hh;
  ll = m_lo * other.m_lo;
  lh = m_lo * other.m_hi;
  hl = m_hi * other.m_lo;
  hh = m_hi * other.m_hi;

  m_lo = std::min(std::min(ll, lh), std::min(hl, hh));
  m_hi = std::max(std::max(ll, lh), std::max(hl, hh));

  return *this;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator/=(const T &value)
{
  return (*this) *= (1 / value);
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator/=(const ShInterval<T> &other)
{
  return (*this) *= rcp(other);
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator%=(const T &value)
{
  // @todo
  m_hi = std::fabs(value);
  m_lo = -m_hi;
}

template<typename T>
ShInterval<T>& ShInterval<T>::operator%=(const ShInterval<T> &other)
{
  // @todo
  m_hi = std::max(std::fabs(other.lo), std::fabs(other.hi));
  m_lo = -m_hi;
}

/** Negation **/
template<typename T>
ShInterval<T> ShInterval<T>::operator-() const 
{
  return ShInterval<T>(-m_hi, -m_lo);
}

template<typename TT>
std::ostream& operator<<(std::ostream &out, const ShInterval<TT> &value)
{
  out << "[" << value.m_lo << "," << value.m_hi << "]";
  return out;
}

template<typename TT>
std::istream& operator>>(std::istream &in, ShInterval<TT> &value)
{
  // @todo this isn't exactly bullet-proof right now...
  in.ignore(1, '[');
  in >> value.m_lo;
  in.ignore(1, ',');
  in >> value.m_hi;
  in.ignore(1, ']');
  return in;
}

/** Arithmetic operators **/
template<typename T>
ShInterval<T> operator+(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return ShInterval<T>(a.m_lo + b.m_lo, a.m_hi + b.m_hi);
}


template<typename T>
ShInterval<T> operator-(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return ShInterval<T>(a.m_lo - b.m_hi, a.m_hi - b.m_lo);
}

template<typename T>
ShInterval<T> operator-(const T &a, const ShInterval<T> &b) 
{
  return ShInterval<T>(a - b.m_hi, a - b.m_lo);
}


template<typename T>
ShInterval<T> operator*(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  T ll, lh, hl, hh;
  ll = a.m_lo * b.m_lo;
  lh = a.m_lo * b.m_hi;
  hl = a.m_hi * b.m_lo;
  hh = a.m_hi * b.m_hi;

  return ShInterval<T>(std::min(std::min(ll, lh), std::min(hl, hh)),
      std::max(std::max(ll, lh), std::max(hl, hh)));
}


template<typename T>
ShInterval<T> operator/(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return a * rcp(b);
}


template<typename T>
ShInterval<T> operator%(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  T large = std::max(std::fabs(b.m_lo), std::fabs(b.m_hi));
  return ShInterval<T>(-large, large); 
}

template<typename T>
ShInterval<T> cbrt(const ShInterval<T> &a) 
{
  // monotonic
  // @todo type check this one 
  return pow(a, ShInterval<T>(1.0f / 3.0f)); 
}


template<typename T>
ShInterval<T> exp(const ShInterval<T> &a) 
{
  // monotonic function
  return ShInterval<T>(std::exp(a.m_lo), std::exp(a.m_hi));
}


template<typename T>
ShInterval<T> exp2(const ShInterval<T> &a) 
{
  // monotonic function
  return ShInterval<T>(exp2f(a.m_lo), exp2f(a.m_hi));
}


template<typename T>
ShInterval<T> exp10(const ShInterval<T> &a) 
{
  // monotonic function
return ShInterval<T>(exp10f(a.m_lo), exp10f(a.m_hi)); 
}


template<typename T>
ShInterval<T> log(const ShInterval<T> &a) 
{
  // monotonic, except <= 0 gives NaN 
  // @todo handle <= 0 with exception?
  return ShInterval<T>(std::log(a.m_lo), std::log(a.m_hi));
}


template<typename T>
ShInterval<T> log2(const ShInterval<T> &a) 
{
  // monotonic, except <= 0 gives NaN 
  // @todo handle <= 0 with exception or clamp?
  return ShInterval<T>(log2f(a.m_lo), log2f(a.m_hi));
}


template<typename T>
ShInterval<T> log10(const ShInterval<T> &a) 
{
  // monotonic, except <= 0 gives NaN 
  // @todo handle <= 0 with exception or clamp?
  return ShInterval<T>(log10f(a.m_lo), log10f(a.m_hi));
}

template<typename T>
ShInterval<T> frac(const ShInterval<T> &a)
{
  // @todo - if width > 1, then return below,
  // otherwise do some more figuring
  return ShInterval<T>(0, 1);
}

template<typename T>
ShInterval<T> fmod(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return a % b;
}


template<typename T>
ShInterval<T> pow(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  // @todo check if this is optimal
  // @todo do integer special cases? - see NuS.cc
  return exp(log(a)*b);
}


template<typename T>
ShInterval<T> rcp(const ShInterval<T> &a) 
{
  if(a.m_lo <= 0 && a.m_hi >= 0) {
    // @todo handle infinities
    // This results in a -inf, inf interval
  }
  return ShInterval<T>(1.0 / a.m_hi, 1.0 / a.m_lo);
}

template<typename T>
ShInterval<T> rsq(const ShInterval<T> &a) 
{
  // @todo something better?
  return rcp(sqrt(a)); 
}

template<typename T>
ShInterval<T> sgn(const ShInterval<T> &a) 
{
  // @todo have constants for positve, negative, zero values

  // the usual sgn function is monotonic:
  return ShInterval<T>(a.m_lo < 0 ? -1 : a.m_lo > 0 ? 1 : 0,
                       a.m_hi < 0 ? -1 : a.m_hi > 0 ? 1 : 0);
}

template<typename T>
ShInterval<T> sqrt(const ShInterval<T> &a) 
{
  // monotonic, except <= 0 gives NaN
  // @todo handle <= 0 with exception or clamp
  return ShInterval<T>(std::sqrt(a.m_lo), std::sqrt(a.m_hi));
}


/** Trig Operators */
template<typename T>
ShInterval<T> acos(const ShInterval<T> &a) 
{
  // @todo
  return ShInterval<T>(0, M_PI);
}


template<typename T>
ShInterval<T> asin(const ShInterval<T> &a) 
{
  // @todo
  return ShInterval<T>(-M_PI/2, M_PI/2);
}


template<typename T>
ShInterval<T> atan(const ShInterval<T> &a) 
{
  // @todo
  return ShInterval<T>(-M_PI/2, M_PI/2);
}


template<typename T>
ShInterval<T> atan2(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  // @todo
  return ShInterval<T>(-M_PI, M_PI);
}


template<typename T>
ShInterval<T> cos(const ShInterval<T> &a) 
{
  //@todo
  return ShInterval<T>(-1, 1);
}

template<typename T>
ShInterval<T> sin(const ShInterval<T> &a) 
{
  // @todo
  return ShInterval<T>(-1, 1);
}


template<typename T>
ShInterval<T> tan(const ShInterval<T> &a) 
{
  // @todo
  return ShInterval<T>(-1, 1); // @todo this is totally wrong...
}


/** Comparison Operators **/
template<typename T>
ShInterval<T> __boolean_op(bool trueCond, bool falseCond) {
  // @todo fix this assumption that false < true
  // @todo special value for maybe? 
  // @todo make this return something special (other than Interval<T>)?
  
  if(trueCond) return ShInterval<T>(1);
  if(falseCond) return ShInterval<T>(0);
  return ShInterval<T>(0, 1);
}

template<typename T>
ShInterval<T> operator<(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return __boolean_op<T>(a.m_hi < b.m_lo, a.m_lo > b.m_hi);
}

template<typename T>
ShInterval<T> operator<=(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return __boolean_op<T>(a.m_hi <= b.m_lo, a.m_lo > b.m_hi);
}

template<typename T>
ShInterval<T> operator>(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return __boolean_op<T>(a.m_hi > b.m_lo, a.m_lo < b.m_hi);
}

template<typename T>
ShInterval<T> operator>(const ShInterval<T> &a, const T &b) 
{
  return __boolean_op<T>(a.m_hi > b, a.m_lo < b);
}

template<typename T>
ShInterval<T> operator>=(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return __boolean_op<T>(a.m_hi > b.m_lo, a.m_lo < b.m_hi);
}

template<typename T>
ShInterval<T> operator==(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return __boolean_op<T>(
      a.m_hi == b.m_hi && a.m_lo == b.m_lo && a.m_lo == a.m_hi,
      a.m_hi < b.m_lo || a.m_lo > b.m_hi);
}

template<typename T>
ShInterval<T> operator!=(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  // @todo should use not(a==b) once we have not
  return __boolean_op<T>(
      a.m_hi < b.m_lo || a.m_lo > b.m_hi,
      a.m_hi == b.m_hi && a.m_lo == b.m_lo && a.m_lo == a.m_hi);
}

template<typename T>
bool boundsEqual(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return (a.lo() == b.lo()) && (a.hi() == b.hi());
}



/** Clamping operators **/
template<typename T>
ShInterval<T> min(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return ShInterval<T>(std::min(a.m_lo, b.m_lo), std::min(a.m_hi, b.m_hi));
}

template<typename T>
ShInterval<T> max(const ShInterval<T> &a, const ShInterval<T> &b) 
{
  return ShInterval<T>(std::max(a.m_lo, b.m_lo), std::max(a.m_hi, b.m_hi));
}

template<typename T>
ShInterval<T> floor(const ShInterval<T> &a) 
{
  // monotonic
  return ShInterval<T>(std::floor(a.m_lo), std::floor(a.m_hi));
}

template<typename T>
ShInterval<T> ceil(const ShInterval<T> &a) 
{
  // monotonic
  return ShInterval<T>(std::ceil(a.m_lo), std::ceil(a.m_hi));
}

template<typename T>
ShInterval<T> rnd(const ShInterval<T> &a) 
{
  // monotonic
  // @todo type handle this in general
  return ShInterval<T>(std::floor(a.m_lo + 0.5f), std::floor(a.m_hi + 0.5f));
}

template<typename T>
ShInterval<T> abs(const ShInterval<T> &a) 
{
  T lo, hi;

  hi = std::max(std::fabs(a.m_lo), std::fabs(a.m_hi));
  lo = std::min(std::fabs(a.m_lo), std::fabs(a.m_hi));
  if(a.m_lo <= 0 && a.m_hi >= 0) lo = 0;
  return ShInterval<T>(lo, hi);
}

/** Misc operators **/
template<typename T>
ShInterval<T> cond(const ShInterval<T> &a, const ShInterval<T> &b, 
    const ShInterval<T> &c)
{
  return lerp(a > (T)(0), b, c);
}

template<typename T>
ShInterval<T> lerp(const ShInterval<T> &a, const ShInterval<T> &b, const ShInterval<T> &c) 
{
  return a * b + ((T)(1) - a) * c;
}

}

#endif
