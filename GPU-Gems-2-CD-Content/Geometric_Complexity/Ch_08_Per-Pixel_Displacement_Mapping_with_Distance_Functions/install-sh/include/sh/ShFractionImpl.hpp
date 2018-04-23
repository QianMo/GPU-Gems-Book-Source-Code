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
#ifndef SHFRACTIONALIMPL_HPP
#define SHFRACTIONALIMPL_HPP

#include <cmath>
#include "ShMath.hpp"
#include "ShFraction.hpp"

namespace SH {

#define _CompType typename ShFraction<T>::CompType 
#define _LongType typename ShFraction<T>::LongType 
#define _SignedLongType typename ShFraction<T>::SignedLongType 

// @todo replace uses of std::fabs 

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>::ShFraction()
{}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>::ShFraction(double value)
  : m_val(clamp_val(value))
{
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> ShFraction<T>::make_fraction(CompType value)
{
  ShFraction result;
  result.m_val = clamp_val(value);
  return result; 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> ShFraction<T>::make_fraction_signed(SignedLongType value)
{
  ShFraction result;
  result.m_val = clamp_val_signed(value);
  return result; 
}

template<typename T/* @todo clamp , bool Clamp */>
template<typename T2>
ShFraction<T>::ShFraction(const ShFraction<T2> &other)
  : m_val(clamp_val(other.get_double())) 
{
}

/** accessor methods **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>::operator double() const
{
  return get_double(); 
}

template<typename T/* @todo clamp , bool Clamp */>
T& ShFraction<T>::val()
{
  return m_val;
}

template<typename T/* @todo clamp , bool Clamp */>
T ShFraction<T>::val() const
{
  return m_val;
}

/** Arithmetic operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator=(double value)
{
  m_val = clamp_val(value);
  return *this;
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator=(const ShFraction& other)
{
  m_val = other.m_val;
  return *this;
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator+=(double value)
{
  return operator+=(ShFraction(value));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator+=(const ShFraction& other)
{
  m_val = clamp_val(CompType(m_val) + CompType(other.m_val));
  return *this;
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator-=(double value)
{
  return operator-=(ShFraction(value));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator-=(const ShFraction& other)
{
  m_val = clamp_val_signed(SignedLongType(m_val) - SignedLongType(other.m_val));
  return *this;
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator*=(double value)
{
  return operator*=(ShFraction(value)); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator*=(const ShFraction& other)
{
  m_val = clamp_val(CompType(m_val) * CompType(other.m_val));
  return *this;
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator/=(double value)
{
  return operator/=(ShFraction(value)); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator/=(const ShFraction& other)
{
  LongType numerator = LongType(m_val) << BITS;   
  LongType denom = LongType(other.m_val);
  m_val = clamp_val(numerator / denom);
  return (*this); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator%=(double value)
{
  return operator%=(ShFraction(value));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T>& ShFraction<T>::operator%=(const ShFraction& other)
{
  // @todo range - should need no clamping for this
  m_val = m_val % other.m_val;
  if(m_val < 0) m_val += other.m_val;
}

/** Negation **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> ShFraction<T>::operator-() const 
{
  if(is_signed /* @todo clamp && Clamp */) return make_fraction(0); 
  return make_fraction(-m_val);
}

template<typename TT>
std::ostream& operator<<(std::ostream &out, const ShFraction<TT> &value)
{
  out << double(value);
  return out;
}

template<typename TT>
std::istream& operator>>(std::istream &in, ShFraction<TT> &value)
{
  double temp;
  in >> temp;
  value = temp;
  return in;
}

template<typename T/* @todo clamp , bool Clamp */>
inline
double ShFraction<T>::get_double() const
{
  return double(m_val) / double(ONE);
}

template<typename T/* @todo clamp , bool Clamp */>
inline
T ShFraction<T>::clamp_val(double value)
{
  double temp = value * ONE;

/* @todo clamp  if(Clamp) { */
    temp = std::max(std::min(temp, double(MAX)), double(MIN));
/* @todo clamp  } */
  return T(temp);
}

template<typename T/* @todo clamp , bool Clamp */>
inline
T ShFraction<T>::clamp_val(CompType value)
{
  /* @todo clamp if(Clamp) { */
    value = std::max(std::min(value, CompType(MAX)), CompType(MIN));
  /* @todo clamp } */
  return T(value);
}

template<typename T/* @todo clamp , bool Clamp */>
inline
T ShFraction<T>::clamp_val_signed(SignedLongType value)
{
  /* @todo clamp if(Clamp) { */
    value = std::max(std::min(value, SignedLongType(MAX)), SignedLongType(MIN));
  /* @todo clamp } */
  return T(value);
}

/** Arithmetic operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator+(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return ShFraction<T>::make_fraction(_CompType(a.m_val) + _CompType(b.m_val));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator-(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return ShFraction<T>::make_fraction_signed(
      _SignedLongType(a.m_val) - 
      _SignedLongType(b.m_val));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator*(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return ShFraction<T>::make_fraction(_CompType(a.m_val) * _CompType(b.m_val));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator/(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  _LongType numerator = _LongType(a.m_val) << ShFraction<T>::BITS;   
  _LongType denom = _LongType(b.m_val);
  return ShFraction<T>::make_fraction(numerator / denom);
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator%(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  T temp = a.m_val % b.m_val;
  if(temp < 0) temp += b.m_val; 
  return ShFraction<T>(temp);
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> cbrt(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::pow(double(a), 1.0 / 3.0)); 
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> exp(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::exp(double(a))); 
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> exp2(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::pow(2.0, double(a))); 
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> exp10(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::pow(10.0, double(a))); 
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> log(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::log(double(a))); 
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> log2(const ShFraction<T> &a) 
{
  return ShFraction<T>(log2f(double(a))); 
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> log10(const ShFraction<T> &a) 
{
  return ShFraction<T>(log10f(double(a))); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> frac(const ShFraction<T> &a)
{
  T result = a.m_val;
  if(result < 0) {
    result += ShFraction<T>::ONE;
  }
  return ShFraction<T>(result);
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> fmod(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return a % b;
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> pow(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  // @todo check if this is optimal
  // @todo do integer special cases? - see NuS.cc
  return ShFraction<T>(std::pow(double(a), double(b))); 
}


// not a good function for fractional types...
// guaranteed to overflow...DOH
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> rcp(const ShFraction<T> &a) 
{
  if(a.m_val > 0) return ShFraction<T>(ShFraction<T>::MAX);
  return ShFraction<T>(ShFraction<T>::MIN);
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> rsq(const ShFraction<T> &a) 
{
  return rcp(a); // same bad behaviour 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> sgn(const ShFraction<T> &a) 
{
  return ShFraction<T>(a.m_val > 0 ? 1 : a.m_val == 0 ? 0 : -1); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> sqrt(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::sqrt(double(a)));
}


/** Trig Operators */
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> acos(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::acos(double(a)));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> asin(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::asin(double(a)));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> atan(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::atan(double(a)));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> atan2(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return ShFraction<T>(std::atan2(double(a), double(b)));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> cos(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::cos(double(a)));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> sin(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::sin(double(a)));
}


template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> tan(const ShFraction<T> &a) 
{
  return ShFraction<T>(std::tan(double(a)));
}


/** Comparison Operators **/
template<typename T/* @todo clamp , bool Clamp */>
bool operator<(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return (a.m_val < b.m_val);
}

template<typename T/* @todo clamp , bool Clamp */>
bool operator<=(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return (a.m_val <= b.m_val);
}

template<typename T/* @todo clamp , bool Clamp */>
bool operator>(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return (a.m_val > b.m_val);
}

template<typename T/* @todo clamp , bool Clamp */>
bool operator>=(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return (a.m_val >= b.m_val);
}

template<typename T/* @todo clamp , bool Clamp */>
bool operator==(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return (a.m_val == b.m_val);
}

template<typename T/* @todo clamp , bool Clamp */>
bool operator!=(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return (a.m_val != b.m_val);
}

/** Clamping operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> min(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return ShFraction<T>(std::min(a.m_val, b.m_val));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> max(const ShFraction<T> &a, const ShFraction<T> &b) 
{
  return ShFraction<T>(std::max(a.m_val, b.m_val));
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> floor(const ShFraction<T> &a) 
{

  T result = 0; 
  if(a.m_val == ShFraction<T>::ONE) {
    result = ShFraction<T>::ONE;
  } else if(ShFraction<T>::is_signed && a.m_val < 0) {
    a.m_val = -ShFraction<T>::ONE;
  }
  return ShFraction<T>(result); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> ceil(const ShFraction<T> &a) 
{
  T ONE = ShFraction<T>::ONE;
  T result = 0; 
  if(a.m_val > 0) {
    result = ONE;
  } else if(ShFraction<T>::is_signed && 
      a.m_val == -ONE) {
    result = -ONE;
  }
  return ShFraction<T>(result);
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> rnd(const ShFraction<T> &a) 
{
  T ONE = ShFraction<T>::ONE;
  T HALF = ONE >> 1; // slightly less than half
  T result;
  if(a.m_val > HALF) {
    result = ONE;
  } else if(!ShFraction<T>::is_signed || result > -HALF) {
    result = 0;
  } else {
    result = -ONE;
  }
  return ShFraction<T>(result); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> abs(const ShFraction<T> &a) 
{
  return ShFraction<T>(a.m_val < 0 ? -a.m_val: a.m_val);
}

/** Misc operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> cond(const ShFraction<T> &a, const ShFraction<T> &b, 
    const ShFraction<T> &c)
{
  return ShFraction<T>(a.m_val > 0 ? b.m_val: c.m_val); 
}

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> lerp(const ShFraction<T> &a, const ShFraction<T> &b, const ShFraction<T> &c) 
{
  T ONE = ShFraction<T>(ShFraction<T>::ONE);
  return a * b + (ONE - a) * c;
}

#undef _CompType
#undef _LongType
#undef _SignedLongType
}

#endif
