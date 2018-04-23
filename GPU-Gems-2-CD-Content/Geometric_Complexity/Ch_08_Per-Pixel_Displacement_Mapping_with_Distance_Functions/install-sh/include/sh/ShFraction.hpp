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
#ifndef SHFRACTION_HPP
#define SHFRACTION_HPP

#include <limits>
#include "ShUtility.hpp"

namespace SH {

/** Determines the computation used for intermediate values 
 * This means that the maximum fractiona type T supported has half as many bits
 * as the largets integer type supported in hardware */ 
template<typename T> struct ShFractionLongType { typedef T type; };

// @todo range - making certain assumptions about sizes of type 
// Make sure these are valid under all circumstances
// May want to use int instead of short in some cases if hardware needs
// to convert shorts.
template<> struct ShFractionLongType<int> { typedef long long type; };
template<> struct ShFractionLongType<short> { typedef int type; };
template<> struct ShFractionLongType<char> { typedef short type; };
template<> struct ShFractionLongType<unsigned int> { typedef unsigned long long type; };
template<> struct ShFractionLongType<unsigned short> { typedef unsigned int type; };
template<> struct ShFractionLongType<unsigned char> { typedef unsigned short type; };

template<typename T> struct ShFractionSignedLongType { typedef T type; };

// @todo range - making certain assumptions about sizes of type; 
// Make sure these are valid under all circumstances
template<> struct ShFractionSignedLongType<int> { typedef long long type; };
template<> struct ShFractionSignedLongType<short> { typedef int type; };
template<> struct ShFractionSignedLongType<char> { typedef short type; };
template<> struct ShFractionSignedLongType<unsigned int> { typedef long long type; };
template<> struct ShFractionSignedLongType<unsigned short> { typedef int type; };
template<> struct ShFractionSignedLongType<unsigned char> { typedef short type; };

/*** Sh class for fraction types represented in integers.  
 *
 * Inspired by NuFixed.hh and NuFixed.cc from the MetaMedia project.
 *
 * @param T integer type to use
 */

/** 
 * This param does not exist any more because the default param broke some stuff 
 * like ShIsFraction in ShStorageType.hpp under VC.NET, and I don't have time to 
 * fix it right now.  
 *
 * All ShFractions are by default clamped.
 * Everything that was commented out has been marked with a @todo clamp
 *
 *
 * @param Clamp whether to clamp to avoid overflow.  If this is true, then
 * during computation we always use a temporary type with enough bits to hold
 * the result.  If this is false, then we only use temporaries with extra bits
 * if an intermediate value may overflow.  If the result itself overflows,
 * then the value stored is implementation defined (wraps around for 2's complement). 
 */

template<typename T /* @todo clamp , bool Clamp=true */>
struct ShFraction {

  // Type to use for operations that require temps with twice the bits
  // (and when clamping is on)
  typedef typename ShFractionLongType<T>::type LongType; 
  typedef typename ShFractionSignedLongType<T>::type SignedLongType; 

  // The usual type used in computation
  // @todo clamp typedef typename SelectType<Clamp, LongType, T>::type CompType; 
  typedef LongType CompType; 

  // Some information about the type and constant values representable by
  // this fraction type
  static const bool is_signed = std::numeric_limits<T>::is_signed;
  static const int BITS = sizeof(T) * 8; ///< number of bits
  static const T ONE; ///< representation of ONE
  static const T MAX; ///< maximum representable value
  static const T MIN;  ///< minumum representable value

  T m_val;

  /** Constructs an fraction with undefined value */
  ShFraction();

  /** Constructs an fraction */
  ShFraction(double value);

  /** Makes a fraction and clamps from the computation type */
  static ShFraction make_fraction(CompType value);

  /** Makes a fraction and clamps from the signed type */
  static ShFraction make_fraction_signed(SignedLongType value);

  template<typename T2>
  ShFraction(const ShFraction<T2> &other);

  /** accessor methods **/
  operator double() const;
  T& val();
  T val() const;


  /** Arithmetic operators **/
  ShFraction& operator=(double value);
  ShFraction& operator=(const ShFraction& other);
  ShFraction& operator+=(double value);
  ShFraction& operator+=(const ShFraction& other);
  ShFraction& operator-=(double value);
  ShFraction& operator-=(const ShFraction& other);
  ShFraction& operator*=(double value);
  ShFraction& operator*=(const ShFraction& other);
  ShFraction& operator/=(double value);
  ShFraction& operator/=(const ShFraction& other);

  /** Float modulus - result is always positive 
   *@{*/
  ShFraction& operator%=(double value);
  ShFraction& operator%=(const ShFraction& other);
  // @}

  /** Scalar arithmetic operators **/

  /** Negation **/
  ShFraction operator-() const;

  /** Output operator **/
  template<typename TT>
  friend std::ostream& operator<<(std::ostream& out, const ShFraction<TT> &value);


  /** Input operator (format matches output) **/
  template<typename TT>
  friend std::istream& operator>>(std::istream& out, ShFraction<TT> &value);

  private:
    // convert value to double
    double get_double() const;

    // convert double to value (may clamp)
    static T clamp_val(double value);

    // convert temporary computation type to value (may clamp) 
    // this works just fine for the non-clamp case as well as well
    static T clamp_val(CompType temp);

    // convert temporary computation type to value (may clamp) 
    static T clamp_val_signed(SignedLongType temp);
};

template<typename T/* @todo clamp , bool Clamp */>
const T ShFraction<T>::ONE = std::numeric_limits<T>::max(); 

template<typename T/* @todo clamp , bool Clamp */>
const T ShFraction<T>::MAX = ShFraction<T>::ONE; 

template<typename T/* @todo clamp , bool Clamp */>
const T ShFraction<T>::MIN = is_signed ? -ShFraction<T>::ONE : 0;  

/** Arithmetic operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator+(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator-(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator*(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator/(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> operator%(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> cbrt(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> exp(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> exp2(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> exp10(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> log(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> log2(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> log10(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> frac(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> fmod(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> pow(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> rcp(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> rsq(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> sgn(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> sqrt(const ShFraction<T> &a);

/** Trig Operators */
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> acos(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> asin(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> atan(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> atan2(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> cos(const ShFraction<T> &a); 
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> sin(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> tan(const ShFraction<T> &a);

/** Comparison Operators **/
template<typename T/* @todo clamp , bool Clamp */>
bool operator<(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
bool operator<=(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
bool operator>(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
bool operator>=(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
bool  operator==(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
bool operator!=(const ShFraction<T> &a, const ShFraction<T> &b);

/** Clamping operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> min(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> max(const ShFraction<T> &a, const ShFraction<T> &b);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> floor(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> ceil(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> rnd(const ShFraction<T> &a);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> abs(const ShFraction<T> &a);

/** Misc Operators **/
template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> cond(const ShFraction<T> &a, const ShFraction<T> &b, const ShFraction<T> &c);

template<typename T/* @todo clamp , bool Clamp */>
ShFraction<T> lerp(const ShFraction<T> &a, const ShFraction<T> &b, const ShFraction<T> &c);

typedef ShFraction<int> ShFracInt;
typedef ShFraction<short> ShFracShort;
typedef ShFraction<char> ShFracByte;

typedef ShFraction<unsigned int> ShFracUInt;
typedef ShFraction<unsigned short> ShFracUShort;
typedef ShFraction<unsigned char> ShFracUByte;

}


#include "ShFractionImpl.hpp"
  
#endif
