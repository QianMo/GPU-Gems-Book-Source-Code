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
#ifndef SHHALF_HPP
#define SHHALF_HPP

#include <limits>
#include "ShUtility.hpp"

namespace SH {

struct ShHalf {
  typedef unsigned short T; 

  static const int S = 1 << 15; // sign bit
  static const int E = 1 << 10; // exponent 

  T m_val;

  /** Constructs an half with undefined value */
  ShHalf();

  /** Constructs an half */
  ShHalf(double value); 

  operator double() const;

  /** Arithmetic operators **/
  ShHalf& operator=(double value);
  ShHalf& operator=(const ShHalf& other);
  ShHalf& operator+=(double value);
  ShHalf& operator+=(const ShHalf& other);
  ShHalf& operator-=(double value);
  ShHalf& operator-=(const ShHalf& other);
  ShHalf& operator*=(double value);
  ShHalf& operator*=(const ShHalf& other);
  ShHalf& operator/=(double value);
  ShHalf& operator/=(const ShHalf& other);

  /** Float modulus - result is always positive 
   *@{*/
  ShHalf& operator%=(double value);
  ShHalf& operator%=(const ShHalf& other);
  // @}

  /** Negation **/
  ShHalf operator-() const;

  /** Output operator **/
  friend std::ostream& operator<<(std::ostream& out, const ShHalf &value);


  /** Input operator (format matches output) **/
  friend std::istream& operator>>(std::istream& out, ShHalf &value);

  private:
    /** Constructs a half */
    static ShHalf make_half(T value);

    static T to_val(double value);
    void set_val(double value);
    double get_double() const;
};

}

#include "ShHalfImpl.hpp"
  
#endif
