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
#ifndef SHDATATYPE_HPP
#define SHDATATYPE_HPP

#include "ShUtility.hpp"
#include "ShVariableType.hpp"
#include "ShInterval.hpp"
#include "ShHalf.hpp"
#include "ShFraction.hpp"
#include "ShStorageType.hpp"

/** @file ShDataType.hpp
 * Defines the host computation and memory storage c++ types associated with
 * each Sh value type.
 *
 * Also defines   
 */
namespace SH {
/**  Used to denote the kinds of C++ data types associated with a Value type. 
 */
enum ShDataType {
  SH_HOST,  //< computation data type on the host
  SH_MEM,  //< data type stored in memory on the host
  SH_DATATYPE_END
};

SH_DLLEXPORT
extern const char* dataTypeName[];

/** Sets the actual host computation and memory data types for a given Value type.
 *  The default is computation and memory types equal the Value type.
 *  Defined below are the exceptions to this rule
 * 
 *  half floats - memory type is 1 sign bit, 5 exponent bits, 10 mantissa bits
 *                see ATI_pixel_format_float for specification
 *                (NV_half_float is similar, except it has NaN and INF)
 * 
 * @{ */
template<typename T, ShDataType DT> struct ShDataTypeCppType; 

template<typename T> struct ShDataTypeCppType<T, SH_HOST> { typedef T type; }; 
template<typename T> struct ShDataTypeCppType<T, SH_MEM> { typedef T type; }; 

// define special cases here
#define SH_VALUETYPE_DATATYPE(T, hostType, memType)\
  template<> struct ShDataTypeCppType<T, SH_HOST> { typedef hostType type; }; \
  template<> struct ShDataTypeCppType<T, SH_MEM> { typedef memType type; }; 

SH_VALUETYPE_DATATYPE(ShHalf, float, ShHalf); 

template<typename T> struct ShHostType { typedef typename ShDataTypeCppType<T, SH_HOST>::type type; };
template<typename T> struct ShMemType { typedef typename ShDataTypeCppType<T, SH_MEM>::type type; };
// @}

/**  Sets the constant values for a given data type.
 * Currently only the additive and multiplicative inverses are here.
 * And with all current types, Zero is a false value and One is a true value
 * (although usually not the only ones).
 * @{ */
template<typename T, ShDataType DT> 
struct ShDataTypeConstant {
    typedef typename ShDataTypeCppType<T, DT>::type type;
    static const type Zero; /* additive identity and also a true value */ \
    static const type One; /* multiplicative identity also a false value */ \
};

template<typename T, ShDataType DT>
const typename ShDataTypeCppType<T, DT>::type ShDataTypeConstant<T, DT>::Zero = 
  (typename ShDataTypeCppType<T, DT>::type)(0.0); 

template<typename T, ShDataType DT>
const typename ShDataTypeCppType<T, DT>::type ShDataTypeConstant<T, DT>::One = 
  (typename ShDataTypeCppType<T, DT>::type)(1.0); 
// @}

/** Returns the boolean cond in the requested data type */
template<typename T, ShDataType DT>
inline
typename ShDataTypeCppType<T, DT>::type shDataTypeCond(bool cond);

/** Returns a whether the two values are exactly the same.
 * This is is useful for the range types.
 * @{ */
template<typename T>
inline
bool shDataTypeEqual(const T &a, const T &b);
// @}

/** Returns whether the value is always greater than zero (i.e. true) 
 */
template<typename T>
inline
bool shDataTypeIsPositive(const T &a);


/** Casts one data type to another data type 
 */
template<typename T1, ShDataType DT1, typename T2, ShDataType DT2>
void shDataTypeCast(typename ShDataTypeCppType<T1, DT1>::type &dest,
                    const typename ShDataTypeCppType<T2, DT2>::type &src);

}

#include "ShDataTypeImpl.hpp"

#endif
