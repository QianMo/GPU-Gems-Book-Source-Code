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
#ifndef SHSTORAGETYPE_HPP
#define SHSTORAGETYPE_HPP

#include "ShUtility.hpp"
#include "ShVariableType.hpp"
#include "ShInterval.hpp"
#include "ShHalf.hpp"
#include "ShFraction.hpp"

/** @file ShDataType.hpp
 * Defines the host computation and memory storage c++ types associated with
 * each Sh value type.
 *
 * Also defines   
 */
namespace SH {

/** A special C++ type used to represent an invalid storage type */
struct ShInvalidStorageType {};

/**  The various value type integers correspond to storage types. 
 *
 * Sh internally uses unsigned integers < 2^16.  User-defined types must use
 * values >= 2^16. 
 *
 * This is idea behind the current set of value types:
 * i) bits 0-2 = B, where 1 << B = sizeof type in bytes 
 * ii) bit 3 = 0 for signed type, 1 for unsigned
 * iii) bits 4-7 = 0 for floating point, 1 for integer, 2 for fraction 
 * iv) bits 8-14 = 0 for regular, 1 for interval, 2 for affine 
 * v) bit 15 = 0 for valid type, 1 for invalid type
 * @{ */
typedef unsigned int ShValueType;
enum __ShValueTypeEnum {
  SH_HALF       = 0x0001, 
  SH_FLOAT      = 0x0002, 
  SH_DOUBLE     = 0x0003, 

  SH_BYTE       = 0x0010,
  SH_SHORT      = 0x0011,
  SH_INT        = 0x0012,

  SH_UBYTE      = 0x0018,
  SH_USHORT     = 0x0019,
  SH_UINT       = 0x001A,

  SH_FBYTE      = 0x0020,
  SH_FSHORT     = 0x0021,
  SH_FINT       = 0x0022,

  SH_FUBYTE     = 0x0028,
  SH_FUSHORT    = 0x0029,
  SH_FUINT      = 0x002A,

  SH_I_HALF    = 0x0101,
  SH_I_FLOAT   = 0x0102,
  SH_I_DOUBLE  = 0x0103,

  SH_A_HALF    = 0x0201,
  SH_A_FLOAT   = 0x0202,
  SH_A_DOUBLE  = 0x0203,

  SH_VALUETYPE_SIZE_MASK = 0x0007,

  SH_VALUETYPE_SIGNED_MASK = 0x0008,
  SH_VALUETYPE_SIGNED = 0x0000,

  SH_VALUETYPE_TYPE_MASK = 0x00F0,
  SH_VALUETYPE_TYPE_FLOAT = 0x0000,
  SH_VALUETYPE_TYPE_INT   = 0x0010,
  SH_VALUETYPE_TYPE_FRAC  = 0x0020,

  SH_VALUETYPE_SPECIAL_MASK = 0x7F00, 
  SH_VALUETYPE_SPECIAL_NONE = 0x0100, 
  SH_VALUETYPE_SPECIAL_I    = 0x0100, 
  SH_VALUETYPE_SPECIAL_A    = 0x0200, 

  SH_VALUETYPE_END = 0xFFFF
};
// @}

/** Functions to retrieve traits about storage types from their ShValueType 
 *
 * @todo range - these should check for user defined value types later
 * and return false.
 * @{ */
bool shIsFloat(ShValueType value_type);
bool shIsInteger(ShValueType value_type);
bool shIsFraction(ShValueType value_type);

bool shIsSigned(ShValueType value_type);

bool shIsRegularValueType(ShValueType value_type);
bool shIsInterval(ShValueType value_type);

bool shIsInvalidValueType(ShValueType value_type);
// @}

/** Mappings from value type to storage type and back. 
 * @{ */ 
template<ShValueType V> struct __ShValueToStorageType { typedef ShInvalidStorageType type; };
template<typename T> struct __ShStorageToValueType { static const ShValueType type = SH_VALUETYPE_END; };

#define SH_VALUE_STORAGE_TYPE_MAPPING(V, T)\
  template<> struct __ShValueToStorageType<V >    { typedef T type; }; \
  template<> struct __ShStorageToValueType<T >    { static const ShValueType type = V; }; 

SH_VALUE_STORAGE_TYPE_MAPPING(SH_HALF,    ShHalf); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_FLOAT,   float); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_DOUBLE,  double); 

SH_VALUE_STORAGE_TYPE_MAPPING(SH_BYTE,    char); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_SHORT,   short); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_INT,     int); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_UBYTE,   unsigned char); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_USHORT,  unsigned short); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_UINT,    unsigned int); 

SH_VALUE_STORAGE_TYPE_MAPPING(SH_FBYTE,   ShFracByte); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_FSHORT,  ShFracShort); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_FINT,    ShFracInt); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_FUBYTE,  ShFracUByte); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_FUSHORT, ShFracUShort); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_FUINT,   ShFracUInt); 

SH_VALUE_STORAGE_TYPE_MAPPING(SH_I_HALF,  ShInterval<ShHalf>); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_I_FLOAT, ShInterval<float>); 
SH_VALUE_STORAGE_TYPE_MAPPING(SH_I_DOUBLE,ShInterval<double>); 

// @}

/** Mapping from Storage Type to a name 
 * @{*/
template<typename T> struct __ShStorageTypeName { static const char* name; };
template<typename T> const char* __ShStorageTypeName<T>::name = "unknown";

#define SH_STORAGETYPE_NAME_SPEC(T)\
  template<> struct SH_DLLEXPORT __ShStorageTypeName<T > { static const char* name; };

SH_STORAGETYPE_NAME_SPEC(ShHalf);
SH_STORAGETYPE_NAME_SPEC(float);
SH_STORAGETYPE_NAME_SPEC(double);

SH_STORAGETYPE_NAME_SPEC(char);
SH_STORAGETYPE_NAME_SPEC(short);
SH_STORAGETYPE_NAME_SPEC(int);
SH_STORAGETYPE_NAME_SPEC(unsigned char);
SH_STORAGETYPE_NAME_SPEC(unsigned short);
SH_STORAGETYPE_NAME_SPEC(unsigned int);

SH_STORAGETYPE_NAME_SPEC(ShFracByte);
SH_STORAGETYPE_NAME_SPEC(ShFracShort);
SH_STORAGETYPE_NAME_SPEC(ShFracInt);
SH_STORAGETYPE_NAME_SPEC(ShFracUByte);
SH_STORAGETYPE_NAME_SPEC(ShFracUShort);
SH_STORAGETYPE_NAME_SPEC(ShFracUInt);

SH_STORAGETYPE_NAME_SPEC(ShInterval<ShHalf>);
SH_STORAGETYPE_NAME_SPEC(ShInterval<float>);
SH_STORAGETYPE_NAME_SPEC(ShInterval<double>);

#undef SH_STORAGETYPE_NAME_SPEC
//@}



/** Returns whether a type is an interval type
 * @{ */
template<typename T>
struct ShIsInterval: public MatchTemplateType<T, ShInterval> {};
//@}

/** Returns whether a type is an interval type
 * @{ */
template<typename T>
struct ShIsFraction: public MatchTemplateType<T, ShFraction> {};
//@}

/** Returns an interval value type corresponding to a type,
 * or SH_VALUETYPE_NONE if no such type is defined; 
 * @{ */

template<typename T>
struct __ShIntervalStorageType
{
  static const bool invalid = MatchType<T, ShInvalidStorageType>::matches;
  static const bool is_interval = ShIsInterval<T>::matches;
  // @todo range - this doesn't quite work once we have other special types,
  // perhaps...
  typedef typename SelectType<invalid, ShInvalidStorageType, 
           typename SelectType<is_interval, T, ShInterval<T> >::type>::type type; 
};

inline 
ShValueType shIntervalValueType(ShValueType value_type); 
// @}

/** Returns the regular value type corresponding to a special templated value type
 * (interval or affine) so far 
 * @{ */
template<ShValueType V>
struct __ShRegularValueType 
{
  static const ShValueType type = (V & ~SH_VALUETYPE_SPECIAL_MASK); 
};

template<typename T>
struct __ShRegularStorageType
{
  static const ShValueType range_value_type = __ShStorageToValueType<T>::type;
  static const ShValueType value_type = __ShRegularValueType<range_value_type>::type; 
  typedef typename __ShValueToStorageType<value_type>::type type; 
};

inline 
ShValueType shRegularValueType(ShValueType value_type); 
// @}

/** Provides a least common ancestor in the automatic promotion tree
 * for use in immediate mode
 * @{ */
template<typename T1, typename T2>
struct ShCommonType;

template<typename T1, typename T2, typename T3>
struct ShCommonType3 {
  typedef typename ShCommonType<typename ShCommonType<T1, T2>::type, T3>::type type; 
};

template<typename T1, typename T2, typename T3, typename T4>
struct ShCommonType4 {
  typedef typename ShCommonType<typename ShCommonType<T1, T2>::type, 
                                typename ShCommonType<T3, T4>::type>::type type; 
};
// @}

/** Holds much of the above information in one place.
 * This is the class to specialize for user-defined types
 * @{ */
template<typename T>
struct ShStorageTypeInfo {
  typedef T storage_type;
  static const ShValueType value_type = __ShStorageToValueType<T>::type; 


  /** Non-special storage type corresponding to T */
  typedef typename __ShRegularStorageType<T>::type RegularType;
  static const ShValueType RegularValueType = __ShStorageToValueType<RegularType>::type;

  // @todo not sure we want all of these here, since there could be user-defined 
  // special types too.  This might be too restrictive if we depend on these. 

  /** Interval storage type corresponding to T (either T itself or uses T as its
   * bounds).  May be ShInvalidStorageType if no proper interval type exists */ 
  typedef typename __ShIntervalStorageType<T>::type IntervalType; 
  static const ShValueType IntervalValueType = __ShStorageToValueType<IntervalType>::type;

  static const char* name; 

  // @todo type include here the ability to set available operations 

  // @todo type include here ShPrograms for transforming from this
  // storage type to a set of different ones, allowing the transformers
  // to choose the best conversions.

  // once the above two things are complete, than user-defined types should
  // work just fine.
};

template<typename T>
const char* ShStorageTypeInfo<T>::name = __ShStorageTypeName<T>::name;


//@}

/** Subclass of ShStorageTypeInfo so we can lookup the same things with a
 * ShValueType.  This is probably not necessary though since most of the time
 * ShValueType is used only internally in the bowels of Sh. */
template<ShValueType V>
struct ShValueTypeInfo: public ShStorageTypeInfo<typename __ShValueToStorageType<V>::type> {
};

}

#include "ShStorageTypeImpl.hpp"

#endif
