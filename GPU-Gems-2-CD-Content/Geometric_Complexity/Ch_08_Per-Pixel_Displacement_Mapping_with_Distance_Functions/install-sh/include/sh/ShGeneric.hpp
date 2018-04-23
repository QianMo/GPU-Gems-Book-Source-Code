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
#ifndef SHGENERIC_HPP
#define SHGENERIC_HPP

#include "ShVariableType.hpp"
#include "ShDataType.hpp"
#include "ShVariable.hpp"
#include "ShVariant.hpp"

namespace SH {

class ShProgram;

/** A variable of length N.
 *
 * This class is provided to make definition of functions that work
 * over n-tuples of particular types easier.
 *
 * ShAttrib derives from ShGeneric. Unlike ShGeneric, which only has
 * two template parameters, ShAttrib has four template
 * parameters. This would make writing functions quite ugly. E.g.:
 *
 * Without Generic:
 *
 * template<int N, typename T, ShBindingType B1, ShBindingType B2,
 *          bool S1, bool S2>
 * ShAttrib<N, SH_TEMP, T> add(const ShAttrib<N, B1, T, S1>& a,
 *                             const ShAttrib<N, B2, T, S2>& b);
 *
 * With Generic:
 *
 * template<int N, typename T>
 * ShAttrib<N, SH_TEMP, T> add(const ShGeneric<N, T>& a,
 *                             const ShGeneric<N, T>& b);
 *
 * This class is explicitly instantiated for T = float with 1 <= N <= 4.
 */
template<int N, typename T>
class ShGeneric : public ShVariable 
{
public:
  typedef T storage_type;
  static const ShValueType value_type = ShStorageTypeInfo<T>::value_type;
  typedef typename ShHostType<T>::type host_type; 
  typedef typename ShMemType<T>::type mem_type; 
  static const int typesize = N;

  typedef ShDataVariant<T, SH_HOST> VariantType; 
  typedef ShPointer<VariantType> VariantTypePtr;
  typedef ShPointer<const VariantType> VariantTypeCPtr;


  ShGeneric(const ShVariableNodePtr& node, ShSwizzle swizzle, bool neg);
  ~ShGeneric();

  // Copy constructor 
  // This should only be used internally.
  // It generates a ShVariableNode of type SH_TEMP 
  // @{

  // @todo type get rid of this. default should be okay for 
  // internal usage
  // ShGeneric(const ShGeneric<N, T> &other);

  template<typename T2>
  ShGeneric(const ShGeneric<N, T2> &other);
  // @}

  // This is needed because the templated assignment op is 
  // non-default, hence C++ spec 12.8.10 says the default
  // is implicitly defined.  Here that doesn't work so well. 
  ShGeneric& operator=(const ShGeneric& other);

  template<typename T2>
  ShGeneric& operator=(const ShGeneric<N, T2>& other);

  ShGeneric& operator=(const ShProgram& other);
  
  template<typename T2>
  ShGeneric& operator+=(const ShGeneric<N, T2>& right);

  template<typename T2>
  ShGeneric& operator-=(const ShGeneric<N, T2>& right);

  template<typename T2>
  ShGeneric& operator*=(const ShGeneric<N, T2>& right);

  template<typename T2>
  ShGeneric& operator/=(const ShGeneric<N, T2>& right);

  template<typename T2>
  ShGeneric& operator%=(const ShGeneric<N, T2>& right);

  template<typename T2>
  ShGeneric& operator+=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator-=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator*=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator/=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator%=(const ShGeneric<1, T2>& right);

  ShGeneric& operator+=(host_type);
  ShGeneric& operator-=(host_type);
  ShGeneric& operator*=(host_type);
  ShGeneric& operator/=(host_type);
  ShGeneric& operator%=(host_type);

  ShGeneric operator-() const;

  ShGeneric operator()() const; ///< Identity swizzle
  ShGeneric<1, T> operator()(int) const;
  ShGeneric<1, T> operator[](int) const;
  ShGeneric<2, T> operator()(int, int) const;
  ShGeneric<3, T> operator()(int, int, int) const;
  ShGeneric<4, T> operator()(int, int, int, int) const;

  /// Range Metadata
  void range(host_type low, host_type high);

  VariantType lowBound() const; 
  host_type lowBound(int index) const;

  VariantType highBound() const;
  host_type highBound(int index) const;
  
  // Arbitrary Swizzle
  template<int N2>
  ShGeneric<N2, T> swiz(int indices[]) const;

  /// Get the values of this variable, with swizzling taken into account
  void getValues(host_type dest[]) const;
  host_type getValue(int index) const;
  
  /// Set the values of this variable, using the swizzle as a
  /// writemask.
  void setValue(int index, const host_type &value); 
  void setValues(const host_type values[]);


protected:
  ShGeneric(const ShVariableNodePtr& node);

};

template<typename T>
class ShGeneric<1, T> : public ShVariable 
{
public:
  typedef T storage_type;
  static const ShValueType value_type = ShStorageTypeInfo<T>::value_type;
  typedef typename ShHostType<T>::type host_type; 
  typedef typename ShMemType<T>::type mem_type; 
  static const int typesize = 1;

  typedef ShDataVariant<T, SH_HOST> VariantType; 
  typedef ShPointer<VariantType> VariantTypePtr;
  typedef ShPointer<const VariantType> VariantTypeCPtr;


  ShGeneric(const ShVariableNodePtr& node, ShSwizzle swizzle, bool neg);
  ~ShGeneric();

  // Copy constructor 
  // This should only be used internally.  It generates a SH_TEMP, 
  // SH_ATTRIB, with the only characteristic copied from other being
  // the storage type.
  // @{

  // @todo type get rid of this
  // ShGeneric(const ShGeneric<1, T> &other);

  template<typename T2>
  ShGeneric(const ShGeneric<1, T2> &other);
  // @}

  ShGeneric& operator=(const ShGeneric<1, T>& other);

  template<typename T2>
  ShGeneric& operator=(const ShGeneric<1, T2>& other);

  ShGeneric& operator=(host_type);
  ShGeneric& operator=(const ShProgram& other);
  
  template<typename T2>
  ShGeneric& operator+=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator-=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator*=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator/=(const ShGeneric<1, T2>& right);

  template<typename T2>
  ShGeneric& operator%=(const ShGeneric<1, T2>& right);

  ShGeneric& operator+=(host_type);
  ShGeneric& operator-=(host_type);
  ShGeneric& operator*=(host_type);
  ShGeneric& operator/=(host_type);
  ShGeneric& operator%=(host_type);

  ShGeneric operator-() const;

  ShGeneric operator()() const; ///< Identity swizzle
  ShGeneric<1, T> operator()(int) const;
  ShGeneric<1, T> operator[](int) const;
  ShGeneric<2, T> operator()(int, int) const;
  ShGeneric<3, T> operator()(int, int, int) const;
  ShGeneric<4, T> operator()(int, int, int, int) const;

  /// Range Metadata
  void range(host_type low, host_type high);

  VariantType lowBound() const; 
  host_type lowBound(int index) const;

  VariantType highBound() const;
  host_type highBound(int index) const;
  
  // Arbitrary Swizzle
  template<int N2>
  ShGeneric<N2, T> swiz(int indices[]) const;

  /// Get the values of this variable, with swizzling taken into account
  void getValues(host_type dest[]) const;
  host_type getValue(int index) const;
  
  /// Set the values of this variable, using the swizzle as a
  /// writemask.
  void setValue(int index, const host_type &value); 
  void setValues(const host_type values[]);

protected:
  ShGeneric(const ShVariableNodePtr& node);

};

}

// This is a hack for ShAttrib.hpp in particular.
// A little dirty, but it works well.
#ifndef SH_DO_NOT_INCLUDE_GENERIC_IMPL
#include "ShGenericImpl.hpp"
#endif

#endif
