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
#ifndef SHVARIANTFACTORY_HPP
#define SHVARIANTFACTORY_HPP

#include <string>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShTypeInfo.hpp"

namespace SH {

struct ShVariant;

struct 
SH_DLLEXPORT 
ShVariantFactory {
  virtual ~ShVariantFactory() {}

  /// Creates a ShDataVariant object with N components 
  virtual ShVariant* generate(int N) const = 0; 

  /// Creates a ShDataVariant object by using the 
  // decode method from the Variant type corresponding
  // to this factory
  virtual ShVariant* generate(std::string s) const = 0;

  /// Creates an ShDataVariant object with the existing
  /// array as data
  /// @param managed Set to true iff this should make a copy
  //                 rather than using the given array internally.
  virtual ShVariant* generate(void *data, int N, bool managed = true) const = 0;  

  /// Creates an ShDataVariant object with N elements set to zero.
  virtual ShVariant* generateZero(int N = 1) const = 0;

  /// Creates an ShDataVariant object with N elements set to one. 
  virtual ShVariant* generateOne(int N = 1) const = 0;
};

template<typename T, ShDataType DT>
struct ShDataVariantFactory: public ShVariantFactory {
  ShVariant* generate(int N) const;

  ShVariant* generate(std::string s) const; 

  ShVariant* generate(void *data, int N, bool managed = true) const;  

  ShVariant* generateZero(int N = 1) const;
  ShVariant* generateOne(int N = 1) const;

  static const ShDataVariantFactory* instance();

  protected:
    static ShDataVariantFactory *m_instance;

    ShDataVariantFactory();
};


}

#include "ShVariantFactoryImpl.hpp"

#endif
