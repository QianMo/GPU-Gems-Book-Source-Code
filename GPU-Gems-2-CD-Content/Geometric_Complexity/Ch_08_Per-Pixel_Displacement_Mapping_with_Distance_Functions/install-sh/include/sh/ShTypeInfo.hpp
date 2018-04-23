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
#ifndef SHTYPEINFO_HPP
#define SHTYPEINFO_HPP

#include <string>
#include <vector>
#include "ShHashMap.hpp"
#include "ShVariableType.hpp"
#include "ShDataType.hpp"
#include "ShRefCount.hpp"
#include "ShInterval.hpp"
#include "ShFraction.hpp"
#include "ShHalf.hpp"

namespace SH {

/// forward declarations 
class ShVariantFactory;


/** A holder of information about a data type and how to allocate it 
 * @see ShDataType.hpp
 * */ 
struct 
SH_DLLEXPORT
ShTypeInfo {
  virtual ~ShTypeInfo() {}

  /** Returns a the name of the value type */
  virtual const char* name() const = 0;

  /** Returns size of type */ 
  virtual int datasize() const = 0;

  /** Returns the factory that generates ShVariant objects of this type */
  virtual const ShVariantFactory* variantFactory() const = 0; 

  /** Initializes the variant factories, automatic promotions, and
   * other variant casters.
   */
  static void init();

  /** Returns the type info with the requested value and data types. */ 
  static const ShTypeInfo* get(ShValueType valueType, ShDataType dataType);

  typedef ShPairHashMap<ShValueType, ShDataType, const ShTypeInfo*> TypeInfoMap;
  private:
    /** Holds ShDataTypeInfo instances for all available valuetype/datatypes */
    static TypeInfoMap m_valueTypes;

    /** Adds automatic promotion and other casts into the ShCastManager */ 
    static void addCasts();

    /** Adds ops to the ShEval class */ 
    static void addOps();
};

// generic level, singleton ShTypeInfo class holding information for
// a particular type
template<typename T, ShDataType DT>
struct ShDataTypeInfo: public ShTypeInfo {
  public:
    typedef typename ShDataTypeCppType<T, DT>::type type;
    static const type Zero;
    static const type One;

    const char* name() const; 
    int datasize() const;
    const ShVariantFactory* variantFactory() const;

    static const ShDataTypeInfo* instance();

  protected:
    static ShDataTypeInfo *m_instance;
    ShDataTypeInfo() {}
};


SH_DLLEXPORT
extern const ShTypeInfo* shTypeInfo(ShValueType valueType, ShDataType dataType = SH_HOST);

SH_DLLEXPORT
extern const ShVariantFactory* shVariantFactory(ShValueType valueType, ShDataType dataType = SH_HOST);

SH_DLLEXPORT
extern const char* shValueTypeName(ShValueType valueType);
}

#include "ShTypeInfoImpl.hpp"

#endif
