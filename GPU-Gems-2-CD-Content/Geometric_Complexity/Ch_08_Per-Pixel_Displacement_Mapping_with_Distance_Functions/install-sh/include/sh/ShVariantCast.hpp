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
#ifndef SHVARIANTCAST_HPP
#define SHVARIANTCAST_HPP

#include "ShDllExport.hpp"
#include "ShVariableType.hpp"
#include "ShDataType.hpp"
#include "ShVariant.hpp"

namespace SH {

// forward declaration
class ShVariant;

/// @file ShVariantCast.hpp
/// Declares a cast between one data type of a storage type and another.  

struct 
SH_DLLEXPORT
ShVariantCast {
  public:
    virtual ~ShVariantCast() {}

    /** Casts data from src into dest.  src must have Src value type and SrcDT
     * data type.  dest similarly must match Dest and DestDT.
     * Also, must guarantee dest != src (it will work if they are the same
     * but what a waste...)
     * @{ */
    virtual void doCast(ShVariant* dest, const ShVariant* src) const = 0;
    // @}

    virtual void getCastTypes(ShValueType &dest, ShDataType &destDT, 
                              ShValueType &src, ShDataType &srcDT) const = 0;

    /** Returns whether the destination of this caster matches the given types
     **/
    virtual void getDestTypes(ShValueType &valueType, ShDataType &dataType) const = 0;
};

/** @brief Handles casting between S and D storage types.
 *
 * The actual data cast will have type ShHostType<SRC> to ShHostType<DEST>
 * and may have some extra conversion code (e.g. clamping) applied
 * in addition to the default C cast for those types. 
 */
template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT>
struct ShDataVariantCast: public ShVariantCast {
  public:
    static const ShValueType DestValueType = ShStorageTypeInfo<Dest>::value_type;
    static const ShValueType SrcValueType = ShStorageTypeInfo<Src>::value_type;
    typedef typename ShDataTypeCppType<Dest, DestDT>::type D;
    typedef typename ShDataTypeCppType<Src, SrcDT>::type S;

    typedef ShDataVariant<Dest, DestDT> DestVariant;
    typedef const ShDataVariant<Src, SrcDT> SrcVariant;

    void doCast(ShVariant* dest, const ShVariant* src) const;

    void getCastTypes(ShValueType &dest, ShDataType &destDT, 
                      ShValueType &src, ShDataType &srcDT) const;

    void getDestTypes(ShValueType &valueType, ShDataType &dataType) const; 

    void doCast(D &dest, const S &src) const;

    static const ShDataVariantCast *instance();
  private:
    static ShDataVariantCast *m_instance;
    ShDataVariantCast() {}
};


}

#include "ShVariantCastImpl.hpp"

#endif
