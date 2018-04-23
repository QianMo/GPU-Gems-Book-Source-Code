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
#ifndef SHVARIANTCASTIMPL_HPP
#define SHVARIANTCASTIMPL_HPP

#include "ShInternals.hpp"
#include "ShVariant.hpp"
#include "ShVariantCast.hpp"

namespace SH {

template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT> 
ShDataVariantCast<Dest, DestDT, Src, SrcDT>* 
ShDataVariantCast<Dest, DestDT, Src, SrcDT>::m_instance = 0;

template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT> 
void ShDataVariantCast<Dest, DestDT, Src, SrcDT>::doCast(
    ShVariant* dest, const ShVariant *src) const
{

  SrcVariant* sv = variant_cast<Src, SrcDT>(src);
  DestVariant* dv = variant_cast<Dest, DestDT>(dest); 

  typename SrcVariant::const_iterator S = sv->begin();
  typename DestVariant::iterator D = dv->begin();
  for(;S != sv->end(); ++S, ++D) doCast(*D, *S);
}

template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT>
void ShDataVariantCast<Dest, DestDT, Src, SrcDT>::getCastTypes(
    ShValueType &dest, ShDataType &destDT, 
    ShValueType &src, ShDataType &srcDT) const
{
  dest = DestValueType;
  destDT = DestDT;
  src = SrcValueType;
  srcDT = SrcDT;
}

template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT>
void ShDataVariantCast<Dest, DestDT, Src, SrcDT>::getDestTypes(
    ShValueType &valueType, ShDataType &dataType) const
{
  valueType = DestValueType; 
  dataType = DestDT;
}

template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT>
void ShDataVariantCast<Dest, DestDT, Src, SrcDT>::doCast(D &dest, const S &src) const
{
  shDataTypeCast<Dest, DestDT, Src, SrcDT>(dest, src);
}

template<typename Dest, ShDataType DestDT, 
  typename Src, ShDataType SrcDT>
const ShDataVariantCast<Dest, DestDT, Src, SrcDT>*
ShDataVariantCast<Dest, DestDT, Src, SrcDT>::instance()
{
  if(!m_instance) m_instance = new ShDataVariantCast();
  return m_instance;
}

}

#endif
