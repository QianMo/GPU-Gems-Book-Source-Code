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
#ifndef SHTYPEINFOIMPL_HPP
#define SHTYPEINFOIMPL_HPP

#include "ShTypeInfo.hpp"
#include "ShVariantFactory.hpp"
#include "ShInterval.hpp"

namespace SH {

template<typename T, ShDataType DT>
const typename ShDataTypeInfo<T, DT>::type
ShDataTypeInfo<T, DT>::Zero = ShDataTypeConstant<T, DT>::Zero; 

template<typename T, ShDataType DT>
const typename ShDataTypeInfo<T, DT>::type
ShDataTypeInfo<T, DT>::One = ShDataTypeConstant<T, DT>::One; 

template<typename T, ShDataType DT>
const char* ShDataTypeInfo<T, DT>::name() const 
{
  return ShStorageTypeInfo<T>::name;
}

template<typename T, ShDataType DT>
int ShDataTypeInfo<T, DT>::datasize() const 
{
  return sizeof(typename ShDataTypeCppType<T, DT>::type); 
}

template<typename T, ShDataType DT>
const ShVariantFactory* ShDataTypeInfo<T, DT>::variantFactory() const
{
  return ShDataVariantFactory<T, DT>::instance();
}

template<typename T, ShDataType DT>
const ShDataTypeInfo<T, DT>* ShDataTypeInfo<T, DT>::instance() 
{
  if(!m_instance) m_instance = new ShDataTypeInfo<T, DT>();
  return m_instance;
}

template<typename T, ShDataType DT>
ShDataTypeInfo<T, DT>* ShDataTypeInfo<T, DT>::m_instance = 0;


}

#endif
