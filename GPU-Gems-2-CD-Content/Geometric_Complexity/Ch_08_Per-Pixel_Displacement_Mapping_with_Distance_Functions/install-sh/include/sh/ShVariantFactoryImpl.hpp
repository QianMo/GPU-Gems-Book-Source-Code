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
#ifndef SH_VARIANTFACTORYIMPL_HPP 
#define SH_VARIANTFACTORYIMPL_HPP 

#include "ShVariant.hpp"
#include "ShVariantFactory.hpp"

namespace SH {

template<typename T, ShDataType DT>
ShVariant* ShDataVariantFactory<T, DT>::generate(int N) const
{
  return new ShDataVariant<T, DT>(N);
}

template<typename T, ShDataType DT>
ShVariant* ShDataVariantFactory<T, DT>::generate(std::string s) const
{
  return new ShDataVariant<T, DT>(s);
}

template<typename T, ShDataType DT>
ShVariant* ShDataVariantFactory<T, DT>::generate(void *data, int N, bool managed) const
{
  return new ShDataVariant<T, DT>(data, N, managed);
}

template<typename T, ShDataType DT>
ShVariant* ShDataVariantFactory<T, DT>::generateZero(int N) const
{
  return new ShDataVariant<T, DT>(N);
}

template<typename T, ShDataType DT>
ShVariant* ShDataVariantFactory<T, DT>::generateOne(int N) const
{
  return new ShDataVariant<T, DT>(N, ShDataTypeConstant<T, DT>::One);
}

template<typename T, ShDataType DT>
ShDataVariantFactory<T, DT>* ShDataVariantFactory<T, DT>::m_instance = 0;

template<typename T, ShDataType DT>
const ShDataVariantFactory<T, DT>*
ShDataVariantFactory<T, DT>::instance() 
{
  if(!m_instance) m_instance = new ShDataVariantFactory<T, DT>();
  return m_instance;
}

template<typename T, ShDataType DT>
ShDataVariantFactory<T, DT>::ShDataVariantFactory()
{}

}

#endif
