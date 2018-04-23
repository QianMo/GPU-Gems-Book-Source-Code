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
#ifndef SHPALETTEIMPL_HPP
#define SHPALETTEIMPL_HPP

namespace SH {

template<typename T>
ShPalette<T>::ShPalette(std::size_t size)
  : m_node(new ShPaletteNode(T::typesize, T::semantic_type, T::value_type, size)),
    m_data(new T[size])
{
  for (std::size_t i = 0; i < size; i++) {
    m_node->set_node(i, m_data[i].node());
  }
}

template<typename T>
ShPalette<T>::~ShPalette()
{
  delete [] m_data;
}

template<typename T>
const T& ShPalette<T>::operator[](std::size_t index) const
{
  return m_data[index];
}

template<typename T>
T& ShPalette<T>::operator[](std::size_t index)
{
  return m_data[index];
}

template<typename T>
template<typename T2>
T ShPalette<T>::operator[](const ShGeneric<1, T2>& index) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable palVar(m_node);
    ShStatement stmt(t, palVar, SH_OP_PAL, index);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    return m_data[(std::size_t)index.getValue(0)];
  }
}

}

#endif
