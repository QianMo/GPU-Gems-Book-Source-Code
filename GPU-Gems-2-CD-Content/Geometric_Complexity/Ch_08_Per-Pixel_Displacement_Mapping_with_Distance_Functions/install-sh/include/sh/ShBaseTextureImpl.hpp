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
#ifndef SHBASETEXTUREIMPL_HPP
#define SHBASETEXTUREIMPL_HPP

#include "ShBaseTexture.hpp"
#include "ShContext.hpp"
#include "ShError.hpp"
#include "ShLibMisc.hpp"

namespace SH {

template<typename T>
ShBaseTexture1D<T>::ShBaseTexture1D(int width, const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_1D, T::typesize, T::value_type, traits, width))
{
}

template<typename T>
ShBaseTexture2D<T>::ShBaseTexture2D(int width, int height, const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_2D, T::typesize, T::value_type, traits, width, height))
{
}

template<typename T>
ShBaseTextureRect<T>::ShBaseTextureRect(int width, int height, const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_RECT, T::typesize, T::value_type, traits, width, height))
{
}

template<typename T>
ShBaseTexture3D<T>::ShBaseTexture3D(int width, int height, int depth, const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_3D, T::typesize, T::value_type, traits, width, height, depth))
{
}

template<typename T>
ShBaseTextureCube<T>::ShBaseTextureCube(int width, int height, const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_CUBE, T::typesize, T::value_type, traits, width, height))
{
}

template<typename T>
ShBaseTexture1D<T>::ShBaseTexture1D(const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_1D, T::typesize, T::value_type, traits, 0))
{
}

template<typename T>
ShBaseTexture2D<T>::ShBaseTexture2D(const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_2D, T::typesize, T::value_type, traits, 0, 0))
{
}

template<typename T>
ShBaseTextureRect<T>::ShBaseTextureRect(const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_RECT, T::typesize, T::value_type, traits, 0, 0))
{
}

template<typename T>
ShBaseTexture3D<T>::ShBaseTexture3D(const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_3D, T::typesize, T::value_type, traits, 0, 0, 0))
{
}

template<typename T>
ShBaseTextureCube<T>::ShBaseTextureCube(const ShTextureTraits& traits)
  : ShBaseTexture(new ShTextureNode(SH_TEXTURE_CUBE, T::typesize, T::value_type, traits, 0, 0))
{
}

template<typename T>
template<typename T2>
T ShBaseTexture1D<T>::operator()(const ShGeneric<1, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEX, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

template<typename T>
template<typename T2>
T ShBaseTexture2D<T>::operator()(const ShGeneric<2, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEX, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

template<typename T>
template<typename T2, typename T3, typename T4>
T ShBaseTexture2D<T>::operator()(const ShGeneric<2, T2>& coords,
                                 const ShGeneric<2, T3>& dx,
                                 const ShGeneric<2, T4>& dy) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, SH_OP_TEXD, texVar, coords, join(dx, dy));
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    shError(ShScopeException("Cannot do derivative texture lookup in immediate mode"));
    T t; return t;
  }
}

template<typename T>
template<typename T2>
T ShBaseTextureRect<T>::operator()(const ShGeneric<2, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEX, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

template<typename T>
template<typename T2>
T ShBaseTexture3D<T>::operator()(const ShGeneric<3, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEX, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
} 

template<typename T>
template<typename T2>
T ShBaseTextureCube<T>::operator()(const ShGeneric<3, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEX, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
} 

template<typename T>
template<typename T2>
T ShBaseTexture1D<T>::operator[](const ShGeneric<1, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEXI, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

template<typename T>
template<typename T2>
T ShBaseTexture2D<T>::operator[](const ShGeneric<2, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEXI, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

template<typename T>
template<typename T2>
T ShBaseTextureRect<T>::operator[](const ShGeneric<2, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEXI, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

template<typename T>
template<typename T2>
T ShBaseTexture3D<T>::operator[](const ShGeneric<3, T2>& coords) const
{
  if (ShContext::current()->parsing()) {
    T t;
    ShVariable texVar(m_node);
    ShStatement stmt(t, texVar, SH_OP_TEXI, coords);
    ShContext::current()->parsing()->tokenizer.blockList()->addStatement(stmt);
    return t;
  } else {
    // TODO!
    T t;
    return t;
  }
}

// setMemory

template<typename T>
void ShBaseTexture1D<T>::memory(ShMemoryPtr memory)
{
  m_node->memory(memory);
}

template<typename T>
void ShBaseTexture2D<T>::memory(ShMemoryPtr memory)
{
  m_node->memory(memory);
}

template<typename T>
void ShBaseTextureRect<T>::memory(ShMemoryPtr memory)
{
  m_node->memory(memory);
}

template<typename T>
void ShBaseTexture3D<T>::memory(ShMemoryPtr memory)
{
  m_node->memory(memory);
}

template<typename T>
void ShBaseTextureCube<T>::memory(ShMemoryPtr memory,
                                  ShCubeDirection face)
{
  m_node->memory(memory, face);
}

// get memory

template<typename T>
ShMemoryPtr ShBaseTexture1D<T>::memory()
{
  return m_node->memory();
}

template<typename T>
ShMemoryPtr ShBaseTexture2D<T>::memory()
{
  return m_node->memory();
}

template<typename T>
ShMemoryPtr ShBaseTextureRect<T>::memory()
{
  return m_node->memory();
}

template<typename T>
ShMemoryPtr ShBaseTexture3D<T>::memory()
{
  return m_node->memory();
}

template<typename T>
ShMemoryPtr ShBaseTextureCube<T>::memory(ShCubeDirection face)
{
  return m_node->memory(face);
}

// get size

template<typename T>
ShAttrib1f ShBaseTexture1D<T>::size() const
{
  ShAttrib1f t(m_node->texSizeVar().node(), ShSwizzle(m_node->texSizeVar().size()), false);
  return t;
}

template<typename T>
ShAttrib2f ShBaseTexture2D<T>::size() const
{
  ShAttrib2f t(m_node->texSizeVar().node(), ShSwizzle(m_node->texSizeVar().size()), false);
  return t;
}

template<typename T>
ShAttrib2f ShBaseTextureRect<T>::size() const
{
  ShAttrib2f t(m_node->texSizeVar().node(), ShSwizzle(m_node->texSizeVar().size()), false);
  return t;
}

template<typename T>
ShAttrib3f ShBaseTexture3D<T>::size() const
{
  ShAttrib3f t(m_node->texSizeVar().node(), ShSwizzle(m_node->texSizeVar().size()), false);
  return t;
}

template<typename T>
ShAttrib2f ShBaseTextureCube<T>::size() const
{
  ShAttrib2f t(m_node->texSizeVar().node(), ShSwizzle(m_node->texSizeVar().size()), false);
  return t;
}

// set size

template<typename T>
void ShBaseTexture1D<T>::size(int width)
{
  m_node->setTexSize(width);
}

template<typename T>
void ShBaseTexture2D<T>::size(int width, int height)
{
  m_node->setTexSize(width, height);
}

template<typename T>
void ShBaseTextureRect<T>::size(int width, int height)
{
  m_node->setTexSize(width, height);
}

template<typename T>
void ShBaseTexture3D<T>::size(int width, int height, int depth)
{
  m_node->setTexSize(width, height, depth);
}

template<typename T>
void ShBaseTextureCube<T>::size(int width, int height)
{
  m_node->setTexSize(width, height);
}

}

#endif
