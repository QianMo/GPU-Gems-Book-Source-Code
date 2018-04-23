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
#ifndef SHBASETEXTURE_HPP
#define SHBASETEXTURE_HPP

#include <string>
#include "ShDllExport.hpp"
#include "ShTextureNode.hpp"
#include "ShMemory.hpp"
#include "ShVariable.hpp"
#include "ShAttrib.hpp"
#include "ShMetaForwarder.hpp"

namespace SH {

class
SH_DLLEXPORT ShBaseTexture : public ShMetaForwarder {
public:
  ShBaseTexture(const ShTextureNodePtr& node);

protected:
  ShTextureNodePtr m_node;
};

/** Base type for 1D Textures.
 */
template<typename T>
class ShBaseTexture1D : public ShBaseTexture {
public:
  ShBaseTexture1D(const ShTextureTraits& traits);
  ShBaseTexture1D(int width, const ShTextureTraits& traits);

  template<typename T2>
  T operator()(const ShGeneric<1, T2>& coords) const;

  template<typename T2>
  T operator[](const ShGeneric<1, T2>& coords) const;

  ShMemoryPtr memory();
  void memory(ShMemoryPtr memory);
  void size(int width);

  ShAttrib1f size() const;
  int width() { return m_node->width(); }

  typedef T return_type;
};

/** Base type for 2D Textures.
 */
template<typename T>
class ShBaseTexture2D : public ShBaseTexture  {
public:
  ShBaseTexture2D(const ShTextureTraits& traits);
  ShBaseTexture2D(int width, int height, const ShTextureTraits& traits);

  template<typename T2>
  T operator()(const ShGeneric<2, T2>& coords) const;

  /// Texture lookup with derivatives
  template<typename T2, typename T3, typename T4>
  T operator()(const ShGeneric<2, T2>& coords,
               const ShGeneric<2, T3>& dx,
               const ShGeneric<2, T4>& dy) const;
  
  template<typename T2>
  T operator[](const ShGeneric<2, T2>& coords) const;

  ShMemoryPtr memory();
  void memory(ShMemoryPtr memory);
  void size(int width, int height);

  ShAttrib2f size() const;

  int width() { return m_node->width(); }
  int height() { return m_node->height(); }

  typedef T return_type;
};

/** Base type for Rectangular Textures.
 */
template<typename T>
class ShBaseTextureRect : public ShBaseTexture  {
public:
  ShBaseTextureRect(const ShTextureTraits& traits);
  ShBaseTextureRect(int width, int height, const ShTextureTraits& traits);

  template<typename T2>
  T operator()(const ShGeneric<2, T2>& coords) const;

  template<typename T2>
  T operator[](const ShGeneric<2, T2>& coords) const;

  ShMemoryPtr memory();
  void memory(ShMemoryPtr memory);
  void size(int width, int height);

  ShAttrib2f size() const;

  int width() { return m_node->width(); }
  int height() { return m_node->height(); }

  typedef T return_type;
};

/** Base type for 3D Textures.
 */
template<typename T>
class ShBaseTexture3D : public ShBaseTexture  {
public:
  ShBaseTexture3D(const ShTextureTraits& traits);
  ShBaseTexture3D(int width, int height, int depth, const ShTextureTraits& traits);

  template<typename T2>
  T operator()(const ShGeneric<3, T2>& coords) const;

  template<typename T2>
  T operator[](const ShGeneric<3, T2>& coords) const;

  ShMemoryPtr memory();
  void memory(ShMemoryPtr memory);
  void size(int width, int height, int depth);

  ShAttrib3f size() const;
  int width() { return m_node->width(); }
  int height() { return m_node->height(); }
  int depth() { return m_node->depth(); }

  typedef T return_type;
};

/** Base type for Cube Textures.
 */
template<typename T>
class ShBaseTextureCube : public ShBaseTexture {
public:
  ShBaseTextureCube(const ShTextureTraits& traits);
  ShBaseTextureCube(int width, int height, const ShTextureTraits& traits);

  template<typename T2>
  T operator()(const ShGeneric<3, T2>& coords) const;

  ShMemoryPtr memory(ShCubeDirection face);
  void memory(ShMemoryPtr memory, ShCubeDirection face);
  void size(int width, int height);

  ShAttrib2f size() const;

  int width() { return m_node->width(); }
  int height() { return m_node->height(); }

  typedef T return_type;
};

}

#include "ShBaseTextureImpl.hpp"

#endif
