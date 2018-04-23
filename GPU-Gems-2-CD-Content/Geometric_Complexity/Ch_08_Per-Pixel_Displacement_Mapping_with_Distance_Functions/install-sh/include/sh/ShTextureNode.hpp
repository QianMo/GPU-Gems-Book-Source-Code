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
#ifndef SHTEXTURENODE_HPP
#define SHTEXTURENODE_HPP

#include "ShDllExport.hpp"
#include "ShVariableNode.hpp"
#include "ShMemory.hpp"
#include "ShRefCount.hpp"
#include "ShVariable.hpp"

namespace SH {

/** Texture formats.
 * An enumeration of the various ways textures can be laid out.
 */
enum ShTextureDims {
  SH_TEXTURE_1D,   // Power of two
  SH_TEXTURE_2D,   // Power of two
  SH_TEXTURE_RECT, // Non power of two
  SH_TEXTURE_3D,   // Power of two, but depth may not be
  SH_TEXTURE_CUBE, // 6 "2D" memory objects, power of two
};

/** Cube map faces.
 * An enumeration of names for the various faces of a cube map.
 */
enum ShCubeDirection {
  SH_CUBE_POS_X = 0,
  SH_CUBE_NEG_X = 1,
  SH_CUBE_POS_Y = 2,
  SH_CUBE_NEG_Y = 3,
  SH_CUBE_POS_Z = 4,
  SH_CUBE_NEG_Z = 5,
};

/** Texture traits.
 * An enumeration of the various wrapping an clamping modes supported
 * by textures.
 */
class 
SH_DLLEXPORT ShTextureTraits {
public:
  enum Filtering {
    SH_FILTER_NONE,
    SH_FILTER_MIPMAP
  };
  
  enum Wrapping {
    SH_WRAP_CLAMP,
    SH_WRAP_CLAMP_TO_EDGE,
    SH_WRAP_REPEAT
  };
  enum Clamping {
    SH_CLAMPED,
    SH_UNCLAMPED
  };

  ShTextureTraits(unsigned int interpolation,
                  Filtering filtering,
                  Wrapping wrapping,
                  Clamping clamping)
    : m_interpolation(interpolation),
      m_filtering(filtering),
      m_wrapping(wrapping),
      m_clamping(clamping)
  {
  }

  bool operator==(const ShTextureTraits& other) const
  {
    return m_interpolation == other.m_interpolation
      && m_filtering == other.m_filtering
      && m_wrapping == other.m_wrapping
      && m_clamping == other.m_clamping;
  }

  bool operator!=(const ShTextureTraits& other) const { return !(*this == other); }
  
  unsigned int interpolation() const { return m_interpolation; }
  ShTextureTraits& interpolation(unsigned int interp) { m_interpolation = interp; return *this; }
  
  Filtering filtering() const { return m_filtering; }
  ShTextureTraits& filtering(Filtering filtering) { m_filtering = filtering; return *this; }
  
  Wrapping wrapping() const { return m_wrapping; }
  ShTextureTraits& wrapping(Wrapping wrapping) { m_wrapping = wrapping; return *this; }
  
  Clamping clamping() const { return m_clamping; }
  ShTextureTraits& clamping(Clamping clamping) { m_clamping = clamping; return *this; }

private:
  unsigned int m_interpolation;
  Filtering m_filtering;
  Wrapping m_wrapping;
  Clamping m_clamping;
};

class 
SH_DLLEXPORT ShTextureNode : public ShVariableNode {
public:
  ShTextureNode(ShTextureDims dims,
                int size, // scalars per tuple 
                ShValueType valueType, // type index 
                const ShTextureTraits&,
                int width, int height = 1, int depth = 1);
  virtual ~ShTextureNode();

  ShTextureDims dims() const;

  // Memory
  ShPointer<const ShMemory> memory(int n = 0) const;
  ShPointer<const ShMemory> memory(ShCubeDirection dir) const;
  ShMemoryPtr memory(int n = 0);
  ShMemoryPtr memory(ShCubeDirection dir);
  void memory(ShMemoryPtr memory, int n = 0);
  void memory(ShMemoryPtr memory, ShCubeDirection dir);

  // Basic properties - not all may be valid for all types
  const ShTextureTraits& traits() const; // valid for all texture nodes
  ShTextureTraits& traits(); // valid for all texture nodes
  int width() const; // valid for all texture nodes
  int height() const; // 1 for SH_TEXTURE_1D
  int depth() const; // 1 unless SH_TEXTURE_3D
  int count() const; // number of elements  

  void setTexSize(int w);
  void setTexSize(int w, int h);
  void setTexSize(int w, int h, int d);
  const ShVariable& texSizeVar() const;
  
private:
  ShTextureDims m_dims;
  
  ShMemoryPtr* m_memory; // array of either 1 or 6 (for cubemaps)
  
  ShTextureTraits m_traits;
  int m_width, m_height, m_depth;

  ShVariable m_texSizeVar;
  
  // NOT IMPLEMENTED
  ShTextureNode(const ShTextureNode& other);
  ShTextureNode& operator=(const ShTextureNode& other);
};

typedef ShPointer<ShTextureNode> ShTextureNodePtr;
typedef ShPointer<const ShTextureNode> ShTextureNodeCPtr;

}
#endif
