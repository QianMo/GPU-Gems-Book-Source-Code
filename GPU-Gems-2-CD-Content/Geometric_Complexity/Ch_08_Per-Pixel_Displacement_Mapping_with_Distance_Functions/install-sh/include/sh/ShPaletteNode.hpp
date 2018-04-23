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
#ifndef SHPALETTENODE_HPP
#define SHPALETTENODE_HPP

#include <cstddef>
#include "ShVariableNode.hpp"

namespace SH {

/** @internal Palette Node Representation
 *
 * Represents a palette (i.e. a uniform array of variables)
 * internally.
 *
 * To use palettes, refer to the ShPalette class instead.
 *
 * @see ShPalette
 */
class SH_DLLEXPORT ShPaletteNode : public ShVariableNode {
public:
  ShPaletteNode(int elements, ShSemanticType semantic, ShValueType valueType, std::size_t length);

  /// Set the VariableNode corresponding to the given index. Only ShPalette should call this.
  void set_node(std::size_t index, const ShVariableNodePtr& node);

  /// Return the number of variables represented by this palette.
  std::size_t palette_length() const;

  /// Return one of the variables represented by this palette.
  ShVariableNodeCPtr get_node(std::size_t index) const;

  /// Return one of the variables represented by this palette.
  ShVariableNodePtr get_node(std::size_t index);

private:
  std::size_t m_length;
  ShVariableNodePtr* m_nodes;
};

typedef ShPointer<ShPaletteNode> ShPaletteNodePtr;
typedef ShPointer<const ShPaletteNode> ShPaletteNodeCPtr;

}

#endif
