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
#ifndef SHBLOCK_HPP
#define SHBLOCK_HPP

#include <vector>
#include <iosfwd>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShStatement.hpp"

namespace SH {

/** A generic block or token.
 * These are either basic blocks or tokens.
 */
class
SH_DLLEXPORT ShBlock : public ShRefCountable {
public:
  virtual ~ShBlock();

  /// Output a textual representation of this control statement.
  virtual void print(std::ostream& out, int indent) const = 0;
};

typedef ShPointer<ShBlock> ShBlockPtr;
typedef ShPointer<const ShBlock> ShBlockCPtr;

/** A list of generic blocks.
 */
class
SH_DLLEXPORT ShBlockList : public ShRefCountable {
public:
  ShBlockList(bool isArgument = false);

  /// True iff this block list is an argument (e.g. to sh_IF)
  bool isArgument();

  /// Add a simple "three variable" statement.
  void addStatement(const ShStatement& statement);
  /// Add a generic block.
  void addBlock(const ShBlockPtr& statement);

  /// Return the front block from the list (does not remove it)
  ShBlockPtr ShBlockList::getFront() const;
  
  /// Remove the front block from the list and return it
  ShBlockPtr ShBlockList::removeFront();

  /// Return true iff this list does not contain any blocks.
  bool empty() const;

  /// Output a token list
  friend SH_DLLEXPORT std::ostream& operator<<(std::ostream& out, const ShBlockList& blockList);

  /// Output a token list at a given indentation
  std::ostream& print(std::ostream& out, int indentation) const;

private:
  bool m_isArgument;
  std::vector<ShBlockPtr> m_blocks;
};

typedef ShPointer<ShBlockList> ShBlockListPtr;
typedef ShPointer<const ShBlockList> ShBlockListCPtr;

}

#endif
