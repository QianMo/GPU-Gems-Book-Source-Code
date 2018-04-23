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
#ifndef SHBASICBLOCK_HPP
#define SHBASICBLOCK_HPP

#include <list>
#include "ShBlock.hpp"
#include "ShStatement.hpp"

namespace SH {

/** A basic block.
 * A basic block is a block composed of simpler statements, but with
 * no loops or conditionals.   It may only contain arithmetic 
 * operations, texture lookups, and assignments (although some of 
 * these assignments might be conditional).
 */
class 
SH_DLLEXPORT
ShBasicBlock : public ShBlock {
public:
  ~ShBasicBlock();

  void print(std::ostream& out, int indent) const;
  void graphvizDump(std::ostream& out) const;
  
  /**@name Add statement at start.
   * Adds the given statement after the statements in this block */
  void addStatement(const ShStatement& stmt);

  /**@name Add statement at end.
   * Adds the given statement before the statements in this block */
  void prependStatement(const ShStatement& stmt);

  typedef std::list<ShStatement> ShStmtList;

  ShStmtList::const_iterator begin() const;
  ShStmtList::const_iterator end() const;
  ShStmtList::iterator begin();
  ShStmtList::iterator end();

  ShStmtList::iterator erase(ShStmtList::iterator I) {
    return m_statements.erase(I);
  }

  // Place all the elements of l before the iterator I and removes them
  // from l
  void splice(ShStmtList::iterator I, ShStmtList &l) {
    m_statements.splice(I, l);
  }

  // Places all the elements starting from lI in l before the iterator I and
  // removes them from l
  void splice(ShStmtList::iterator I, ShStmtList &l, ShStmtList::iterator lI) {
    m_statements.splice(I, l, lI);
  }
  
  ShStmtList m_statements;
//private:
};

typedef ShPointer<ShBasicBlock> ShBasicBlockPtr;
typedef ShPointer<const ShBasicBlock> ShBasicBlockCPtr;

}

#endif
