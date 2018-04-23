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
#ifndef SHSTRUCTURAL_HPP
#define SHSTRUCTURAL_HPP

#include <list>
#include <utility>
#include <iosfwd>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShCtrlGraph.hpp"
#include "ShVariable.hpp"

namespace SH {

class
SH_DLLEXPORT ShStructuralNode : public ShRefCountable {
public:
  friend class ShStructural;
  
  enum NodeType {
    UNREDUCED,
    BLOCK,
    IF,
    IFELSE,
    SELFLOOP,
    WHILELOOP
  };

  ShStructuralNode(const ShCtrlGraphNodePtr& node);
  ShStructuralNode(NodeType type);

  // Graphviz-format dump of this node and its children
  std::ostream& dump(std::ostream& out, int nodes = -1) const;

  // Structural information
  NodeType type;
  ShStructuralNode* container;
  typedef std::list< ShPointer< ShStructuralNode> > StructNodeList;
  StructNodeList structnodes; ///< Nodes in this region
  
  // Graph structure
  ShCtrlGraphNodePtr cfg_node;
  typedef std::pair<ShVariable, ShPointer<ShStructuralNode> > SuccessorEdge;
  typedef std::list<SuccessorEdge> SuccessorList;
  SuccessorList succs;
  typedef std::pair<ShVariable, ShStructuralNode*> PredecessorEdge;
  typedef std::list<PredecessorEdge> PredecessorList;
  PredecessorList preds;

  // Spanning tree
  ShStructuralNode* parent;
  typedef std::list< ShPointer<ShStructuralNode> > ChildList;
  ChildList children;

};

typedef ShPointer<ShStructuralNode> ShStructuralNodePtr;
typedef ShPointer<const ShStructuralNode> ShStructuralNodeCPtr;

class
SH_DLLEXPORT ShStructural {
public:
  ShStructural(const ShCtrlGraphPtr& graph);

  // Graphviz-format dump of the structural tree.
  std::ostream& dump(std::ostream& out) const;

  const ShStructuralNodePtr& head();
  
private:
  ShCtrlGraphPtr m_graph;
  ShStructuralNodePtr m_head;

  typedef std::list<ShStructuralNode*> PostorderList;
  PostorderList m_postorder;

  ShStructuralNodePtr build_tree(const ShCtrlGraphNodePtr& node);
  void build_postorder(const ShStructuralNodePtr& node);
};

}

#endif
