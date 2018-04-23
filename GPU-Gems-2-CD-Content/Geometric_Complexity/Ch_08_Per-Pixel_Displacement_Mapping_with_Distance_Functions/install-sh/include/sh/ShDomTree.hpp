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

#ifndef SHDOMTREE_HPP
#define SHDOMTREE_HPP

#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include "ShDllExport.hpp"
#include "ShCtrlGraph.hpp"
#include "ShRefCount.hpp"

namespace SH {

  /** A dominator tree in a flowgraph.
   */
  class
  SH_DLLEXPORT ShDomTree {
  public:
    ShDomTree(ShCtrlGraphPtr ctrlGraph);

    /// Postorder (bottom-up) traversal.
    template<typename T> void postorder(T& f);

    /// Preorder traversal.
    template<typename T> void preorder(T& f);
  
    void debugDump();

    int numbering(ShCtrlGraphNodePtr node) const {
      return (int)(std::find( m_vertex.begin(), m_vertex.end(), node) - m_vertex.begin());
    }
  
  private:
    void dfs(ShCtrlGraphNodePtr v);
    ShCtrlGraphNodePtr eval(ShCtrlGraphNodePtr v);
    void compress(ShCtrlGraphNodePtr v);
    void link(ShCtrlGraphNodePtr v, ShCtrlGraphNodePtr w);
  
    ShCtrlGraphPtr m_graph;

    // See [TODO Ref Langauer & Tarjan pg 127] for more information on these

    typedef std::map<ShCtrlGraphNodePtr, ShCtrlGraphNodePtr> ParentMap;
    ParentMap m_parent;
    typedef std::set<ShCtrlGraphNodePtr> PredSet;
    typedef std::map<ShCtrlGraphNodePtr, PredSet> PredMap;
    PredMap m_pred;
    typedef std::map<ShCtrlGraphNodePtr, int> SemiMap;
    SemiMap m_semi;
    std::vector<ShCtrlGraphNodePtr> m_vertex;
    typedef std::set<ShCtrlGraphNodePtr> BucketSet;
    typedef std::map<ShCtrlGraphNodePtr, BucketSet> BucketMap;
    BucketMap m_bucket;
    typedef std::map<ShCtrlGraphNodePtr, ShCtrlGraphNodePtr> DomMap;
    DomMap m_dom;

    typedef std::map<ShCtrlGraphNodePtr, ShCtrlGraphNodePtr> AncestorMap;
    AncestorMap m_ancestor;
    typedef std::map<ShCtrlGraphNodePtr, ShCtrlGraphNodePtr> LabelMap;
    LabelMap m_label;

    typedef std::set<ShCtrlGraphNodePtr> ChildrenSet;
    typedef std::map<ShCtrlGraphNodePtr, ChildrenSet> ChildrenMap;
    ChildrenMap m_children;
  
    int m_n;

    template<typename T> void postorderNode(T& f, ShCtrlGraphNodePtr root, int level = 0);
    template<typename T> void preorderNode(T& f, ShCtrlGraphNodePtr root, int level = 0);
  };

  template<typename T>
  void ShDomTree::postorder(T& f)
  {
    postorderNode(f, m_graph->entry());
  }

  template<typename T>
  void ShDomTree::postorderNode(T& f, ShCtrlGraphNodePtr root, int level)
  {
    ChildrenSet& children = m_children[root];
    for (ChildrenSet::iterator I = children.begin(); I != children.end(); ++I) {
      postorderNode(f, *I, level + 1);
    }
    f(root, level);
  }

  template<typename T>
  void ShDomTree::preorder(T& f)
  {
    preorderNode(f, m_graph->entry());
  }

  template<typename T>
  void ShDomTree::preorderNode(T& f, ShCtrlGraphNodePtr root, int level)
  {
    f(root, level);
    ChildrenSet& children = m_children[root];
    for (ChildrenSet::iterator I = children.begin(); I != children.end(); ++I) {
      preorderNode(f, *I, level + 1);
    }
  }

}

#endif
