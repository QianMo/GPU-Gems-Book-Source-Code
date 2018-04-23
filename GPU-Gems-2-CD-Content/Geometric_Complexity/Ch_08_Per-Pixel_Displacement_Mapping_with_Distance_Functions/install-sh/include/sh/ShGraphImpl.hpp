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
#ifndef SHGRAPHIMPL_HPP 
#define SHGRAPHIMPL_HPP 

#include "ShGraph.hpp"
#include <stack>
#include <map>

namespace SH {

// TODO should clear arguments passed in by reference to the different
// algs like shortest path and trans closuer


template<typename G>
ShGraphVertex<G>::ShGraphVertex()
  : marked(false)
{}

template<typename G>
ShGraphVertex<G>::ShGraphVertex(const ShGraphVertex<G> &other)
  : marked(other.marked)
{}

template<typename G>
std::ostream& ShGraphVertex<G>::graphvizDump(std::ostream &out) const
{
  out << " [label=\"\", shape=circle, height=0.25]";
  return out;
}

template<typename G>
ShGraphEdge<G>::ShGraphEdge()
  : start(0), end(0)
{}

template<typename G>
ShGraphEdge<G>::ShGraphEdge(typename G::Vertex *start, typename G::Vertex *end)
  : start(start), end(end)
{}

template<typename G>
ShGraphEdge<G>::ShGraphEdge(const ShGraphEdge<G> &other)
  : start(0), end(0) 
{}

template<typename G>
std::ostream& ShGraphEdge<G>::graphvizDump(std::ostream &out) const
{
  return out; // use default edge
}

template<typename G>
ShGraph<G>::ShGraph()
{}

template<typename G>
ShGraph<G>::ShGraph(const ShGraph<G> &other)
{
  *this = other;
}

template<typename G>
ShGraph<G>::~ShGraph()
{
  clear();
}

template<typename G>
void ShGraph<G>::addVertex(typename G::Vertex *v)
{
  verts.insert(v);
}

template<typename G>
void ShGraph<G>::addEdge(typename G::Edge *e)
{
  edges.insert(e);
  e->start->edges.push_back(e);
}

template<typename G>
void ShGraph<G>::removeVertex(typename G::Vertex *v)
{
  for(typename EdgeList::iterator E = v->edges.begin(); E != v->edges.end(); ++E) {
    edges.erase(*E);
  }
  verts.erase(v);
}

template<typename G>
void ShGraph<G>::removeEdge(typename G::Edge *e)
{
  edges.erase(e);
  e->start->edges.remove(e);
  delete e;
}

template<typename G>
void ShGraph<G>::clear()
{
  for(typename VertexSet::iterator V = verts.begin(); V != verts.end(); ++V) delete *V;
  for(typename EdgeSet::iterator E = edges.begin(); E != edges.end(); ++E) delete *E;
 
  verts.clear();
  edges.clear();
}

template<typename G>
void ShGraph<G>::clearMarked()
{
  for(typename VertexSet::iterator V = verts.begin(); V != verts.end(); ++V) (*V)->marked = false;
}

template<typename G>
ShGraph<G>& ShGraph<G>::operator=(const ShGraph<G> &other)
{
  clear();

  VertexMap<Vertex *> vmap;

  // mappings for null pointers
  vmap[0] = 0;

  for(typename VertexSet::const_iterator V = other.verts.begin(); V != other.verts.end(); ++V) {
    typename G::Vertex *newv = new Vertex(**V);
    vmap[*V] = newv;
    addVertex(newv);
  }

  for(typename EdgeSet::const_iterator E = other.edges.begin(); E != other.edges.end(); ++E) {
    typename G::Edge *newe = new Edge(**E);
    newe->start = vmap[(*E)->start];
    newe->end = vmap[(*E)->end];
    addEdge(newe);
  }

  return *this;
}

template<typename G>
template<typename F>
void ShGraph<G>::dfs(typename G::Vertex *start, F &functor)
{
  std::stack<typename G::Vertex *> worklist;

  for(worklist.push(start); !worklist.empty(); worklist.pop()) {
    typename G::Vertex *workVertex = worklist.top();
    if(workVertex->marked) continue; // useful for cycle detection
    workVertex->marked = true;

    for(typename EdgeList::iterator E = workVertex->edges.begin(); E != workVertex->edges.end(); ++E) {
      worklist.push(E->end);
    }
    functor(workVertex);
  }
}

template<typename G>
template<typename W>
typename W::WeightType ShGraph<G>::bellmanFord(typename G::Vertex *start, typename G::Vertex *end, W &weigher, EdgeList *path) 
{
  // essentially the same as above, with predecessors
  // TODO refactor so the algorithm shows up only once
  std::map<typename G::Vertex *, typename W::WeightType> dist;
  std::map<typename G::Vertex *, typename G::Edge *> pred; // edge going back to predecessor in SP so far

  for(typename VertexSet::const_iterator V = verts.begin(); V != verts.end(); ++V) {
    dist[*V] = W::LARGE;
    pred[*V] = 0;
  }

  dist[start] = W::ZERO; 

  for(int i = 0; i < verts.size(); ++i) {
    for(typename EdgeSet::iterator E = edges.begin(); E != edges.end(); ++E) {
      Edge &e = **E;

      typename W::WeightType testDist = dist[e.start] + weigher(e);
      if(testDist < dist[e.end]) {
        dist[e.end] = testDist;
        pred[e.end] = &e;
      }
    }
  }

  // trace shortest path
  if(path) {
    path->clear();
    for(typename G::Edge *finger = pred[end]; finger != 0; finger = pred[finger->start]) {
      path->push_front(finger);
    }
  }

  return dist[end];
}

template<typename G>
template<typename W>
void ShGraph<G>::floydWarshall(W &weigher, VertexPairMap<typename W::WeightType> &dist, FirstStepMap *path) 
{
  typename VertexSet::const_iterator V, U, X; 
  typename EdgeSet::iterator E;

  dist.clear();
  if(path) path->clear();

  // initialize distances to LARGE for distinct pairs
  for(V = verts.begin(); V != verts.end(); ++V) for(U = verts.begin(); U != verts.end(); ++U) {
    if(V == U) dist(*V, *U) = W::ZERO; 
    else dist(*V, *U) = W::LARGE;
  }

  // initialize to minimum edge weight for adjacent vertices
  for(E = edges.begin(); E != edges.end(); ++E) {
    Edge& e = **E;
    typename W::WeightType &testDist = dist(e.start, e.end);
    if(testDist > weigher(e)) {
      testDist = weigher(e);
      if(path) (*path)(e.start, e.end) = &e;
    }
  }

  // do the bottom-up dynamic programming
  // loop invariant of the outer loop
  //    dist(A, B) contains the shortest path from A to B using only the
  //        nodes in verts.begin() to V as intermediates
  //    path(A, B) contains the first edge in the above shortest path
  for(V = verts.begin(); V != verts.end(); ++V) {
    for(U = verts.begin(); U != verts.end(); ++U) {
      for(X = verts.begin(); X != verts.end(); ++X) {
        typename W::WeightType &oldDist = dist(*U, *X);
        typename W::WeightType testDist = dist(*U, *V) + dist(*V, *X);
        if(oldDist > testDist && (dist(*U, *V) != W::LARGE && 
              dist(*V, *X) != W::LARGE)) {
          oldDist = testDist;
          if(path) (*path)(*U, *X) = (*path)(*U, *V);
        }
      }
    }
  }
}


template<typename G>
void ShGraph<G>::transitiveClosure(TransitiveClosureMap &tcm)
{
  typename VertexSet::const_iterator V, U, X; 
  typename EdgeSet::iterator E;
  typedef VertexPairMap<bool> TCMap; // transitive closure map

  // initialize to minimum edge weight for adjacent vertices
  tcm.clear();
  for(V = verts.begin(); V != verts.end(); ++V) tcm(*V, *V) = true;

  for(E = edges.begin(); E != edges.end(); ++E) {
    Edge& e = **E;
    tcm(e.start, e.end) = true;
  }

  // do bottom-up dynamic programming
  for(V = verts.begin(); V != verts.end(); ++V) {
    for(U = verts.begin(); U != verts.end(); ++U) {
      for(X = verts.begin(); X != verts.end(); ++X) {
        tcm(*U, *X) = tcm(*U, *X) || (tcm(*U, *V) && tcm(*V, *X));
      }
    }
  }
}

template<typename G>
void ShGraph<G>::rootSet(VertexSet &roots) {
  roots = verts;
  typename EdgeSet::const_iterator E;
  for(E = edges.begin(); E != edges.end(); ++E) {
    roots.erase((*E)->end);
  }
}

template<typename G>
struct NegativeWeigher {
  typedef int WeightType;
  static const int LARGE=2000000000; // TODO MAX_INT
  static const int ZERO=0;
  int operator()(const typename G::Edge &e) { return -1; }
}; 

template<typename G>
void ShGraph<G>::vertexHeight(const VertexSet &roots, HeightMap &heights) {
  heights.clear();
  VertexPairMap<int> dist;
  NegativeWeigher<G> weigher;
  typename VertexSet::const_iterator U, V;

  floydWarshall(weigher, dist);
  for(U = verts.begin(); U != verts.end(); ++U) {
    for(V = roots.begin(); V != roots.end(); ++V) {
      heights[*U] = std::max(heights[*U], -dist(*V, *U));
    }
  }
}


template<typename G>
void ShGraph<G>::leastCommonAncestor(LCAMap &ancestor)
{
  typename VertexSet::const_iterator U, V, W;
  typename EdgeSet::const_iterator E;

  // First step, identify the "roots" of the DAG (those that have no
  // ancestors) 
  VertexSet roots;
  rootSet(roots);

  // Find the height of each node by longest path distance from any root
  HeightMap heights;
  vertexHeight(roots, heights);

  // Calculate the transitive closure matrix 
  TransitiveClosureMap tcm;
  transitiveClosure(tcm);

  // Identify a candidate least common ancestor (greatest height)
  // for each pair of verts 
  ancestor.clear();
  for(U = verts.begin(); U != verts.end(); ++U) {
    for(V = verts.begin(); V != verts.end(); ++V) {
      int maxHeight = -1;
      for(W = verts.begin(); W != verts.end(); ++W) {
        // if W is an ancestor of both U and V, check if its the LCA so far 
        if(tcm(*W, *U) && tcm(*W, *V) && heights[*W] > maxHeight) {
          ancestor(*U, *V) = ancestor(*V, *U) = *W;
          maxHeight = heights[*W]; 
        }
      }
    }
  }
}

template<typename G>
std::ostream& ShGraphDefaultDumper<G>::operator()(std::ostream& out, const typename G::Vertex *v)
{
  return v->graphvizDump(out);
}

template<typename G>
std::ostream& ShGraphDefaultDumper<G>::operator()(std::ostream& out, const typename G::Edge *e)
{
  return e->graphvizDump(out);
}

template<typename G>
std::ostream& graphvizDump(std::ostream &out, const ShGraph<G> &g)
{
  ShGraphDefaultDumper<G> dumper;
  return graphvizDump(out, g, dumper); 
}

template<typename G, typename D>
std::ostream& graphvizDump(std::ostream &out, const ShGraph<G> &g, D &dumpFunctor)
{
  out << "digraph {" << std::endl;
    typename ShGraph<G>::VertexSet::const_iterator V = g.verts.begin();
    for(; V != g.verts.end(); ++V) {
      out << "\"" << *V << "\" ";
      dumpFunctor(out, *V);
      out << ";" << std::endl;
      out << std::endl;
    }

    typename ShGraph<G>::EdgeSet::const_iterator E = g.edges.begin();
    for(; E != g.edges.end(); ++E) {
      const typename G::Edge &e = **E;
      out << "\"" << e.start << "\" ";
      out << "-> ";
      out << "\"" << e.end << "\" ";
      dumpFunctor(out, *E);
      out << ";" << std::endl;
      out << std::endl;
    }
  out << "}" << std::endl;

  return out;
}

}

#endif
