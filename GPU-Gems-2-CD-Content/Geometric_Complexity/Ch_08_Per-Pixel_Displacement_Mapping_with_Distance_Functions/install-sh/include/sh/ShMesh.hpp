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
#ifndef SHUTIL_SHMESH_HPP
#define SHUTIL_SHMESH_HPP

#include <list>
#include <map>
#include <set>

#include "sh.hpp"

namespace ShUtil {

/** \file ShMesh.hpp
 * A basic mesh structure based on half-edges.
 *
 * To use this class, define CopyConstructible 
 * vertex, face, and edge classes to hold your vertex/face/edge specific data
 * that are subclasses of ShMeshVertex, ShMeshFace, and ShMeshEdge
 * respectively.
 *
 * The half-edge ShMesh class always keeps the following invariants: 
 * * Edge pointers: 
 *   For any edge e,
 *   a) if e.next, then e.next->prev = e
 *   b) if e.next, then e.next->start == e->end
 *   c) if e.prev, then e.prev->next = e
 *   d) if e.prev, then e.prev->end == e->start
 *   e) if e.sym != 0, then e.sym->sym = e
 *
 * * Vertex edge:
 *   For any vertex v, v.edge.start = v 
 *
 * * Face edge:
 *   For any face f, f.edge.face = f, and
 *   f.edge->next->next...  = f.edge after following enough next pointers.
 *
 * All the public ShMesh class functions maintain these invariants.
 *
 * Null Pointers:
 * For any edge e, e.start and e.end are always != 0.
 * For any face f, f.edge is always != 0.
 * All other pointers can be 0. 
 */

template<typename VertexType, typename FaceType, typename EdgeType>
struct ShMeshType {
  typedef VertexType Vertex;
  typedef FaceType Face;
  typedef EdgeType Edge;
};

template<typename M>
struct ShMeshVertex {
  typedef typename M::Edge Edge;
  Edge *edge; //< Edge that starts at this vertex 

  /** \brief Constructor that sets edge to 0 */ 
  ShMeshVertex();

  /** \brief Constructor that sets edge to 0 */ 
  ShMeshVertex(const ShMeshVertex<M> &other);
};

template<typename M>
struct ShMeshFace {
  typedef typename M::Edge Edge;
  Edge *edge; //< Edge in this face 

  /** \brief Constructor that sets edge to 0 */ 
  ShMeshFace();

  /** \brief Constructor that sets edge to 0 */ 
  ShMeshFace(const ShMeshFace<M> &other);
};

// A half-edge going from start to end that is part of face.
template<typename M>
struct ShMeshEdge {
  typedef typename M::Vertex Vertex;
  typedef typename M::Face Face;
  typedef typename M::Edge Edge;
  Vertex *start; //< Start vertex 
  Vertex *end;  //< End vertex
  Face *face; //< Face  
  Edge *sym; //< Edge paired with this edge.
  Edge *next; //< Next edge in the face 
  Edge *prev; //< Previous edge in the face

  /** \brief Constructs a edge with all pointers = 0 */ 
  ShMeshEdge();

  /** \brief Constructor that sets all pointers = 0 */ 
  ShMeshEdge(const ShMeshEdge<M> &other);

  /** \brief Constructs a edge with pointers to the given objects 
   * (any may be 0 except start and end) */
  void setLinks(Vertex *s, Vertex *e, Face *f, 
      Edge *next, Edge *prev, Edge *sym);

  /** \brief Sets next, updating next->prev if next is non-null. */ 
  void setNext(Edge *n);

  /** \brief Sets prev, updating prev->next if prev is non-null */ 
  void setPrev(Edge *p);

  /** \brief Sets sym, updating sym->sym if sym is non-null */ 
  void setSym(Edge *s);
};


/** ShMesh class stores a mesh using a half-edge data structure */
template<typename M>
class ShMesh {
  public:
    typedef M MeshType; 
    typedef typename M::Vertex Vertex; 
    typedef typename M::Edge Edge; 
    typedef typename M::Face Face; 

    typedef std::set<Vertex*> VertexSet;
    typedef std::set<Edge*> EdgeSet;
    typedef std::set<Face*> FaceSet;
    typedef std::list<Vertex*> VertexList;

    /** \brief Empty mesh constructor */
    ShMesh();

    /** \brief Copy constructor 
     * Makes copies of vertices, edges, and faces and builds a mesh isomorphic other */
    ShMesh(const ShMesh<M>& other);

    /** \brief ShMesh destructor */
    ~ShMesh();

    /** \brief Assignment Operator  */
    ShMesh<M>& operator=(const ShMesh<M>& other); 

    /** \brief removes all verts, edges, and faces in this mesh & deletes them*/
    void clear();

    /** \brief Adds a face to the mesh.
     * The face contains the given vertices in order (do not repeat first vertex).
     * Adds required edges and faces. 
     * The edge corresponding to vl(0) -> vl(1) is set to result->edge
     */
    Face* addFace(const VertexList &vl);

    /** \brief Removes a face from the mesh.
     * Deletes edges involed in the face, but not the vertices. 
     */
    void removeFace(Face *f);

    /** \brief Vertex merging function.
     * Merges any vertices that are "equal" according to the 
     * StrictWeakOrdering functor VertLess
     */
    template<typename VertLess>
    void mergeVertices();

    /** \brief Edge merging function.
     * Pairs up half-edges that match each other (i.e. e1.start = e2.end, e1.end = e2.start) 
     *
     * Note that if there are multiple edges between start->end 
     * that match up with an edge, e, from end->start, one of them will be  
     * set to e->sym.  Which one gets matched is undefined.
     */
    void mergeEdges();

    /** \brief Triangulates by ear. 
     * returnst true if any triangles removed */
    bool earTriangulate();

    VertexSet verts;
    EdgeSet edges;
    FaceSet faces;

  protected:
    typedef std::map<Vertex*, Vertex*> VertexMap;
    typedef std::map<Edge*, Edge*> EdgeMap;
    typedef std::map<Face*, Face*> FaceMap;

    // TODO this is a real hack...figure out how to do removeHalfEdge(e) in log or 
    // constant time without the incidence map for weird meshes 
    // (e.g. with articulation points).
    
    // On certain meshes, all edges incident to a vertex v can be found
    // by traversing v.edge->sym pointers, but this is not always
    // the case.
    typedef std::multimap<Vertex*, Edge*> IncidenceMap;
    typedef typename IncidenceMap::value_type Incidence;
    typedef typename IncidenceMap::iterator IncidenceIterator;
    typedef std::pair<typename IncidenceMap::iterator, 
              typename IncidenceMap::iterator> IncidenceRange;

    IncidenceMap m_incidences; // m_incidences[v] holds all edges e with e.start = v

    /** \brief Removes a half-edge from the mesh.
     * If e->start->edge == this, then e->start->edge is set
     * to a different element in the m_startMap;
     *
     * This is a private utility function that does not update 
     * e->face if e->face->edge == e. 
     */
    void removeHalfEdge(Edge *e);

    /** \brief Adds e to the edges set and m_incidenceEdges incidence map */
    void insertHalfEdge(Edge *e);
};


}

#include "ShMeshImpl.hpp"

#endif
