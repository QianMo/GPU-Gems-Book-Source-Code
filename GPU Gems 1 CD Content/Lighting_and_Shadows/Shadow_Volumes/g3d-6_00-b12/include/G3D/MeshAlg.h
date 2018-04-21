/**
 @file MeshAlg.h

 Indexed Mesh algorithms.

 @maintainer Morgan McGuire, matrix@graphics3d.com

 @created 2003-09-14
 @edited  2003-12-09
*/

#ifndef G3D_MESHALG_H
#define G3D_MESHALG_H

#include "G3D/platform.h"
#include "G3D/Array.h"
#include "G3D/Vector3.h"

namespace G3D {

/**
  Indexed mesh algorithms.  You have to build your own mesh class.
  <P>
  (No mesh class is provided with G3D because there isn't an "ideal" 
  mesh format-- you may want keyframed animation, skeletal animation,
  texture coordinates, etc.)

  */
class MeshAlg {
public:

    /**
     Oriented, indexed triangle.
     */
    class Face {
    public:
        Face();

        /**
         Used by Edge::faceIndex to indicate a missing face.
         This is a large negative value.
         */
        static const int        NONE;


        /**
         Vertices in the face in counter-clockwise order
         */
        int                     vertexIndex[3];

        /**
         If the edge index is negative, ~index is in this face
         but is directed oppositely.  The index on the edgeIndex
         array has no correspondence with the index on the vertexIndex
         array.  
         */
        // Temporarily takes on the value Face::NONE during adjacency
        // computation to indicate an edge that has not yet been assigned.
        int                     edgeIndex[3];
    };


    /** Oriented, indexed edge */
    class Edge {
    public:
        Edge();

        int                     vertexIndex[2];

        /**
         The edge is directed forward in the first face and
         backward in the second face. Face index of MeshAlg::Face::NONE
         indicates a broken edge.
         */
        int                     faceIndex[2];

        /**
         Returns true if either faceIndex is NONE.
         */
        inline bool broken() const {
            return (faceIndex[0] == Face::NONE) ||
                   (faceIndex[1] == Face::NONE);
        }
    };
    

    /**
     Convenient for passing around the data that changes under
     animation.
     The faces and edges are needed to interpret these values.
     */
    class Geometry {
    public:        
        Array<Vector3>          vertexArray;
        Array<Vector3>          normalArray;

        /**
         Assignment is optimized using SSE.
         */
        Geometry& operator=(const Geometry& src);

        void clear() {
            vertexArray.clear();
            normalArray.clear();
        }
    };

    /**
     Given a set of vertices and a set of indices for traversing them
     to create triangles, computes other mesh properties.  
     
     Faces consisting of fewer than 3 discrete vertex indices will not
     appear in the face array and their edges will not appear in
     the edge array.

     Where two faces meet, there are two opposite directed edges.  These
     are collapsed into a single bidirectional edge in the geometricEdgeArray.
     If four faces meet exactly at the same edge, that edge will appear
     twice in the array.  If an edge is broken (has only one adjacent face)
     it will appear in the array with one face index set to NONE.

     @param vertexArray Vertex positions to use when deciding colocation.
     @param indexArray  Order to traverse vertices to make triangles
     @param faceArray   Output
     @param geometricEdgeArray Output.  These edges automatically weld 
            colocated vertices and remove degenerate edges.
     @param adjacentFaceArray Output. adjacentFaceArray[v] is an array of
                        indices for faces touching vertex index v
     */
    static void computeAdjacency(
        const Array<Vector3>&   vertexArray,
        const Array<int>&       indexArray,
        Array<Face>&            faceArray,
        Array<Edge>&            geometricEdgeArray,
        Array< Array<int> >&    adjacentFaceArray);


    static void computeAreaStatistics(
        const Array<Vector3>&   vertexArray,
        const Array<int>&       indexArray,
        double&                 minEdgeLength,
        double&                 meanEdgeLength,
        double&                 medianEdgeLength,
        double&                 maxEdgeLength,
        double&                 minFaceArea,
        double&                 meanFaceArea,
        double&                 medianFaceArea,
        double&                 maxFaceArea);

   /**
     @param vertexNormalArray Output. Computed by averaging adjacent face normals
     @param faceNormalArray Output. 
    */
    static void computeNormals(
        const Array<Vector3>&   vertexArray,
        const Array<Face>&      faceArray,
        const Array< Array<int> >& adjacentFaceArray,
        Array<Vector3>&         vertexNormalArray,
        Array<Vector3>&         faceNormalArray);

    /**
     Computes face normals only.  Significantly faster (especially if
     normalize is true) than computeNormals.
     */
    static void computeFaceNormals(
        const Array<Vector3>&           vertexArray,
        const Array<Face>&              faceArray,
        Array<Vector3>&                 faceNormals,
        bool                            normalize = true);

    /**
     Classifies each face as a backface or a front face relative
     to the observer point P (which is at infinity when P.w = 0).
     A face with normal exactly perpendicular to the observer vector
     may be classified as either a front or a back face arbitrarily.
     */
    static void identifyBackfaces(
        const Array<Vector3>&           vertexArray,
        const Array<Face>&              faceArray,
        const Vector4&                  P,
        Array<bool>&                    backface);

    /** A faster version of identifyBackfaces for the case where
        face normals have already been computed */
    static void identifyBackfaces(
        const Array<Vector3>&           vertexArray,
        const Array<Face>&              faceArray,
        const Vector4&                  P,
        Array<bool>&                    backface,
        const Array<Vector3>&           faceNormals);

    /**
     Welds nearby and colocated elements of the <I>oldVertexArray</I> together so that
     <I>newVertexArray</I> contains no vertices within <I>radius</I> of one another.
     This is useful for downsampling meshes and welding cracks created by artist errors
     or numerical imprecision.  It is not guaranteed to fix cracks but the probability of
     not fixing them (with a suitablly large radius) approaches zero.

     The two integer arrays map indices back and forth between the arrays according to:
     <PRE>
     oldVertexArray[toOld[ni]] == newVertexArray[ni]
     oldVertexArray[oi] == newVertexArray[toNew[ni]]
     </PRE>

     Note that newVertexArray is never longer than oldVertexArray and is shorter when
     vertices are welded.

     Welding with a large radius will effectively compute a lower level of detail for
     the mesh.

     The welding method runs in roughly linear time in the length of oldVertexArray--
     a uniform spatial grid is used to achieve nearly constant time vertex collapses
     for uniformly distributed vertices.

     @cite The method is that described as the 'Grouper' in Baum, Mann, Smith, and Winget, 
     Making Radiosity Usable: Automatic Preprocessing and Meshing Techniques for
     the Generation of Accurate Radiosity Solutions, Computer Graphics vol 25, no 4, July 1991.
     */
    static void computeWeld(
        const Array<Vector3>& oldVertexArray,
        Array<Vector3>&       newVertexArray,
        Array<int>&           toNew,
        Array<int>&           toOld,
        double                radius);


    /**
     Counts the number of edges (in an edge array returned from 
     MeshAlg::computeAdjacency) that have only one adjacent face.
     */
    static int countBrokenEdges(const Array<Edge>& edgeArray);

    /**
     Generates an array of integers from start to start + n - 1 that have run numbers
     in series then omit the next skip before the next run.  Useful for turning
     a triangle list into an indexed face set.

     Example: 
     <PRE>
       createIndexArray(10, x);
       // x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

       createIndexArray(5, x, 2);
       // x = [2, 3, 4, 5, 6, 7] 

       createIndexArray(6, x, 0, 2, 1);
       // x = [0, 1, 3, 4, 6, 7]
     </PRE>
     */
    static void createIndexArray(
        int                 n, 
        Array<int>&         array,
        int                 start   = 0,
        int                 run     = 1,
        int                 skip    = 0);

    /**
     Computes a conservative, near-optimal axis aligned bounding box and sphere. 

      @cite The bounding sphere uses the method from J. Ritter. An effcient bounding sphere. In Andrew S. Glassner, editor, Graphics Gems. Academic Press, Boston, MA, 1990.

    */
    static void computeBounds(const Array<Vector3>& vertex, class Box& box, class Sphere& sphere);

protected:

    /**
     Helper for computeAdjacency.  If a directed edge with index e already
     exists from i0 to i1 then e is returned.  If a directed edge with index e
     already exists from i1 to i0, ~e is returned (the complement) and
     edgeArray[e] is set to f.  Otherwise, a new edge is created from i0 to i1
     with first face index f and its index is returned.
    
     @param vertexArray Vertex positions to use when deciding colocation.
     
     @param area Area of face f.  When multiple edges of the same direction 
       are found between the same vertices (usually because of degenerate edges)
       the face with larger area is kept in the edge table.
     */
    static int findEdgeIndex(
        const Array<Vector3>&   vertexArray,
        Array<Edge>&            geometricEdgeArray,
        int i0, int i1, int f, double area);
};
}
#endif

