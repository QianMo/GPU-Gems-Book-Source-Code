/**
  @file MeshAlgAdjacency.cpp

  @maintainer Morgan McGuire, matrix@graphics3d.com
  @created 2003-09-14
  @edited  2003-10-22

  Copyright 2000-2003, Morgan McGuire.
  All rights reserved.

 */

#include "G3D/MeshAlg.h"
#include "G3D/Table.h"
#include "G3D/Set.h"

namespace G3D {

/**
 An edge directed according to the "size" of the vertices 
 for use in edgeTable. 

 Let the "smaller" vertex of A and B be the one with 
 the smaller x component, with ties broken by the y 
 and z components as needed.  This gives a canonical
 ordering on any set of vertices.
 */
class MeshDirectedEdgeKey {
public:

    /**
     vertex[0] < vertex[1]
     */
    Vector3 vertex[2];

    /**
     Corresponding indices in the vertex array.
     Colocated vertices are indistinguishible
     to the edge table, so *any* two vertex 
     indices for vertices are the right position
     are sufficient.
     */
    int     vertexIndex[2];

    MeshDirectedEdgeKey() {}
    
    MeshDirectedEdgeKey(
        const int        i0,
        const Vector3&   v0,
        const int        i1,
        const Vector3&   v1) {

        vertexIndex[0] = i0;
        vertexIndex[1] = i1;
        vertex[0]    = v0;
        vertex[1]    = v1;

        // Find the smaller vertex.
        for (int i = 0; i < 3; ++i) {
            if (v0[i] < v1[i]) {
                break;
            } else if (v0[i] > v1[i]) {
                // Swap
                vertexIndex[0] = i1;
                vertexIndex[1] = i0;
                vertex[0]    = v1;
                vertex[1]    = v0;
                break;
            }
        }
    }


    bool operator==(const MeshDirectedEdgeKey& e2) const {
        for (int i = 0; i < 2; ++i) {
            if (vertex[i] != e2.vertex[i]) {
                return false;
            }
        }
        return true;
    }
};

}

unsigned int hashCode(const G3D::MeshDirectedEdgeKey& e) {
    return (e.vertex[0].hashCode() + 1) ^ e.vertex[1].hashCode();
}

namespace G3D {

/**
 A 2-key hashtable mapping edges to lists of face indices.  
 Used only for MeshAlg::computeAdjacency.

 In the face lists, index <I>f</I> >= 0 indicates that
 <I>f</I> contains the edge as a forward edge.  Index <I>f</I> < 0 
 indicates that ~<I>f</I> contains the edge as a backward edge.
 */
class MeshEdgeTable {
public:
    typedef Table<MeshDirectedEdgeKey, Array<int> > ET;

private:
    
    ET                   table;

public:
    
    /**
     Clears the table.
     */
    void clear() {
        table.clear();
    }
    
    /**
     Inserts the faceIndex into the edge's face list.
     The index may be a negative number indicating a backface.
     */
    void insert(const MeshDirectedEdgeKey& edge, int faceIndex) {
        
        // debugAssertM((table.size() > 20) && (table.debugGetLoad() < 0.5 || table.debugGetNumBuckets() < 20),
        //    "MeshEdgeTable is using a poor hash function.");

        if (! table.containsKey(edge)) {
            // First time
            Array<int> x(1);
            x[0] = faceIndex;
            table.set(edge, x);
        } else {
            table[edge].append(faceIndex);
        }
    }

    /**
     Returns the face list for a given edge
     */
    const Array<int>& get(const MeshDirectedEdgeKey& edge) {
        return table[edge];
    }

    ET::Iterator begin() {
        return table.begin();
    }

    const ET::Iterator end() const {
        return table.end();
    }

};


/**
 Used and cleared by MeshModel::computeAdjacency()
 */
static MeshEdgeTable            edgeTable;

/**
 Assigns the edge index into the next unassigned edge
 index.  The edge index may be negative, indicating
 a reverse edge.
 */
static void assignEdgeIndex(MeshAlg::Face& face, int e) {
    for (int i = 0; i < 3; ++i) {
        if (face.edgeIndex[i] == MeshAlg::Face::NONE) {
            face.edgeIndex[i] = e;
            return;
        }
    }

    debugAssertM(false, "Face has already been assigned 3 edges");
}


void MeshAlg::computeAdjacency(
    const Array<Vector3>&   vertexArray,
    const Array<int>&       indexArray,
    Array<Face>&            faceArray,
    Array<Edge>&            edgeArray,
    Array< Array<int> >&    adjacentFaceArray) {

    edgeArray.resize(0);
    adjacentFaceArray.resize(0);
    adjacentFaceArray.resize(vertexArray.size());

    faceArray.resize(0);
    edgeTable.clear();
    
    // Face normals
    Array<Vector3> faceNormal;

    // Iterate through the triangle list
    for (int q = 0; q < indexArray.size(); q += 3) {

        // Don't allow degenerate faces
        if ((indexArray[q + 0] != indexArray[q + 1]) &&
            (indexArray[q + 1] != indexArray[q + 2]) &&
            (indexArray[q + 2] != indexArray[q + 0])) {
            Vector3 vertex[3];
            int f = faceArray.size();
            MeshAlg::Face& face = faceArray.next();

            // Construct the face
            for (int j = 0; j < 3; ++j) {
                int v = indexArray[q + j];
                face.vertexIndex[j] = v;
                face.edgeIndex[j]   = Face::NONE;
                adjacentFaceArray[v].append(f);
                vertex[j]           = vertexArray[v];
            }

            faceNormal.append((vertex[1] - vertex[0]).cross(vertex[2] - vertex[0]).direction());

            static const int nextIndex[] = {1, 2, 0};

            // Add each edge to the edge table.
            for (int j = 0; j < 3; ++j) {
                const int      i0 = indexArray[q + j];
                const int      i1 = indexArray[q + nextIndex[j]];
                const Vector3& v0 = vertexArray[i0];
                const Vector3& v1 = vertexArray[i1];

                const MeshDirectedEdgeKey edge(i0, v0, i1, v1);

                if (v0 == edge.vertex[0]) {
                    // The edge was directed in the same manner as in the face
                    edgeTable.insert(edge, f);
                } else {
                    // The edge was directed in the opposite manner as in the face
                    edgeTable.insert(edge, ~f);
                }
            }
        }
    }
    
    // For each edge in the edge table, create an edge in the edge array.
    // Collapse every 2 edges from adjacent faces.

    MeshEdgeTable::ET::Iterator cur = edgeTable.begin();
    MeshEdgeTable::ET::Iterator end = edgeTable.end();

    while (cur != end) {
        MeshDirectedEdgeKey&  edgeKey        = cur->key; 
        Array<int>&           faceIndexArray = cur->value;

        // Process this edge
        while (faceIndexArray.size() > 0) {

            // Remove the last index
            int f0 = faceIndexArray.pop();

            // Find the normal to that face
            const Vector3& n0 = faceNormal[(f0 >= 0) ? f0 : ~f0];

            bool found = false;

            // We try to find the matching face with the closest
            // normal.  This ensures that we don't introduce a lot
            // of artificial ridges into flat parts of a mesh.
            double ndotn = -2;
            int f1, i1;
            
            // Try to Find the face with the matching edge
            for (int i = faceIndexArray.size() - 1; i >= 0; --i) {
                int f = faceIndexArray[i];

                if ((f >= 0) != (f0 >= 0)) {
                    // This face contains the oppositely oriented edge
                    // and has not been assigned too many edges

                    const Vector3& n1 = faceNormal[(f >= 0) ? f : ~f];
                    double d = n1.dot(n0);

                    if (found) {
                        // We previously found a good face; see if this
                        // one is better.
                        if (d > ndotn) {
                            // This face is better.
                            ndotn = d;
                            f1    = f;
                            i1    = i;
                        }
                    } else {
                        // This is the first face we've found
                        found = true;
                        ndotn = d;
                        f1    = f;
                        i1    = i;
                    }
                }
            }

            // Create the new edge
            int e = edgeArray.size();
            Edge& edge = edgeArray.next();
            
            edge.vertexIndex[0] = edgeKey.vertexIndex[0];
            edge.vertexIndex[1] = edgeKey.vertexIndex[1];

            if (f0 >= 0) {
                edge.faceIndex[0] = f0;
                edge.faceIndex[1] = Face::NONE;
                assignEdgeIndex(faceArray[f0], e); 
            } else {
                edge.faceIndex[1] = ~f0;
                edge.faceIndex[0] = Face::NONE;
                assignEdgeIndex(faceArray[~f0], ~e); 
            }

            if (found) {
                // We found a matching face; remove both
                // faces from the active list.
                faceIndexArray.fastRemove(i1);

                if (f1 >= 0) {
                    edge.faceIndex[0] = f1;
                    assignEdgeIndex(faceArray[f1], e); 
                } else {
                    edge.faceIndex[1] = ~f1;
                    assignEdgeIndex(faceArray[~f1], ~e); 
                }
            }
        }

        ++cur;
    }

    edgeTable.clear();
}

} // G3D namespace
