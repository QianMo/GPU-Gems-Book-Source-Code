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
#ifndef SHUTIL_SHOBJMESH_HPP
#define SHUTIL_SHOBJMESH_HPP

#include <iosfwd>

#include "sh.hpp"
#include "ShMesh.hpp"

namespace ShUtil {

using namespace SH;

struct ShObjVertex; 
struct ShObjFace;
struct ShObjEdge;

typedef ShMeshType<ShObjVertex, ShObjFace, ShObjEdge> ShObjMeshType;

struct ShObjVertex: public ShMeshVertex<ShObjMeshType> {
  ShPoint3f pos; // vertex position

  /** \brief Constructs a vertex with the given position */
  ShObjVertex(const ShPoint3f &p);
};

struct ShObjFace: public ShMeshFace<ShObjMeshType> {
  ShNormal3f normal; // face normal
};

struct ShObjEdge: public ShMeshEdge<ShObjMeshType> {
  // properties for start vertex in this edge's face
  ShNormal3f normal;  
  ShVector3f tangent;
  ShTexCoord2f texcoord;
};

/* OBJ file mesh where each vertex stores its position,
 * Each edge stores the normal/tc/tangent for its start vertex,
 * and each face stores its face normal
 *
 * Each face in the object mesh is a triangle. (Triangulation
 * happens on loading from the OBJ file)
 */
class ShObjMesh: public ShMesh<ShObjMeshType> {
  public:
    typedef ShMesh<ShObjMeshType> ParentType;

    /** \brief Constructs an empty mesh */
    ShObjMesh();

    /** \brief Constructs ShObjMesh from an input stream of an OBJ file */
    ShObjMesh(std::istream &in);

    /** \brief Deletes current mesh and reads in a new mesh from an OBJ file */
    std::istream& readObj(std::istream &in);

    /** \brief Generates face normals by cross product  */
    void generateFaceNormals();

    /** \brief Generates normals by averaging adjacent face normals
     * for vertices that have zero normals (or all vertices if force = true) 
     *
     * returns the number of fixed vertex normals */ 
    int generateVertexNormals(bool force = false);

    /** \brief Generates tangents by cross product with (0,1,0)
     * for vertices that have zero tangents (or all vertices if force = true)
     *
     * returns the number of tangents generated
     */
    int generateTangents(bool force = false);

    /** \brief Generates texcoords in a "spherical" shrink map centered at
     * the average of all vertex positions only when all vertex texcoords are 0. 
     * (or if force = true)
     *
     * returns the number of texture coordinates generated */ 
    int generateSphericalTexCoords(bool force = false);

    /** \brief Normalizes all the normals held in this mesh */
    void normalizeNormals();

    /** \brief Consolidates vertices whose coordinates are within 1e-5 of each 
     * other componentwise
     */
    void consolidateVertices();

    /** \brief Sets mesh data to data from an OBJ file */
    friend std::istream& operator>>(std::istream &in, ShObjMesh &mesh);
};

}

#endif
