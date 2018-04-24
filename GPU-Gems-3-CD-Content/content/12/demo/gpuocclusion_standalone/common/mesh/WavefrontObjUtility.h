/*! \file WavefrontObjUtility.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a utility class for performing I/O on Wavefront OBJ model files.
 */

#ifndef WAVEFRONT_OBJ_UTILITY_H
#define WAVEFRONT_OBJ_UTILITY_H

#include "IndexFaceMesh.h"
#include <ostream>

/*! \class WavefrontObjUtility
 *  \brief WavefrontObjUtility is basically a namespace with some I/O methods for Wavefront .OBJ files.
 */
template<typename P3D, typename P2D, typename N3D> class WavefrontObjUtility
{
  public:
    /*! \typedef Mesh
     *  \brief Shorthand.
     */
    typedef IndexFaceMesh<P3D, P2D, N3D> Mesh;


    /*! This static method writes an IndexFaceMesh in Wavefront .OBJ format to an ostream.
     *  \param os The ostream to write to.
     *  \param m The IndexFaceMesh to write.
     *  \return os
     *  \note This method is templatized on 3D point type, 2D parametric type, and 3D normal type.
     */
    static std::ostream &writeObj(std::ostream &os, const Mesh &m);

    /*! This static method reads an IndexFaceMesh in Wavefront .OBJ format from a file on disk.
     *  \param filename The name of the file to read from.
     *  \param m The IndexFaceMesh read from filename is returned here.
     *  \param triangulate Whether or not to triangulate non-triangular faces.
     *                     false by default.
     *  \return true if m could be successfully loaded from filename; false, otherwise.
     */
    static bool readObj(const char *filename, Mesh &m, const bool triangulate = false);

    /*! This static method reads an IndexFaceMesh in Wavefront .OBJ format from an istream.
     *  \param is The istream to read from.
     *  \param m The IndexFaceMesh read from is returned here.
     *  \param triangulate Whether or not to triangulate non-triangular faces.
     *                     false by default.
     *  \return is
     *  \note This method is templatized on 3D point type, 2D parametric type, and 3D normal type.
     */
    static std::istream &readObj(std::istream &is, Mesh &m, const bool triangulate = false);
}; // end class WavefrontObjUtility

/*! This function reads a single Face from an istream.
 *  \param is The istream containing the Face definition.
 *  \param f The Face is returned here.
 *  \return is.
 */
template<typename P3D, typename P2D, typename N3D>
  std::istream &readFace(std::istream &is, typename WavefrontObjUtility<P3D,P2D,N3D>::Mesh::Face &f);

/*! This function reads a single Vertex from an istream.
 *  \param is The string containing the Vertex definition.
 *  \param v The Vertex is returned here.
 *  \return is
 */
template<typename P3D, typename P2D, typename N3D>
  std::istream &readVertex(std::istream &is, typename typename WavefrontObjUtility<P3D,P2D,N3D>::Mesh::Vertex &v);

#include "WavefrontObjUtility.inl"

#endif // WAVEFRONT_OBJ_UTILITY_H
