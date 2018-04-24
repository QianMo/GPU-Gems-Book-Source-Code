/*! \file IndexFaceMesh.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting a no-frills index face mesh.
 *         This is a bare-bones, baseline mesh class.
 */

#ifndef INDEX_FACE_MESH_H
#define INDEX_FACE_MESH_H

#include <vector>

/*! \class IndexFaceMesh
 *  \brief An IndexFaceMesh is a collection of Faces, where each Face is simply a
 *         list of Position, ParametricCoordinate, and Normal indices.
 *         IndexFaceMesh is templatized on a user-defined type for 3d points, 2d parametric coordinates,
 *         and normals.  NormalType defaults to PositionType if the user does not supply it.
 */
template<typename P3D, typename P2D, typename N3D = P3D> class IndexFaceMesh
{
  public:
    /*! \typedef Position
     *  \brief Shorthand.
     */
    typedef P3D Position;

    /*! \typedef ParametricCoordinate
     *  \brief Shorthand.
     */
    typedef P2D ParametricCoordinate;

    /*! \typedef Normal
     *  \brief Shorthand.
     */
    typedef N3D Normal;

    /*! \struct Vertex
     *  \brief A Vertex is simply an index to a Position, ParametricCoordinate, and a Normal.
     *  \note If any index is -1, this indicates no data for this attribute.
     */
    struct Vertex
    {
      /*! \fn Vertex
       *  \brief Null constructor does nothing.
       */
      inline Vertex(void);

      /*! \fn Vertex
       *  \brief Convenience constructor accepts indices.
       *  \param p Sets mPositionIndex.
       *  \param uv Sets mParametricCoordinateIndex.
       *  \param n Sets mNormalIndex.
       */
      inline Vertex(const int p,
                    const int uv,
                    const int n);

      /*! A Vertex has an index to a position.
       */
      int mPositionIndex;

      /*! A Vertex has an index to a ParametricCoordinate.
       */
      int mParametricCoordinateIndex;

      /*! A Vertex has an index to a Normal.
       */
      int mNormalIndex;
    }; // end struct Vertex

    /*! \typedef Face
     *  \brief A Face is simply a list of Vertices.
     */
    typedef std::vector<Vertex> Face;

    /*! \typedef PositionList
     *  \brief Shorthand.
     */
    typedef std::vector<Position> PositionList;

    /*! \typedef ParametricList
     *  \brief Shorthand.
     */
    typedef std::vector<ParametricCoordinate> ParametricList;

    /*! \typedef NormalList
     *  \brief Shorthand.
     */
    typedef std::vector<Normal> NormalList;

    /*! \typedef FaceList
     *  \brief Shorthand.
     */
    typedef std::vector<Face> FaceList;

    /*! This method returns the list of Positions.
     *  \return mPositions
     */
    inline PositionList &getPositions(void);

    /*! This method returns a const reference to the list of Positions.
     *  \return mPositions
     */
    inline const PositionList &getPositions(void) const;

    /*! This method returns the list of ParametricCoordinates.
     *  \return mParametricCoordinates.
     */
    inline ParametricList &getParametricCoordinates(void);

    /*! This method returns a const reference to the list of ParametricCoordinates.
     *  \return mParametricCoordinates.
     */
    inline const ParametricList &getParametricCoordinates(void) const;

    /*! This method returns the list of Normals.
     *  \return mNormals
     */
    inline NormalList &getNormals(void);

    /*! This method returns a const reference to the list of Normals.
     *  \return mNormals
     */
    inline const NormalList &getNormals(void) const;

    /*! This method returns the list of Faces.
     *  \return mFaces
     */
    inline FaceList &getFaces(void);

    /*! This method returns the list of Faces.
     *  \return mFaces
     */
    const inline FaceList &getFaces(void) const;

    /*! This method clears each list.
     */
    inline void clear(void);

    /*! This method triangulates this Mesh.
     */
    void triangulate(void);

  protected:
    /*! An IndexFaceMesh keeps a list of vertex Positions.
     */
    PositionList mPositions;

    /*! An IndexFaceMesh keeps a list of vertex ParametricCoordinates.
     */
    ParametricList mParametricCoordinates;

    /*! An IndexFaceMesh keeps a list of vertex Normals.
     */
    NormalList mNormals;

    /*! An IndexFaceMesh keeps a list of Faces.
     */
    FaceList mFaces;
}; // end class IndexFaceMesh

#include "IndexFaceMesh.inl"

#endif // INDEX_FACE_MESH_H

