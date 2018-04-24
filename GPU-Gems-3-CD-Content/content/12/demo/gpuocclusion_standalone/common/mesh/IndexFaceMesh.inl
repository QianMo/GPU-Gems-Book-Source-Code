/*! \file IndexFaceMesh.inl
 *  \author Jared Hoberock
 *  \brief Inline file for IndexFaceMesh.h.
 */

template<typename P3D, typename P2D, typename N3D>
  typename IndexFaceMesh<P3D, P2D, N3D>::PositionList &IndexFaceMesh<P3D, P2D, N3D>::getPositions(void)
{
  return mPositions;
} // end IndexFaceMesh::getPositions()

template<typename P3D, typename P2D, typename N3D>
  const typename IndexFaceMesh<P3D, P2D, N3D>::PositionList &IndexFaceMesh<P3D, P2D, N3D>::getPositions(void) const
{
  return mPositions;
} // end IndexFaceMesh::getPositions()

template<typename P3D, typename P2D, typename N3D>
  typename IndexFaceMesh<P3D, P2D, N3D>::ParametricList &IndexFaceMesh<P3D, P2D, N3D>::getParametricCoordinates(void)
{
  return mParametricCoordinates;
} // end IndexFaceMesh::getParametricCoordinates()

template<typename P3D, typename P2D, typename N3D>
  const typename IndexFaceMesh<P3D, P2D, N3D>::ParametricList &IndexFaceMesh<P3D, P2D, N3D>::getParametricCoordinates(void) const
{
  return mParametricCoordinates;
} // end IndexFaceMesh::getParametricCoordinates()

template<typename P3D, typename P2D, typename N3D>
  typename IndexFaceMesh<P3D, P2D, N3D>::NormalList &IndexFaceMesh<P3D, P2D, N3D>::getNormals(void)
{
  return mNormals;
} // end IndexFaceMesh::getNormals()

template<typename P3D, typename P2D, typename N3D>
  const typename IndexFaceMesh<P3D, P2D, N3D>::NormalList &IndexFaceMesh<P3D, P2D, N3D>::getNormals(void) const
{
  return mNormals;
} // end IndexFaceMesh::getNormals()

template<typename P3D, typename P2D, typename N3D>
  void IndexFaceMesh<P3D, P2D, N3D>::clear(void)
{
  getPositions().clear();
  getParametricCoordinates().clear();
  getNormals().clear();
  getFaces().clear();
} // end IndexFaceMesh::clear()

template<typename P3D, typename P2D, typename N3D>
  typename IndexFaceMesh<P3D, P2D, N3D>::FaceList &IndexFaceMesh<P3D, P2D, N3D>::getFaces(void)
{
  return mFaces;
} // end IndexFaceMesh::getFaces()

template<typename P3D, typename P2D, typename N3D>
  const typename IndexFaceMesh<P3D, P2D, N3D>::FaceList &IndexFaceMesh<P3D, P2D, N3D>::getFaces(void) const
{
  return mFaces;
} // end IndexFaceMesh::getFaces()

template<typename P3D, typename P2D, typename N3D>
  IndexFaceMesh<P3D, P2D, N3D>::Vertex::Vertex(const int p, const int uv, const int n)
{
  mPositionIndex = p;
  mParametricCoordinateIndex = uv;
  mNormalIndex = n;
} // end IndexFaceMesh::Vertex::Vertex()

template<typename P3D, typename P2D, typename N3D>
  IndexFaceMesh<P3D, P2D, N3D>::Vertex::Vertex(void)
{
  ;
} // end IndexFaceMesh::Vertex::Vertex()

template<typename P3D, typename P2D, typename N3D>
  void IndexFaceMesh<P3D, P2D, N3D>
    ::triangulate(void)
{
  Face::iterator v0;
  Face::iterator v1;
  Face::iterator v2;

  // push back additional faces which have too many
  // vertices
  // do a stupid triangulation
  std::vector<Face> newFaces;
  for(std::vector<Face>::iterator f = mFaces.begin();
      f != mFaces.end();
      ++f)
  {
    if(f->size() > 3)
    {
      v0 = f->begin();
      v1 = v0 + 1;
      v2 = v1 + 1;

      // skip the first triangle
      for(++v1, ++v2;
          v2 != f->end();
          ++v1, ++v2)
      {
        newFaces.push_back(Face());
        newFaces.back().push_back(*v0);
        newFaces.back().push_back(*v1);
        newFaces.back().push_back(*v2);
      } // end for f

      // resize the original face to include
      // only the first 3 vertices
      f->resize(3);
    } // end if
  } // end for f

  for(std::vector<Face>::iterator f = newFaces.begin();
      f != newFaces.end();
      ++f)
  {
    mFaces.push_back(*f);
  } // end for f
} // end IndexFaceMesh::triangulate()

