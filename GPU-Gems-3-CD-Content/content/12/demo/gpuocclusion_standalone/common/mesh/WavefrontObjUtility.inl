/*! \file WavefrontObjUtility.inl
 *  \author Jared Hoberock
 *  \brief Inline file for WavefrontObjUtility.h.
 */

#include <string>
#include <strstream>
#include <string.h>
#include <fstream>

#include "WavefrontObjUtility.h"

template<typename P3D, typename P2D, typename N3D>
  std::ostream &WavefrontObjUtility<P3D, P2D, N3D>::writeObj(std::ostream &os, const typename WavefrontObjUtility<P3D,P2D,N3D>::Mesh &m)
{
  // write each vertex (position) first
  os << "# " << static_cast<unsigned int>(m.getPositions().size()) << " vertices." << std::endl;
  for(std::vector<Mesh::Position>::const_iterator p = m.getPositions().begin();
      p != m.getPositions().end();
      ++p)
  {
    os << "v " << (*p)[0] << " " << (*p)[1] << " " << (*p)[2] << std::endl;
  } // end for p

  os << std::endl;

  // write each parametric coordinate next
  os << "# " << static_cast<unsigned int>(m.getParametricCoordinates().size()) << " parametric coordinates." << std::endl;
  for(std::vector<Mesh::ParametricCoordinate>::const_iterator coord = m.getParametricCoordinates().begin();
      coord != m.getParametricCoordinates().end();
      ++coord)
  {
    os << "vt " << (*coord)[0] << " " << (*coord)[1] << " " << std::endl;
  } // end for coord

  os << std::endl;

  // write each normal next
  os << "# " << static_cast<unsigned int>(m.getNormals().size()) << " normals." << std::endl;
  for(std::vector<Mesh::Normal>::const_iterator n = m.getNormals().begin();
      n != m.getNormals().end();
      ++n)
  {
    os << "vn " << (*n)[0] << " " << (*n)[1] << " " << (*n)[2] << std::endl;
  } // end for n

  // now write each face
  // add +1 to each index
  os << "# " << static_cast<unsigned int>(m.getFaces().size()) << " faces." << std::endl;
  for(std::vector<Mesh::Face>::const_iterator f = m.getFaces().begin();
      f != m.getFaces().end();
      ++f)
  {
    os << "f ";

    for(Mesh::Face::const_iterator v = f->begin();
        v != f->end();
        ++v)
    {
      if(v->mPositionIndex != -1) os << v->mPositionIndex + 1;

      if(v->mParametricCoordinateIndex != -1) os << "/" << v->mParametricCoordinateIndex + 1;

      if(v->mNormalIndex != -1) os << "/" << v->mNormalIndex + 1;

      os << " ";
    } // end for v

    os << std::endl;
  } // end for f

  return os;
} // end WavefrontObjUtility::writeObj()

template<typename P3D, typename P2D, typename N3D>
  bool WavefrontObjUtility<P3D, P2D, N3D>
    ::readObj(const char *filename,
              Mesh &m,
              const bool triangulate)
{
  std::ifstream is;
  is.open(filename);
  if(is.is_open())
  {
    readObj(is, m, triangulate);
    is.close();
    return true;
  } // end if

  return false;
} // end WavefrontObjUtility::readObj()

template<typename P3D, typename P2D, typename N3D>
  std::istream &WavefrontObjUtility<P3D, P2D, N3D>::
    readObj(std::istream &is,
            Mesh &m,
            const bool triangulate)
{
  m.clear();

  Mesh::Position p;
  Mesh::ParametricCoordinate uv;
  Mesh::Normal n;
  Mesh::Face f;

  // get references to the different lists
  Mesh::PositionList &positions = m.getPositions();
  Mesh::ParametricList &parms = m.getParametricCoordinates();
  Mesh::NormalList &normals = m.getNormals();
  Mesh::FaceList &faces = m.getFaces();

  std::string token;

  while(!is.eof())
  {
    // get the token
    token.clear();
    is >> token;

    if(token[0] == '#')
    {
      // slurp the rest of the line
      getline(is, token);
    } // end if
    else if(token == "v")
    {
      // we're reading a vertex position
      is >> p[0] >> p[1] >> p[2];
      positions.push_back(p);
    } // end else if
    else if(token == "vt")
    {
      // we're reading a texture coordinate
      is >> uv[0] >> uv[1];
      parms.push_back(uv);
    } // end else if
    else if(token == "vn")
    {
      // we're reading a vertex normal
      is >> n[0] >> n[1] >> n[2];
      normals.push_back(n);
    } // end else if
    else if(token == "f")
    {
      // we're reading a face
      readFace<P3D, P2D, N3D>(is, f);
      faces.push_back(f);
    } // end else if
  } // end while

  // triangulate if requested
  if(triangulate)
  {
    m.triangulate();
  } // end if

  return is;
} // end WavefrontObjUtility::readObj()

template<typename P3D, typename P2D, typename N3D>
  std::istream &readFace(std::istream &is, typename WavefrontObjUtility<P3D,P2D,N3D>::Mesh::Face &f)
{
  std::string line;

  // slurp the entire line
  getline(is,line);

  // turn the line into a stream
  std::istrstream lineStream(line.c_str());

  typename WavefrontObjUtility<P3D,P2D,N3D>::Mesh::Vertex v;

  // clear the face
  f.clear();

  // read vertices until we exhaust the line
  while(!(lineStream >> std::ws).eof())
  {
    readVertex<P3D,P2D,N3D>(lineStream, v);
    f.push_back(v);
  } // end while

  return is;
} // end readFace()

template<typename P3D, typename P2D, typename N3D>
  std::istream &readVertex(std::istream &is, typename WavefrontObjUtility<P3D,P2D,N3D>::Mesh::Vertex &v)
{
  // start with an invalid Vertex
  v.mPositionIndex = v.mParametricCoordinateIndex = v.mNormalIndex = -1;

  // get a token
  std::string token;
  is >> token;

  int pos, parm, norm;

  char *buf = &token[0];
  char *word = 0;

  if(strstr(buf, "//"))
  {
    // we have v//n
    word = strtok(buf, " ");
    sscanf(word, "%d//%d", &pos, &norm);
    v.mPositionIndex = pos - 1;
    v.mNormalIndex = norm - 1;
  } // end if v//n
  else if(sscanf(buf, "%d/%d/%d", &pos, &parm, &norm) == 3)
  {
    // v/t/n
    word = strtok(buf, " ");
    sscanf(word, "%d/%d/%d", &pos, &parm, &norm);
    v.mPositionIndex = pos - 1;
    v.mParametricCoordinateIndex = parm - 1;
    v.mNormalIndex = norm - 1;
  } // end else if v/t/n
  else if(sscanf(buf, "%d/%d", &pos, &parm) == 2)
  {
    word = strtok(buf, " ");
    sscanf(word, "%d/%d", &pos, &parm);
    v.mPositionIndex = pos - 1;
    v.mParametricCoordinateIndex = parm - 1;
  } // end else if v/t
  else if(sscanf(buf, "%d", &pos)== 1)
  {
    // v
    word = strtok(buf, " ");

    sscanf(word, "%d", &pos);
    v.mPositionIndex = pos - 1;
  } // end else if v

  return is;
} // end readVertex()

