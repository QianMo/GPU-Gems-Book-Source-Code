/*! \file IndexFaceMeshViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for IndexFaceMeshViewer.h.
 */

#include "IndexFaceMeshViewer.h"
#include "WavefrontObjUtility.h"

#ifdef QT_VERSION
#include <qstring.h>
#endif // QT_VERSION

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::init(void)
{
  glewInit();
  Parent::init();

#ifdef QT_VERSION
  restoreStateFromFile();
#endif // QT_VERSION

  mDrawMesh.create();

  mUseSmoothNormals = false;
  mLabelVertices = false;
  mLabelFaces = false;
  mPolygonMode = GL_FILL;
} // end IndexFaceMeshViewer::init()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::draw(void)
{
  glPushAttrib(GL_POLYGON_BIT | GL_LIGHTING_BIT);

  glPolygonMode(GL_FRONT_AND_BACK, mPolygonMode);
  if(mPolygonMode != GL_FILL) glDisable(GL_LIGHTING);

  mDrawMesh();

  if(mLabelVertices)
  {
    drawVertexLabels();
  } // end if

  if(mLabelFaces)
  {
    drawFaceLabels();
  } // end if

  glPopAttrib();
} // end IndexFaceMeshViewer::draw()

template<typename BaseViewer, typename KeyEvent>
  bool IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::loadMeshFile(const char *filename)
{
  bool result = false;
  if(WavefrontObjUtility<float3,float2,float3>::readObj(filename, mMesh, true))
  {
    updateDisplayLists();

#ifdef QT_VERSION
    float3 m, M;
    getBoundingBox(mMesh, m, M);
    float3 c = (m + M) / 2.0;
    setSceneCenter(qglviewer::Vec(c[0], c[1], c[2]));
    setSceneBoundingBox(qglviewer::Vec(m[0], m[1], m[2]),
                        qglviewer::Vec(M[0], M[1], M[2]));
#endif // QT_VERSION

    result = true;
  } // end if

  return result;
} // end IndexFaceMeshViewer::loadMeshFile()

template<typename BaseViewer, typename KeyEvent>
  bool IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::loadMeshFile(void)
{
  // get the filename
  std::string filename = getOpenFileName("Choose a mesh file",
                                         "c:/dev/data/geometry/obj",
                                         "Meshes (*.obj *.off)");

  if(filename.size() > 0)
  {
    return loadMeshFile(filename.c_str());
  } // end if

  return false;
} // end IndexFaceMeshViewer::loadMeshFile()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::updateDisplayLists(void)
{
  mDrawMesh.bind();
  drawMeshImmediate();
  mDrawMesh.unbind();
} // end IndexFaceMeshViewer::updateDisplayLists()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::drawMeshImmediate(void) const
{
  const Mesh::FaceList &faces = mMesh.getFaces();
  const Mesh::PositionList &positions = mMesh.getPositions();
  const Mesh::NormalList &normals = mMesh.getNormals();

  float3 e1,e2,n;
  glBegin(GL_TRIANGLES);
  for(Mesh::FaceList::const_iterator f = faces.begin();
      f != faces.end();
      ++f)
  {
    e1 = positions[(*f)[1].mPositionIndex] - positions[(*f)[0].mPositionIndex];
    e2 = positions[(*f)[2].mPositionIndex] - positions[(*f)[0].mPositionIndex];
    n = e1.cross(e2);
    n = n.normalize();
    glNormal3fv(n);
    for(Mesh::Face::const_iterator v = f->begin();
        v != f->end();
        ++v)
    {
      if(mUseSmoothNormals && v->mNormalIndex != -1)
      {
        glNormal3fv(normals[v->mNormalIndex]);
      } // end if

      glVertex3fv(positions[v->mPositionIndex]);
    } // end for v
  } // end for f
  glEnd();
} // end IndexFaceMeshViewer::drawImmediate()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 'O':
    {
      if(loadMeshFile())
      {
        updateGL();
      } // end if
      break;
    } // end case O

    case 'N':
    {
      mUseSmoothNormals = !mUseSmoothNormals;
      updateDisplayLists();
      updateGL();
      break;
    } // end case N

    case 'L':
    {
      if(e->modifiers() & SHIFT_MODIFIER)
      {
        mLabelFaces = !mLabelFaces;
      } // end if
      else
      {
        mLabelVertices = !mLabelVertices;
      } // end else

      updateGL();
      break;
    } // end case L

    case 'P':
    {
      if(mPolygonMode == GL_FILL) mPolygonMode = GL_POINT;
      else if(mPolygonMode == GL_POINT) mPolygonMode = GL_LINE;
      else if(mPolygonMode == GL_LINE) mPolygonMode = GL_FILL;
      updateGL();
      break;
    } // end case P

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end IndexFaceMeshViewer::keyPressEvent()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::getBoundingBox(const Mesh &m,
                     float3 &minCorner,
                     float3 &maxCorner)
{
  float inf = std::numeric_limits<float>::infinity();
  minCorner = float3(inf,inf,inf);
  maxCorner = -minCorner;

  const Mesh::PositionList &positions = m.getPositions();
  for(Mesh::PositionList::const_iterator p = positions.begin();
      p != positions.end();
      ++p)
  {
    for(int i = 0; i < 3; ++i)
    {
      minCorner[i] = std::min<float>(minCorner[i], (*p)[i]);
      maxCorner[i] = std::max<float>(maxCorner[i], (*p)[i]);
    } // end for i
  } // end for p
} // end IndexFaceMeshViewer::getBoundingBox()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::drawVertexLabels(void)
{
#ifdef QT_VERSION
  glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);
  glColor3f(1,0,0);
  glDisable(GL_LIGHTING);

  QString label;
  const Mesh::PositionList &positions = mMesh.getPositions();
  for(unsigned int i = 0;
      i != positions.size();
      ++i)
  {
    label = QString::number(i);
    renderText(positions[i][0], positions[i][1], positions[i][2], label);
  } // end for i

  glPopAttrib();
#endif // QT_VERSION
} // end IndexFaceMeshViewer::drawVertexLabels()

template<typename BaseViewer, typename KeyEvent>
  void IndexFaceMeshViewer<BaseViewer,KeyEvent>
    ::drawFaceLabels(void)
{
#ifdef QT_VERSION
  glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT);
  glColor3f(0,0,1);
  glDisable(GL_LIGHTING);

  QString label;
  for(unsigned int f = 0;
      f != mMesh.getFaces().size();
      ++f)
  {
    label = QString::number(f);

    float3 centroid(0,0,0);
    for(Mesh::Face::const_iterator v = mMesh.getFaces()[f].begin();
        v != mMesh.getFaces()[f].end();
        ++v)
    {
      centroid += mMesh.getPositions()[v->mPositionIndex];
    } // end for v

    centroid /= mMesh.getFaces()[f].size();

    renderText(centroid[0], centroid[1], centroid[2], label);
  } // end for i

  glPopAttrib();
#endif // QT_VERSION
} // end IndexFaceMeshViewer::drawVertexLabels()


