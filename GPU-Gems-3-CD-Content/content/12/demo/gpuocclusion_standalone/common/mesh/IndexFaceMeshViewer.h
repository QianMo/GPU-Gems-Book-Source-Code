/*! \file IndexFaceMeshViewer.h
 *  \author Jared Hoberock
 *  \brief Defines a QGLViewer class for IndexFaceMesh.
 */

#ifndef INDEX_FACE_MESH_VIEWER_H
#define INDEX_FACE_MESH_VIEWER_H

#include <GL/glew.h>
#include <commonviewer/CommonViewer.h>
#include <gl++/displaylist/DisplayList.h>
#include "IndexFaceMesh.h"
#include <gpcpu/Vector.h>

template<typename BaseViewer, typename KeyEvent>
  class IndexFaceMeshViewer
    : public CommonViewer<BaseViewer, KeyEvent>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CommonViewer<BaseViewer, KeyEvent> Parent;

    /*! \typedef Mesh
     *  \brief Shorthand.
     */
    typedef IndexFaceMesh<float3, float2, float3> Mesh;

    /*! This method calls the mDrawMesh display list.
     */
    inline virtual void draw(void);

    /*! This method calls Parent::init() and mDrawMesh.create().
     */
    inline virtual void init(void);

    inline virtual void keyPressEvent(KeyEvent *e);

  protected:
    /*! This method computes the axis-aligned bounding box of the
     *  given Mesh.
     *  \param m The Mesh of interest.
     *  \param minCorner The minimal corner of m's bounding box.
     *  \param maxCorner The maximal corner of m's bounding box.
     */
    inline static void getBoundingBox(const Mesh &m,
                                      float3 &minCorner,
                                      float3 &maxCorner);

    /*! This method updates mDrawMesh.
     */
    inline virtual void updateDisplayLists(void);

    /*! This method loads mMesh.
     */
    inline virtual bool loadMeshFile(void);

    /*! This method loads mMesh given a path to a Wavefront .OBJ.
     *  \param filename The path to the mesh to load.
     */
    inline virtual bool loadMeshFile(const char *filename);

    /*! This method draws mMesh in OpenGL immediate mode.
     */
    inline virtual void drawMeshImmediate(void) const;

    /*! This method draws vertex index labels.
     */
    inline virtual void drawVertexLabels(void);

    /*! This method draws face index labels.
     */
    inline virtual void drawFaceLabels(void);

    /*! DisplayList for mMesh.
     */
    DisplayList mDrawMesh;

    /*! A Mesh to render.
     */
    Mesh mMesh;

    /*! Use smooth or float normals?
     */
    bool mUseSmoothNormals;

    /*! Label vertices?
     */
    bool mLabelVertices;

    /*! Label faces?
     */
    bool mLabelFaces;

    /*! Polygon mode for rendering.
     */
    GLenum mPolygonMode;
}; // end IndexFaceMeshViewer

#include "IndexFaceMeshViewer.inl"

#endif // INDEX_FACE_MESH_VIEWER_H

