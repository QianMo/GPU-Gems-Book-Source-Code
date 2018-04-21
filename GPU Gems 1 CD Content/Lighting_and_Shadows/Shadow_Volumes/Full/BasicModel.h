/**
  @file BasicModel.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#ifndef BASIC_MODEL_H
#define BASIC_MODEL_H

#include <G3DAll.h>


class EdgeHash : public G3D::Hashable
{
public:
    EdgeHash();
    EdgeHash(int v0, int v1);

    unsigned int hashCode() const;
    bool operator==(const EdgeHash& rhs);

    int m_vertexIndices[2];
    Array<Vector3>*     m_vertexArray;
};


struct BasicEdge
{
    int faceIndices[2];
    int vertexIndices[2];
};


struct BasicFace
{
    int                             vertexIndices[3];
};


class BasicModel
{
public:

    // this is mostly used for sub-classes
    BasicModel();

    BasicModel(
        const Array<Vector3>&       vertexArray, 
        const Array<Vector2>&       texCoordArray,
        const Array<BasicFace>&     faceArray,
        const Array<BasicEdge>&     edgeArray,
        bool                        castShadow,
        Color3                      color);

    virtual ~BasicModel();


    /**
     * Can used by sub-classes of BasicModel
     */
    virtual void useTextures(bool texturesOn);

    const Box& getBoundingBox() const;

    const Sphere& getBoundingSphere() const;

    /**
     * This collapses points that are close to each-other and
     * then gets rid of any degenerate triangles produced
     */
    void compact();

    /**
     * This splits each triangle into four pieces numLevels
     * times, so that the final triangle count is
     * 4 ^ (numLevels).  This is useful if you are only doing per-vertex
     * lighting.  This does nothing clever, a much better solution
     * would be to test for edge lengths over a certain size and then
     * split a triangle that had any edge that was too long.  This was
     * going to be used with Quake 3 levels until it became apparent
     * that their geometry wasn't going to work well with shadows
     * (see README.txt).
     */
    void retesselateFaces(
        int                         numLevels);

    void computeStaticFaceNormals(
        const Array<Vector3>&       vertexArray,
        Array<Vector3>&             faceNormal) const;

    virtual void updateModel(int milliTime);

    virtual void drawFaces(
            int&                    polyCount);

    void drawShadowVolume(
            const Vector4&          light,
            bool                    frontCap,
            bool                    extrusions,
            bool                    endCap,
            int&                    polyCount,
            bool                    shadowOptimization);

    bool doesCastShadow() const;


    void splitFaces(
        const Array<BasicFace>&     oldFaceArray,
        const Array<Vector3>&       oldVertexArray,
        Array<BasicFace>&           newFaceArray,
        Array<Vector3>&             newVertexArray);

    void computeEdges(
            int                         startFace,
            int                         endFace);

    void computeEdges();

    void addEdge(
            int                         vertex0,
            int                         vertex1,
            int                         face,
            Table<EdgeHash, int>&       edgeTable,
            Array<BasicEdge>&           edgeArray,
            Array<Vector3>&             vertexArray);

    void computeBoundingBox();

    void computeBoundingSphere();


    // this is public for sheer ease of use
    CoordinateFrame         m_transformation;
    Color3                  m_modelColor;


    Array<Vector3>        m_vertexArray;
    Array<Vector2>        m_texCoordArray;
    Array<Vector3>        m_extrudedVertexArray;
    Array<Vector3>        m_faceNormalArray;

    // Geometric edges used for shadow volume computation.
    Array<BasicEdge>      m_edgeArray;
    Array<BasicFace>      m_faceArray;

    // This array is filled every time silhouettes edges are computed
    Array<bool>           m_isBackfaceArray;
    Array<int>            m_silhouetteEdgeArray;
    Box*                  m_boundingBox;
    Sphere*               m_boundingSphere;
    bool                  m_castShadow;

    // This says if the last calculation of extrusions is still
    // not valid.  An extrusion is definitely valid there is one
    // light in the scene and neither the object or light has moved.
    bool                  m_extrusionDirty;
};

#endif

