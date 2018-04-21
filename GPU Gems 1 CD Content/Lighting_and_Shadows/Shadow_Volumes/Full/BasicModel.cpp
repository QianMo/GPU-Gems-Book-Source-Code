/**
  @file BasicModel.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#include <G3DAll.h>
#include "BasicModel.h"
#include "Renderer.h"


EdgeHash::EdgeHash()
{
    m_vertexIndices[0] = -1;
    m_vertexIndices[1] = -1;
    m_vertexArray = NULL;
}


EdgeHash::EdgeHash(
        int                         v0,
        int                         v1)
{
    m_vertexIndices[0] = v0;
    m_vertexIndices[1] = v1;
    m_vertexArray = NULL;
}


unsigned int EdgeHash::hashCode() const
{
    return (*m_vertexArray)[m_vertexIndices[0]].hashCode() +
        (*m_vertexArray)[m_vertexIndices[1]].hashCode();
}

bool EdgeHash::operator==(const EdgeHash& rhs)
{
    // return true for any edge that has vertices in equivilant positions
    return ((*m_vertexArray)[m_vertexIndices[0]].fuzzyEq(
        (*m_vertexArray)[rhs.m_vertexIndices[0]]) &&
        (*m_vertexArray)[m_vertexIndices[1]].fuzzyEq(
        (*m_vertexArray)[rhs.m_vertexIndices[1]])) ||
        ((*m_vertexArray)[m_vertexIndices[0]].fuzzyEq(
        (*m_vertexArray)[rhs.m_vertexIndices[1]]) &&
        (*m_vertexArray)[m_vertexIndices[1]].fuzzyEq(
        (*m_vertexArray)[rhs.m_vertexIndices[0]]));
}



BasicModel::BasicModel()
{
    m_boundingBox = NULL;
    m_boundingSphere = NULL;
    m_extrusionDirty = true;
}


BasicModel::BasicModel(
    const Array<Vector3>&           vertexArray, 
    const Array<Vector2>&           texCoordArray,
    const Array<BasicFace>&         faceArray,
    const Array<BasicEdge>&         edgeArray,
    bool                            castShadow,
    Color3                          color)
{
    m_vertexArray       = vertexArray;
    m_texCoordArray     = texCoordArray;
    m_faceArray         = faceArray;
    m_edgeArray         = edgeArray;
    m_castShadow		= castShadow;
    m_modelColor        = color;

    m_isBackfaceArray.resize(m_faceArray.size());
    m_silhouetteEdgeArray.resize(m_edgeArray.size());
    m_extrudedVertexArray.resize(m_vertexArray.size());

    computeStaticFaceNormals(m_vertexArray, m_faceNormalArray);

    m_extrusionDirty = true;
    m_boundingBox = NULL;
    m_boundingSphere = NULL;
    computeBoundingBox();
    computeBoundingSphere();
}


BasicModel::~BasicModel()
{
    delete m_boundingBox;
    delete m_boundingSphere;
}


void BasicModel::useTextures(bool texturesOn)
{
    // do nothing
}


void BasicModel::computeBoundingBox()
{
    int i;
    Vector3 minPosition;
    Vector3 maxPosition;

    if (m_boundingBox != NULL) {
        delete m_boundingBox;
        m_boundingBox = NULL;
    }

    minPosition = m_vertexArray[0];
    maxPosition = m_vertexArray[0];

    // compute min/max values of vertices along x, y, z axes
    for (i = 1; i < m_vertexArray.size(); i++) {
        int j;
        for (j = 0; j < 3; j++) {
            minPosition[j] = min(minPosition[j], m_vertexArray[i][j]);
            maxPosition[j] = max(maxPosition[j], m_vertexArray[i][j]);
        }
    }

    m_boundingBox = new Box(minPosition, maxPosition);
}


void BasicModel::computeBoundingSphere()
{
    int i;
    Vector3 center(0, 0, 0);

    if (m_boundingSphere != NULL) {
        delete m_boundingSphere;
        m_boundingSphere = NULL;
    }

    // average points to get approximate center
    for (i = 0; i < m_vertexArray.size(); i++) {
        center += m_vertexArray[i];
    }
    center /= m_vertexArray.size();

    // find maximum distance from center (sphere radius)
    float maxDistance = -1.0;
    for (i = 0; i < m_vertexArray.size(); i++) {
        Vector3 centerToVertex = m_vertexArray[i] - center;
        maxDistance = max(maxDistance, centerToVertex.squaredLength());
    }
    maxDistance = sqrt(maxDistance);

    m_boundingSphere = new Sphere(center, maxDistance);
}


void BasicModel::compact()
{
    int i;
    int j;
    Array<int> oldToNewVectorMap(m_vertexArray.size());
    Table<Vector3, int> newVertexTable;
    Array<Vector3> newVertexArray;

    for (i = 0; i < m_vertexArray.size(); i++) {
        const Vector3& oldVector = m_vertexArray[i];
        if (newVertexTable.containsKey(oldVector)) {
            oldToNewVectorMap[i] = newVertexTable[oldVector];
        } else {
            int curIndex = newVertexArray.size();
            newVertexArray.push(oldVector);
            oldToNewVectorMap[i] = curIndex;
            newVertexTable.set(oldVector, curIndex);
        }
    }

    Array<BasicFace> newFaceArray;
    for (i = 0; i < m_faceArray.size(); i++) {
        const BasicFace& oldFace = m_faceArray[i];
        BasicFace newFace;
        for (j = 0; j < 3; j++) {
            newFace.vertexIndices[j] =
                oldToNewVectorMap[oldFace.vertexIndices[j]];
        }

        if ((newFace.vertexIndices[0] != newFace.vertexIndices[1]) &&
            (newFace.vertexIndices[1] != newFace.vertexIndices[2]) &&
            (newFace.vertexIndices[2] != newFace.vertexIndices[0])) {

            newFaceArray.push(newFace);
        }
    }


    m_vertexArray = newVertexArray;
    m_faceArray = newFaceArray;
}


void BasicModel::splitFaces(
        const Array<BasicFace>&     oldFaceArray,
        const Array<Vector3>&       oldVertexArray,
        Array<BasicFace>&           newFaceArray,
        Array<Vector3>&             newVertexArray)
{
    /* vertices are referred to as such:
     *           2
     *       5       4
     *    0      3      1
     */
    static const int faceIndices[4][3] = { { 0, 3, 5 },
                                           { 3, 4, 5 },
                                           { 3, 1, 4 },
                                           { 5, 4, 2 } };

    static const float newVertexWeights[3][3] = { { 0.5, 0.5, 0   },
                                                  { 0,   0.5, 0.5 },
                                                  { 0.5, 0,   0.5 } };

    // copy all of the old vertices into the new vertex array
    newVertexArray = oldVertexArray;

    int i;
    int j;
    int k;
    int m;
    for (i = 0; i < oldFaceArray.size(); i++) {
        const BasicFace& oldFace = oldFaceArray[i];
        int newVertexIndices[6] = { oldFace.vertexIndices[0],
                                    oldFace.vertexIndices[1],
                                    oldFace.vertexIndices[2],
                                    newVertexArray.size() + 0,
                                    newVertexArray.size() + 1,
                                    newVertexArray.size() + 2 };
        // add 3 new vertices
        for (j = 0; j < 3; j++) {
            Vector3 newVertex = Vector3::ZERO;
            // for each old vertex
            for (k = 0; k < 3; k++) {
                // average in one element 
                for (m = 0; m < 3; m++) {
                    newVertex[m] += oldVertexArray[newVertexIndices[k]][m] *
                        newVertexWeights[j][k];
                }
            }
            newVertexArray.push(newVertex);
        }

        // add 4 new faces
        for (j = 0; j < 4; j++) {
            BasicFace newFace;
            for (k = 0; k < 3; k++) {
                newFace.vertexIndices[k] = newVertexIndices[faceIndices[j][k]];
            }
            newFaceArray.push(newFace);
        }
    }
}

void BasicModel::retesselateFaces(
        int                         numLevels)
{
    int i;
    for (i = 0; i < numLevels; i++) {
        Array<BasicFace>    newFaceArray;
        Array<Vector3>      newVertexArray;

        splitFaces(m_faceArray, m_vertexArray, newFaceArray, newVertexArray);
        m_faceArray = newFaceArray;
        m_vertexArray = newVertexArray;
    }
}




void BasicModel::addEdge(
        int                         vertex0,
        int                         vertex1,
        int                         face,
        Table<EdgeHash, int>&       edgeTable,
        Array<BasicEdge>&           edgeArray,
        Array<Vector3>&             vertexArray)
{
    EdgeHash edgeKey(vertex0, vertex1);
    edgeKey.m_vertexArray = & vertexArray;

    if (edgeTable.containsKey(edgeKey)) {
        // if this is the second face referencing this edge
        int edgeIndex = edgeTable[edgeKey];
        BasicEdge& existingEdge = edgeArray[edgeIndex];

        // make sure the vertices are wound correctly
        // (ie in opposite directions for the two different faces)
        debugAssert(vertexArray[existingEdge.vertexIndices[0]].fuzzyEq(
            vertexArray[vertex1]));
        debugAssert(vertexArray[existingEdge.vertexIndices[1]].fuzzyEq(
            vertexArray[vertex0]));

        // make sure this is only the second face to reference this edge
        debugAssert(existingEdge.faceIndices[0] >= 0);
        debugAssert(existingEdge.faceIndices[0] != face);
        debugAssert(existingEdge.faceIndices[1] == -1);

        // set second face
        existingEdge.faceIndices[1] = face;
    } else {
        // if this is the first face referencing this edge
        BasicEdge newEdge;
        newEdge.vertexIndices[0] = vertex0;
        newEdge.vertexIndices[1] = vertex1;
        newEdge.faceIndices[0] = face;
        newEdge.faceIndices[1] = -1;

        // add edge to lookup table and array
        edgeTable.set(edgeKey, edgeArray.size());
        edgeArray.push(newEdge);
    }
}


/** compute edges in groups because sometimes (ie Quake 3)
 * the models are disconnected (leg, torso, head) and have seams 
 * along texture boundaries.  The seams we can deal with by
 * using fuzzyEq() to test if two points are equal.  However
 * if we use fuzzyEq() and then test the leg and the torso
 * we may have four triangles meeting at one edge which is
 * basically impossible to sort out.
 */
void BasicModel::computeEdges(
    int                         startFace,
    int                         endFace)
{
    int i;
    Table<EdgeHash, int> edgeTable;

    for (i = startFace; i < endFace; i++) {
        // make sure this is not a degenerate triangle
        // (ie the vertices are so close the triangle is extremely small)
        debugAssert(!m_vertexArray[m_faceArray[i].vertexIndices[0]].fuzzyEq(
            m_vertexArray[m_faceArray[i].vertexIndices[1]]));
        debugAssert(!m_vertexArray[m_faceArray[i].vertexIndices[1]].fuzzyEq(
            m_vertexArray[m_faceArray[i].vertexIndices[2]]));
        debugAssert(!m_vertexArray[m_faceArray[i].vertexIndices[2]].fuzzyEq(
            m_vertexArray[m_faceArray[i].vertexIndices[0]]));

        // add three edges for each face
        addEdge(m_faceArray[i].vertexIndices[0],
                m_faceArray[i].vertexIndices[1], i, 
                edgeTable, m_edgeArray, m_vertexArray);
        addEdge(m_faceArray[i].vertexIndices[1],
                m_faceArray[i].vertexIndices[2], i, 
                edgeTable, m_edgeArray, m_vertexArray);
        addEdge(m_faceArray[i].vertexIndices[2],
                m_faceArray[i].vertexIndices[0], i, 
                edgeTable, m_edgeArray, m_vertexArray);
    }

    // double check that every edge has been referenced by two faces
    for (i = 0; i < m_edgeArray.size(); i++) {
        debugAssert(m_edgeArray[i].faceIndices[0] >= 0);
        debugAssert(m_edgeArray[i].faceIndices[1] >= 0);
        debugAssert(m_edgeArray[i].vertexIndices[0] >= 0);
        debugAssert(m_edgeArray[i].vertexIndices[1] >= 0);
    }
}


void BasicModel::computeEdges()
{
    debugAssert(m_edgeArray.size() == 0);
    computeEdges(0, m_faceArray.size());
}


const Box& BasicModel::getBoundingBox() const
{
    return (*m_boundingBox);
}


bool BasicModel::doesCastShadow() const
{
    return m_castShadow;
}


void BasicModel::updateModel(int milliTime)
{
	// do nothing by default
}

void BasicModel::computeStaticFaceNormals(
    const Array<Vector3>&           vertexArray,
    Array<Vector3>&                 faceNormal) const
{

    faceNormal.resize(m_faceArray.size());

    for (int f = m_faceArray.size() - 1; f >= 0; --f) { 
        const int* index = m_faceArray[f].vertexIndices;

        const Vector3& A = vertexArray[index[0]];
        const Vector3& B = vertexArray[index[1]];
        const Vector3& C = vertexArray[index[2]];

        faceNormal[f] = (B - A).cross(C - A).fastDirection();
    }
}


void BasicModel::drawFaces(
        int&                        polyCount)
{
    glBegin(GL_TRIANGLES);
        int numFaces = m_faceArray.size();
        for(int g = 0; g < numFaces; ++g) {
            glNormal(m_faceNormalArray[g]);
            for (int w = 0; w < 3; ++w) {
                int i = m_faceArray[g].vertexIndices[w];
                glVertex(m_vertexArray[i]);
            }
        }
    glEnd();

    polyCount = numFaces;
}


void BasicModel::drawShadowVolume(
        const Vector4&              light,
        bool                        frontCap,
        bool                        extrusions,
        bool                        endCap,
        int&                        polyCount,
        bool                        shadowOptimization)
{
    polyCount = 0;

    if (m_extrusionDirty) {
        // extrude vertices

        // NOTE that for directional lights all vertices will be extruded
        // to the same point, so in that case you do not need to
        // calculate this array.  We do not implement this optimization.
        for (int v = 0; v < m_vertexArray.size(); ++v) {
            m_extrudedVertexArray[v] =
                directionToPoint(light, m_vertexArray[v]);
        }

        // find polygons that are backfacing to the light
        for (int f = 0; f < m_faceArray.size(); ++f) {
            Vector3& lightDirection =
                    m_extrudedVertexArray[m_faceArray[f].vertexIndices[0]];
            m_isBackfaceArray[f] = 
                (m_faceNormalArray[f].dot(lightDirection) > 0);
        }

        m_silhouetteEdgeArray.resize(0);
        for(int e = 0; e < m_edgeArray.size(); ++e) {
            if(m_isBackfaceArray[m_edgeArray[e].faceIndices[0]] ^
                m_isBackfaceArray[m_edgeArray[e].faceIndices[1]]) {

                m_silhouetteEdgeArray.append(e);
            }
        }
    }
    m_extrusionDirty = false;
   

    if (extrusions) {
        // find silhouette edges and extrude them into faces
        glBegin(GL_QUADS);
            
            for (int e = 0; e < m_silhouetteEdgeArray.size(); ++e) {
                int eIndex = m_silhouetteEdgeArray[e];
                //this is a silhouette edge.  winding direction is
                //given by which face is the backface.
                Vector4 reg0(m_vertexArray[
                        m_edgeArray[eIndex].vertexIndices[0]], 1);
                Vector4 reg1(m_vertexArray[
                        m_edgeArray[eIndex].vertexIndices[1]], 1);
                Vector4 ext0(m_extrudedVertexArray[
                        m_edgeArray[eIndex].vertexIndices[0]], 0);
                Vector4 ext1(m_extrudedVertexArray[
                        m_edgeArray[eIndex].vertexIndices[1]], 0);

                if(m_isBackfaceArray[m_edgeArray[eIndex].faceIndices[0]]) {
                    glVertex(reg0);
                    glVertex(reg1);
                    glVertex(ext1);
                    glVertex(ext0);
                } else {
                    glVertex(reg1);
                    glVertex(reg0);
                    glVertex(ext0);
                    glVertex(ext1);
                }

                polyCount += 2;
            }
            
        glEnd();
    }


    // draw front cap
    if(frontCap) {
        glBegin(GL_TRIANGLES);
            for(int g = 0; g < m_faceArray.size(); ++g) {
                if(!m_isBackfaceArray[g]) {
                    for (int w = 0; w < 3; ++w) {
                        int i = m_faceArray[g].vertexIndices[w];
                        glVertex(m_vertexArray[i]);
                    }

                    polyCount += 1;
                }
            }
        glEnd();
    }

    bool useBackCapOptimization;
    if (!shadowOptimization) {
        useBackCapOptimization = false;
    } else if (isDirectionalLight(light)) {
        // we can actually skip the back cap entirely
        // in this case
        useBackCapOptimization = true;
    } else {
        // if we have a point light and we are doing shadow
        // optimizations then we can use the fan optimization
        // if the point light is outside of the bounding sphere
        // of the model, (the OpenGL specifications do not require
        // correct rendering of triangles at inifinity that span
        // larger then 180 degrees)
        Vector3 lightPos = vector4to3(light);

        // light and lightPos are in object space
        useBackCapOptimization =
            ((lightPos - m_boundingSphere->center).length() > m_boundingSphere->radius);
    }

    // draw back cap
    // if we are using a directional light the end cap
    // will be infinitely small and we don't have to draw it
    if(endCap && !isDirectionalLight(light)) {

       // draw back cap
       if (useBackCapOptimization) {
            // optimized triangle fan for back cap
            Vector4 firstPoint = Vector4(m_extrudedVertexArray[m_edgeArray[0].vertexIndices[0]], 0);
            glBegin(GL_TRIANGLES);
                for (int e = 1; e < m_silhouetteEdgeArray.size(); ++e) {
                    int eIndex = m_silhouetteEdgeArray[e];
                    int vIndex0 = m_edgeArray[eIndex].vertexIndices[0];
                    int vIndex1 = m_edgeArray[eIndex].vertexIndices[1];
                    glVertex(firstPoint);
                    if (m_isBackfaceArray[m_edgeArray[eIndex].faceIndices[0]]) {
                        glVertex(Vector4(m_extrudedVertexArray[vIndex0], 0));
                        glVertex(Vector4(m_extrudedVertexArray[vIndex1], 0));
                    } else {
                        glVertex(Vector4(m_extrudedVertexArray[vIndex1], 0));
                        glVertex(Vector4(m_extrudedVertexArray[vIndex0], 0));
                    }
                    polyCount += 1;
                }
            glEnd();
        } else {
            // unoptimized back faces for back cap
            glBegin(GL_TRIANGLES);
            for(int g = 0; g < m_faceArray.size(); ++g) {
                if(m_isBackfaceArray[g]) {
                    for (int w = 0; w < 3; ++w) {
                        int i = m_faceArray[g].vertexIndices[w];
                        glVertex(Vector4(m_extrudedVertexArray[i], 0));
                    }

                    polyCount += 1;
                }
            }
            glEnd();
        }
    }
}

