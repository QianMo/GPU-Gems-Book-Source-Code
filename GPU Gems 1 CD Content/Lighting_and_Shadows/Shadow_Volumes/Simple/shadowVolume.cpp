/**
 @file shadowVolume.cpp
 
 @maintainer Morgan McGuire, morgan@graphics3d.com
 
 @created 2001-12-16
 @edited  2004-01-15
 */

#include "GLG3D/shadowVolume.h"
#include "GLG3D/VAR.h"

namespace G3D {

/**
 Used by begin/end/markShadows
 */
static bool inMarkShadows = false;

void beginMarkShadows(RenderDevice* renderDevice) {
    debugAssert(! inMarkShadows);
    inMarkShadows = true;

    renderDevice->pushState();
        // Render only to the stencil buffer.
        renderDevice->disableDepthWrite();
        renderDevice->disableColorWrite();
        app->renderDevice->setDepthTest(RenderDevice::DEPTH_LESS);

        // z-fail; decrement for front faces and increment
        // for back faces.
        if (renderDevice->supportsTwoSidedStencil()) {
            renderDevice->setCullFace(RenderDevice::CULL_NONE);
            renderDevice->setStencilOp(
                RenderDevice::STENCIL_KEEP,
                RenderDevice::STENCIL_DECR_WRAP,
                RenderDevice::STENCIL_KEEP,

                RenderDevice::STENCIL_KEEP,
                RenderDevice::STENCIL_INCR_WRAP,
                RenderDevice::STENCIL_KEEP);
        } else {

            // Front face pass
            renderDevice->setCullFace(RenderDevice::CULL_BACK);
            renderDevice->setStencilOp(
                RenderDevice::STENCIL_KEEP,
                RenderDevice::STENCIL_DECR_WRAP,
                RenderDevice::STENCIL_KEEP);
        }
}


void endMarkShadows(RenderDevice* renderDevice) {
    debugAssert(inMarkShadows);
    inMarkShadows = false;
    renderDevice->popState();
}


/**
 Copies the source array over the dest array,
 setting w = 1 for all points.
 Helper for markShadows.
 */
static void vertexCopy(
    const Array<Vector3>& _source,
    Array<Vector4>&       _dest) {

    alwaysAssertM(_source.size() <= _dest.size(), 
        "Destination array must be at least as large as source");

    // Extract the underlying pointers for speed
    const Vector3* source = _source.getCArray();
    Vector4*       dest   = _dest.getCArray();

    for (int v = _source.size() - 1; v >= 0; --v) {
        dest[v].x = source[v].x;
        dest[v].y = source[v].y;
        dest[v].z = source[v].z;
        dest[v].w = 1;
    }
}


/**
 Helper for markShadows.  Extrudes source to infinity
 away from L.
 */
static void vertexExtrude(
    const Array<Vector3>&   _source,
    Array<Vector4>&         _dest,
    int                     shift,
    const Vector4&          L) {

    debugAssert(L.w == 1.0);
    
    // Extract arrays for efficiency
    const Vector3* source = _source.getCArray();
    Vector4*       dest   = _dest.getCArray() + shift;

    for (int i = _source.size() - 1; i >= 0; --i) {
        dest[i].x = source[i].x - L.x;
        dest[i].y = source[i].y - L.y;
        dest[i].z = source[i].z - L.z;
        dest[i].w = 0.0;
    }
}


void markShadows(
    RenderDevice*           renderDevice, 
    const PosedModelRef&    model,
    const Vector4&          light) {

    debugAssertM(inMarkShadows, "Must call beginMarkShadows before markShadows");

    if (model->numBrokenEdges() > 0) {
        // Can't cast shadows for such an object
        return;
    }

    // TODO: point light

    // Move to object space
    CoordinateFrame cframe;
    model->getCoordinateFrame(cframe);

    const Vector4 L = cframe.toObjectSpace(light);

    // Get the relevant object space mesh.
    // This is faster than getting world space
    // geometry for most PosedModel objects.

    const MeshAlg::Geometry& geometry = model->objectSpaceGeometry();
    const Array<Vector3>& vertexArray = geometry.vertexArray;
    const Array<MeshAlg::Edge>& edgeArray = model->edges();
    const Array<MeshAlg::Face>& faceArray = model->faces();

    Array<bool> backface;
    MeshAlg::identifyBackfaces(vertexArray, faceArray, L, backface);

    bool directional = (light.w == 0);

    int numPts;
    int n = vertexArray.size();
        
    // Directional lights need all of the vertices plus one at
    // infinity.  Point lights need a full copy of the object
    // at infinity.
    if (directional) {
        numPts = vertexArray.size() + 1;
    } else {
        numPts = vertexArray.size() * 2;
    }

    // Create an array of float4 for use on the graphics card.
    Array<Vector4> cpuVertex(numPts);
    vertexCopy(vertexArray, cpuVertex);

    if (directional) {
        cpuVertex[n] = -L;
    } else {
        // Extrude to infinity
        vertexExtrude(vertexArray, cpuVertex, n, L);
    }

    // Upload to graphics card
    VARAreaRef varArea = VARArea::create(cpuVertex.size() * sizeof(Vector4));
    VAR gpuVertex(cpuVertex, varArea);

    // Triangle list indices
    static Array<int> index;
    index.resize(0, DONT_SHRINK_UNDERLYING_ARRAY);

    // Shadow volume sides
    for (int e = edgeArray.size() - 1; e >= 0; --e) {
        const MeshAlg::Edge& edge = edgeArray[e];
        if (backface[edge.faceIndex[0]] != backface[edge.faceIndex[1]]) {
            // Silhouette edge

            int i0 = edge.vertexIndex[0];
            int i1 = edge.vertexIndex[1];

            // Wind in the direction of the backface
            if (directional) {
                // Triangle
                if (backface[edge.faceIndex[0]]) {
                    index.append(i0, i1, n);
                } else {
                    index.append(n, i1, i0);
                }
            } else {
                // Quad
                if (backface[edge.faceIndex[0]]) {
                    index.append(i0, i1, i1 + n);
                    index.append(i0, i1 + n, i0 + n);
                } else {
                    index.append(i1 + n, i1, i0);
                    index.append(i0 + n, i1 + n, i0);
                }
            }
        }
    }

    // Shadow volume caps
    for (int f = faceArray.size() - 1; f >= 0; --f) {
        const MeshAlg::Face& face = faceArray[f];
        if (! backface[f]) {
            index.append(face.vertexIndex[0], face.vertexIndex[1], face.vertexIndex[2]);
        } else if (! directional) {
            // Point light requires dark cap as well 
            index.append(face.vertexIndex[0] + n, face.vertexIndex[1] + n, face.vertexIndex[2] + n);
        }
    }

    renderDevice->setObjectToWorldMatrix(cframe);
    renderDevice->beginIndexedPrimitives();
        renderDevice->setVertexArray(gpuVertex);
        renderDevice->sendIndices(RenderDevice::TRIANGLES, index);

        if (! renderDevice->supportsTwoSidedStencil()) {
            // Render a second pass for the back faces
            renderDevice->setCullFace(RenderDevice::CULL_FRONT);
            renderDevice->setStencilOp(
                RenderDevice::STENCIL_KEEP,
                RenderDevice::STENCIL_INCR_WRAP,
                RenderDevice::STENCIL_KEEP);

            renderDevice->sendIndices(RenderDevice::TRIANGLES, index);

            renderDevice->setCullFace(RenderDevice::CULL_BACK);
            renderDevice->setStencilOp(
                RenderDevice::STENCIL_KEEP,
                RenderDevice::STENCIL_DECR_WRAP,
                RenderDevice::STENCIL_KEEP);
        }

    renderDevice->endIndexedPrimitives();

}


}
