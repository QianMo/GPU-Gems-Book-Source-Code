/**
  @file Renderer.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/


#include "BasicModel.h"
#include "Viewport.h"
#include "BasicCamera.h"
#include "DemoSettings.h"
#include "Renderer.h"
#include "Light.h"


static int s_polygonCount[POLY_COUNT_NUM_CATEGORIES];


void incrementPolyCount(
        int                                 count,
        ePolyCount                          type)
{
    s_polygonCount[type] += count;
}


int getPolyCount(
        ePolyCount                          type)
{
    return s_polygonCount[type];
}


void resetPolyCount(
        ePolyCount                          type)
{
    s_polygonCount[type] = 0;
}


Vector3 vector4to3(
        const Vector4&                      vec)
{
    return Vector3(vec[0], vec[1], vec[2]);
}

bool isDirectionalLight(
        const Vector4&                      light)
{
    return Light::staticIsDirectionalLight(light);
}


Vector3 directionToPoint(
        const Vector4&                      light,
        const Vector3&                      vertex)
{
    Vector3 light3 = vector4to3(light);
    if (isDirectionalLight(light)) {
        return -light3;
    } else {
        return (vertex - light3);
    }
}




bool checkOcclusionCull(
        const Box &                         box,
        const BasicCamera&                  camera)
{
#ifndef _NVIDIA_EXTENSIONS_
    // always returned not culled
    return false;
#else

    Vector3 eye = camera.m_transformation.translation;

    // if we are inside of the bounding box then we
    // cannot cull the geometry
    if (box.contains(eye)) {
        return false;
    }

    unsigned int queryID;
    unsigned int pixelCount;

    // make a query identifier and start put
    // all rendering in the occlusion query state
    glGenOcclusionQueriesNV(1, & queryID);
    glBeginOcclusionQueryNV(queryID);

    // only render faces with normals pointing towards us
    // other faces will point away and rendering them is wasteful
    for (int i = 0; i < 6; i++) {
        Vector3 p1;
        Vector3 p2;
        Vector3 p3;
        Vector3 p4;

        box.getFaceCorners(i, p1, p2, p3, p4);
        Plane curFace(p1, p2, p3);
        if (curFace.halfSpaceContains(eye)) {
            glBegin(GL_QUADS);
                glVertex(p1);
                glVertex(p2);
                glVertex(p3);
                glVertex(p4);
            glEnd();
        }
    }

    // end the occlusion query state
    glEndOcclusionQueryNV();

    // query as to whether any of the stuff we
    // rendered passed the depth buffer
    glGetOcclusionQueryuivNV(queryID,
        GL_PIXEL_COUNT_NV, & pixelCount);

    return (pixelCount == 0);
#endif // _NVIDIA_EXTENSIONS_
}




Vector3 getClipPoint(
        const Vector3&                      point,
        const CoordinateFrame&              objectToCamera,
        const double*                       cameraToClip)
{
    Vector3 cameraPoint = objectToCamera.pointToWorldSpace(point);
    Vector4 clipPoint;

    // the viewing matrix is a 4x4 matrix which doesn't have
    // any simple form in the graphics3d library so we do the
    // the matrix multiplication out manually
    int i;
    for (i = 0; i < 4; i++) {
        clipPoint[i] = cameraToClip[i + 0] * cameraPoint[0];
        clipPoint[i] += cameraToClip[i + 4] * cameraPoint[1];
        clipPoint[i] += cameraToClip[i + 8] * cameraPoint[2];
        clipPoint[i] += cameraToClip[i + 12] * 1.0;
    }

    clipPoint /= clipPoint[3];
    return clipPoint;
}


void getLightCubeBoundingPoints(
        const Vector3&                      lightPos,
        double                              lightRadius,
        Array<Vector3>&                     outPoints)
{
    int i;

    for (i = 0; i < 8; i++) {
        // add corners of bounding cube that have coords of pos +/-
        // radius, this axis-aligned cube would match cube-maps
        // used for per-pixel lighting
        float x = lightPos[0] + lightRadius * (((i / 1) % 2) ? (-1) : (1));
        float y = lightPos[1] + lightRadius * (((i / 2) % 2) ? (-1) : (1));
        float z = lightPos[2] + lightRadius * (((i / 4) % 2) ? (-1) : (1));
        outPoints.push(Vector3(x, y, z));
    }
}


void getScreenBoundingRectangle(
        const Array<Vector3>&               boundingPoints,
        const BasicCamera&                  camera,
        const double*                       viewMatrix,
        int                                 screenWidth,
        int                                 screenHeight,
        int&                                outX,
        int&                                outY,
        int&                                outWidth,
        int&                                outHeight)
{
    int i;
    
    const CoordinateFrame& transform = camera.getWorldToCamera();

    int maxX;
    int maxY;

    for (i = 0; i < boundingPoints.size(); i++) {
        Vector3 clipPoint = getClipPoint(boundingPoints[i],
            transform, viewMatrix);

        // all points should be projected in front of near clip plane
        debugAssert(clipPoint[2] >= -1.0);
        float newX = (clipPoint[0] + 1.0) * screenWidth / 2.0;
        float newY = (clipPoint[1] + 1.0) * screenHeight / 2.0;
        
        if (i == 0) {
            outX = floor(newX);
            outY = floor(newY);
            maxX = ceil(newX);
            maxY = ceil(newY);
        } else {
            outX = min(outX, floor(newX));
            outY = min(outY, floor(newY));
            maxX = max(maxX, ceil(newX));
            maxY = max(maxY, ceil(newY));
        }
    }

    outWidth = maxX - outX;
    outHeight = maxY - outY;
}


void getScreenBoundingRectangle(
        const Vector3&                      lightPos,
        double                              lightRadius,
        const BasicCamera&                  camera,
        const Viewport&                     view,
        int                                 screenWidth,
        int                                 screenHeight,
        int&                                outX,
        int&                                outY,
        int&                                outWidth,
        int&                                outHeight)
{
    Array<Vector3> boundingPoints;

    debugAssert(lightRadius > 0);
    
    static const int edgeIndices[12][2] =
    { { 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 },   // plane (z=-radius) edges
      { 4, 5 }, { 5, 7 }, { 7, 6 }, { 6, 4 },   // plane (z=radius) edges
      { 0, 4 }, { 1, 5 }, { 3, 7 }, { 2, 6 }    // across z planes
    };

    getLightCubeBoundingPoints(lightPos, lightRadius, boundingPoints);

    /* vertices... shown looking along the negative z-axis, this
     * shows vertices on the z=radius and z=-radius planes
     *
     * -z     +z
     * 2 3    6 7
     * 0 1    4 5
     *
     * all coordinates are in world space (light position is already
     * in world space)
     */
    int i;
    for (i = 0; i < 8; i++) {
        // add corners of bounding cube that have coords of pos +/-
        // radius, this axis-aligned cube would match cube-maps
        // used for per-pixel lighting
        float x = lightPos[0] + lightRadius * (((i / 1) % 2) ? (-1) : (1));
        float y = lightPos[1] + lightRadius * (((i / 2) % 2) ? (-1) : (1));
        float z = lightPos[2] + lightRadius * (((i / 4) % 2) ? (-1) : (1));
        boundingPoints.push(Vector3(x, y, z));
    }
    
    Vector3 ur, lr, ll, ul;
    getViewportCorners(camera, view, ur, lr, ll, ul);

    // we want to project points a little in front of the near plane
    // so that we can be absolutely sure they won't be clipped due to
    // rounding error
    double x, y, z;
    view.getNearXYZ(x, y, z);
    Vector3 lookVector = camera.getViewVector();
    Vector3 clipOffset = lookVector * z * 1.0001;
    Plane nearClipPlane(ul + clipOffset, ur + clipOffset, lr + clipOffset);

    // add any points that are already on correct side of near clip plane
    Array<Vector3> projectedPoints;
    for (i = 0; i < 8; i++) {
        if (nearClipPlane.halfSpaceContains(boundingPoints[i])) {
            projectedPoints.push(boundingPoints[i]);
        }
    }

    // project any points along edges into near clipping plane
    for (i = 0; i < 12; i++) {
        int index1 = edgeIndices[i][0];
        int index2 = edgeIndices[i][1];

        // if one or the other but not both points of an edge on
        // the cube is behind the near clip plane, then we want
        // to intersect the edge line segment with the near clip plane
        if ((nearClipPlane.halfSpaceContains(boundingPoints[index1])) ^
            (nearClipPlane.halfSpaceContains(boundingPoints[index2]))) {

            Vector3 edge = boundingPoints[index2] - boundingPoints[index1];
            Vector3 planeNormal;
            float planeDistance;
            float tLine;
            nearClipPlane.getEquation(planeNormal, planeDistance);
            tLine = -(planeNormal.dot(boundingPoints[index1]) + planeDistance) /
                planeNormal.dot(edge);

            projectedPoints.push(boundingPoints[index1] + edge * tLine);
        }
    }
    
    if (projectedPoints.size() > 0) {
        double viewMatrix[16];
        view.getInfiniteFrustumMatrix(viewMatrix);

        getScreenBoundingRectangle(projectedPoints, camera, viewMatrix,
            screenWidth, screenHeight, outX, outY, outWidth, outHeight);
    } else {
        // all points are behind near clip plane, doesn't matter what
        // scissor region is
        outX = 0;
        outY = 0;
        outWidth = 0;
        outHeight = 0;
    }
}


bool viewportMaybeShadowed(
        const BasicCamera&                  camera,
        const BasicModel&                   model,
        const Vector4&                      light,
        const Vector3&                      ur,
        const Vector3&                      lr,
        const Vector3&                      ll,
        const Vector3&                      ul)
{
    Array<Plane> occlusionPyramid(5);

    // all operations are in object space
    const CoordinateFrame& transform = model.m_transformation;
    Vector3 objectUR, objectLR, objectLL, objectUL;
    objectUR = transform.pointToObjectSpace(ur);
    objectUL = transform.pointToObjectSpace(ul);
    objectLL = transform.pointToObjectSpace(ll);
    objectLR = transform.pointToObjectSpace(lr);

    occlusionPyramid[0] = Plane(objectUL, objectUR, objectLR);

    double k = (isDirectionalLight(light)) ? (1) : (0);

    Vector4 objectLight = transform.toObjectSpace(light);
    Vector3 light3 = vector4to3(objectLight);
    occlusionPyramid[1] = Plane(objectUR * k + light3, objectUR, objectUL);
    occlusionPyramid[2] = Plane(objectLR * k + light3, objectLR, objectUR);
    occlusionPyramid[3] = Plane(objectLL * k + light3, objectLL, objectLR);
    occlusionPyramid[4] = Plane(objectUL * k + light3, objectUL, objectLL);

    Vector3 viewCenter = (objectUR + objectLR + objectLL + objectUL) / 4.0;
    Vector3 lightDirection = directionToPoint(objectLight, viewCenter);

    if (!isDirectionalLight(objectLight)) {
        // we can put an extra culling plane right behind the light
        occlusionPyramid.append(Plane(lightDirection, light3));
    }

    Vector3 objectViewVector =
        transform.vectorToObjectSpace(camera.getViewVector());
    if (lightDirection.dot(objectViewVector) > 0) {
        // light is behind us, flip all occlusion planes
        for (int i = 0; i < occlusionPyramid.size(); ++i) {
            occlusionPyramid[i].flip();
        }
    }

    Box bounds = model.getBoundingBox();
    return ! bounds.culledBy(occlusionPyramid);
}


bool receivesIllumination(
        const BasicModel&                   model,
        const Vector4&                      light,
        float                               lightRadius,
        bool                                lightAttenuation)
{
    if (!isDirectionalLight(light) && (lightRadius > 0) &&
            lightAttenuation) {
        Vector3 vec3 = vector4to3(light);
        // test if object is outside of point light bounding sphere
        Vector3 lightToObject =
            model.m_transformation.pointToObjectSpace(vec3) -
            model.m_boundingSphere->center;
        float distance = lightToObject.length();
        if (distance > model.m_boundingSphere->radius + lightRadius) {
            // light does not reach the object
            return false;
        }
    }

    return true;
}


void getViewportCorners(
        const BasicCamera&                  camera,
        const Viewport&                     view,
        Vector3&                            outUR,
        Vector3&                            outLR,
        Vector3&                            outLL,
        Vector3&                            outUL)
{
    double x, y, z;
    view.getNearXYZ(x, y, z);
    camera.get3DViewportCorners(x, y, z, outUR, outLR, outLL, outUL);
}


void makeFrustumPlanes(
        const Vector3&                      eyePosition,
        const Vector3&                      ur,
        const Vector3&                      lr,
        const Vector3&                      ll,
        const Vector3&                      ul,
        Array<Plane>&                       outPlanes)
{
    outPlanes.push(Plane(ul, ur, lr));
    outPlanes.push(Plane(eyePosition, ur, lr));
    outPlanes.push(Plane(eyePosition, lr, ll));
    outPlanes.push(Plane(eyePosition, ll, ul));
    outPlanes.push(Plane(eyePosition, ul, ur));
}


bool pointsCulled(
        const Array<Vector4>&               boundingPoints,
        const Array<Plane>&                 planes)
{

    // See if there is one plane for which all
    // of the vertices are on the wrong side
    for (int p = 0; p < planes.size(); p++) {
        bool culled = true;
        int v = 0;

        // Assume this plane culls all points.  See if there is a point
        // not culled by the plane.
        while ((v < boundingPoints.size()) && culled) {
            // this dot product formulation works for both homogeneous values
            // of 1 and 0
            Vector4 planeVec;
            planes[p].getEquation(planeVec[0], planeVec[1], planeVec[2],
                    planeVec[3]);
            culled = (planeVec.dot(boundingPoints[v]) < 0);
            v++;
        }

        if (culled) {
            // This plane culled the box
            return true;
        }
    }

    // None of the planes could cull this box
    return false;
}


bool lightInFrustum(
        const Vector4&                      lightPos,
        double                              lightRadius,
        const BasicCamera&                  camera,
        const Viewport&                     view,
        bool                                lightAttenuation,
        bool                                shadowOptimizations)
{
    Vector3 ur;
    Vector3 lr;
    Vector3 ll;
    Vector3 ul;
    Array<Plane> planes;
    Array<Vector3> boundingPoints3;
    Array<Vector4> boundingPoints4;

    if (!shadowOptimizations || isDirectionalLight(lightPos) ||
           !(lightRadius >= 0)) {
        return true;
    }

    getViewportCorners(camera, view, ur, lr, ll, ul);
    makeFrustumPlanes(camera.m_transformation.translation,
        ur, lr, ll, ul, planes);

    getLightCubeBoundingPoints(vector4to3(lightPos), lightRadius,
        boundingPoints3);

    for (int i = 0; i < boundingPoints3.size(); i++) {
        boundingPoints4.append(Vector4(boundingPoints3[i], 1));
    }
    return !pointsCulled(boundingPoints4, planes);
}



void modelInFrustum(
        const BasicModel&                   model,
        const Array<Plane>&                 frustumPlanes,
        const Vector4&                      light,
        bool                                shadowOptimizations,
        bool&                               outModelInFrustum,
        bool&                               outExtrusionInFrustum,
        bool&                               outProjectionInFrustum)
{
    int i;

    if (!shadowOptimizations) {
        outModelInFrustum = true;
        outExtrusionInFrustum = true;
        outProjectionInFrustum = true;
        return;
    }

    // all operations are in world space (because frustum planes
    // are in world space)
    Array<Vector4> modelBounds;
    const Box& box = (*model.m_boundingBox);

    // check if model is in frustum
    for (i = 0; i < 8; i++) {
        const CoordinateFrame& transformation = model.m_transformation;
        modelBounds.push(
            Vector4(transformation.pointToWorldSpace(box.getCorner(i)), 1));
    }
    outModelInFrustum = !pointsCulled(modelBounds, frustumPlanes);

    // check if all of the models infinitely projected vertices are
    // in the frustum
    Array<Vector4> projectionBounds;
    for (i = 0; i < 8; i++) {
        projectionBounds.push(
            Vector4(directionToPoint(light, vector4to3(modelBounds[i])), 0));
    }
    outProjectionInFrustum = !pointsCulled(projectionBounds, frustumPlanes);

    // check if all of the models extruded quads are in the frustum
    for (i = 0; i < 8; i++) {
        modelBounds.push(projectionBounds[i]);
    }
    outExtrusionInFrustum = !pointsCulled(modelBounds, frustumPlanes);
}


eShadowMethod calculateShadowMethod(
        const BasicModel&                   model,
        const BasicCamera&                  camera,
        const Vector4&                      light,
        float                               lightRadius,
        bool                                lightAttenuation,
        bool                                shadowOptimizations,
        const Vector3&                      ur,
        const Vector3&                      lr,
        const Vector3&                      ll,
        const Vector3&                      ul)
{

    if (model.m_castShadow == false) {
        // this object casts no shadow
        return SHADOW_NONE;
    }

    if (shadowOptimizations &&
        (!receivesIllumination(model, light, lightRadius, lightAttenuation))) {
        // no shadow necessary, light does not reach the object
        // so everything behind the object will get no light
        // whether or not we cast a shadow
        return SHADOW_NONE;
    }

    if (shadowOptimizations &&
        (!viewportMaybeShadowed(camera, model, light, ur, lr, ll, ul))) {
        // we can do the faster z-pass method
        return SHADOW_Z_PASS;
    } else {
        // we need to do the slower z-fail method
        return SHADOW_Z_FAIL;
    }
}


eShadowMethod calculateShadowMethod(
        const BasicModel&                   model,
        const BasicCamera&                  camera,
        const Vector4&                      light,
        float                               lightRadius,
        bool                                lightAttenuation,
        bool                                shadowOptimizations,
        const Viewport&                     view)
{
    Vector3 ur;
    Vector3 lr;
    Vector3 ll;
    Vector3 ul;

    getViewportCorners(camera, view, ur, lr, ll, ul);

    return calculateShadowMethod(model, camera, light,
        lightRadius, lightAttenuation, shadowOptimizations,
        ur, lr, ll, ul);
}


void zPassShadow(
        BasicModel&                         model,
        const Vector4&                      light,
        int&                                polyCount,
        bool                                frontCapInFrustum,
        bool                                extrusionInFrustum,
        bool                                backCapInFrustum,
        bool                                shadowOptimization)
{
    glCullFace(GL_FRONT);
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR_WRAP_EXT);
    model.drawShadowVolume(model.m_transformation.toObjectSpace(light),
            false, extrusionInFrustum, false, polyCount, shadowOptimization);

    glCullFace(GL_BACK);
    glStencilOp(GL_KEEP, GL_KEEP, GL_DECR_WRAP_EXT);
    model.drawShadowVolume(model.m_transformation.toObjectSpace(light),
            false, extrusionInFrustum, false, polyCount, shadowOptimization);
}


void zFailShadow(
        BasicModel&                         model,
        const Vector4&                      light,
        int&                                polyCount,
        bool                                frontCapInFrustum,
        bool                                extrusionInFrustum,
        bool                                backCapInFrustum,
        bool                                shadowOptimization)
{
    glCullFace(GL_BACK);
    glStencilOp(GL_KEEP, GL_INCR_WRAP_EXT, GL_KEEP);
    model.drawShadowVolume(model.m_transformation.toObjectSpace(light),
            frontCapInFrustum, extrusionInFrustum, backCapInFrustum,
            polyCount, shadowOptimization);

    glCullFace(GL_FRONT);
    glStencilOp(GL_KEEP, GL_DECR_WRAP_EXT, GL_KEEP);
    model.drawShadowVolume(model.m_transformation.toObjectSpace(light),
            frontCapInFrustum, extrusionInFrustum, backCapInFrustum,
            polyCount, shadowOptimization);
}


void ambientPass(
        const Array<BasicModel*>&           modelArray, 
        const BasicCamera&                  camera,
        const Viewport&                     view,
        bool                                shadowOptimizations)
{

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LIGHT0);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glDepthMask(1);
    glColorMask(1, 1, 1, 1);
    glDepthFunc(GL_LESS);
    
    Vector3 ur, lr, ll, ul;
    Array<Plane> frustumPlanes;

    getViewportCorners(camera, view, ur, lr, ll, ul);
    makeFrustumPlanes(camera.m_transformation.translation,
            ur, lr, ll, ul, frustumPlanes);

    for(int i = 0; i < modelArray.size(); i++) {
        bool frontCapInFrustum, extrusionInFrustum, backCapInFrustum;
        Vector4 dummyLight;
        modelInFrustum(*modelArray[i], frustumPlanes, dummyLight,
            shadowOptimizations, frontCapInFrustum, extrusionInFrustum,
            backCapInFrustum);
        if (frontCapInFrustum) {
            int polyCount;
            glLoadMatrix(camera.getWorldToCamera() *
                modelArray[i]->m_transformation);
            glColor(modelArray[i]->m_modelColor);
            modelArray[i]->useTextures(true);
            modelArray[i]->drawFaces(polyCount);
            incrementPolyCount(polyCount, POLY_COUNT_TOTAL);
            incrementPolyCount(polyCount, POLY_COUNT_VISIBLE);
        }
    }
    glPopAttrib();

}


void shadowPass(
        const Array<BasicModel*>&               modelArray,
        const BasicCamera&                      camera,
        const Viewport&                         view,
        const Vector4&                          light,
        float                                   lightRadius,
        DemoSettings&                           vars)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glClear(GL_STENCIL_BUFFER_BIT);
    glEnable(GL_STENCIL_TEST);          // write to stencil buffer
    glStencilMask(0xff);                
    glStencilFunc(GL_ALWAYS, 0, 0xff);  // always pass stencil test
    glDepthMask(0);                     // do not write to z-buffer
    glEnable(GL_CULL_FACE);             // enable face culling
    glColorMask(0, 0, 0, 0);            // do not write to frame buffer
    glDepthFunc(GL_LESS);


    // calculate data for optimizations
    Vector3 ur, lr, ll, ul;
    Array<Plane> frustumPlanes;

    getViewportCorners(camera, view, ur, lr, ll, ul);
    makeFrustumPlanes(camera.m_transformation.translation,
            ur, lr, ll, ul, frustumPlanes);


    // scissor region optimization
    if (vars.m_shadowOptimizations && (!isDirectionalLight(light)) &&
            (lightRadius > 0) && vars.m_lightAttenuation) {
        int x, y, width, height;

        getScreenBoundingRectangle(vector4to3(light),
            lightRadius, camera, view, vars.m_winWidth, vars.m_winHeight,
            x, y, width, height);
        glEnable(GL_SCISSOR_TEST);
        glScissor(x, y, width, height);
    }

    glDisable(GL_SCISSOR_TEST);

    for(int i = 0 ; i < modelArray.size(); i++) {
        int polyCount = 0;
        bool frontCapInFrustum, extrusionInFrustum, backCapInFrustum;
        eShadowMethod method;

        glLoadMatrix(camera.getWorldToCamera() *
            modelArray[i]->m_transformation);
        modelArray[i]->useTextures(false);

        modelInFrustum(*modelArray[i], frustumPlanes, light,
            vars.m_shadowOptimizations, frontCapInFrustum,
            extrusionInFrustum, backCapInFrustum);

        method = calculateShadowMethod(*modelArray[i],
                camera, light, lightRadius, vars.m_lightAttenuation,
                vars.m_shadowOptimizations, ur, lr, ll, ul);
        switch (method) {
        case SHADOW_NONE:
            // do nothing
            break;
        case SHADOW_Z_PASS:
            zPassShadow(*modelArray[i], light, polyCount,
                frontCapInFrustum, extrusionInFrustum,
                backCapInFrustum, vars.m_shadowOptimizations);
            break;
        case SHADOW_Z_FAIL:
            zFailShadow(*modelArray[i], light, polyCount,
                frontCapInFrustum, extrusionInFrustum,
                backCapInFrustum, vars.m_shadowOptimizations);
            break;
        default:
            debugAssert(false);
            break;
        }

        incrementPolyCount(polyCount, POLY_COUNT_TOTAL);
    }
    glPopAttrib();
}


void illuminationPass(
        const Array<BasicModel*>&           modelArray,
        const BasicCamera&                  camera,
        const Viewport&                     view,
        const Vector4&                      light,
        bool                                shadowOptimizations)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glEnable(GL_STENCIL_TEST);          // read from stencil buffer
    glStencilMask(0);                   // do not write to stencil buffer
    glStencilFunc(GL_EQUAL, 0, 0xff);   // set stencil test function
    glDepthMask(0);                     // do not write to z-buffer
    glEnable(GL_CULL_FACE);             // enable face culling
    glCullFace(GL_BACK);
    glColorMask(1,1,1,1);
    glEnable(GL_BLEND);         // add light contribution to frame buffer
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_LIGHT0);        // light0 should be set to have the
                                // characteristics of the light
                                // we want to use for this pass

    Vector3 ur, lr, ll, ul;
    Array<Plane> frustumPlanes;

    getViewportCorners(camera, view, ur, lr, ll, ul);
    makeFrustumPlanes(camera.m_transformation.translation,
            ur, lr, ll, ul, frustumPlanes);

    for(int i = 0; i < modelArray.size(); i++) {
        bool frontCapInFrustum, extrusionInFrustum, backCapInFrustum;
        modelInFrustum(*modelArray[i], frustumPlanes, light,
            shadowOptimizations, frontCapInFrustum, extrusionInFrustum,
            backCapInFrustum);
        if (frontCapInFrustum) {
            int polyCount;
            glLoadMatrix(camera.getWorldToCamera() *
                modelArray[i]->m_transformation);
            glColor(modelArray[i]->m_modelColor);
            modelArray[i]->useTextures(true);
            modelArray[i]->drawFaces(polyCount);
            incrementPolyCount(polyCount, POLY_COUNT_TOTAL);
        }
    }
    glPopAttrib();
}



