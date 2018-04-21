/**
  @file Renderer.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#ifndef _RENDERER_H_
#define _RENDERER_H_

#include <graphics3D.h>



class BasicCamera;
class Viewport;
class BasicModel;
struct DemoSettings;


/**
 * the different types of polygon counts maintained by the program
 */
enum ePolyCount {
    POLY_COUNT_TOTAL = 0,
    POLY_COUNT_VISIBLE,
    POLY_COUNT_NUM_CATEGORIES
};

enum eShadowMethod {
    SHADOW_NONE = 0,
    SHADOW_Z_PASS,
    SHADOW_Z_FAIL
};



bool checkOcclusionCull(
        const Box&                          box,
        const BasicCamera&                  camera);



/**
 * increment the polygon count for a specific type of count
 * (such as POLY_COUNT_VISIBLE)
 */
void incrementPolyCount(
        int                                 count,
        ePolyCount                          type);


/**
 * get the current polygon count for a specific type of count
 */
int getPolyCount(
        ePolyCount                          type);


/**
 * set the polygon count to 0 for a specific type of count
 */
void resetPolyCount(
        ePolyCount                          type);


/**
 * this returns a Vector3 parameter which merely ignores the
 * fourth element of the vec passed in
 */
Vector3 vector4to3(
        const Vector4&                      vec);


/**
 * wrapper around Light::staticIsDirectionalLight()
 */
bool isDirectionalLight(
        const Vector4&                      light);


/**
 * Given a light position (which may either be a point light or directional
 * light depending on the homogeneous coordinate), return a vector
 * that goes from the light source to the vertex
 */
Vector3 directionToPoint(
        const Vector4&                      light,
        const Vector3&                      vertex);

/**
 * fills the outPoints array with eight corners of the cube
 * defined with lightPos and lightRadius
 */
void getLightCubeBoundingPoints(
        const Vector3&                      lightPos,
        double                              lightRadius,
        const Array<Vector3>&               outPoints);


/**
 * returns a (x, y, z) point in clip space (all points are homogenized)
 * based on the projection matrix
 */
Vector3 getScreenPoint(
        const Vector3&                      point,
        const BasicCamera&                  camera,
        const Viewport&                     view);


/**
 * Return x, y, width, height of screen bounding rectangle for world
 * space points.  All points should be in front of near clip plane.
 */
void getScreenBoundingRectangle(
        const Array<Vector3>&               boundingPoints,
        const BasicCamera&                       camera,
        const double*                       viewMatrix,
        int                                 screenWidth,
        int                                 screenHeight,
        int&                                outX,
        int&                                outY,
        int&                                outWidth,
        int&                                outHeight);


/**
 * Makes axis-aligned cube of bounding points around light.
 * Projects points to be in front of near clip plane.
 */
void getScreenBoundingRectangle(
        const Vector3&                      lightPos,
        double                              lightRadius,
        const BasicCamera&                       camera,
        const Viewport&                     view,
        int                                 screenWidth,
        int                                 screenHeight,
        int&                                outX,
        int&                                outY,
        int&                                outWidth,
        int&                                outHeight);

/**
 * used to project points to near clip plane for
 * getScreenBoundingRectangle()
 */
void getNearClipPlane(
        const BasicCamera&                       camera,
        const Viewport&                     view,
        Plane&                              outPlane);


/**
 * This method tests if the view volume is conservatively
 * shadowed by the model.  It is conservative in a fashion
 * where it may return a false positive result (ie you will
 * have to do the slower z-fail method when you mnight not need to).
 * The light parameter is the light position in world space.
 * The last four parameters are the four corners of the viewport
 * in object space.
 */
bool viewportMaybeShadowed(
        const BasicCamera&                       camera,
        const BasicModel&                   model,
        const Vector4&                      light,
        const Vector3&                      ur,
        const Vector3&                      lr,
        const Vector3&                      ll,
        const Vector3&                      ul);


/**
 * whether or not model receives illumination
 */
bool receivesIllumination(
        const BasicModel&                   model,
        const Vector4&                      light,
        float                               lightRadius,
        bool                                lightAttenuation);


/**
 * wrapper around camera::get3DViewportCorners
 */
void getViewportCorners(
        const BasicCamera&                       camera,
        const Viewport&                     view,
        Vector3&                            outUR,
        Vector3&                            outLR,
        Vector3&                            outLL,
        Vector3&                            outUL);


/**
 * make planes used for frustum culling
 */
void makeFrustumPlanes(
        const Vector3&                      eyePosition,
        const Vector3&                      ur,
        const Vector3&                      lr,
        const Vector3&                      ll,
        const Vector3&                      ul,
        Array<Plane>&                       outPlanes);


/**
 * see if all bounding points are culled by any one single plane,
 * generalization of Box::culledBy()
 */
bool pointsCulled(
        const Array<Vector4>&               boundingPoints,
        const Array<Plane>&                 planes);


/**
 * called for each light once per frame
 */
bool lightInFrustum(
        const Vector4&                      lightPos,
        double                              lightRadius,
        const BasicCamera&                       camera,
        const Viewport&                     view,
        bool                                lightAttenuation,
        bool                                shadowOptimizations);

/**
 * called for each model once per light pass
 */
void modelInFrustum(
        const BasicModel&                   model,
        const Array<Plane>&                 frustumPlanes,
        const Vector4&                      light,
        bool                                shadowOptimizations,
        bool&                               outModelInFrustum,
        bool&                               outExtrusionInFrustum,
        bool&                               outProjectionInFrustum);


/**
 * This returns which shadowing method should be used for a specific model
 */
eShadowMethod calculateShadowMethod(
        const BasicModel&                   model,
        const BasicCamera&                       camera,
        const Vector4&                      light,
        float                               lightRadius,
        bool                                lightAttenuation,
        bool                                shadowOptimizations,
        Viewport&                           view);


eShadowMethod calculateShadowMethod(
        const BasicModel&                   model,
        const BasicCamera&                       camera,
        const Vector4&                      light,
        float                               lightRadius,
        bool                                lightAttenuation,
        bool                                shadowOptimizations,
        const Vector3&                      ur,
        const Vector3&                      lr,
        const Vector3&                      ll,
        const Vector3&                      ul);

/**
 * This method is faster because we do not have to render front or
 * end caps, (the two false values passed to renderShadowVolume)
 * This (or zPassShadow) is called for each shadow-casting
 * model during each shadow pass.
 */
void zPassShadow(
        BasicModel&                         model,
        const Vector4&                      light,
        int&                                polyCount,
        bool                                frontCapInFrustum,
        bool                                extrusionInFrustum,
        bool                                backCapInFrustum,
        bool                                shadowOptimization);


/**
 * This method is slower because we need to render front and
 * end caps, (the two true values passed to renderShadowVolume).
 * This (or zPassShadow) is called for each shadow-casting
 * model during each shadow pass.
 */
void zFailShadow(
        BasicModel&                         model,
        const Vector4&                      light,
        int&                                polyCount,
        bool                                frontCapInFrustum,
        bool                                extrusionInFrustum,
        bool                                backCapInFrustum,
        bool                                shadowOptimization);


/**
 * In this pass ambient light contribution for all models is written
 * to the frame buffer, and the z-buffer is is also set (it should
 * not need to be modified after this stage).  This happens
 * once per frame.
 */
void ambientPass(
        const Array<BasicModel*>&           modelArray, 
        const BasicCamera&                       camera,
        const Viewport&                     view,
        bool                                shadowOptimizations);


/**
 * This pass renders shadow volume geometry into the stencil buffer
 * for one light.  This happens once per light pass.
 */
void shadowPass(
        const Array<BasicModel*>&           modelArray,
        const BasicCamera&                       camera,
        const Viewport&                     view,
        const Vector4&                      light,
        float                               lightRadius,
        DemoSettings&                       vars);


/**
 * This pass adds the contribution of every model for one light to
 * the framebuffer, it uses the stencil buffer initialized in shadowPass()
 * to disregard any light contribution for pixels that are in shadow.
 * This happens once per light pass.
 */
void illuminationPass(
        const Array<BasicModel*>&           modelArray,
        const BasicCamera&                       camera,
        const Viewport&                     view,
        const Vector4&                      light,
        bool                                shadowOptimizations);


#endif

