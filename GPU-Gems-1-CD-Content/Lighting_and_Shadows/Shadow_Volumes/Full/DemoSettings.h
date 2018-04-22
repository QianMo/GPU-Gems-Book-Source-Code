/**
  @file Renderer.h

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/

#ifndef _DEMO_SETTINGS_H_
#define _DEMO_SETTINGS_H_

#include <G3DAll.h>
#include "Light.h"
#include "BasicModel.h"
#include "BasicCamera.h"
#include "Viewport.h"

struct DemoSettings
{
    bool            m_endProgram;
    bool            m_drawShadowVolumes;
    bool            m_highlightOccluders;
    bool            m_wireframeMode;
    bool            m_drawShadows;
    bool            m_volumeTransparent;

    bool            m_shadowOptimizations;

    bool            m_cameraForward;
    bool            m_cameraBack;
    bool            m_cameraLeft;
    bool            m_cameraRight;

    int             m_directionX, m_directionZ;

    bool            m_lightTransXUp;
    bool            m_lightTransXDown;
    bool            m_lightTransZUp;
    bool            m_lightTransZDown;
    bool            m_lightTransYUp;
    bool            m_lightTransYDown;
    bool            m_modelAngleUp;
    bool            m_modelAngleDown;

    int             m_mouseX, m_mouseY;

    int             m_winWidth, m_winHeight;

    double          m_modelAngle;

    bool            m_lightAttenuation;
    int             m_lightModify;
    Array<Light>    m_lightArray;
    Array<bool>     m_lightAtCameraArray;
    double          m_lightAreaChange;
    double          m_lightAreaRadius;

    int*            m_timeArray;
    int             m_curTimeIndex;
    int             m_numTimeSamples;

    bool            m_animateModels;

    bool            m_occlusionCull;
};


/**
 * initialize DemoSettings struct
 */
void initializeSettings(
        DemoSettings&                       vars);

/**
 * set all mutable attributes to their default settings
 */
void setSettingsDefault(
        DemoSettings&                       vars);


/**
 * set rotation of models to be identity
 */
void setSceneDefault(
        Array<BasicModel*>&                 modelArray);


/**
 * function pointer typedef for scenePresetN() functions
 */
typedef void (*presetPointer)(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);

void scenePreset1(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);

void scenePreset2(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);

void scenePreset3(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);

void scenePreset4(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);

void scenePreset5(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);


/**
 * set a few GL parameters that will be constant through the program
 */
void demoInitializeGL(
        DemoSettings&                       vars,
        Viewport&                           viewport);


/**
 * test to see if the application has focus
 */
bool appHasFocus();



/**
 * load the non-model textures that we need
 */
void loadTextures(
        const std::string&                  baseDir,
        TextureRef&                         fontTexture,
        Array<TextureRef>&                  skyboxArray);

/**
 * add cubes in various positions (making a cube of cubes)
 */
void addCubes(
        Array<BasicModel*>&                 modelArray,
        const std::string&                  filename);


/**
 * called at start of program
 */
void initializeScene(
        const std::string&                  baseDir,
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);

/**
 * called once per frame
 */
void handleEvents(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);


/**
 * called once per frame
 */
void updateScene(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray);


/**
 * reset the GL state at the beginning of every frame, this
 * includes clearing the frame buffer and depth buffer 
 */
void resetGL(
        DemoSettings&                       vars,
        BasicCamera&                             camera);


/**
 * set GL lighting to be a soft ambient light
 */
void setAmbientLight();


/**
 * set GL lighting according to parameters
 */
void setLight(
        Light&                      light,
        Vector4&                    lightPosition,
        Color4&                     lightColor,
        bool                        useAttenuation,
        BasicCamera&                     camera);



/**
 * directional area lights are simulated by many passes of directional
 * lights each with a perturbed direction.  This assumes that the light
 * is pointing somewhat along the y-axis because only x and z are
 * perturbed.  The passNum should be a number from 0 to 8 which specifies
 * which pass is being applied (we do this currently does 9 passes).
 * The newPosition parameter is set by this method.
 */
void setLightPass(
        int                         lightNum,
        int                         passNum,
        int                         numPasses,
        Vector4&                    newPosition,
        BasicCamera&                     camera,
        DemoSettings&               vars);


/**
 * Called if we are rendering with more then one light pass,
 * or if the light source has moved.  In these cases
 * the extruded data the models last computed is almost
 * definitely not right.
 */
void dirtyAllExtrusions(
         Array<BasicModel*>&        modelArray);

/**
 * called on program exit
 */
void deleteAllModels(
         Array<BasicModel*>&        modelArray);


#endif


