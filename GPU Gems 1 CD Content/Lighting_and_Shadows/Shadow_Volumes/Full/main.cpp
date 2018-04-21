/**
  @file main.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block (smblock@cs.brown.edu)
  @cite Portions written by Peter Sibely (pgs@cs.brown.edu)
  @cite Portions written by Morgan McGuire (morgan@cs.brown.edu)

  See the README.txt file for information on camera controls and
  other tips on running the program.
  
  See the README for links to the necessary libraries and for all
  citations in one place.

  This program is based on the "Fast, Practical and Robust Shadows" paper by:

  Morgan McGuire  (Brown University, morgan@cs.brown.edu)
  John F. Hughes  (Brown University, jfh@cs.brown.edu)
  Kevin Egan      (Brown University, ktegan@cs.brown.edu)
  Mark Kilgard    (NVIDIA Corporation, mjk@nvidia.com)
  Cass Everitt    (NVIDIA Corporation, ceveritt@nvidia.com)

  (This demo uses G3D for initialization but uses OpenGL calls for
   most of the implementation to stay close to the code in the paper)
*/

#include <G3DAll.h>
#include <SDL.h>
#include "BasicModel.h"
#include "BasicCamera.h"
#include "Viewport.h"
#include "Renderer.h"
#include "DemoSettings.h"
#include "Effects.h"
#include "initGL.h"

#if G3D_VER != 60012
    #error Requires G3D 6.00 b12
#endif

int main(
        int                         argc,
        char**                      argv) {
    int i;
    int j;

    DemoSettings vars;
    Viewport viewport(800, 600, toRadians(50), 1.0);

    // set demo options to the default, initialize GL and
    // set it to a default state
    initializeSettings(vars);
    initGL(vars.m_winWidth, vars.m_winHeight);
    demoInitializeGL(vars, viewport);


    // base is the directory in which all sub-directories
    // (ie models, textures, skybox) are contained
    std::string baseDir = "./";

    // load textures
    TextureRef fontTexture;
    Array<TextureRef> skyboxArray(6);
    loadTextures(baseDir, fontTexture, skyboxArray);

    // initialize scene
    BasicCamera camera;
    Array<BasicModel*> modelArray;
    initializeScene(baseDir, vars, camera, modelArray);

    // main loop
    int spinCounter = 0;

    while (!vars.m_endProgram) {

        // record events, update scene, and reset GL
        handleEvents(vars, camera, modelArray);
        updateScene(vars, camera, modelArray);
        resetGL(vars, camera);

        // tell all models to update themselves (for animating models)
        int milliTime = SDL_GetTicks();
        if (vars.m_animateModels) {
            for (i = 0; i < modelArray.size(); i++) {
                modelArray[i]->updateModel(milliTime);
            }
        }

        // reset polygon counts
        resetPolyCount(POLY_COUNT_TOTAL);
        resetPolyCount(POLY_COUNT_VISIBLE);

        // draw sky box
        drawSkyBox(camera, skyboxArray);

        // do ambient pass
        setAmbientLight();
        ambientPass(modelArray, camera, viewport,
            vars.m_shadowOptimizations);

        int numLightsOn = 0;
        for (i = 0; i < vars.m_lightArray.size(); ++i) {
            if (vars.m_lightArray[i].m_on) {
                numLightsOn++;
            }
        }

        // do a separate pass for each light
        for (i = 0; i < vars.m_lightArray.size(); ++i) {
            if (vars.m_lightArray[i].m_on) {
                int numPasses;
                numPasses = (vars.m_lightArray[i].m_areaLight) ? (5) : (1);

                // do multiple passes per light if we have an area light
                for (j = 0; j < numPasses; j++) {
                    Vector4 newPosition;
                    setLightPass(i, j, numPasses, newPosition, camera, vars);

                    if (!lightInFrustum(newPosition,
                            vars.m_lightArray[i].m_radius,
                            camera, viewport, vars.m_lightAttenuation,
                            vars.m_shadowOptimizations)) {
                        // jump to the next light
                        continue;
                    }

                    if (!vars.m_shadowOptimizations ||
                            (numLightsOn > 1) || (numPasses > 1)) {
                        dirtyAllExtrusions(modelArray);
                    }

                    if (vars.m_drawShadows) {
                        // set stencil buffer
                        shadowPass(modelArray, camera, viewport, newPosition,
                                vars.m_lightArray[i].m_radius, vars);
                    }

                    // add light contribution to frame buffer
                    illuminationPass(modelArray, camera, viewport,
                            newPosition, vars.m_shadowOptimizations);
                }
            }
        }

        // draw any special effects that don't need shadows
        finalPass(modelArray, camera, viewport, fontTexture, vars);

        SDL_GL_SwapBuffers();
    }

    // release allocated memory
    delete[] vars.m_timeArray;
    deleteAllModels(modelArray);

    return 0;
}


