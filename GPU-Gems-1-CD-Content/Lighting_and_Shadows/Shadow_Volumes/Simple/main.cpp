/**
  @file main.cpp

  This demo shows how to structure the basic shadow volume algorithm.
  It employs none of the optimizations (except two-sided stencil testing)
  from the chapter and only works for directional lights.  Most of the
  details of model loading and rendering are actually implemented 
  inside the Open Source G3D library (which is linked to the demo) to
  keep them from cluttering the code.

  The adjacency code and shadow volume creation code are inside the library
  but are included in the directory as shadowVolume.cpp and MeshAlgAdjacency.cpp
  so the details are clear.  G3D is available under the BSD license so you
  can use that code directly in your own projects as long a small copyright
  statement appears in your documentation.

  @author Morgan McGuire, matrix@graphics3d.com
 */

#include "definitions.h"


void Demo::doGraphics() {

    // Capture the current geometry of the objects in the scene
    poseObjects();

    app->renderDevice->setProjectionAndCameraMatrix(app->debugCamera);

    // Clear the stencil and depth buffers and draw the background
    app->renderDevice->clear(true, true, true);
    app->sky->render(app->lighting);

    // Draw the objects
    app->renderDevice->enableLighting();

        ambientPass();

        markShadows();

        directionalPass();

    app->renderDevice->disableLighting();

    // Add a lens flare for the sun
    app->sky->renderLensFlare(app->lighting);
}


void Demo::ambientPass() {
    // Turn on ambient illumination
	app->renderDevice->setAmbientLightColor(app->lighting.ambient);

    // Turn off the directional light
	app->renderDevice->setLight(0, NULL);

    drawObjects();

    // Disable writing to the depth buffer; we don't need it after
    // this pass
    app->renderDevice->disableDepthWrite();
}


void Demo::markShadows() {
    // Create a directional light vector
    Vector4 L(app->lighting.lightDirection, 0);

    app->renderDevice->setDepthTest(RenderDevice::DEPTH_LESS);

    // Set up z-fail state (as described in the chapter)
    G3D::beginMarkShadows(app->renderDevice);
        for (int i = 0; i < modelArray.size(); ++i) {
            // Create and render shadow volumes for this object
            G3D::markShadows(app->renderDevice, modelArray[i], L);
        }
    G3D::endMarkShadows(app->renderDevice);
}


void Demo::directionalPass() {

    // Enable additive blending to sum the results of both passes
    app->renderDevice->setBlendFunc(RenderDevice::BLEND_ONE, RenderDevice::BLEND_ONE);

    // Only render where there are no shadows (stencil == 0)
    app->renderDevice->setStencilTest(RenderDevice::STENCIL_EQUAL);
    app->renderDevice->setDepthTest(RenderDevice::DEPTH_LEQUAL);

    // Disable the ambient illumination
    app->renderDevice->setAmbientLightColor(Color3::BLACK);

    // Turn on the light
	app->renderDevice->setLight(0, 
        GLight::directional(app->lighting.lightDirection, 
                            app->lighting.lightColor));

    drawObjects();
}

////////////////////////////////////////////////////////////////////////////////

Demo::Demo(App* _app) : GApplet(_app), app(_app) {
}


void Demo::doSimulation(SimTime dt) {
    app->charPose.doSimulation(dt,
        false, false, false, false,
        iRandom(0, 500) == 0, iRandom(0, 800) == 0, false, false,
        false, false, false, false,
        false, false, false, false);
}


void Demo::doLogic() {
    if (app->userInput->keyPressed(SDLK_ESCAPE)) {
        // Even when we aren't in debug mode, quit on escape.
        endApplet = true;
        app->endProgram = true;
    }
}


void Demo::poseObjects() {
    modelArray.clear();
    textureArray.clear();

    // The ground has no texture
    textureArray.append((TextureRef)NULL);
    modelArray.append(app->ground->pose(CoordinateFrame()));

    textureArray.append(app->charTexture);
    modelArray.append(app->character->pose(CoordinateFrame(Vector3(0,1.7,0)), app->charPose));
}


void Demo::drawObjects() {
    for (int i = 0; i < modelArray.size(); ++i) {
        app->renderDevice->setTexture(0, textureArray[i]);
        modelArray[i]->render(app->renderDevice);
    }
}




void App::main() {
    debugCamera.setPosition(Vector3(0, 2, -7));
    debugCamera.lookAt(Vector3(0, 2, 0));

    // Move the far clipping plane to infinity
    debugCamera.setFarPlaneZ(-inf);

	setDebugMode(true);
	debugController.setActive(false);

    // Load objects
    sky         = Sky::create(renderDevice, dataDir + "sky/");
    ground      = IFSModel::create(dataDir + "ifs/octagon.ifs", 50);
    character   = MD2Model::create(dataDir + "quake2/players/pknight/tris.md2");
    charTexture = Texture::fromFile(dataDir + "quake2/players/pknight/ctf_b.pcx",
        TextureFormat::AUTO, Texture::CLAMP, Texture::TRILINEAR_MIPMAP, Texture::DIM_2D, 2.0);

    Demo(this).run();
}


App::App(const GAppSettings& settings) : GApp(settings), lighting(G3D::toSeconds(11, 00, 00, AM)) {
    dataDir = "../g3d-6_00-b12/data/";
}


int main(int argc, char** argv) {
    GAppSettings settings;
    App(settings).run();
    return 0;
}
