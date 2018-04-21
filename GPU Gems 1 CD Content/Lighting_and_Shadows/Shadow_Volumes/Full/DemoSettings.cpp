/**
  @file DemoSettings.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/


#include "DemoSettings.h"
#include <SDL.h>
#include "Effects.h"
#include "loaders/Quake3Md3.h"
#include "loaders/Quake3Bsp.h"
#include "loaders/SMLoader.h"


void initializeSettings(
        DemoSettings&                       vars)
{
    int i;
    
    setSettingsDefault(vars);

    vars.m_winWidth                 = 800;
    vars.m_winHeight                = 680;

    vars.m_curTimeIndex             = 0;
    vars.m_numTimeSamples           = 20;

    // initialize time samplings to record 1 fps
    vars.m_timeArray = new int[vars.m_numTimeSamples];
    vars.m_timeArray[0] = SDL_GetTicks();
    for (i = 1; i < vars.m_numTimeSamples; i++) {
        vars.m_timeArray[i] = vars.m_timeArray[0] -
            (vars.m_numTimeSamples - i) * 1000;
    }
}


void setSettingsDefault(
        DemoSettings&                       vars)
{
    vars.m_endProgram               = false;
    vars.m_drawShadowVolumes        = false;
    vars.m_highlightOccluders       = false;
    vars.m_wireframeMode            = false;
    vars.m_drawShadows              = true;
    vars.m_volumeTransparent        = true;

    vars.m_shadowOptimizations      = true;

    vars.m_cameraForward            = false;
    vars.m_cameraBack               = false;
    vars.m_cameraLeft               = false;
    vars.m_cameraRight              = false;

    vars.m_directionX               = 0;
    vars.m_directionZ               = 0;

    vars.m_lightTransXUp            = false;
    vars.m_lightTransXDown          = false;
    vars.m_lightTransZUp            = false;
    vars.m_lightTransZDown          = false;
    vars.m_lightTransYUp            = false;
    vars.m_lightTransYDown          = false;
    vars.m_modelAngleUp             = false;
    vars.m_modelAngleDown           = false;

    vars.m_mouseX                   = vars.m_winWidth / 2;
    vars.m_mouseY                   = vars.m_winHeight / 2;

    vars.m_modelAngle               = 0.0;

    vars.m_lightModify              = 0;
    vars.m_lightArray.resize(5);
    vars.m_lightAtCameraArray.resize(vars.m_lightArray.size());

    vars.m_lightAttenuation         = true;
    vars.m_lightArray[0].m_color    = Color4(1,   1,   1);
    vars.m_lightArray[1].m_color    = Color4(1,   0,   0);
    vars.m_lightArray[2].m_color    = Color4(0,   1,   0);
    vars.m_lightArray[3].m_color    = Color4(0,   0,   1);
    vars.m_lightArray[4].m_color    = Color4(1,   1,   1);

    for (int i = 0; i < vars.m_lightArray.size(); ++i) {
        vars.m_lightArray[i].m_position = Vector4(0, 15, 15, 1);
        vars.m_lightArray[i].m_on   = false;
        vars.m_lightArray[i].m_areaLight = false;
        vars.m_lightArray[i].m_attenuation = Vector3(0, 0.005, 0.007);
        vars.m_lightArray[i].calculateRadius(0.1);

        vars.m_lightAtCameraArray[i] = false;
    }

    // simulated sun in last light
    vars.m_lightArray[vars.m_lightArray.size() - 1].m_position =
        Vector4(-70, 20, -40, 0);
    vars.m_lightArray[0].m_on       = true;

    vars.m_lightAreaChange          = 0.0;
    vars.m_lightAreaRadius          = 0.5;

    vars.m_animateModels            = true;

    vars.m_occlusionCull            = false;
}


void setSceneDefault(
        Array<BasicModel*>&                 modelArray)
{
    int i;

    dirtyAllExtrusions(modelArray);
    for (i = 0; i < modelArray.size(); i++) {
        modelArray[i]->m_transformation.rotation = Matrix3::IDENTITY;
    }
}


void scenePreset1(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    setSettingsDefault(vars);
    setSceneDefault(modelArray);
    
    camera.orient(Vector3(15, 2, 20), Vector3(-0.787, 0, -0.354).direction());
}


void scenePreset2(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    setSettingsDefault(vars);
    setSceneDefault(modelArray);
    
    vars.m_lightArray[0].m_position = Vector4(0, 17, -65, 1);
    camera.orient(Vector3(15, 15, -45), Vector3(-0.450, 0.030, -0.892));
}


void scenePreset3(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    setSettingsDefault(vars);
    setSceneDefault(modelArray);

    vars.m_drawShadows = false;
    vars.m_drawShadowVolumes = true;
    
    camera.orient(Vector3(225, 67, 137), Vector3(-0.812, -0.305, -0.497));
}


void scenePreset4(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    setSettingsDefault(vars);
    setSceneDefault(modelArray);
    
    vars.m_lightArray[0].m_position = Vector4(0, 40, -40, 0);
    camera.orient(Vector3(17, 55, -122), Vector3(-0.322, -0.433, 0.842));
}


void scenePreset5(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    scenePreset1(vars, camera, modelArray);

    vars.m_lightArray[1].m_position = Vector4( 5, 10, 20, 1);
    vars.m_lightArray[2].m_position = Vector4(-5, 10, 20, 1);

    vars.m_lightArray[1].m_on = true;
    vars.m_lightArray[2].m_on = true;
}


void demoInitializeGL(
        DemoSettings&                       vars,
        Viewport&                           viewport)
{
    GLfloat mat_specular[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat mat_shininess[] = { 25.0 };

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    
    glActiveTextureARB(GL_TEXTURE0_ARB);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glClearStencil(0);
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    viewport.setInfiniteFrustum();

    glMatrixMode(GL_MODELVIEW);

    SDL_ShowCursor(0);
}


bool appHasFocus()
{
    // taken from the UserInput class in G3D
    unsigned int s = SDL_GetAppState();

    return ((s & SDL_APPMOUSEFOCUS) != 0) &&
           ((s & SDL_APPINPUTFOCUS) != 0) &&
           ((s & SDL_APPACTIVE) != 0);
}



void loadTextures(
        const std::string&                  baseDir,
        TextureRef&                         fontTexture,
        Array<TextureRef>&                  skyboxArray)
{
	int i;

    // load sky box textures
    std::string skyDir = baseDir + std::string("data/skybox/");
    char* ext[] = { "bk", "lt", "ft", "rt", "tp", "dn" };
    for (i = 0; i < 6; i++) {
        //skyboxArray[i] = new Texture(ext[i], skyDir + ext[i] + ".jpg",
        //        "", Texture::BILINEAR_NO_MIPMAP, Texture::CLAMP);
        skyboxArray[i] = Texture::fromFile(skyDir + ext[i] + ".jpg",
			TextureFormat::AUTO, Texture::CLAMP, Texture::BILINEAR_NO_MIPMAP);
    }

    // load font texture
    fontTexture = loadFontTexture("data/font/font.jpg");
}


void addCubes(
         Array<BasicModel*>&        modelArray,
         const std::string&         filename)
{
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            for (int z = 0; z < 5; z++) {
                BasicModel* newModel = importSMFile(filename);
                debugAssert(newModel != NULL);
                newModel->m_transformation.translation =
                    Vector3(-5 * 3 / 2.0, 5, -90) + Vector3(x, y, z) * 3;
                modelArray.push(newModel);
            }
        }
    }
}


void initializeScene(
        const std::string&                  baseDir,
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    int i;

    // load cathedral model and add it to scene
    std::string filename = baseDir + "data/models/cathedral.sm";
    BasicModel* newModel = importSMFile(filename);
    debugAssert(newModel != NULL);
    newModel->m_transformation.translation = Vector3::ZERO;
    modelArray.push(newModel);
    
    static float tickHeight[6] = { 0.9, -0.25, -1.2, -0.5, -0.4, -0.2 };
    
    // load tick model and add it to scene
    for (i = 0; i < 6; i++) {
        modelArray.push(loadQuake3Md3("default", baseDir +
                std::string("data/thetick/"), i * 7));
        modelArray.last()->m_transformation.translation =
            Vector3(0, tickHeight[i], 10 * i - 2);
    }

    // load cubes and add them to scene
    filename = baseDir + "data/models/cube.sm";
    addCubes(modelArray, filename);

    scenePreset1(vars, camera, modelArray);

}


void handleEvents(
        DemoSettings&                       vars,
        BasicCamera&                             camera,
        Array<BasicModel*>&                 modelArray)
{
    // Event handling
    SDL_Event event;

    static const presetPointer presetArray[5] =
    { scenePreset1, scenePreset2, scenePreset3, scenePreset4, scenePreset5 };

    // test if the application has focus
    if (!appHasFocus()) {
        SDL_ShowCursor(SDL_ENABLE);
    } else {
        SDL_ShowCursor(SDL_DISABLE);
    }

    while (SDL_PollEvent(&event))
    {

        //screenshot vars
        CImage* im;
        int i = 0;
        std::string filename;

        switch(event.type) {
        case SDL_QUIT:
            vars.m_endProgram = true;
            break;
        case SDL_KEYUP:
            switch(event.key.keysym.sym) {
            case SDLK_LEFT:
			case SDLK_a:
                vars.m_cameraLeft = false;
                break;
            case SDLK_RIGHT:
			case SDLK_d:
                vars.m_cameraRight = false;
                break;
            case SDLK_UP:
			case SDLK_w:
                vars.m_cameraForward = false;
                break;
            case SDLK_DOWN:
			case SDLK_s:
                vars.m_cameraBack = false;
                break;
            case SDLK_j:
                vars.m_lightTransXDown = false;
                break;
            case SDLK_l:
                vars.m_lightTransXUp = false;
                break;
            case SDLK_i:
                vars.m_lightTransZDown = false;
                break;
            case SDLK_k:
                vars.m_lightTransZUp = false;
                break;
            case SDLK_y:
                vars.m_lightTransYUp = false;
                break;
            case SDLK_h:
                vars.m_lightTransYDown = false;
                break;
            case SDLK_m:
                vars.m_modelAngleUp = false;
                break;
            case SDLK_n:
                vars.m_modelAngleDown = false;
                break;                                                        
            case SDLK_e:
            case SDLK_r:
                vars.m_lightAreaChange = 0.0;
                break;
            }
            break;

        case SDL_KEYDOWN:
            switch (event.key.keysym.sym) {
            case SDLK_ESCAPE:
            case SDLK_q:
                vars.m_endProgram = true;
                break;
            case SDLK_EQUALS:

                // Read back the front buffer
                glReadBuffer(GL_FRONT);

                im = new CImage(vars.m_winWidth, vars.m_winHeight);
                glReadPixels(0, 0, vars.m_winWidth, vars.m_winHeight,
                        GL_RGB, GL_UNSIGNED_BYTE, im->byte());

                // Flip right side up
                flipRGBVertical(im->byte(), im->byte(),
                        vars.m_winWidth, vars.m_winHeight);

                // Restore the read buffer to the back
                glReadBuffer(GL_BACK);

                // Save the file
                createDirectory("screenshots"); 
                do {
                    filename = std::string("screenshots/") + "screenshot_" + 
                        format("%03d", i) + ".jpg";
                    ++i;
                } while (fileExists(filename)); 
                im->save(filename);
                delete im;
                break;
            case SDLK_v:
                vars.m_drawShadowVolumes = !vars.m_drawShadowVolumes;
                break;
            case SDLK_t:
                vars.m_volumeTransparent = !vars.m_volumeTransparent;
                break;
            case SDLK_f:
                vars.m_wireframeMode = !vars.m_wireframeMode;
                break;
            case SDLK_z:
                vars.m_drawShadows = !vars.m_drawShadows;
                break;
            case SDLK_p:
                vars.m_animateModels = !vars.m_animateModels;
                break;
            case SDLK_o:
                vars.m_highlightOccluders = !vars.m_highlightOccluders;
                break;
            case SDLK_x:
                vars.m_shadowOptimizations = !vars.m_shadowOptimizations;
                break;
            case SDLK_0:
            case SDLK_1:
            case SDLK_2:
            case SDLK_3:
            case SDLK_4:
                vars.m_lightModify = event.key.keysym.sym - SDLK_0;
                break;
            case SDLK_5:
            case SDLK_6:
            case SDLK_7:
            case SDLK_8:
            case SDLK_9:
                // call one of 5 present functions (for example: scenePreset1)
                (presetArray[event.key.keysym.sym - SDLK_5])(vars,
                             camera, modelArray);
                break;
            case SDLK_u:
                vars.m_lightAttenuation = !vars.m_lightAttenuation;
                dirtyAllExtrusions(modelArray);
                break;
            case SDLK_c:
                vars.m_lightAtCameraArray[vars.m_lightModify] =
                    !vars.m_lightAtCameraArray[vars.m_lightModify];
                break;
            case SDLK_b:
                vars.m_lightArray[vars.m_lightModify].m_areaLight =
                    !vars.m_lightArray[vars.m_lightModify].m_areaLight;
                dirtyAllExtrusions(modelArray);
                break;
            case SDLK_g:
                vars.m_lightArray[vars.m_lightModify].m_position[3] =
                    1.0 - vars.m_lightArray[vars.m_lightModify].m_position[3];
                dirtyAllExtrusions(modelArray);
                break;
            case SDLK_e:
                vars.m_lightAreaChange = -0.02;
                dirtyAllExtrusions(modelArray);
                break;
            case SDLK_r:
                vars.m_lightAreaChange = 0.02;
                dirtyAllExtrusions(modelArray);
                break;
            case SDLK_SPACE:
                vars.m_lightArray[vars.m_lightModify].m_on =
                    !vars.m_lightArray[vars.m_lightModify].m_on;
                // if we switch to a new light that has the same
                // effect of moving the current light source,
                // all cached extrusions are invalid
                dirtyAllExtrusions(modelArray);
                break;
            case SDLK_LEFT:
			case SDLK_a:
                vars.m_cameraLeft = true;
                break;
            case SDLK_RIGHT:
			case SDLK_d:
                vars.m_cameraRight = true;
                break;
            case SDLK_UP:
			case SDLK_w:
                vars.m_cameraForward = true;
                break;
            case SDLK_DOWN:
			case SDLK_s:
                vars.m_cameraBack = true;
                break;
            case SDLK_j:
                vars.m_lightTransXDown = true;
                break;
            case SDLK_l:
                vars.m_lightTransXUp = true;
                break;
            case SDLK_i:
                vars.m_lightTransZDown = true;
                break;
            case SDLK_k:
                vars.m_lightTransZUp = true;
                break;
            case SDLK_y:
                vars.m_lightTransYUp = true;
                break;
            case SDLK_h:
                vars.m_lightTransYDown = true;
                break;
            case SDLK_m:
                vars.m_modelAngleUp = true;
                break;
            case SDLK_n:
                vars.m_modelAngleDown = true;
                break;
            }
            break;
        }
    }
}


void updateScene(
        DemoSettings&                           vars,
        BasicCamera&                                 camera,
        Array<BasicModel*>&                     modelArray)
{
    int i;
    Matrix3 rotationMatrix(Matrix3::IDENTITY);

    if(vars.m_cameraForward && !vars.m_cameraBack) {
        vars.m_directionZ = 1;
    }
    if(vars.m_cameraBack && !vars.m_cameraForward) {
        vars.m_directionZ = -1;
    }
    if(vars.m_cameraLeft && !vars.m_cameraRight) {
        vars.m_directionX = -1;
    }
    if(vars.m_cameraRight && !vars.m_cameraLeft) {
        vars.m_directionX = 1;
    }

    if(vars.m_lightTransXUp) {
        vars.m_lightArray[vars.m_lightModify].m_position[0]++;
    }
    if(vars.m_lightTransXDown) {
        vars.m_lightArray[vars.m_lightModify].m_position[0]--;
    }
    if(vars.m_lightTransYUp) {
        vars.m_lightArray[vars.m_lightModify].m_position[1]++;
    }
    if(vars.m_lightTransYDown) {
        vars.m_lightArray[vars.m_lightModify].m_position[1]--;
    }
    if(vars.m_lightTransZUp) {
        vars.m_lightArray[vars.m_lightModify].m_position[2]++;
    }
    if(vars.m_lightTransZDown) {
        vars.m_lightArray[vars.m_lightModify].m_position[2]--;
    }

    if(vars.m_modelAngleUp) {
        vars.m_modelAngle++;
    }
    if(vars.m_modelAngleDown) {
        vars.m_modelAngle--;
    }

    // if the light moves cached extrusions are not valid
    if (vars.m_lightTransXUp || vars.m_lightTransXDown ||
        vars.m_lightTransYUp || vars.m_lightTransYDown ||
        vars.m_lightTransZUp || vars.m_lightTransZDown ||
        vars.m_modelAngleUp  || vars.m_modelAngleDown) {

        dirtyAllExtrusions(modelArray);
    }

    SDL_GetMouseState(&vars.m_mouseX, &vars.m_mouseY);

    camera.updateCamera(vars.m_directionX, vars.m_directionZ,
            vars.m_mouseX, vars.m_mouseY);
    if (vars.m_lightAtCameraArray[vars.m_lightModify]) {
        dirtyAllExtrusions(modelArray);
        Vector3 position = camera.m_transformation.translation +
            camera.getViewVector() * 3;
        vars.m_lightArray[vars.m_lightModify].m_position = Vector4(position, 1);
    }

    
    rotationMatrix.fromAxisAngle(Vector3(0.0, 0.0, 1.0),
        toRadians(vars.m_modelAngle));

    for (i = 0; i < modelArray.size(); i++) {
        modelArray[i]->m_transformation.rotation = rotationMatrix;
    }

    vars.m_lightAreaRadius += vars.m_lightAreaChange;

    vars.m_directionX = 0;
    vars.m_directionZ = 0;

}


void resetGL(
        DemoSettings&                           vars,
        BasicCamera&                                 camera)
{
    debugAssert(glGetError() == GL_NO_ERROR);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // stencil buffer is normally cleared every shadow pass
    // but if we aren't drawing shadows then we should clear it here
    if (!vars.m_drawShadows) {
        glClear(GL_STENCIL_BUFFER_BIT);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (vars.m_wireframeMode) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glDisable(GL_STENCIL_TEST);
    glEnable(GL_DEPTH_TEST);

}


void setAmbientLight()
{
    GLfloat ambient_on[] = { 0.2, 0.2, 0.2, 1.0 };

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // turn on ambient lighting
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_on);

    // turn off diffuse lighting
    GLfloat diffuse_off[] = { 0.0, 0.0, 0.0, 1.0 };
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_off);
}


void setLight(
        Light&                      light,
        Vector4&                    lightPosition,
        Color4&                     lightColor,
        bool                        useAttenuation,
        BasicCamera&                     camera)
{
    GLfloat vec[4];
    GLfloat ambient_off[] = { 0.0, 0.0, 0.0, 1.0 };
    int i;

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // turn off ambient lighting
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_off);

    // set position
    glLoadMatrix(camera.getWorldToCamera());
    for (i = 0; i < 4; i++) {
        vec[i] = lightPosition[i];
    }
    glLightfv(GL_LIGHT0, GL_POSITION, vec);

    // set diffuse color
    for (i = 0; i < 4; i++) {
        vec[i] = lightColor[i];
    }
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec);

    // set light attenuation
    if (useAttenuation) {
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, light.m_attenuation[0]);
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, light.m_attenuation[1]);
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, light.m_attenuation[2]);
    } else {
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1);
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0);
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0);
    }
}


void setLightPass(
        int                         lightNum,
        int                         passNum,
        int                         numPasses,
        Vector4&                    newPosition,
        BasicCamera&                     camera,
        DemoSettings&               vars)
{
    Vector3 offset;
    double radius = vars.m_lightAreaRadius;
    switch (passNum) {
        case 0: offset = Vector3(0,         0,          0); break;

        case 1: offset = Vector3(-radius,   -radius,    0); break;
        case 2: offset = Vector3(-radius,   radius,     0); break;
        case 3: offset = Vector3(radius,    -radius,    0); break;
        case 4: offset = Vector3(radius,    radius,     0); break;
    }

    newPosition = Vector4(offset[0] +
        vars.m_lightArray[lightNum].m_position[0],
        offset[1] + vars.m_lightArray[lightNum].m_position[1],
        offset[2] + vars.m_lightArray[lightNum].m_position[2],
        vars.m_lightArray[lightNum].m_position[3]);
    Color4 newColor(vars.m_lightArray[lightNum].m_color[0] / numPasses,
        vars.m_lightArray[lightNum].m_color[1] / numPasses,
        vars.m_lightArray[lightNum].m_color[2] / numPasses,
        vars.m_lightArray[lightNum].m_color[3]);
    setLight(vars.m_lightArray[lightNum], newPosition, newColor,
            vars.m_lightAttenuation, camera);
}



void dirtyAllExtrusions(
         Array<BasicModel*>&        modelArray)
{
    for (int i = 0; i < modelArray.size(); i++) {
        modelArray[i]->m_extrusionDirty = true;
    }
}




void deleteAllModels(
         Array<BasicModel*>&        modelArray)
{
    for (int i = 0; i < modelArray.size(); i++) {
        delete modelArray[i];
    }
}



