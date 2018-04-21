#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <G3DAll.h>

#if G3D_VER != 60012
    #error Requires G3D 6.00 b12
#endif

class App : public GApp {
protected:
    void main();
public:

    /** Background */
    SkyRef              sky;

    /** Something to cast shadows on */
    IFSModelRef         ground;

    /** A Quake2 character to cast a shadow */
    MD2ModelRef         character;
    TextureRef          charTexture;
    MD2Model::Pose      charPose;

    /** Directional and ambient lighting information */
    LightingParameters  lighting;

    App(const GAppSettings& settings);
};


class Demo : public GApplet {
public:
    class App*                  app;

    /** The poseObjects method collects
       the scene graph into these two arrays */
    Array<PosedModelRef>        modelArray;
    Array<TextureRef>           textureArray;

    Demo(App* app);    

    virtual void doLogic();

    /** Animates the character */
    virtual void doSimulation(SimTime dt);

    /** Renders the entire scene */
    virtual void doGraphics();

    /** Makes one rendering pass, drawing everything in the scene. 
        Called from doGraphics. */
    void drawObjects();

    /** Called from doGraphics to capture the scene graph. */
    void poseObjects();

    void ambientPass();
    
    /** Marks shadows with positive values in the stencil buffer */
    void markShadows();

    void directionalPass();
};

#endif
