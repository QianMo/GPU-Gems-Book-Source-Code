#ifndef __CRENDERER_HPP__
#define __CRENDERER_HPP__

#include <cg/cgGL.h>


#define TEXTUREWIDTH 512

extern float rRotationX;
extern float rRotationY;
extern float CamZ, CamY, CamX;
extern float Weather;

struct TTexParam {
    CGparameter tmp[20];
    int Count;
    TTexParam() { Count = 0;}
    CGparameter Add(CGparameter p, int TexID = 0) { 
      if (TexID != 0) cgGLSetTextureParameter(p, TexID);
      tmp[Count++] = p;
    }
    void Enable(int bEnable) {
      for (int k = 0; k < Count; k++)
        if (bEnable) cgGLEnableTextureParameter(tmp[k]);
                else cgGLDisableTextureParameter(tmp[k]);
        
    }
    
};


class CRenderer
{
public:
  CRenderer() {};

  void Initialize();
  void Render(int bReflection);
  GLuint GetReflID() {return WaterReflID;}

/*
  void Update();
  void Reset();
  void Shutdown();
*/

private:

  CGcontext Context;
  
  CGprogram fragmentProgram;
  CGprogram vertexProgram;
  struct {
    //CGparameter environmentMap;
    //CGparameter NMap, NMap1, WRefl, Freshel;
    CGparameter c[10];
    CGparameter dXYMap, dXYMap1;
    CGparameter EnvCMap, FoamMap;
  } fpVars;

  struct {
    CGparameter VOfs, CPos;
    CGparameter Gabarites;
    CGparameter HMap0, HMap1;
  } vpVars;

  void RenderSky();
  void RenderSea();
  void RenderIsland();
  


  void CreateNoiseTexture();
  float PRNGenerator(int x);
  GLuint m_TextureID, Cube0ID, NMapID[16], LandCID, WaterReflID, FreshelID, NoiseID;
  GLfloat m_pfNoiseTextureMap[TEXTUREWIDTH][TEXTUREWIDTH];
  GLuint FoamID, WaterReflDMID[128], WaterRefldXYID[128];
  
  float m_LightPos[3];
  float m_specExp;
};

#endif // __CRENDERER_HPP__