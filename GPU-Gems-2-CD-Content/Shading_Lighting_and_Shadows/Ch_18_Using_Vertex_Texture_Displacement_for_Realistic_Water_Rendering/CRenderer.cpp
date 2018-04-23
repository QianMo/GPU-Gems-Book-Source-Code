
#include "CRenderer.hpp"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

//#include <iostream.h>


#include <gl/glut.h>
#include <gl/gl.h>
#include "glext.h"


#define _USE_MATH_DEFINES
#include <math.h>

PFNGLACTIVETEXTUREARBPROC  glActiveTextureARB = 0;
#define GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB   0x8B4C
/* GL_ATI_texture_float */
#define GL_RGBA_FLOAT32_ATI                 0x8814
#define GL_RGB_FLOAT32_ATI                  0x8815
#define GL_ALPHA_FLOAT32_ATI                0x8816
#define GL_INTENSITY_FLOAT32_ATI            0x8817
#define GL_LUMINANCE_FLOAT32_ATI            0x8818
#define GL_LUMINANCE_ALPHA_FLOAT32_ATI      0x8819
#define GL_RGBA_FLOAT16_ATI                 0x881A
#define GL_RGB_FLOAT16_ATI                  0x881B
#define GL_ALPHA_FLOAT16_ATI                0x881C
#define GL_INTENSITY_FLOAT16_ATI            0x881D
#define GL_LUMINANCE_FLOAT16_ATI            0x881E
#define GL_LUMINANCE_ALPHA_FLOAT16_ATI      0x881F

typedef int  intptrARB;
typedef unsigned int sizeiptrARB;

typedef void (APIENTRY * glBindBufferARBPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRY * glDeleteBuffersARBPROC) (GLsizei n, const GLuint *buffers);
typedef void (APIENTRY * glGenBuffersARBPROC) (GLsizei n, GLuint *buffers);
typedef GLboolean (APIENTRY * glIsBufferARBPROC) (GLuint buffer);

typedef void (APIENTRY * glBufferDataARBPROC) (GLenum target, sizeiptrARB size, const void *data, GLenum usage);
typedef void (APIENTRY * glBufferSubDataARBPROC) (GLenum target, intptrARB offset, sizeiptrARB size, const void *data);
typedef void (APIENTRY * glGetBufferSubDataARBPROC) (GLenum target, intptrARB offset, sizeiptrARB size, void *data);

typedef void * (APIENTRY * glMapBufferARBPROC) (GLenum target, GLenum access);
typedef GLboolean (APIENTRY * glUnmapBufferARBPROC) (GLenum target);

typedef void (APIENTRY * glGetBufferParameterivARBPROC) (GLenum target, GLenum pname, int *params);
typedef void (APIENTRY * glGetBufferPointervARBPROC) (GLenum target, GLenum pname, void **params);


extern glBindBufferARBPROC glBindBufferARB;
extern glDeleteBuffersARBPROC glDeleteBuffersARB;
extern glGenBuffersARBPROC glGenBuffersARB;
extern glIsBufferARBPROC glIsBufferARB;
extern glBufferDataARBPROC glBufferDataARB;
extern glBufferSubDataARBPROC glBufferSubDataARB;
extern glGetBufferSubDataARBPROC glGetBufferSubDataARB;
extern glMapBufferARBPROC glMapBufferARB;
extern glUnmapBufferARBPROC glUnmapBufferARB;
extern glGetBufferParameterivARBPROC glGetBufferParameterivARB;
extern glGetBufferPointervARBPROC glGetBufferPointervARB;

glBindBufferARBPROC glBindBufferARB = 0;
glDeleteBuffersARBPROC glDeleteBuffersARB = 0;
glGenBuffersARBPROC glGenBuffersARB = 0;
glIsBufferARBPROC glIsBufferARB = 0;
glBufferDataARBPROC glBufferDataARB = 0;
glBufferSubDataARBPROC glBufferSubDataARB = 0;
glGetBufferSubDataARBPROC glGetBufferSubDataARB = 0;
glMapBufferARBPROC glMapBufferARB = 0;
glUnmapBufferARBPROC glUnmapBufferARB = 0;
glGetBufferParameterivARBPROC glGetBufferParameterivARB = 0;
glGetBufferPointervARBPROC glGetBufferPointervARB = 0;



#define LandSize 128
BYTE LandH[LandSize][LandSize];
unsigned int VBO_ID;
unsigned int IBO_ID;
int nInd, nVert;


extern float CamX;
extern float CamY;
extern float CamZ;


CGprofile  fpProfile = CG_PROFILE_FP40;
CGprofile  vpProfile = CG_PROFILE_VP40;



#define WaterImages 16
#define WaterTexSize 256

struct TVertexV1C {
  float XYZ[3];
  DWORD RGBA;
  void Set(float X, float Y, float Z, DWORD aRGBA = 0) { 
    XYZ[0] = X; XYZ[1] = Y; XYZ[2] = Z;
  }
};

float GetTime() {
  return glutGet(GLenum(GLUT_ELAPSED_TIME)) * .001f;
}

void LoadRaw(char *RawFN, void *Data, int SizeBytes) {
  FILE *F = fopen(RawFN,"rb");
  fread(Data, SizeBytes, 1, F);
  fclose(F);
}

int make_tex(BYTE *pBuf, int texSize, int Wrap = 1, int nChan = 3) {
  GLuint ID; glGenTextures(1, &ID);

  glBindTexture(GL_TEXTURE_2D, ID);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (Wrap? GL_REPEAT : GL_CLAMP_TO_EDGE));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (Wrap? GL_REPEAT : GL_CLAMP_TO_EDGE));
  // SGIS_generate_mipmap extension must exist
  glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

  glTexImage2D( GL_TEXTURE_2D, 0, nChan==3?GL_RGB8:GL_RGBA8, texSize, texSize, 0, 
     nChan==3?GL_RGB:GL_RGBA, GL_UNSIGNED_BYTE, pBuf);
  
	return ID;
}


int load_tex(char *string, int texSize, int Wrap = 1, int nChan = 3) {
  BYTE *pBuf = new BYTE[texSize*texSize*nChan];
  LoadRaw(string, pBuf, texSize*texSize*nChan);
  GLuint ID = make_tex(pBuf, texSize, Wrap, nChan);
  delete[] pBuf;
  return ID;
}

int load_texlist(char *stringmask, GLuint *IDS, int Count, int texSize, int Wrap = 1) {
  for (int i = 0; i < Count; i++) {
    char FN[80]; sprintf(FN, stringmask, i);
    IDS[i] = load_tex(FN, texSize, Wrap);
  }
  return 1;
}


int BuildFreshel_tex(int texSize) {
  float refractionIndexRatio = 1.3333333f; // Water refraction koef
  float FreshelK0 = pow(1.0-refractionIndexRatio, 2.0) / pow(1.0+refractionIndexRatio, 2.0);
  

  BYTE *pBuf = new BYTE[texSize*texSize*3];
  BYTE *pB = pBuf;
  for (int j = 0; j < texSize; j++) {
    float L = float(j)/texSize;
    for (int i = 0; i < texSize; i++) {
      float CosT = float(i)/texSize;
      float K = (FreshelK0+(1-FreshelK0)*pow((1-CosT),5))*L;
      pB[0] = pB[1] = pB[2] = int(K*255.0f);
      pB+=3;
    }
  }

  GLuint ID = make_tex(pBuf, texSize, 0);
  delete[] pBuf;
	return ID;
}



int load_cubemap(const char *string, int texSize) {
  GLenum CubeSides[6] = {
     GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB, 
     GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB,
     GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB,
     GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB,
     GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB,
     GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB
  };

  char CN[6][5] = {{"posx"}, {"negx"}, {"negy"}, {"posy"}, {"posz"}, {"negz"}};

  GLuint CubeMapID; glGenTextures(1, &CubeMapID);
  

  glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, CubeMapID);	
  glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);


  BYTE *pBuf = new BYTE[texSize*texSize*3];
  for (int i = 0; i < 6; i++) {

    char buff[1024];
	  sprintf(buff, string, CN[i]);
    LoadRaw(buff, pBuf, texSize*texSize*3);
    // SGIS_generate_mipmap extension must exist
    

    glTexImage2D( CubeSides[i], 0, GL_RGB8, texSize, texSize, 0, GL_RGB, GL_UNSIGNED_BYTE, pBuf);
  }
  delete[] pBuf;
	return CubeMapID;
}

inline void EnableTexGenScreen() {
    float mtxTex[16] = {1.0, 0.0, 0.0, 0.0, 
                        0.0, 1.0, 0.0, 0.0, 
            				    0.0, 0.0, 1.0, 0.0, 
					         	    0.0, 0.0, 0.0, 1.0
                       };
   glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
   glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
   glTexGeni (GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
   glTexGeni (GL_Q, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
   glTexGenfv(GL_S, GL_OBJECT_PLANE, mtxTex);
   glTexGenfv(GL_T, GL_OBJECT_PLANE, mtxTex+4);
   glTexGenfv(GL_R, GL_OBJECT_PLANE, mtxTex+8);
   glTexGenfv(GL_Q, GL_OBJECT_PLANE, mtxTex+12);
   glEnable(GL_TEXTURE_GEN_S);
   glEnable(GL_TEXTURE_GEN_T);
   glEnable(GL_TEXTURE_GEN_R);
   glEnable(GL_TEXTURE_GEN_Q);
}

inline void DisableTexGenScreen() {
  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);
  glDisable(GL_TEXTURE_GEN_R);
  glDisable(GL_TEXTURE_GEN_Q);
}


void SetScreenPrjMatrix() {
	//Use convertion to homogenies coord
  float mtxMView[16];
  float mtxPrj[16];
   
  glGetFloatv(GL_MODELVIEW_MATRIX, mtxMView);
  glGetFloatv(GL_PROJECTION_MATRIX, mtxPrj);   
  
	float ScaleMtx[16] = {0.5, 0.0, 0.0, 0.0, 
                        0.0, 0.5, 0.0, 0.0, 
           						  0.0, 0.0, 1.0, 0.0, 
					          	  0.5, 0.5, 0.0, 1.0
  };
  glLoadMatrixf(ScaleMtx);
  glMultMatrixf(mtxPrj);
  glMultMatrixf(mtxMView);
}

  



static void CheckCgError(CGcontext Context = 0)
{
  CGerror err = cgGetError();

  if (err != CG_NO_ERROR) {
     printf("CG error: %s \n", cgGetErrorString(err));
     printf("%s \n", cgGetLastListing(Context));
  }
}

// ***************************************************************************
// ** Name: InitResources
// ** Desc: Initialize all resources that we need
// ***************************************************************************
void CRenderer::Initialize()
{

  m_LightPos[0] = 0.0;
  m_LightPos[1] = 1.0;
  m_LightPos[2] = 0.0;


  // Create cgContext.
  Context = cgCreateContext();


  CheckCgError();

  //fragmentProfile = CG_PROFILE_ARBFP1;
  //fragmentProfile = cgGetProfile("fp40");
  //fragmentProfile = CG_PROFILE_FP40;

  

  
  fragmentProgram = cgCreateProgramFromFile(Context,
    CG_SOURCE, "fpWaterDM.cg", fpProfile, NULL, NULL);
  CheckCgError(Context);
  cgGLLoadProgram(fragmentProgram);
  CheckCgError(Context);

  for(int k = 0; k < 10; k++) {
    char S[4]; sprintf(S, "c%i", k);
    fpVars.c[k] = cgGetNamedParameter(fragmentProgram, S);
  }
  fpVars.dXYMap = cgGetNamedParameter(fragmentProgram, "dXYMap");
  fpVars.dXYMap1 = cgGetNamedParameter(fragmentProgram, "dXYMap1");
  fpVars.EnvCMap = cgGetNamedParameter(fragmentProgram, "EnvCMap");
  fpVars.FoamMap = cgGetNamedParameter(fragmentProgram, "FoamMap");

  CheckCgError(Context);


  
  // VBO ext must exists
  glBindBufferARB = (glBindBufferARBPROC) wglGetProcAddress("glBindBufferARB");
  glDeleteBuffersARB = (glDeleteBuffersARBPROC) wglGetProcAddress("glDeleteBuffersARB");
  glGenBuffersARB = (glGenBuffersARBPROC) wglGetProcAddress("glGenBuffersARB");
  glIsBufferARB = (glIsBufferARBPROC) wglGetProcAddress("glIsBufferARB");
  glBufferDataARB = (glBufferDataARBPROC) wglGetProcAddress("glBufferDataARB");
  glBufferSubDataARB = (glBufferSubDataARBPROC) wglGetProcAddress("glBufferSubDataARB");
  glGetBufferSubDataARB = (glGetBufferSubDataARBPROC) wglGetProcAddress("glGetBufferSubDataARB");
  glMapBufferARB = (glMapBufferARBPROC) wglGetProcAddress("glMapBufferARB");
  glUnmapBufferARB = (glUnmapBufferARBPROC) wglGetProcAddress("glUnmapBufferARB");
  glGetBufferParameterivARB = (glGetBufferParameterivARBPROC) wglGetProcAddress("glGetBufferParameterivARB");
  glGetBufferPointervARB = (glGetBufferPointervARBPROC) wglGetProcAddress("glGetBufferPointervARB");


  
  CheckCgError();

  vertexProgram = cgCreateProgramFromFile(Context,
    CG_SOURCE,
    "vpWaterDM.cg",
    vpProfile, NULL,
     NULL//(const char **)CharArray //NULL
    );
  CheckCgError(Context);

  if(vertexProgram != NULL){
    cgGLLoadProgram(vertexProgram);
    CheckCgError();  
  }

  vpVars.VOfs = cgGetNamedParameter(vertexProgram, "VOfs");
  vpVars.CPos = cgGetNamedParameter(vertexProgram, "CPos");
  vpVars.Gabarites = cgGetNamedParameter(vertexProgram, "Gabarites");
  vpVars.HMap0 = cgGetNamedParameter(vertexProgram, "HMap0");
  vpVars.HMap1 = cgGetNamedParameter(vertexProgram, "HMap1");
  


  
  CreateNoiseTexture();

  // Create texture surface
  glGenTextures(1, &m_TextureID);
  glBindTexture(GL_TEXTURE_2D, m_TextureID);
  glEnable(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);

  // SGIS_generate_mipmap extension must exist
  glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

  glTexImage2D(GL_TEXTURE_2D,0, GL_LUMINANCE, TEXTUREWIDTH, TEXTUREWIDTH, 0, GL_LUMINANCE, GL_FLOAT, m_pfNoiseTextureMap);
  glDisable(GL_TEXTURE_2D);


  // Freshel Texture
  FreshelID = BuildFreshel_tex(128);


  Cube0ID = load_cubemap("Data/Terrain_%s.raw", 512);
//  load_texlist("Data/WaterNoise%02iDOT3.raw", NMapID, 16, 256);

  NoiseID = load_tex("Data/Noise.raw", 256);
  LoadRaw("Data/Land_H.raw", LandH, sizeof(LandH));
  LandCID  = load_tex("Data/Land_C.raw", 512);

  glGenTextures(1, &WaterReflID);
  glBindTexture(GL_TEXTURE_2D, WaterReflID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


  glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)wglGetProcAddress("glActiveTextureARB");
#if 1
  if (1) { //GL.NV_vertex_program3
    GLint vtex_units;
    glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS_ARB, &vtex_units);
    printf("VS3.0: Vertex texture units: %d\n", vtex_units);

    FoamID = load_tex("Data/FoamNV40.raw", WaterTexSize, 1, 4);

    float *BMap  = new float[WaterTexSize*WaterTexSize];
    BYTE *BMap1 = new BYTE[WaterTexSize*WaterTexSize];
    BYTE *BMap2 = new BYTE[WaterTexSize*WaterTexSize];
    float *pBuf = new float[WaterTexSize*WaterTexSize*4];
    float *pBuf2 = new float[WaterTexSize*WaterTexSize*4];
    float *pBuf4 = new float[WaterTexSize*WaterTexSize*4];

    { // 8 step interpolation between animation frames
      for(int l = 0; l < WaterImages*8; l++)  {
        // Load Water maps, scale H to 0..1
        // To do quadric /\  waves distorsion
        char LoadFN[256];
        if ((l%8)==0) {
          if (l==0) {
            sprintf(LoadFN, "Data/HMaps/WaterH" "%02i" ".raw", l/8);
            LoadRaw(LoadFN, BMap1, WaterTexSize*WaterTexSize);
          }
          else { // Copy previous
            memcpy(BMap1, BMap2, WaterTexSize*WaterTexSize);
          }
        } 

        {// Interpolate intermediate map
          int l1= (l/8+1)&(WaterImages-1);
          if ((l%8)==0) {
            sprintf(LoadFN, "Data/HMaps/WaterH" "%02i" ".raw", l1);
            LoadRaw(LoadFN, BMap2, WaterTexSize*WaterTexSize);

          }
          float R = float((l&7))/8.0f;
          for (int y = 0; y < WaterTexSize; y++) {
            for (int x = 0; x < WaterTexSize; x++) {
              int PixAddr = y*WaterTexSize+x;
              float t = float(BMap1[PixAddr])*(1-R)+float(BMap2[PixAddr])*R;	 
              float H = pow((t/255.0f),1.7f);
              if (H < 0) H = 0; if(H > 1.0f) H = 1.0f; 
              BMap[PixAddr] = H;
            }
          }
        }

        float *pB = pBuf;
        // float MinV = 1.0f, MaxV = 0;
        int Mask = WaterTexSize-1;
        for (int y = 0; y < WaterTexSize; y++)
          for (int x = 0; x < WaterTexSize; x++) {
            float V = BMap[y*WaterTexSize+x];
            //   SetMinMax(V, MinV, MaxV);
            pB[0] = V; 
            pB[1] = BMap[y*WaterTexSize+((x+1) & Mask)]-BMap[y*WaterTexSize+((x-1) & Mask)];
            pB[2] = BMap[x+((y+1) & Mask)*WaterTexSize]-BMap[x+((y-1) & Mask)*WaterTexSize];

            pB[1] = pB[1]*0.5*0.5+0.5;
            pB[2] = pB[2]*0.5*0.5+0.5;

            pB+=3;
          }
          pB = pBuf;

          glGenTextures(1, &WaterReflDMID[l]); 
          glBindTexture(GL_TEXTURE_2D, WaterReflDMID[l]);
          //       if (GL.SGIS_generate_mipmap) glTexParameteri(GL_TEXTURE_2D,GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);   

          int SZ = WaterTexSize;

          for (y = 0; y < SZ; y++) {
            for (int x = 0; x < SZ; x++) {
              pBuf2[(y*SZ + x)*4+0] = pBuf[(y*SZ + x)*3+0];
            }
          }
          float *pBuf3 = new float[(SZ/2)*(SZ/2)*4], *pB3 = pBuf3;
          { 
            for (int k = 0; k < 10; k++) {
              for (y = 0; y < SZ; y++) {
                for (int x = 0; x < SZ; x++) {
                  //pBuf2[(y*SZ + x)*4+0] = pBuf[(y*SZ + x)*4+0];
                  pBuf2[(y*SZ + x)*4+1] = pBuf2[(y*SZ + (x+1)%SZ)*4+0];
                  pBuf2[(y*SZ + x)*4+2] = pBuf2[(((y+1)%SZ)*SZ + x)*4+0];
                  pBuf2[(y*SZ + x)*4+3] = pBuf2[(((y+1)%SZ)*SZ + (x+1)%SZ)*4+0];
                }
              }


              // Load 2x downscaled - save memory 4x times
              if (k > 0) {
                glTexImage2D(GL_TEXTURE_2D, k-1, GL_RGBA_FLOAT32_ATI/*GL_LUMINANCE_FLOAT32_ATI*/, 
                  SZ, SZ, 0, /*GL_LUMINANCE*/GL_RGBA, GL_FLOAT, pBuf2);

              }
              SZ/=2;  int SZ2 = SZ*2;
              if (SZ < 1) break;
              for (int y = 0; y < SZ; y++) {
                for (int x = 0; x < SZ; x++) {
                  float F = pBuf2[((y*2)*SZ2+x*2)*4+0];
                  F += pBuf2[((y*2)*SZ2+x*2+1)*4+0];
                  F += pBuf2[((y*2+1)*SZ2+x*2)*4+0]; 
                  F += pBuf2[((y*2+1)*SZ2+x*2+1)*4+0]; 
                  F/=4;
                  pBuf3[(y*SZ+x)*4+0] = F;
                }
              }
              float *t = pBuf2; pBuf2 = pBuf3; pBuf3 = t;
            }
          }
          pBuf3 = pB3; delete[] pB3;


          glGenTextures(1, &WaterRefldXYID[l]); 
          glBindTexture(GL_TEXTURE_2D, WaterRefldXYID[l]);
          glTexParameteri(GL_TEXTURE_2D,GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);   


          {
            for (int y = 0; y < WaterTexSize; y++) {
              for (int x = 0; x < WaterTexSize; x++) {
                int ii = (y*WaterTexSize + x);
                pBuf4[ii*4+0] = pBuf[ii*3+0];
                pBuf4[ii*4+1] = pBuf[ii*3+1];
                pBuf4[ii*4+2] = pBuf[ii*3+2];
                pBuf4[ii*4+3] = 1.0f;
              }
            }
          }

          // Load texture - with rejecting A-channel 
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB_FLOAT16_ATI, 
            WaterTexSize, WaterTexSize, 0, GL_RGBA, GL_FLOAT, pBuf4);

      }
    }
    delete [] pBuf;
    delete [] pBuf2;
    delete [] pBuf4;
    delete [] BMap;
    delete [] BMap1;
    delete [] BMap2;

    // Create radial grid ring
    {
        
        TVertexV1C *pVB = new TVertexV1C[512*384];
        int *pIB = new int[(512+1)*(384+1)*6];
        
        nVert = 0;
        int j;
        for(j = 0; j < 512; j++) {
            float R0 = 1;//(SQR(j*RScale))*MaxR;
            for (int i = 0; i < 384; i++) {
              float A = (i*M_PI*2.0f)/384.0f;
              float dX = cos(A), dY = sin(A); nVert++;
              pVB[j*384+i].Set(R0*dX, R0*dY,  j); //glVertex3fv(A.Get());
            }
         }  
         
        nInd = 0;
  	    for (int i = 0; i < 384+1; i++) { 
          for(j = 0; j < 512-1; j++) {
            int Ind[4] = {j*384+(i%384), (j+1)*384+(i%384), (j+1)*384+((i+1)%384), (j)*384+((i+1)%384)};
            pIB[nInd++] = Ind[0];  pIB[nInd++] = Ind[1];  pIB[nInd++] = Ind[2];
            pIB[nInd++] = Ind[0];  pIB[nInd++] = Ind[2];  pIB[nInd++] = Ind[3];
          }
         }
    

        glGenBuffersARB(1, &VBO_ID);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, VBO_ID);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(TVertexV1C)*nVert, pVB, GL_STATIC_DRAW_ARB); 

        glGenBuffersARB(1, &IBO_ID);
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, IBO_ID);
        glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, sizeof(int)*nInd, pIB, GL_STATIC_DRAW_ARB); 


    
      } 
  }
#endif

}




  
#define GridSize 128
#define FarR (4*4000.0)


void CRenderer::RenderSky() {
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, Cube0ID);	
    glEnable(GL_TEXTURE_CUBE_MAP_ARB);


    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP_EXT);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP_EXT);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP_EXT);

    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);
    glEnable(GL_TEXTURE_GEN_R);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_TEXTURE);
     glPushMatrix();
      glLoadIdentity();
    
      //glRotatef(-90.0, 1.0, 1.0, 0);    
      //glScalef(-1.0, -1.0, -1.0);
      

      
      //glScalef(+1,+1,-1);
      glRotatef(-rRotationX, 0, 1, 0);
      glRotatef(-rRotationY, 1, 0, 0);
      
      
      
      glRotatef(180, 0, 0, 1);
      //glRotatef(-90.0, 1.0, 0.0, 0);    
      
      
      //glRotatef(-rRotationY, 1, 0, 0);
    
      glutSolidSphere(FarR, 10, 10);

    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);

    glDisable(GL_TEXTURE_GEN_S);
    glDisable(GL_TEXTURE_GEN_T);
    glDisable(GL_TEXTURE_GEN_R);

    glDisable(GL_TEXTURE_CUBE_MAP_ARB);

}


void CRenderer::RenderSea() {
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  float Time = GetTime();

  int Frame0 = int(0.13*Time*128) & (128-1);
  int Frame1 = int(0.19*Time*128+127) & (128-1);

  float MaxR = 36000/8, RScale = 1.0f/512.0f;
  float Weather = 0.5;

  float _S = (1-Weather)*0.5 + 0.25;
  float H = 2.5*_S;  //H *= 3;
  H*=1.5;
  float TexScale = 0.0025*(1.0+(0.5-abs(Weather-0.5))*4);//*2.5;
      
  glPushMatrix();
  
  
  glTranslatef(CamX, CamY, 0);// Move grid Center
  
  
  cgGLEnableProfile(vpProfile);
  cgGLEnableProfile(fpProfile);
  //cgGLBindProgram(fragmentProgram);
  cgGLBindProgram(vertexProgram);
  
  cgGLSetParameter4f(vpVars.VOfs, CamX, CamY, pow(RScale, 4)*MaxR, 0);
  cgGLSetParameter4f(vpVars.CPos, CamX*0, CamY*0, CamZ, 0);
  cgGLSetParameter4f(vpVars.Gabarites, TexScale, 256.0f/2.0f, H, 1.0/TexScale);
  cgGLSetTextureParameter(vpVars.HMap0,  WaterReflDMID[Frame0]);
  cgGLEnableTextureParameter(vpVars.HMap0);
  cgGLSetTextureParameter(vpVars.HMap1,  WaterReflDMID[Frame1]);
  cgGLEnableTextureParameter(vpVars.HMap1);
  CheckCgError();
  cgGLBindProgram(fragmentProgram);
  cgGLSetTextureParameter(fpVars.dXYMap,  WaterRefldXYID[Frame0]);
  cgGLEnableTextureParameter(fpVars.dXYMap);

  cgGLSetTextureParameter(fpVars.dXYMap1,  WaterRefldXYID[Frame1]);
  cgGLEnableTextureParameter(fpVars.dXYMap1);
  cgGLSetTextureParameter(fpVars.EnvCMap, Cube0ID);
  cgGLEnableTextureParameter(fpVars.EnvCMap);

  cgGLSetTextureParameter(fpVars.FoamMap,  FoamID);
  cgGLEnableTextureParameter(fpVars.FoamMap);


  cgGLSetParameter4f(fpVars.c[0], 0.2*0.15,1*0.15,0.85*0.15,0.15); //diffuse+Ambient in w 
  cgGLSetParameter4f(fpVars.c[2], +0.9,-0.2, +0.1,0); //Sun direction

  float cMaxIntensity  = 0.8, cMinIntensity = 0.05, FreshelCosClamp=-0.1;
  float Wavy = 2.125;// Wavy = 1 Normal, < 1 Wild
  float FoamB = +0.50+2*0.15, FoamKx = 0.8/(1-FoamB);
  cgGLSetParameter4f(fpVars.c[3], cMinIntensity, cMaxIntensity-cMinIntensity, FreshelCosClamp, Wavy*2);
  cgGLSetParameter4f(fpVars.c[9], FoamKx, -FoamB*FoamKx, 0, 0);

  CheckCgError();

   
  //glDisable(GL_TEXTURE_2D);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB, VBO_ID);
  glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, IBO_ID);  

   
  glVertexPointer(3, GL_FLOAT, sizeof(TVertexV1C), 0/*CylinderS*/);
  glEnableClientState(GL_VERTEX_ARRAY);
  glColor3f(1,0,0);   
  glDrawElements(GL_TRIANGLES, nInd, GL_UNSIGNED_INT, 0);  
  glDisableClientState(GL_VERTEX_ARRAY);

  cgGLDisableProfile(vpProfile);
  cgGLDisableProfile(fpProfile);
  cgGLDisableTextureParameter(vpVars.HMap0);
  cgGLDisableTextureParameter(vpVars.HMap1);
  cgGLDisableTextureParameter(fpVars.dXYMap);
  cgGLDisableTextureParameter(fpVars.dXYMap1);
  cgGLDisableTextureParameter(fpVars.EnvCMap);
  cgGLDisableTextureParameter(fpVars.FoamMap);

  glPopMatrix();

  glDisable(GL_BLEND);
  
}

void CRenderer::RenderIsland() {
//return; 
  GLfloat s1_vector[4] = {1.0f/LandSize/4.0f, 0.0, 0.0, 0.0};
  GLfloat t1_vector[4] = {0.0,  0.0, 1.0f/LandSize/4.0f,0.0};
  glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  
  glTexGenfv(GL_S, GL_OBJECT_PLANE, s1_vector);
  glTexGenfv(GL_T, GL_OBJECT_PLANE, t1_vector);

  glEnable(GL_TEXTURE_GEN_S);
  glEnable(GL_TEXTURE_GEN_T);
  
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, LandCID);

  glActiveTextureARB(GL_TEXTURE1_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, NoiseID);

  GLfloat s2_vector[4] = {11.0f/LandSize/4.0f, 0.0, 0.0, 0.0};
  GLfloat t2_vector[4] = {0.0,  0.0, 11.0f/LandSize/4.0f,0.0};
  glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  
  glTexGenfv(GL_S, GL_OBJECT_PLANE, s2_vector);
  glTexGenfv(GL_T, GL_OBJECT_PLANE, t2_vector);

  glEnable(GL_TEXTURE_GEN_S);
  glEnable(GL_TEXTURE_GEN_T);
  
 
  // Render Landscape 
  for (int y = 0; y < LandSize-1; y++) {
    glBegin(GL_TRIANGLE_STRIP);
      for (int x = 0; x < LandSize-1; x++) {
         glVertex3f(x*4, y*4, LandH[y][x]-6.0f);
         glVertex3f(x*4, y*4+4, LandH[y+1][x]-6.0f);
      }
    glEnd();
  }

  glDisable(GL_TEXTURE_2D);
  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glDisable(GL_TEXTURE_GEN_S);
  glDisable(GL_TEXTURE_GEN_T);

}

void CRenderer::Render(int Reflection) {
  if (!Reflection) RenderSky();
  if (!Reflection)RenderSea();
  if (!Reflection) RenderIsland();
}

void CRenderer::CreateNoiseTexture()
{

	for(int i = 0; i < TEXTUREWIDTH; i++)
		for(int j = 0; j < TEXTUREWIDTH; j++)
			m_pfNoiseTextureMap[i][j] = (PRNGenerator(i*TEXTUREWIDTH + j) + 1.0) / 2;
}

float CRenderer::PRNGenerator(int x)
{
  x = (x << 13) ^ x;

  int Prime1 = 15731;
  int Prime2 = 789221;
  int Prime3 = 1376312589;

  return (1.0 - ((x * (x*x*Prime1 + Prime2) + Prime3) & 0x7fffffff) / 1073741824.0);
}





