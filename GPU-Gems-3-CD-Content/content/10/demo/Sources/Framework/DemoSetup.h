#pragma once

#include "Application.h"
#include "SceneObject.h"
#include "Camera.h"
#include "Light.h"
#include "ShadowMap.h"
#include "Mesh.h"

// light
extern Light g_Light;
// camera
extern Camera g_Camera;

// scene
extern std::vector<SceneObject *> g_SceneObjects;
extern std::vector<Mesh *> g_Meshes;

// split scheme
extern int g_iNumSplits;
extern float g_fSplitSchemeWeight;
extern float g_fSplitPos[10]; // g_iNumSplits + 1 elements needed

// rendering method
enum RenderingMethod
{
  METHOD_MULTIPASS = 0,
  METHOD_DX9 = 1,
  METHOD_DX10_GSC = 2, // geometry shader cloning
  METHOD_DX10_INST = 3, // instancing
  NUM_METHODS = 4,
};
extern int g_iRenderingMethod;
extern bool g_bUseSceneDependentProjection;
extern bool g_bMethodSupported[4];
enum VisiblityTest
{
  VISTEST_ACCURATE = 0,
  VISTEST_CHEAP = 1,
  VISTEST_NONE = 2
};
extern int g_iVisibilityTest;

// misc
extern int g_iMeshTypes;
extern int g_iNumSceneObjects;
extern float g_fSceneArea;
extern int g_iTrisPerFrame;
extern bool g_bHUDStats;
extern bool g_bHUDFPS;
extern bool g_bHUDTextures;

// shadowmap
extern int g_iShadowMapSize;
extern std::vector<ShadowMap *> g_ShadowMaps;
template<class ShadowMapType>
ShadowMapType *GetShadowMap(int iIndex=0) { return (ShadowMapType *)g_ShadowMaps[iIndex]; }

// shaders
#define NUM_SPLITS_IN_SHADER 4

// API independent functions:
//
// Prints the stats string drawn in HUD
extern void PrintStats(char *strText);
// Returns the largest possible split count
extern int GetMaxSplitCount(void);
// Updates menu states
extern void UpdateMenus(void);
// Change user controlled settings
extern void DoControls(void);
// Creates stuff from command line parameters
extern bool CreateAll(LPSTR lpCmdLine);
// Destroys everything
extern void DestroyAll(void);
// Creates the meshes
extern bool CreateMeshes(void);
// Creates the scene
extern void CreateScene(void);
// Destroys the scene
extern void DestroyScene(void);
// Destroys the meshes
extern void DestroyMeshes(void);
// Destroys all shadow maps
extern void DestroyShadowMaps(void);

// API specific functions:
//
// Creates a new mesh and returns it
extern Mesh *CreateNewMesh();
// Creates an application and returns it
extern Application *CreateApplication(void);
// Changes current rendering method
extern bool ChangeRenderingMethod(int iNewMethod);