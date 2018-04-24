#include "Common.h"
#include "DemoSetup.h"
#include "Application.h"
#include "Resources.h"
#include "IntersectionTests.h"
#include <string>
#include <algorithm>

// light
Light g_Light;
// camera
Camera g_Camera;

// scene
std::vector<SceneObject *> g_SceneObjects;
std::vector<Mesh *> g_Meshes;

// split scheme
int g_iNumSplits = 4;
float g_fSplitSchemeWeight = 0.5f;
float g_fSplitPos[10]; // g_iNumSplits + 1 elements needed

// rendering method
int g_iRenderingMethod = METHOD_MULTIPASS;
bool g_bUseSceneDependentProjection = true;
bool g_bMethodSupported[4] = {true, true, false, false};
int g_iVisibilityTest = VISTEST_ACCURATE;

// misc
int g_iMeshTypes = 0;
int g_iNumSceneObjects = 30;
float g_fSceneArea = 200.0f;
int g_iTrisPerFrame = 0;
bool g_bHUDStats = true;
bool g_bHUDFPS = true;
bool g_bHUDTextures = true;

// shadowmap
int g_iShadowMapSize = 1024;
std::vector<ShadowMap *> g_ShadowMaps;

// Prints the stats string drawn in HUD
//
//
void PrintStats(char *strText)
{
  strText[0]=0;
  char strStats[1024];
  strStats[0]=0;
  char strFPS[1024];
  strFPS[0]=0;

  size_t iSize=1024;

  char *strMethod = "Multi-pass method (n+n passes, 1 texture)";
  if(g_iRenderingMethod == METHOD_DX9) strMethod = "DX9-level method (n+1 passes, n textures)";
  if(g_iRenderingMethod == METHOD_DX10_GSC) strMethod = "DX10-level method (GS cloning) (1+1 passes, n textures)";
  if(g_iRenderingMethod == METHOD_DX10_INST) strMethod = "DX10-level method (instancing) (1+1 passes, n textures)";

  int iTexMem = 0;
  for(unsigned int i = 0; i < g_ShadowMaps.size(); i++)
  {
    iTexMem += g_ShadowMaps[i]->GetMemoryInMB();
  }

  if(g_bHUDStats)
    _snprintf(strStats,iSize,"%s\n%i splits, texture type: %s\n%i objects, %ik tris per frame, %iMB texture memory\n", strMethod, g_iNumSplits, g_ShadowMaps[0]->GetInfoString(), g_iNumSceneObjects, g_iTrisPerFrame/1000, iTexMem);
  if(g_bHUDFPS)
    _snprintf(strFPS,iSize,"FPS: %i", GetAppBase()->GetFPS());
  _snprintf(strText,iSize,"%s%s\n",strStats, strFPS);
}


// Returns the largest possible split count
//
//
int GetMaxSplitCount(void)
{
  if(g_iRenderingMethod == METHOD_MULTIPASS)
    // with multi-pass allow 9
    return 9;

  // others use precompiled shaders
  return NUM_SPLITS_IN_SHADER;
}


// Updates menu states
//
//
void UpdateMenus(void)
{
  // shadow map size menu
  int iSMSize = g_ShadowMaps[0]->GetSize();
  for(int i=ID_SM256; i<=ID_SM4096; i++)
  {
    if(1 << (i - ID_SM256 + 8) == iSMSize)
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_CHECKED);
    else
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_UNCHECKED);
  }

  // split menu
  int iMaxSplits = GetMaxSplitCount();
  for(int i=ID_SPLITS1; i<=ID_SPLITS9; i++)
  {
    if(i == g_iNumSplits + ID_SPLITS1 - 1)
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_CHECKED);
    else
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_UNCHECKED);
    if(i > iMaxSplits + ID_SPLITS1 -1)
      EnableMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_GRAYED);
    else
      EnableMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_ENABLED);
  }

  // scene-dependent projection
  CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_SCENEDEPENDENTPROJ, g_bUseSceneDependentProjection ? MF_CHECKED : MF_UNCHECKED);

  // method menu
  for(int i=ID_METHOD_MULTIPASS; i<=ID_METHOD_DX10_INST; i++)
  {
    if(i == g_iRenderingMethod + ID_METHOD_MULTIPASS)
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_CHECKED);
    else
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_UNCHECKED);
    if(!g_bMethodSupported[i - ID_METHOD_MULTIPASS])
      EnableMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_GRAYED);
    else
      EnableMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_ENABLED);
  }

  // visibility test menu
  for(int i=ID_VISTEST_ACCURATE; i<=ID_VISTEST_NONE; i++)
  {
    if(i - ID_VISTEST_ACCURATE == g_iVisibilityTest)
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_CHECKED);
    else
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_UNCHECKED);
  }

  // light menu
  if(g_Light.m_Type == Light::TYPE_ORTHOGRAPHIC)
  {
    CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_LIGHT_ORTH, MF_CHECKED);
    CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_LIGHT_PERS, MF_UNCHECKED);
  }
  else
  {
    CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_LIGHT_ORTH, MF_UNCHECKED);
    CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_LIGHT_PERS, MF_CHECKED);
  }

  // mesh type
  for(int i=ID_SCENE_MESHES_RND; i<=ID_SCENE_MESHES_LP; i++)
  {
    if(i - ID_SCENE_MESHES_RND == g_iMeshTypes)
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_CHECKED);
    else
      CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), i, MF_UNCHECKED);
  }

  // HUD
  CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_HUD_STATS, g_bHUDStats ? MF_CHECKED : MF_UNCHECKED);
  CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_HUD_FPS, g_bHUDFPS ? MF_CHECKED : MF_UNCHECKED);
  CheckMenuItem(GetMenu(GetAppBase()->GetHWND()), ID_HUD_TEXTURES, g_bHUDTextures ? MF_CHECKED : MF_UNCHECKED);
}


// Menu was clicked
//
//
void MenuFunc(int iID)
{
  // changing shadow map size
  if(iID >= ID_SM256 && iID <= ID_SM4096)
  {
    int iOldSize = g_iShadowMapSize;
    g_iShadowMapSize = 1 << (iID - ID_SM256 + 8);
    // re-create shadow maps
    if(!ChangeRenderingMethod(g_iRenderingMethod))
    {
      // if failed, return to old size
      g_iShadowMapSize = iOldSize;
      ChangeRenderingMethod(g_iRenderingMethod);
    }
  }

  // changing split count
  if(iID >= ID_SPLITS1 && iID <= ID_SPLITS9)
  {
    g_iNumSplits = Min(iID - ID_SPLITS1 + 1, GetMaxSplitCount());
  }

  // scene-dependent projection
  if(iID == ID_SCENEDEPENDENTPROJ) g_bUseSceneDependentProjection = !g_bUseSceneDependentProjection;

  // changing method
  if(iID >= ID_METHOD_MULTIPASS && iID <= ID_METHOD_DX10_INST)
  {
    ChangeRenderingMethod(iID - ID_METHOD_MULTIPASS);
    g_iNumSplits = Min(g_iNumSplits, GetMaxSplitCount());
  }

  // changing visibility test
  if(iID >= ID_VISTEST_ACCURATE && iID <= ID_VISTEST_NONE)
  {
    g_iVisibilityTest = iID - ID_VISTEST_ACCURATE;
  }

  // changing light type
  if(iID == ID_LIGHT_ORTH) g_Light.m_Type = Light::TYPE_ORTHOGRAPHIC;
  else if(iID == ID_LIGHT_PERS) g_Light.m_Type = Light::TYPE_PERSPECTIVE;

  // HUD
  if(iID == ID_HUD_STATS) g_bHUDStats = !g_bHUDStats;
  if(iID == ID_HUD_FPS) g_bHUDFPS = !g_bHUDFPS;
  if(iID == ID_HUD_TEXTURES) g_bHUDTextures = !g_bHUDTextures;

  // Scene
  //
  if(iID == ID_SCENE_LARGER)
  {
    g_fSceneArea += 50.0f;
    DestroyScene(); CreateScene();
  }
  if(iID == ID_SCENE_SMALLER)
  {
    g_fSceneArea = Max(g_fSceneArea - 50.0f, 50.0f);
    DestroyScene(); CreateScene();
  }
  if(iID == ID_SCENE_ADD)
  {
    g_iNumSceneObjects += 30;
    DestroyScene(); CreateScene();
  }
  if(iID == ID_SCENE_REM)
  {
    g_iNumSceneObjects = Max(g_iNumSceneObjects - 30, 10);
    DestroyScene(); CreateScene();
  }

  // mesh type
  if(iID >= ID_SCENE_MESHES_RND && iID <= ID_SCENE_MESHES_LP)
  {
    g_iMeshTypes = iID - ID_SCENE_MESHES_RND;
    DestroyScene(); CreateScene();
  }

  UpdateMenus();
}


// Change user controlled settings
//
//
void DoControls(void)
{
  static double _fLastUpdate = 0.0;
  float fDeltaTime = DeltaTimeUpdate(_fLastUpdate);

  // Adjust number of splits
  //
  for(int i=0;i<9;i++)
  {
    if(GetKeyDown('1'+i)) g_iNumSplits = Min(i+1, GetMaxSplitCount());
  }

  // Adjust split scheme weight
  //
  if(GetKeyDown(VK_ADD)) g_fSplitSchemeWeight += 0.01f * fDeltaTime;
  else if(GetKeyDown(VK_SUBTRACT)) g_fSplitSchemeWeight -= 0.01f * fDeltaTime;

  // Switch render mode
  //
  static bool _bSwitchingMode = false;
  if(GetKeyDown('M'))
  {
    if(!_bSwitchingMode)
    {
      int iMethods;
      for(iMethods = 0; iMethods < NUM_METHODS; iMethods++)
      {
        if(!g_bMethodSupported[iMethods]) break;
      }
      ChangeRenderingMethod((g_iRenderingMethod + 1) % iMethods);
      g_iNumSplits = Min(g_iNumSplits, GetMaxSplitCount());
      _bSwitchingMode = true;
    }
  } else {
    _bSwitchingMode = false;
  }

  // Switch texture res
  //
  static bool _bSwitchingRes = false;
  if(GetKeyDown('R'))
  {
    if(!_bSwitchingRes)
    {
      int iOldSize = g_iShadowMapSize;
      g_iShadowMapSize *= 2;
      if(g_iShadowMapSize > 4096) g_iShadowMapSize = 256;
      // re-create shadow maps
      if(!ChangeRenderingMethod(g_iRenderingMethod))
      {
        // if failed, return to old size
        g_iShadowMapSize = iOldSize;
        ChangeRenderingMethod(g_iRenderingMethod);
      }
      _bSwitchingRes = true;
    }
  } else {
    _bSwitchingRes = false;
  }

  // Switch visibility test
  //
  static bool _bSwitchingVisTest = false;
  if(GetKeyDown('V'))
  {
    if(!_bSwitchingVisTest)
    {
      g_iVisibilityTest = (g_iVisibilityTest+1)%3;
      _bSwitchingVisTest = true;
    }
  } else {
    _bSwitchingVisTest = false;
  }

  // Switch scene-dependent projection
  //
  static bool _bSwitchingSDP = false;
  if(GetKeyDown('P'))
  {
    if(!_bSwitchingSDP)
    {
      g_bUseSceneDependentProjection = !g_bUseSceneDependentProjection;
      _bSwitchingSDP = true;
    }
  } else {
    _bSwitchingSDP = false;
  }

  // Hide HUD
  //
  static bool _bHidingHUD = false;
  if(GetKeyDown('H'))
  {
    if(!_bHidingHUD)
    {
      if(g_bHUDStats || g_bHUDFPS || g_bHUDTextures)
        g_bHUDStats = g_bHUDFPS = g_bHUDTextures = false;
      else
        g_bHUDStats = g_bHUDFPS = g_bHUDTextures = true;

      _bHidingHUD = true;
    }
  } else {
    _bHidingHUD = false;
  }

  // Scene
  //
  static double _fLastSceneUpdate = 0.0;
  if(_fLastUpdate - _fLastSceneUpdate > 0.1)
  {
    if(GetKeyDown(VK_INSERT))
    {
      g_iNumSceneObjects += 10;
      DestroyScene(); CreateScene();
      _fLastSceneUpdate = _fLastUpdate;
    }
    if(GetKeyDown(VK_DELETE))
    {
      g_iNumSceneObjects = Max(g_iNumSceneObjects - 10, 10);
      DestroyScene(); CreateScene();
      _fLastSceneUpdate = _fLastUpdate;
    }
    if(GetKeyDown(VK_PRIOR))
    {
      g_fSceneArea += 50.0f;
      DestroyScene(); CreateScene();
      _fLastSceneUpdate = _fLastUpdate;
    }
    if(GetKeyDown(VK_NEXT))
    {
      g_fSceneArea = Max(g_fSceneArea - 50.0f, 50.0f);
      DestroyScene(); CreateScene();
      _fLastSceneUpdate = _fLastUpdate;
    }
  }

  // Quit
  //
  if(GetKeyDown(VK_ESCAPE))
  {
    PostQuitMessage(0);
  }

  UpdateMenus();
}


// Creates stuff from command line parameters
//
//
bool CreateAll(LPSTR lpCmdLine)
{
  int iWindowWidth = 800;
  int iWindowHeight = 600;
  int iFullScreen = 0;
  int iVSyncEnabled = 0;
  int iReferenceRasterizer = 0;

  sscanf(lpCmdLine,"%i %i %i %i %i",
    &iWindowWidth,
    &iWindowHeight,
    &iFullScreen,
    &iVSyncEnabled,
    &iReferenceRasterizer);

  // setup application parameters
  Application::CreationParams acp;
  acp.iWidth = iWindowWidth;
  acp.iHeight = iWindowHeight;
  acp.bFullScreen = iFullScreen == 1 ? true : false;
  acp.bVSync = iVSyncEnabled == 1 ? true : false;
  acp.bReferenceRasterizer = iReferenceRasterizer == 1 ? true : false;
  acp.strTitle = TEXT("Parallel-Split Shadow Maps");

  CreateApplication();

  // create window, initialize OGL etc.
  if(!GetAppBase()->Create(acp)) return false;

  // create meshes
  if(!CreateMeshes()) return false;

  // create scene
  CreateScene();

  // create textures
  if(!ChangeRenderingMethod(g_iRenderingMethod)) return false;

  return true;
}


// Destroys everything
//
//
void DestroyAll(void)
{
  DestroyScene();
  DestroyMeshes();
  DestroyShadowMaps();
  GetAppBase()->Destroy();
  delete GetAppBase();
}


// Creates a new mesh
//
//
inline bool CreateNewMesh(std::string strName)
{
  if(strName == "floor.lwo") return true;

  strName = "Models\\" + strName;

  Mesh *pMesh = CreateNewMesh();
  if(!pMesh->LoadFromLWO(strName.c_str())) return false;
  g_Meshes.push_back(pMesh);
  return true;
}


// Creates the meshes
//
//
struct MeshSizeSort
{
  bool operator()(Mesh *a, Mesh *b)
  {
    Vector3 A = a->m_OOBB.GetSize();
    A.y = 0;
    Vector3 B = b->m_OOBB.GetSize();
    B.y = 0;
    return Dot(A,A) < Dot(B,B);
  }
};
bool CreateMeshes(void)
{
  Mesh *pMesh;

  // floor mesh first
  pMesh = CreateNewMesh();
  if(!pMesh->LoadFromLWO("Models\\floor.lwo")) return false;
  g_Meshes.push_back(pMesh);

  // find all lwo files from Models directory
  //
  WIN32_FIND_DATAA FindFileData;
  HANDLE hFind = INVALID_HANDLE_VALUE;
  DWORD dwError;

  hFind = FindFirstFileA("Models\\*.lwo", &FindFileData);

  if(hFind != INVALID_HANDLE_VALUE)
  {
    if(!CreateNewMesh(FindFileData.cFileName)) return false;

    while (FindNextFileA(hFind, &FindFileData) != 0)
    {
      if(!CreateNewMesh(FindFileData.cFileName)) return false;
    }
    dwError = GetLastError();
    FindClose(hFind);
  }

  if(g_Meshes.size() > 1)
  {
    std::sort(g_Meshes.begin() + 1, g_Meshes.end(), MeshSizeSort());
  }

  return true;
}


// Creates the scene
//
//
void CreateScene(void)
{
  // find special mesh indices
  unsigned int iHPMesh = 1;
  unsigned int iLPMesh = 1;
  for(unsigned int i = 1; i < g_Meshes.size(); i++)
  {
    // high polycount mesh
    if(g_Meshes[i]->m_iNumTris > g_Meshes[iHPMesh]->m_iNumTris) iHPMesh = i;
    // low polycount mesh
    if(g_Meshes[i]->m_iNumTris < g_Meshes[iLPMesh]->m_iNumTris) iLPMesh = i;
  }

  int iMeshCounter = (int)g_Meshes.size() - 1;
  float fSwitchPercent = 0.05f;
  float fSwitchStep = powf(1.0f/0.1f, 1.0f/(g_Meshes.size()));

  for(int i = 0; i < g_iNumSceneObjects; i++)
  {
    SceneObject *pObject;
    pObject = new SceneObject();

    // pick meshes
    unsigned int iMesh;
    // randomly starting from index 1
    if(g_iMeshTypes == 0)
    {
      // after certain percentage switch to next mesh type
      if(i/(float)g_iNumSceneObjects >= fSwitchPercent)
      {
        if(fSwitchPercent == 0.05f) fSwitchPercent = 0.1f;
        fSwitchPercent *= fSwitchStep;
        if(fSwitchPercent > 1) fSwitchPercent = 1;
        iMeshCounter--;
        if(iMeshCounter <= 1) iMeshCounter = 1;
      }

      iMesh = iMeshCounter;
    }
    // we want highest polycount
    else if(g_iMeshTypes == 1) iMesh = iHPMesh;
    // we want lowest polycount
    else if(g_iMeshTypes == 2) iMesh = iLPMesh;

    pObject->SetMesh(g_Meshes[iMesh]);
    g_SceneObjects.push_back(pObject);


    // try to find empty spot (but not too hard)
    for(int tries = 0; tries < 5; tries++)
    {
      pObject->m_mWorld.SetRotation(Vector3(DegreeToRadian((rand()%4)*90), 0, 0));
      pObject->CalculateAABB();

      // position is quantized by size of mesh
      Vector3 vSize = pObject->m_AABB.GetSize();
      int iMaxStepsX = (int)(g_fSceneArea / vSize.x);
      int iMaxStepsZ = (int)(g_fSceneArea / vSize.z);

      // object is too large for scene
      if(iMaxStepsX == 0 || iMaxStepsZ == 0)
      {
        // remove it
        g_SceneObjects.pop_back();
        delete pObject;
        break;
      }

      float fPosX = -0.5f * g_fSceneArea + vSize.x * 0.5f + vSize.x * (rand()%iMaxStepsX);
      float fPosZ = -0.5f * g_fSceneArea + vSize.z * 0.5f + vSize.z * (rand()%iMaxStepsZ);

      pObject->m_mWorld.SetTranslation(Vector3(fPosX, 0, fPosZ));
      pObject->CalculateAABB();

      unsigned int j;
      for(j = 0; j < g_SceneObjects.size(); j++)
      {
        if(pObject == g_SceneObjects[j]) continue;
        // intersects with existing geometry
        if(IntersectionTest(pObject->m_AABB, g_SceneObjects[j]->m_AABB)) break;
      }

      // found empty location
      if(j == g_SceneObjects.size()) break;
    }
  }

  // floor object
  SceneObject *pFloor = new SceneObject();
  pFloor->SetMesh(g_Meshes[0]);
  pFloor->m_mWorld.Scale(Vector3(1.0f, 1.0f, 1.0f) * (g_fSceneArea + 8.0f));
  pFloor->CalculateAABB();
  pFloor->m_bOnlyReceiveShadows = true;
  g_SceneObjects.push_back(pFloor);

  // set light
  float fMaxLength = 1.414214f * g_fSceneArea;
  float fHeight = (fMaxLength + 32.0f) / tanf(g_Light.m_fFOV * 0.5f);
  g_Light.m_vSource = Vector3(0, fHeight, 0);
  g_Light.m_vTarget = Vector3(0, 0, 0);

  // set camera
  g_Camera.m_fFarMax = fMaxLength + 300.0f;
}


// Destroys the scene
//
//
void DestroyScene(void)
{
  // delete objects
  for(unsigned int i = 0; i < g_SceneObjects.size(); i++)
  {
    delete g_SceneObjects[i];
  }
  g_SceneObjects.clear();
}


// Destroys the meshes
//
//
void DestroyMeshes(void)
{
  // delete meshes
  for(unsigned int i = 0; i < g_Meshes.size(); i++)
  {
    delete g_Meshes[i];
  }
  g_Meshes.clear();
}


// Destroys all shadow maps
//
//
void DestroyShadowMaps(void)
{
  for(unsigned int i = 0; i < g_ShadowMaps.size(); i++)
  {
    g_ShadowMaps[i]->Destroy();
    delete g_ShadowMaps[i];
  }
  g_ShadowMaps.clear();
}
