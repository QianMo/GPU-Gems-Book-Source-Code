#pragma once

#include <cmath>
#include <vector>
#include <fstream>

#include "SDKmisc.h"

#include "Scene.hpp"
#include "PostProcess.hpp"
#include "Filtering.hpp"

//--------------------------------------------------------------------------------------
class App
{
public:
  enum SceneType {
    S_CAR = 0,
    S_COMMANDO,
    S_SPHERES,
    S_LARGE,
    S_NUM
  };

  enum FilteringTechnique {
    FT_VSM = 0,
    FT_PSVSM,
    FT_SAVSMINT,
    FT_SAVSMFP,
    FT_PCF,
    FT_POINT,
    FT_NUM
  };
  
  enum ControlMode {
    CM_CAMERA = 0,
    CM_LIGHT
  };

public:
  App();

  // Callbacks
  void OnD3D10CreateDevice(ID3D10Device* d3dDevice);
  void OnD3D10ResizedSwapChain(ID3D10Device* d3dDevice, const DXGI_SURFACE_DESC* BackBufferDesc);
  void OnD3D10DestroyDevice();

  LRESULT HandleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

  void Move(double Time, float ElapsedTime);

  // Do any pre-rendering tasks
  void PreRender(ID3D10Device* d3dDevice);

  // Should be called between BeginScene/EndScene
  void Render(ID3D10Device* d3dDevice);

  void BeginBenchmarking();
  bool IsBenchmarking() const { return m_Benchmarking; }

  // UI
  void SetShadowDimensions(int w, int h);
  void SetScene(int s);
  void SetControlMode(int m);
  void SetAnimateLight(bool b);
  void SetLightingOnly(bool b);
  void SetFilteringTechnique(int t);
  void SetSoftness(float p);
  void SetLBR(bool b);
  void SetLBRAmount(float p);
  void SetDistributePrecision(bool s);
  void SetPSSMSplits(int s);
  void SetPSSMSplitLambda(float p);
  void SetPSSMVisualizeSplits(bool b);
  void SaveScreenshot(const std::wstring& FileName);

  // Provide a few query mechanisms to avoid duplicating error-prone defaults...
  int GetShadowWidth() const           { return m_ShadowWidth; }
  int GetShadowHeight() const          { return m_ShadowHeight; }
  int GetScene() const                 { return m_SceneIndex; }
  int GetControlMode() const           { return m_Control; }
  bool GetAnimateLight() const         { return m_AnimateLight; }
  bool GetLightingOnly() const         { return m_LightingOnly; }
  int GetFilteringTechnique() const    { return m_Technique; }
  float GetSoftness() const            { return m_Softness; }
  bool GetLBR() const                  { return m_LBR; }
  float GetLBRAmount() const           { return m_LBRAmount; }
  bool GetDistributePrecision() const         { return m_DistributePrecision; }
  int GetPSSMSplits() const            { return m_PSSMSplits; }
  float GetPSSMSplitLambda() const     { return m_PSSMSplitLambda; }
  bool GetPSSMVisualizeSplits() const  { return m_PSSMVisualizeSplits; }
  bool GetResetMSAAUI() const          { return m_ResetMSAAUI; }

  // Shadow multisampling
  // Should be called between PreRender (i.e. after resource creation) and Render
  Filtering::MSAAList QueryShadowMSAAModes(ID3D10Device* d3dDevice) const;

  // Should be called outside of rendering (after for example)
  // Also resets the MSAA UI flag
  // NOTE: Assumes that this mode is indeed supported for the current shadow technique!
  void SetShadowMSAA(const DXGI_SAMPLE_DESC& SampleDesc);

  // Names of enumerations
  static const wchar_t* GetSceneName(int s);
  static const wchar_t* GetFilteringTechniqueName(int t);

private:
  void InitEffect(ID3D10Device* d3dDevice);
  void InitScene(ID3D10Device* d3dDevice);

  // Invalidate and force recreate of Effect/Filter
  void InvalidateEffectFilter(bool ResetMultisampling = false);

  // Effect parameter functions
  void UpdateLightParameters();
  void UpdateCameraParameters();
  void UpdateUIParameters();
  // At a minimum, calls all of the above
  void UpdateParameters();

  void RenderText(ID3D10Device* d3dDevice);

  CFirstPersonCamera* GetCurrentCamera();
  const CFirstPersonCamera* GetCurrentCamera() const;

  // Renders the scene normally using the currently set technique
  void RenderScene(ID3D10Device* d3dDevice,
                   ID3D10EffectTechnique* RenderTechnique,
                   const D3DXMATRIXA16& View,
                   const D3DXMATRIXA16& Proj);

  void ProcessBenchmarkingFrame(float Time);
  void EndBenchmarking();

  void DoSaveScreenshot(ID3D10Device* d3dDevice, const std::wstring& FileName) const;

  D3D10_VIEWPORT               m_BackBufferViewport;    // Back buffer viewport

  ID3DX10Font*                 m_Font;                  // Font for drawing text
  ID3DX10Sprite*               m_Sprite;                // To speed up drawing text
  CDXUTTextHelper*             m_TextHelper;            // Makes drawing text simple

  CFirstPersonCamera           m_ViewCamera;            // View camera
  CFirstPersonCamera           m_LightCamera;           // Light camera

  Scene*                       m_Scene;                 // Current scene
  PostProcess*                 m_PostProcess;           // Post-processing helper
  Filtering*                   m_Filter;                // Our current filtering technique implementation

  float                        m_LightFOV;
  float                        m_LightLinNear;
  float                        m_LightLinFar;

  // Effect and shader constants
  ID3D10Effect*                m_Effect;                // D3D10 effect interface
  ID3D10EffectVectorVariable*  m_EffectViewPos;         // View position
  ID3D10EffectMatrixVariable*  m_EffectLightViewProj;   // Light view projection matrix
  ID3D10EffectVectorVariable*  m_EffectLightLinNearFar; // Linear near/far of the light projection
  ID3D10EffectVectorVariable*  m_EffectLightPos;        // Light position
  ID3D10EffectVectorVariable*  m_EffectLightDir;        // Light direction
  ID3D10EffectScalarVariable*  m_EffectLightFOV;        // Light field of view
  ID3D10EffectVectorVariable*  m_EffectLightDistFalloff;// Light distance falloff
  ID3D10EffectScalarVariable*  m_EffectLightingOnly;    // Lighting only enabled
  ID3D10EffectScalarVariable*  m_EffectLBR;             // Lighting bleeding reduction enabled
  ID3D10EffectScalarVariable*  m_EffectLBRAmount;       // Lighting bleeding reduction amount
  ID3D10EffectScalarVariable*  m_EffectDistributePrecision;    // Floating point distribute components enabled

  // Benchmarking mode stuff
  bool                         m_Benchmarking;          // Currently benchmarking state
  std::wofstream               m_BenchOut;              // Output file for benchmark results
  int                          m_BenchTechnique;        // Current batch filtering technique
  int                          m_BenchFilterWidth;      // Current batch minimum filter width
  int                          m_BenchFrames;           // Frames so far
  double                       m_BenchGoodTime;         // Used (non-dropped) frame times so far
  typedef std::vector<double>  TimeList;
  TimeList                     m_BenchResults;          // Record results for one test
  int                          m_PrevTechnique;         // The previous setting before benchmarking

  // UI
  int                          m_ShadowWidth;           // Width of shadow map
  int                          m_ShadowHeight;          // Height of shadow map
  int                          m_SceneIndex;            // Current scene
  int                          m_Technique;             // Current filtering technique
  int                          m_Control;               // Current control mode
  bool                         m_AnimateLight;          // Animate light or not
  bool                         m_LightingOnly;          // Show only shadow
  float                        m_Softness;              // Softness of the shadow edge
  bool                         m_LBR;                   // Light bleeding reduction
  float                        m_LBRAmount;             // LBR tweakable setting
  bool                         m_DistributePrecision;          // Distribute precision to two fp32's
  int                          m_PSSMSplits;            // Number of PSSM splits
  float                        m_PSSMSplitLambda;       // PSSM practice split scheme lambda value
  bool                         m_PSSMVisualizeSplits;   // Whether or not to visualize the splits
  DXGI_SAMPLE_DESC             m_ShadowMSAA;            // MSAA mode
  std::wstring                 m_NextScreenshotName;    // Flag to save a screenshot
  bool                         m_ResetMSAAUI;           // Flag that indicates MSAA UI needs update
};
