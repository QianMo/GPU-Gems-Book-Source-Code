#include "DXUT.h"
#include "App.hpp"

//--------------------------------------------------------------------------------------
void App::InvalidateEffectFilter(bool ResetMultisampling)
{
  SAFE_DELETE(m_PostProcess);
  SAFE_DELETE(m_Filter);
  SAFE_RELEASE(m_Effect);

  if (ResetMultisampling) {
    // Reset to no multisampling, since texture format may have changed
    m_ShadowMSAA.Count = 1;
    m_ShadowMSAA.Quality = 0;
    m_ResetMSAAUI = true;
  }
}

//--------------------------------------------------------------------------------------
void App::SetShadowDimensions(int w, int h)
{
  m_ShadowWidth = w;
  m_ShadowHeight = h;
  InvalidateEffectFilter();
}

//--------------------------------------------------------------------------------------
void App::SetScene(int s)
{
  // Release any old scene
  SAFE_DELETE(m_Scene);
  m_SceneIndex = s;
}

//--------------------------------------------------------------------------------------
void App::SetControlMode(int m)
{
  m_Control = m;
  UpdateCameraParameters();
}

//--------------------------------------------------------------------------------------
const CFirstPersonCamera* App::GetCurrentCamera() const
{
  switch (m_Control) {
    case CM_LIGHT: return &m_LightCamera;
    default:       return &m_ViewCamera;
  };
}

//--------------------------------------------------------------------------------------
CFirstPersonCamera* App::GetCurrentCamera()
{
  switch (m_Control) {
    case CM_LIGHT: return &m_LightCamera;
    default:       return &m_ViewCamera;
  };
}

//--------------------------------------------------------------------------------------
void App::SetAnimateLight(bool b)
{
  m_AnimateLight = b;
}

//--------------------------------------------------------------------------------------
void App::SetLightingOnly(bool b)
{
  m_LightingOnly = b;
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::SetFilteringTechnique(int t)
{
  m_Technique = t;
  InvalidateEffectFilter(true);
}

//--------------------------------------------------------------------------------------
void App::SetSoftness(float p)
{
  m_Softness = p;
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::SetLBR(bool b)
{
  m_LBR = b;
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::SetLBRAmount(float p)
{
  m_LBRAmount = p;
  UpdateUIParameters();
}  

//--------------------------------------------------------------------------------------
void App::SetDistributePrecision(bool s)
{
  m_DistributePrecision = s;
  InvalidateEffectFilter(true);    // Texture format may change => new MSAA options
}

//--------------------------------------------------------------------------------------
void App::SetPSSMSplits(int s)
{
  m_PSSMSplits = s;
  InvalidateEffectFilter();
}

//--------------------------------------------------------------------------------------
void App::SetPSSMSplitLambda(float p)
{
  m_PSSMSplitLambda = p;
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::SetPSSMVisualizeSplits(bool b)
{
  m_PSSMVisualizeSplits = b;
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::SaveScreenshot(const std::wstring& FileName)
{
  m_NextScreenshotName = FileName;
}

//--------------------------------------------------------------------------------------
Filtering::MSAAList App::QueryShadowMSAAModes(ID3D10Device* d3dDevice) const
{
  return m_Filter->QueryMSAASupport(d3dDevice);
}

//--------------------------------------------------------------------------------------
void App::SetShadowMSAA(const DXGI_SAMPLE_DESC& SampleDesc)
{
  InvalidateEffectFilter();
  // Now setup for this mode on the next creation
  m_ShadowMSAA = SampleDesc;
  m_ResetMSAAUI = false;
}

//--------------------------------------------------------------------------------------
const wchar_t* App::GetSceneName(int s)
{
  switch (s) {
    case S_CAR:      return L"Car";
    case S_SPHERES:  return L"Spheres";
    case S_COMMANDO: return L"Commando";
    case S_LARGE:    return L"Convoy";
    default:         return L"Unknown";
  }
}

//--------------------------------------------------------------------------------------
const wchar_t* App::GetFilteringTechniqueName(int t)
{
  switch (t) {
    case FT_POINT:    return L"Shadow Map";
    case FT_PCF:      return L"PCF";
    case FT_VSM:      return L"VSM";
    case FT_PSVSM:    return L"PSVSM";
    case FT_SAVSMFP:  return L"SAVSM (FP)";
    case FT_SAVSMINT: return L"SAVSM (INT)";
    default:          return L"Unknown";
  }
}
