#include "DXUT.h"
#include "App.hpp"
#include "Point.hpp"
#include "PCF.hpp"
#include "StandardSAT.hpp"
#include "Hardware.hpp"
#include "PSVSM.hpp"

//--------------------------------------------------------------------------------------
// Constants
namespace {
  // Camera parameters
  const float c_CameraFOV = D3DX_PI / 4;
  const float c_CameraNear =  0.01f;
  const float c_CameraFar  = 4.5f;
  const float c_CameraRotSpeed  = 0.008f;
  const float c_CameraMoveSpeed = 0.6f;

  // Light parameters
  const float c_LightMinNear = 0.04f;

  // Animation parameters
  const float c_LightAnimateSpeed = 0.15f;    // radians/sec

  // Font parameters
  const wchar_t c_Font[] = L"Verdana";
  const unsigned int c_FontHeight = 18;
  const unsigned int c_FontWidth = 0;         // Auto
  const unsigned int c_FontSpriteBufferSize = 512;

  // Benchmarking
  const wchar_t c_BenchOutFile[] = L"Bench.csv";
  const int c_BenchMaxFilterWidth = 16;
  const int c_BenchDropFrames = 50;           // Should always be at least 1!
  const double c_BenchTime = 2.0f;            // Per test, in seconds
  const int c_BenchMinGoodFrames = 100;
}
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
App::App()
  : m_Font(0), m_Sprite(0), m_TextHelper(0)
  , m_Effect(0), m_PostProcess(0), m_Filter(0), m_Benchmarking(false)
  , m_ShadowWidth(512), m_ShadowHeight(512)
  , m_Technique(FT_VSM), m_Control(CM_CAMERA)
  , m_LightingOnly(false), m_Softness(0.25f)
  , m_LBR(true), m_LBRAmount(0.18f)
  , m_DistributePrecision(true), m_PSSMSplits(3), m_PSSMSplitLambda(0.65f)
  , m_PSSMVisualizeSplits(false)
  , m_ResetMSAAUI(true)
{
  // Other initialization
  m_ShadowMSAA.Count = 1;
  m_ShadowMSAA.Quality = 0;

  SetScene(S_CAR);

  D3DXVECTOR3 Target(0.0f, 0.0f, 0.0f);

  // Camera parameters
  m_ViewCamera.SetScalers(c_CameraRotSpeed, c_CameraMoveSpeed);
  m_ViewCamera.SetRotateButtons(true, false, false);

  // Light parameters
  m_LightCamera.SetScalers(c_CameraRotSpeed, c_CameraMoveSpeed);
  m_LightCamera.SetRotateButtons(true, false, false);
}

//--------------------------------------------------------------------------------------
void App::OnD3D10CreateDevice(ID3D10Device* d3dDevice)
{
  HRESULT hr;

  // These will be recreated on demand
  SAFE_RELEASE(m_Effect);
  SAFE_DELETE(m_Filter);
  SAFE_DELETE(m_PostProcess);
  SAFE_DELETE(m_Scene);

  // Initialize the font
  V(D3DX10CreateFont(d3dDevice, c_FontHeight, c_FontWidth, FW_BOLD, 1, FALSE,
    DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
    c_Font, &m_Font));

  // Sprite object for speeding up font rendering
  V(D3DX10CreateSprite(d3dDevice, c_FontSpriteBufferSize, &m_Sprite));

  // Helper for rendering text
  m_TextHelper = new CDXUTTextHelper(0, 0, m_Font, m_Sprite, c_FontHeight);
}

//--------------------------------------------------------------------------------------
void App::OnD3D10ResizedSwapChain(ID3D10Device* d3dDevice, const DXGI_SURFACE_DESC* BackBufferDesc)
{
  // Setup back buffer viewport
  m_BackBufferViewport.Width    = BackBufferDesc->Width;
  m_BackBufferViewport.Height   = BackBufferDesc->Height;
  m_BackBufferViewport.MinDepth = 0.0f;
  m_BackBufferViewport.MaxDepth = 1.0f;
  m_BackBufferViewport.TopLeftX = 0;
  m_BackBufferViewport.TopLeftY = 0;

  // Setup the camera projection parameters
  float AspectRatio = BackBufferDesc->Width / static_cast<float>(BackBufferDesc->Height);
  m_ViewCamera.SetProjParams(c_CameraFOV, AspectRatio, c_CameraNear, c_CameraFar);
}

//--------------------------------------------------------------------------------------
void App::OnD3D10DestroyDevice()
{
  SAFE_RELEASE(m_Effect);
  SAFE_DELETE(m_Filter);
  SAFE_DELETE(m_PostProcess);
  SAFE_DELETE(m_Scene);
  
  SAFE_RELEASE(m_Font);
  SAFE_RELEASE(m_Sprite);
  SAFE_DELETE(m_TextHelper);
}

//--------------------------------------------------------------------------------------
LRESULT App::HandleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
  // Handle camera input
  return GetCurrentCamera()->HandleMessages(hWnd, uMsg, wParam, lParam);
}

//--------------------------------------------------------------------------------------
void App::Move(double Time, float ElapsedTime)
{
  // Animate light if enabled
  if (m_AnimateLight && m_Scene) {
    D3DXVECTOR3 Eye = *m_LightCamera.GetEyePt();
    D3DXVECTOR3 LookAt = *m_LightCamera.GetLookAtPt();
    D3DXVECTOR3 SceneCenter = m_Scene->GetCenter();

    // Rotate around center of the scene
    float Amount = c_LightAnimateSpeed * ElapsedTime;
    Eye -= SceneCenter;
    LookAt -= SceneCenter;
    D3DXMATRIX Rotate;
    D3DXMatrixRotationY(&Rotate, Amount);
    D3DXVECTOR4 NewEye;
    D3DXVECTOR4 NewLookAt;
    D3DXVec3Transform(&NewEye, &Eye, &Rotate);
    D3DXVec3Transform(&NewLookAt, &LookAt, &Rotate);
    Eye = D3DXVECTOR3(NewEye);
    LookAt = D3DXVECTOR3(NewLookAt);
    Eye += SceneCenter;
    LookAt += SceneCenter;

    m_LightCamera.SetViewParams(&Eye, &LookAt);
  }

  // Pass it on anywhere
  m_ViewCamera.FrameMove(ElapsedTime);
  m_LightCamera.FrameMove(ElapsedTime);

  // Update parameters
  UpdateCameraParameters();
  UpdateLightParameters();

  // If we're benchmarking, process our last frame time
  ProcessBenchmarkingFrame(ElapsedTime);
}

//--------------------------------------------------------------------------------------
void App::UpdateCameraParameters()
{
  if (!m_Effect) return;

  HRESULT hr;
  V(m_EffectViewPos->SetFloatVector(D3DXVECTOR4(*GetCurrentCamera()->GetEyePt(), 0)));
}

//--------------------------------------------------------------------------------------
void App::UpdateLightParameters()
{  
  if (!m_Effect || !m_Scene) return;

  // Get scene constants
  D3DXVECTOR2 DistFalloff;
  m_Scene->GetLightConstants(m_LightFOV, DistFalloff);

  // Compute near and far
  float Near = c_LightMinNear;
  float Far  = DistFalloff.y;

  // Compute linear (distance to light) near and min/max for this perspective matrix
  float CosLightFOV = std::cos(0.5f * m_LightFOV);
  m_LightLinNear = Near;
  m_LightLinFar  = Far / (CosLightFOV * CosLightFOV);

  // Note that aspect is always 1.0 since we always use circular lights
  m_LightCamera.SetProjParams(m_LightFOV, 1.0f, Near, Far);
  D3DXMATRIXA16 LightViewProj = *m_LightCamera.GetViewMatrix() *
                                *m_LightCamera.GetProjMatrix();

  // Update global parameters
  D3DXVECTOR3 LightPos    = *m_LightCamera.GetEyePt();
  D3DXVECTOR3 LightTarget = *m_LightCamera.GetLookAtPt();
  D3DXVECTOR3 LightDir    = LightTarget - LightPos;
  D3DXVec3Normalize(&LightDir, &LightDir);

  // Set the effect constants
  HRESULT hr;
  V(m_EffectLightViewProj->SetMatrix(LightViewProj));
  V(m_EffectLightLinNearFar->SetFloatVector(D3DXVECTOR4(m_LightLinNear,
                                                        m_LightLinFar, 0, 0)));
  V(m_EffectLightPos->SetFloatVector(D3DXVECTOR4(LightPos, 0)));
  V(m_EffectLightDir->SetFloatVector(D3DXVECTOR4(LightDir, 0)));
  V(m_EffectLightFOV->SetFloat(m_LightFOV));
  V(m_EffectLightDistFalloff->SetFloatVector(D3DXVECTOR4(DistFalloff.x,
                                                         DistFalloff.y, 0, 0)));
}

//--------------------------------------------------------------------------------------
void App::UpdateUIParameters()
{
  if (!m_Effect || !m_Filter) return;

  HRESULT hr;
  V(m_EffectLightingOnly->SetBool(m_LightingOnly));
  V(m_EffectLBR->SetBool(m_LBR));
  V(m_EffectLBRAmount->SetFloat(m_LBRAmount));
  V(m_EffectDistributePrecision->SetBool(m_DistributePrecision));

  // PSSM related UI
  if (m_Technique == FT_PSVSM) {
    PSVSM* PSSMFilter = dynamic_cast<PSVSM*>(m_Filter);
    assert(PSSMFilter);
    PSSMFilter->SetSplitLambda(m_PSSMSplitLambda);
    PSSMFilter->SetVisualizeSplits(m_PSSMVisualizeSplits);
  }

  if (m_Benchmarking) {
    // Use the benchmarking parameters
    m_Filter->SetMinFilterWidth(m_BenchFilterWidth);
  } else {
    // Compute the maximum LOD from the softness setting
    // Note that this effectively makes it a logarithmic slider of filter width,
    // which is convenient and intuitive.
    float MaxDim = static_cast<float>(std::max(m_ShadowWidth, m_ShadowHeight));
    float LogMaxDim = std::log(MaxDim) / std::log(2.0f);
    float MaxLOD = m_Softness * LogMaxDim;

    m_Filter->SetMaxLOD(MaxLOD);
  }
}

//--------------------------------------------------------------------------------------
void App::UpdateParameters()
{
  if (!m_Effect) return;

  UpdateCameraParameters();
  UpdateLightParameters();
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::PreRender(ID3D10Device* d3dDevice)
{
  // (Re)initialize the effect if necessary
  if (!m_Effect || !m_Filter) {
    InitEffect(d3dDevice);
  }

  // (Re)initialize the scene if necessary
  if (!m_Scene) {
    InitScene(d3dDevice);
  }
}

//--------------------------------------------------------------------------------------
void App::Render(ID3D10Device* d3dDevice)
{
  // Grab references to our back and depth buffers
  ID3D10RenderTargetView *BackBuffer = DXUTGetD3D10RenderTargetView();
  ID3D10DepthStencilView *DepthBuffer = DXUTGetD3D10DepthStencilView();

  const CBaseCamera* Camera = GetCurrentCamera();

  m_Filter->BeginFrame(d3dDevice); {
    D3DXMATRIXA16 View, Proj;

    // Render and setup the shadow map
    ID3D10EffectTechnique* ShadowTechnique;
    do {
      D3DXMATRIXA16 LightView = *m_LightCamera.GetViewMatrix();
      D3DXMATRIXA16 LightProj = *m_LightCamera.GetProjMatrix();
      ShadowTechnique = m_Filter->BeginShadowMap(LightView, LightProj,
                                                 *m_LightCamera.GetEyePt(),
                                                 m_LightLinFar,
                                                 *Camera);
      RenderScene(d3dDevice, ShadowTechnique, LightView, LightProj);
    } while (m_Filter->EndShadowMap(ShadowTechnique));

    // Render the scene
    ID3D10EffectTechnique* ShadingTechnique = m_Filter->BeginShading(); {
      // Clear buffers
      d3dDevice->ClearRenderTargetView(BackBuffer, D3DXVECTOR4(0.0f, 0.0f, 0.0f, 0.0f));
      d3dDevice->ClearDepthStencilView(DepthBuffer, D3D10_CLEAR_DEPTH, 1.0f, 0);

      // Setup rendering to back buffer
      d3dDevice->OMSetRenderTargets(1, &BackBuffer, DepthBuffer);
      d3dDevice->RSSetViewports(1, &m_BackBufferViewport);

      // Render and shade scene
      RenderScene(d3dDevice, ShadingTechnique,
                  *Camera->GetViewMatrix(), *Camera->GetProjMatrix());
    } m_Filter->EndShading(ShadingTechnique);

    // Optionally visualize shadow map(s)
    if (m_PSSMVisualizeSplits) {
      D3D10_VIEWPORT ShadowViewport(m_BackBufferViewport);
      ShadowViewport.TopLeftX = 25;
      ShadowViewport.Width = m_BackBufferViewport.Width - 2 * ShadowViewport.TopLeftX;
      ShadowViewport.Height = 150;
      ShadowViewport.TopLeftY = m_BackBufferViewport.Height - ShadowViewport.Height -
                                ShadowViewport.TopLeftX;
      m_Filter->DisplayShadowMap(ShadowViewport);
    }
  } m_Filter->EndFrame();

  // Restore viewport for any HUD drawing, etc.
  d3dDevice->RSSetViewports(1, &m_BackBufferViewport);

  // Save a screenshot before text/UI if requested
  if (!m_NextScreenshotName.empty()) {
    DoSaveScreenshot(d3dDevice, m_NextScreenshotName);
    m_NextScreenshotName.clear();
  }

  // Any text
  RenderText(d3dDevice);
}

//--------------------------------------------------------------------------------------
void App::InitEffect(ID3D10Device* d3dDevice)
{
  HRESULT hr;

  SAFE_RELEASE(m_Effect);
  SAFE_DELETE(m_Filter);
  SAFE_DELETE(m_PostProcess);

  // Select the proper shader file
  std::wstring Shader = L"Shaders\\";
  switch (m_Technique) {
  case FT_PCF:      Shader += L"PCF";
                    break;
  case FT_POINT:    Shader += L"Point";
                    break;
  case FT_VSM:      Shader += L"Hardware";
                    break;
  case FT_PSVSM:    Shader += L"PSVSM";
                    break;
  case FT_SAVSMFP:  Shader += L"SATFP";
                    break;
  case FT_SAVSMINT: Shader += L"SATINT";
                    break;
  };
  
  // Append precompiled shader extension
  Shader += L".fxo";

  // Load!
  V(D3DX10CreateEffectFromFile(Shader.c_str(), NULL, NULL, "fx_4_0",
                               0, 0, d3dDevice, NULL, NULL, &m_Effect,
                               NULL, NULL));

  // Grab all effect interfaces
  m_EffectViewPos = m_Effect->GetVariableByName("g_ViewPos")->AsVector();
  assert(m_EffectViewPos && m_EffectViewPos->IsValid());
  m_EffectLightViewProj = m_Effect->GetVariableByName("g_LightViewProjMatrix")->AsMatrix();
  assert(m_EffectLightViewProj && m_EffectLightViewProj->IsValid());
  m_EffectLightLinNearFar = m_Effect->GetVariableByName("g_LightLinNearFar")->AsVector();
  assert(m_EffectLightLinNearFar && m_EffectLightLinNearFar->IsValid());
  m_EffectLightPos = m_Effect->GetVariableByName("g_LightPosition")->AsVector();
  assert(m_EffectLightPos && m_EffectLightPos->IsValid());
  m_EffectLightDir = m_Effect->GetVariableByName("g_LightDirection")->AsVector();
  assert(m_EffectLightDir && m_EffectLightDir->IsValid());
  m_EffectLightFOV = m_Effect->GetVariableByName("g_LightFOV")->AsScalar();
  assert(m_EffectLightFOV && m_EffectLightFOV->IsValid());
  m_EffectLightDistFalloff = m_Effect->GetVariableByName("g_LightDistFalloff")->AsVector();
  assert(m_EffectLightDistFalloff && m_EffectLightDistFalloff->IsValid());
  m_EffectLightFOV = m_Effect->GetVariableByName("g_LightFOV")->AsScalar();
  assert(m_EffectLightFOV && m_EffectLightFOV->IsValid());
  m_EffectLightingOnly = m_Effect->GetVariableByName("g_LightingOnly")->AsScalar();
  assert(m_EffectLightingOnly && m_EffectLightingOnly->IsValid());
  m_EffectLBR = m_Effect->GetVariableByName("g_LBR")->AsScalar();
  assert(m_EffectLBR && m_EffectLBR->IsValid());
  m_EffectLBRAmount = m_Effect->GetVariableByName("g_LBRAmount")->AsScalar();
  assert(m_EffectLBRAmount && m_EffectLBRAmount->IsValid());
  m_EffectDistributePrecision = m_Effect->GetVariableByName("g_DistributePrecision")->AsScalar();
  assert(m_EffectDistributePrecision && m_EffectDistributePrecision->IsValid());

  // Update with new effect interface
  if (m_Scene) {
    m_Scene->SetEffect(d3dDevice, m_Effect);
  }

  // (Re)initialize the post processing helper
  m_PostProcess = new PostProcess(d3dDevice, m_Effect);

  // Recreate filtering
  switch (m_Technique) {
  case FT_POINT:    m_Filter = new Point(d3dDevice, m_Effect,
                                         m_ShadowWidth, m_ShadowHeight);
                    break;
  case FT_PCF:      m_Filter = new PCF(d3dDevice, m_Effect,
                                       m_ShadowWidth, m_ShadowHeight);
                    break;
  case FT_VSM:      m_Filter = new Hardware(d3dDevice, m_Effect,
                                            m_ShadowWidth, m_ShadowHeight,
                                            m_PostProcess, &m_ShadowMSAA);
                    break;
  case FT_PSVSM:    m_Filter = new PSVSM(d3dDevice, m_Effect,
                                         m_ShadowWidth, m_ShadowHeight,
                                         m_PostProcess, &m_ShadowMSAA,
                                         m_PSSMSplits);
                    break;
  case FT_SAVSMFP:  m_Filter = new StandardSAT(d3dDevice, m_Effect,
                                               m_ShadowWidth, m_ShadowHeight,
                                               m_PostProcess, &m_ShadowMSAA,
                                               false, m_DistributePrecision);
                    break;
  // NOTE: INT distribute precision not implemented since it does not seem
  // to be necessary most of the time.
  case FT_SAVSMINT: m_Filter = new StandardSAT(d3dDevice, m_Effect,
                                               m_ShadowWidth, m_ShadowHeight,
                                               m_PostProcess, &m_ShadowMSAA,
                                               true, false);
                    break;
  };

  // Now (re)setup all parameters
  UpdateParameters();
}

//--------------------------------------------------------------------------------------
void App::RenderText(ID3D10Device* d3dDevice)
{
  // Text output non-trivially affects benchmark results...
  if (m_Benchmarking) {
    return;
  }

  m_TextHelper->Begin();

  m_TextHelper->SetInsertionPos(5, 2);
  m_TextHelper->SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 0.0f, 1.0f));
  
  // Normal output
  m_TextHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));
  m_TextHelper->DrawTextLine(DXUTGetDeviceStats());

  if (m_Filter) {
    {
      std::wostringstream oss;
      oss << L"Shadow Format: "
          << DXUTDXGIFormatToString(m_Filter->GetShadowFormat(), false);
      m_TextHelper->DrawTextLine(oss.str().c_str());
    }
  }

  m_TextHelper->End();
}

//--------------------------------------------------------------------------------------
void App::BeginBenchmarking()
{
  if (m_Benchmarking) return;

  // Open up the output file and setup the stream
  m_BenchOut.open(c_BenchOutFile);
  m_BenchOut.precision(8);

  // Write column headers (element 0,0 is blank)
  for (int i = 1; i <= c_BenchMaxFilterWidth; ++i) {
    m_BenchOut << "," << i;
  }
  m_BenchOut << std::endl;

  m_PrevTechnique = m_Technique;
  m_Benchmarking = true;
  m_BenchTechnique = -1;      // Init flag
  m_BenchResults.resize(0);
  m_BenchResults.reserve(c_BenchMaxFilterWidth);
}

//--------------------------------------------------------------------------------------
void App::ProcessBenchmarkingFrame(float Time)
{
  if (!m_Benchmarking) return;

  bool ResetCounters = false;

  // Process this frame time
  if (m_BenchTechnique >= 0) {
    ++m_BenchFrames;
    if (m_BenchFrames > c_BenchDropFrames) {
      m_BenchGoodTime += Time;
    }

    // Done this test?
    int GoodFrames = m_BenchFrames - c_BenchDropFrames;
    if (m_BenchGoodTime >= c_BenchTime && GoodFrames >= c_BenchMinGoodFrames) {
      double AvgFrameTime = m_BenchGoodTime / static_cast<double>(GoodFrames);
      double AvgFrameTimeMs = AvgFrameTime * 1000.0;
      m_BenchResults.push_back(AvgFrameTimeMs);
      
      // Go on to the next test
      ++m_BenchFilterWidth;
      UpdateUIParameters();
      ResetCounters = true;
    }
  }

  if (m_BenchTechnique < 0 || m_BenchFilterWidth > c_BenchMaxFilterWidth) {
    // Record previous test results
    if (m_BenchTechnique >= 0) {
      m_BenchOut << GetFilteringTechniqueName(m_BenchTechnique);
      for (TimeList::const_iterator i = m_BenchResults.begin();
           i != m_BenchResults.end(); ++i) {
        m_BenchOut << "," << *i;
      }
      m_BenchOut << std::endl;
    }

    // Next technique
    ++m_BenchTechnique;
    m_BenchFilterWidth = 1;
    m_BenchResults.clear();
    ResetCounters = true;

    // Are we done?
    if (m_BenchTechnique >= FT_NUM) {
      EndBenchmarking();
    } else {
      SetFilteringTechnique(m_BenchTechnique);
    }
  }

  if (ResetCounters) {
    m_BenchFrames = 0;
    m_BenchGoodTime = 0.0f;
  }
}

//--------------------------------------------------------------------------------------
void App::EndBenchmarking()
{
  if (!m_Benchmarking) return;

  // Close the output file
  m_BenchOut.close();

  // Restore user settings
  m_Benchmarking = false;
  SetFilteringTechnique(m_PrevTechnique);
  UpdateUIParameters();
}

//--------------------------------------------------------------------------------------
void App::DoSaveScreenshot(ID3D10Device* d3dDevice, const std::wstring& FileName) const
{
  HRESULT hr;
 
  // Grab back buffer surface info
  ID3D10Resource *BackBufferResource;
  DXUTGetD3D10RenderTargetView()->GetResource(&BackBufferResource);
  const DXGI_SURFACE_DESC* BBDesc = DXUTGetDXGIBackBufferSurfaceDesc();

  // Create a non multisampled surface (to resolve to)
  D3D10_TEXTURE2D_DESC Desc;
  ZeroMemory(&Desc, sizeof(Desc));
  Desc.Width            = BBDesc->Width;
  Desc.Height           = BBDesc->Height;
  Desc.MipLevels        = 1;
  Desc.ArraySize        = 1;
  Desc.Format           = BBDesc->Format;
  Desc.SampleDesc.Count = 1;
  
  ID3D10Texture2D *Texture;
  V(d3dDevice->CreateTexture2D(&Desc, 0, &Texture));

  // Resolve
  d3dDevice->ResolveSubresource(Texture, D3D10CalcSubresource(0, 0, 0),
                                BackBufferResource, D3D10CalcSubresource(0, 0, 0),
                                BBDesc->Format);

  V(D3DX10SaveTextureToFile(Texture, D3DX10_IFF_BMP,
                            (m_NextScreenshotName + L".bmp").c_str()));

  // Cleanup
  SAFE_RELEASE(BackBufferResource);
  SAFE_RELEASE(Texture);
}