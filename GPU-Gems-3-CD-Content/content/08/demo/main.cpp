#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"

#include "App.hpp"
#include "resource.h"
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Constants
//--------------------------------------------------------------------------------------
static const bool c_EnableShadowMultisampling = true;
static const unsigned int c_FactorSliderRes = 1000;


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
// This is also the draw order, so choose it wisely!
enum HUDEnum {
  HUD_SATFP = 0,
  HUD_PSSM,
  HUD_VSM,
  HUD_SOFTNESS,
  HUD_SHADOW,
  HUD_GENERIC,
  HUD_NUM
};

CDXUTDialogResourceManager   g_DialogResourceManager;   // Manager for shared resources of dialogs
CD3DSettingsDlg              g_SettingsDlg;             // Device settings dialog
App                          g_App;                     // Application interface
CDXUTDialog                  g_HUD[HUD_NUM];            // UI dialogs
Filtering::MSAAList          g_ShadowMSAAModes;         // Supported shadow msaa modes

// Multisampling has to be done at a certain time in the frame, so it needs state
bool                         g_UpdateSelectedMultisamplingType;


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
enum UIControls {
  UI_TOGGLEFULLSCREEN = 1,
  UI_CHANGEDEVICE,
  UI_SCENE,
  UI_CONTROLMODE,
  UI_ANIMATELIGHT,
  UI_LIGHTINGONLY,
  UI_FILTERINGTECHNIQUE,
  UI_DIMENSIONS,
  UI_SOFTNESSTEXT,
  UI_SOFTNESS,
  UI_SHADOWMSAATEXT,
  UI_MULTISAMPLING,
  UI_LBR,
  UI_LBRAMOUNT,
  UI_PSSMSPLITSTEXT,
  UI_PSSMSPLITS,
  UI_PSSMSPLITLAMBDATEXT,
  UI_PSSMSPLITLAMBDA,
  UI_PSSMVISUALIZESPLITS,
  UI_SATDISTRIBUTEPRECISION
};


//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
void    InitializeDialogs();
void    InitializeDXUT();
void    UpdateHUDVisibility();

// DXUT general callbacks
void    CALLBACK OnFrameMove(double Time, float ElapsedTime, void* UserContext);
LRESULT CALLBACK MsgProc(HWND Wnd, UINT Msg, WPARAM wParam, LPARAM lParam, bool* NoFurtherProcessing, void* UserContext);
void    CALLBACK OnKeyboard(UINT Char, bool KeyDown, bool AltDown, void* UserContext);
void    CALLBACK OnGUIEvent(UINT Event, int ControlID, CDXUTControl* Control, void* UserContext);

// DXUT Direct3D 10 callbacks
bool    CALLBACK IsD3D10DeviceAcceptable(UINT Adapter, UINT Output, D3D10_DRIVER_TYPE DeviceType, DXGI_FORMAT BackBufferFormat, bool Windowed, void* UserContext);
HRESULT CALLBACK OnD3D10CreateDevice(ID3D10Device* d3dDevice, const DXGI_SURFACE_DESC* BackBufferSurfaceDesc, void* UserContext);
HRESULT CALLBACK OnD3D10ResizedSwapChain(ID3D10Device* d3dDevice, IDXGISwapChain* SwapChain, const DXGI_SURFACE_DESC* BackBufferSurfaceDesc, void* UserContext);
void    CALLBACK OnD3D10ReleasingSwapChain(void* UserContext);
void    CALLBACK OnD3D10DestroyDevice(void* UserContext);
void    CALLBACK OnD3D10FrameRender(ID3D10Device* d3dDevice, double Time, float ElapsedTime, void* UserContext);


//--------------------------------------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
  // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
  _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

  // Set DXUT callbacks
  DXUTSetCallbackMsgProc(MsgProc);
  DXUTSetCallbackKeyboard(OnKeyboard);
  DXUTSetCallbackFrameMove(OnFrameMove);

  DXUTSetCallbackD3D10DeviceAcceptable(IsD3D10DeviceAcceptable);
  DXUTSetCallbackD3D10DeviceCreated(OnD3D10CreateDevice);
  DXUTSetCallbackD3D10SwapChainResized(OnD3D10ResizedSwapChain);
  DXUTSetCallbackD3D10FrameRender(OnD3D10FrameRender);
  DXUTSetCallbackD3D10SwapChainReleasing(OnD3D10ReleasingSwapChain);
  DXUTSetCallbackD3D10DeviceDestroyed(OnD3D10DestroyDevice);

  InitializeDialogs();
  InitializeDXUT();

  // Pass control to DXUT for handling the message pump and 
  // dispatching render calls. DXUT will call your FrameMove 
  // and FrameRender callback when there is idle time between handling window messages.
  try {
    DXUTMainLoop();
  } catch (std::exception &e) {
    // Convert error message to wstring and display
    std::string e_str = e.what();
    std::wstring e_wstr(e_str.begin(), e_str.end());
    MessageBox(DXUTGetHWND(), e_wstr.c_str(), L"Runtime Exception", MB_OK | MB_ICONERROR);
  }

  return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Sets up the dialogs
//--------------------------------------------------------------------------------------
void InitializeDialogs()
{
  g_SettingsDlg.Init(&g_DialogResourceManager);
  
  // Create HUDs
  for (int i = 0; i < HUD_NUM; ++i) {
    g_HUD[i].Init(&g_DialogResourceManager);
    g_HUD[i].SetCallback(OnGUIEvent);
  }
  
  // Generic HUD
  {
    int y = 10;
    CDXUTDialog &HUD = g_HUD[HUD_GENERIC];
    
    HUD.AddButton(UI_TOGGLEFULLSCREEN, L"Toggle Full Screen", 40, y, 160, 22);
    y += 24;
    HUD.AddButton(UI_CHANGEDEVICE,     L"Change Device (F2)", 40, y, 160, 22, VK_F2);
    y += 35;

    CDXUTComboBox* Scene;
    HUD.AddComboBox(UI_SCENE, 40, y, 160, 22, 0, false, &Scene);
    y += 30;
    for (int i = 0; i < App::S_NUM; ++i) {
      Scene->AddItem(App::GetSceneName(i), IntToPtr(i));
    }
    Scene->SetSelectedByIndex(g_App.GetScene());

    CDXUTComboBox* ControlMode;
    HUD.AddComboBox(UI_CONTROLMODE, 40, y, 160, 22, 0, false, &ControlMode);
    y += 30;
    ControlMode->AddItem(L"Camera", IntToPtr(App::CM_CAMERA));
    ControlMode->AddItem(L"Light", IntToPtr(App::CM_LIGHT));
    ControlMode->SetSelectedByData(IntToPtr(g_App.GetControlMode()));

    HUD.AddCheckBox(UI_ANIMATELIGHT, L"Animate Light (Spacebar)", 40, y, 160, 22,
                    g_App.GetAnimateLight(), VK_SPACE);
    y += 30;
  }

  // Shadow HUD
  {
    int y = 10;
    CDXUTDialog &HUD = g_HUD[HUD_SHADOW];

    CDXUTComboBox* Technique;
    HUD.AddComboBox(UI_FILTERINGTECHNIQUE, 40, y, 160, 22, VK_F5, false, &Technique);
    for (int i = 0; i < App::FT_NUM; ++i) {
      Technique->AddItem(App::GetFilteringTechniqueName(i), IntToPtr(i));
    }
    Technique->SetSelectedByIndex(g_App.GetFilteringTechnique());
    y += 30;

    // Only bother to support square shadow map sizes for UI simplicity
    CDXUTComboBox *Dim;
    HUD.AddComboBox(UI_DIMENSIONS, 40, y, 160, 22, VK_F6, false, &Dim);
    y += 30;
    Dim->AddItem(L"128x128", IntToPtr(128));
    Dim->AddItem(L"256x256", IntToPtr(256));
    Dim->AddItem(L"512x512", IntToPtr(512));
    Dim->AddItem(L"1024x1024", IntToPtr(1024));
    Dim->AddItem(L"2048x2048", IntToPtr(2048));
    Dim->SetSelectedByData(IntToPtr(g_App.GetShadowWidth()));

    HUD.AddCheckBox(UI_LIGHTINGONLY, L"Lighting Only", 40, y, 160, 22, g_App.GetLightingOnly(), 0);
    y += 30;
  }

  // Softness HUD
  {
    int y = 10;
    CDXUTDialog &HUD = g_HUD[HUD_SOFTNESS];

    HUD.AddStatic(UI_SOFTNESSTEXT, L"Softness:",  0, y, 136, 18);
    y += 22;
    // Need a special range including negatives... this effectively chooses the LOD from
    // which minimum filter width can be computed.
    HUD.AddSlider(UI_SOFTNESS, 45, y, 150, 22, 
                  static_cast<int>(-0.25f * c_FactorSliderRes),
                  static_cast<int>(0.60f * c_FactorSliderRes),
                  static_cast<int>(g_App.GetSoftness() * c_FactorSliderRes));
    y += 40;
  }

  // VSM HUD
  {
    int y = 10;
    CDXUTDialog &HUD = g_HUD[HUD_VSM];

    HUD.AddCheckBox(UI_LBR, L"Light Bleeding Reduction:",  40, y, 160, 22, g_App.GetLBR(), 0);
    y += 24;
    // Only allow us to go up to 90% on this one
    HUD.AddSlider(UI_LBRAMOUNT, 45, y, 150, 22, 0,
                  static_cast<int>(0.9f * c_FactorSliderRes),
                  static_cast<int>(g_App.GetLBRAmount() * c_FactorSliderRes));
    y += 40;

    if (c_EnableShadowMultisampling) {
      // Full contents will be set up later
      HUD.AddComboBox(UI_MULTISAMPLING, 40, y, 160, 22, VK_F7);
      y += 30;
    }
  }

  // PSSM HUD
  {
    int y = 10;
    CDXUTDialog &HUD = g_HUD[HUD_PSSM];

    HUD.AddStatic(UI_PSSMSPLITSTEXT, L"Number of Splits:",  0, y, 170, 18);
    y += 22;
    HUD.AddSlider(UI_PSSMSPLITS, 45, y, 150, 22, 1, 4, g_App.GetPSSMSplits());
    y += 40;

    HUD.AddStatic(UI_PSSMSPLITLAMBDATEXT, L"Split Distribution (Uniform / Log):", 0, y, 240, 18);
    y += 22;
    HUD.AddSlider(UI_PSSMSPLITLAMBDA, 45, y, 150, 22, 0,
                  c_FactorSliderRes,
                  static_cast<int>(g_App.GetPSSMSplitLambda() * c_FactorSliderRes));
    y += 40;

    HUD.AddCheckBox(UI_PSSMVISUALIZESPLITS,  L"Visualize Splits", 40, y, 160, 22,
                    g_App.GetPSSMVisualizeSplits(), 0);
  }

  // SAT HUD
  {
    int y = 10;
    CDXUTDialog &HUD = g_HUD[HUD_SATFP];
    
    HUD.AddCheckBox(UI_SATDISTRIBUTEPRECISION, L"Distribute Precision", 40, y, 160, 22,
                    g_App.GetDistributePrecision(), 0);
  }

  // Setup initial visibility
  UpdateHUDVisibility();
}


//--------------------------------------------------------------------------------------
// Chooses a mode and initializes DXUT
//--------------------------------------------------------------------------------------
void InitializeDXUT()
{
  // Initialize DXUT
  DXUTInit();
  DXUTSetCursorSettings(true, true);
  DXUTCreateWindow(L"Variance Shadow Maps (D3D10)");
  
  // Default settings
  DXUTDeviceSettings s;
  ZeroMemory(&s, sizeof(s));
  s.ver                     = DXUT_D3D10_DEVICE;
  s.d3d10.DriverType        = D3D10_DRIVER_TYPE_HARDWARE;
  s.d3d10.CreateFlags       = 0;
#ifdef _DEBUG
  s.d3d10.CreateFlags      |= D3D10_CREATE_DEVICE_DEBUG;
#endif
  s.d3d10.SyncInterval      = 0;
  s.d3d10.AutoCreateDepthStencil = true;
  s.d3d10.AutoDepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;

  // Back buffer
  s.d3d10.sd.BufferDesc.Width       = 1024;
  s.d3d10.sd.BufferDesc.Height      = 768;
  s.d3d10.sd.BufferDesc.Format      = DXGI_FORMAT_R8G8B8A8_UNORM;
  s.d3d10.sd.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
  s.d3d10.sd.BufferDesc.Scaling     = DXGI_MODE_SCALING_UNSPECIFIED;
  s.d3d10.sd.BufferUsage            = 0;
  s.d3d10.sd.OutputWindow           = NULL;
  s.d3d10.sd.Windowed               = true;
  s.d3d10.sd.SwapEffect             = DXGI_SWAP_EFFECT_DISCARD;
  s.d3d10.sd.Flags                  = 0;
  // Multisampling
  s.d3d10.sd.SampleDesc.Count       = 4;   // 4x MSAA
  s.d3d10.sd.SampleDesc.Quality     = 16;  // 16x CSAA
    
  // Fill in the rest with DXUT defaults
  // NOTE: This may no longer be necessary, but leave it in here in case DXUT
  // doesn't like the omitted data.
  DXUTMatchOptions mo;
  mo.eAPIVersion       = DXUTMT_PRESERVE_INPUT;
  mo.eAdapterOrdinal   = DXUTMT_IGNORE_INPUT;
  mo.eOutput           = DXUTMT_IGNORE_INPUT;
  mo.eDeviceType       = DXUTMT_PRESERVE_INPUT;
  mo.eWindowed         = DXUTMT_CLOSEST_TO_INPUT;
  mo.eAdapterFormat    = DXUTMT_CLOSEST_TO_INPUT;
  mo.eVertexProcessing = DXUTMT_CLOSEST_TO_INPUT;
  mo.eResolution       = DXUTMT_CLOSEST_TO_INPUT;
  mo.eBackBufferFormat = DXUTMT_CLOSEST_TO_INPUT;
  mo.eBackBufferCount  = DXUTMT_CLOSEST_TO_INPUT;
  mo.eMultiSample      = DXUTMT_CLOSEST_TO_INPUT;
  mo.eSwapEffect       = DXUTMT_PRESERVE_INPUT;
  mo.eDepthFormat      = DXUTMT_CLOSEST_TO_INPUT;
  mo.eStencilFormat    = DXUTMT_IGNORE_INPUT;
  mo.ePresentFlags     = DXUTMT_IGNORE_INPUT;
  mo.eRefreshRate      = DXUTMT_IGNORE_INPUT;
  mo.ePresentInterval  = DXUTMT_CLOSEST_TO_INPUT;
  DXUTFindValidDeviceSettings(&s, &s, &mo);

  // Create the device
  DXUTCreateDeviceFromSettings(&s);
}


//--------------------------------------------------------------------------------------
// Reject any D3D10 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D10DeviceAcceptable(UINT Adapter, UINT Output, D3D10_DRIVER_TYPE DeviceType, DXGI_FORMAT BackBufferFormat, bool Windowed, void* UserContext)
{
  return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10CreateDevice(ID3D10Device* d3dDevice, const DXGI_SURFACE_DESC* BackBufferSurfaceDesc, void* UserContext)
{
  HRESULT hr;

  V_RETURN(g_DialogResourceManager.OnD3D10CreateDevice(d3dDevice));
  V_RETURN(g_SettingsDlg.OnD3D10CreateDevice(d3dDevice));
  g_App.OnD3D10CreateDevice(d3dDevice);

  return S_OK;
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10DestroyDevice(void* UserContext)
{
  g_App.OnD3D10DestroyDevice();
  g_SettingsDlg.OnD3D10DestroyDevice();
  g_DialogResourceManager.OnD3D10DestroyDevice();
  DXUTGetGlobalResourceCache().OnDestroyDevice();
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10ResizedSwapChain(ID3D10Device* d3dDevice, IDXGISwapChain* SwapChain, const DXGI_SURFACE_DESC* BackBufferSurfaceDesc, void* UserContext)
{
  HRESULT hr;

  g_App.OnD3D10ResizedSwapChain(d3dDevice, BackBufferSurfaceDesc);
  V_RETURN(g_DialogResourceManager.OnD3D10ResizedSwapChain(d3dDevice, BackBufferSurfaceDesc));
  V_RETURN(g_SettingsDlg.OnD3D10ResizedSwapChain(d3dDevice, BackBufferSurfaceDesc));

  // TODO: Sync this "y" with the resulting sizes of the HUDs
  int y = 0;

  // Device HUD
  {
    CDXUTDialog &HUD = g_HUD[HUD_GENERIC];
    HUD.SetLocation(BackBufferSurfaceDesc->Width-210, y);
    HUD.SetSize(195, 150);
    y += 150;
  }

  // Shadow HUD
  {
    CDXUTDialog &HUD = g_HUD[HUD_SHADOW];
    HUD.SetLocation(BackBufferSurfaceDesc->Width-210, y);
    HUD.SetSize(195, 90);
    y += 90;
  }

  // Softness HUD
  {
    CDXUTDialog &HUD = g_HUD[HUD_SOFTNESS];
    HUD.SetLocation(BackBufferSurfaceDesc->Width-210, y);
    HUD.SetSize(195, 55);
    y += 55;
  }

  // VSM HUD
  {
    CDXUTDialog &HUD = g_HUD[HUD_VSM];
    HUD.SetLocation(BackBufferSurfaceDesc->Width-210, y);
    if (c_EnableShadowMultisampling) {
      y += 100;
      HUD.SetSize(195, 100);
    } else {
      y += 60;
      HUD.SetSize(195, 60);
    }
  }

  // PSSM HUD
  {
    CDXUTDialog &HUD = g_HUD[HUD_PSSM];
    HUD.SetLocation(BackBufferSurfaceDesc->Width-210, y);
    HUD.SetSize(195, 60);
    // Overlaps with SAT HUD
  }

  // SAT HUD
  {
    CDXUTDialog &HUD = g_HUD[HUD_SATFP];
    HUD.SetLocation(BackBufferSurfaceDesc->Width-210, y);
    HUD.SetSize(195, 60);
    y += 60;
  }

  return S_OK;
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10ReleasingSwapChain(void* UserContext)
{
  g_DialogResourceManager.OnD3D10ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove(double Time, float ElapsedTime, void* UserContext)
{
  g_App.Move(Time, ElapsedTime);
}


//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender(ID3D10Device* d3dDevice, double Time, float ElapsedTime, void* UserContext)
{
  // If the settings dialog is being shown, then
  // render it instead of rendering the app's scene
  if(g_SettingsDlg.IsActive()) {
    g_SettingsDlg.OnRender(ElapsedTime);
  } else {
    // Setup application
    g_App.PreRender(d3dDevice);

    if (c_EnableShadowMultisampling) {
      // Update our multisampling UI if necessary
      // Only do this now if we're using a relevant technique
      int CurTech = g_App.GetFilteringTechnique();
      if (g_App.GetResetMSAAUI() && (CurTech == App::FT_VSM ||
                                     CurTech == App::FT_SAVSMFP ||
                                     CurTech == App::FT_SAVSMINT ||
                                     CurTech == App::FT_PSVSM)) {
        // Store previous selection
        CDXUTComboBox *Box = g_HUD[HUD_VSM].GetComboBox(UI_MULTISAMPLING);
        std::wstring SelectedText;
        if (Box->GetSelectedIndex() >= 0) {
          SelectedText = Box->GetSelectedItem()->strText;
        }

        // Clear everything out and add the default element back
        Box->RemoveAllItems();
        Box->AddItem(L"No Shadow MSAA", 0);

        // Query for supported modes and add them to our combo box
        g_ShadowMSAAModes = g_App.QueryShadowMSAAModes(d3dDevice);
        for (Filtering::MSAAList::iterator i = g_ShadowMSAAModes.begin();
             i != g_ShadowMSAAModes.end(); ++i) {
          Box->AddItem((i->Name + L" Shadow MSAA").c_str(), &(*i));
        }

        // Reset the selected item, if it's still available
        if (!SelectedText.empty()) {
          Box->SetSelectedByText(SelectedText.c_str());
        } else {
          Box->SetSelectedByIndex(0);
        }

        g_UpdateSelectedMultisamplingType = true;
      }
    }

    // Standard rendering
    // Application
    g_App.Render(d3dDevice);

    // Render HUDs
    if (!g_App.IsBenchmarking()) {
      for (int i = 0; i < HUD_NUM; ++i) {
        g_HUD[i].OnRender(ElapsedTime);
      }
    }

    // Time to update multisampling?
    if (c_EnableShadowMultisampling && g_UpdateSelectedMultisamplingType) {
      // Update our application setting based on our selection
      CDXUTComboBox *Box = g_HUD[HUD_VSM].GetComboBox(UI_MULTISAMPLING);
      const Filtering::MSAAMode *Mode = static_cast<Filtering::MSAAMode *>(Box->GetSelectedData());
      if (Mode) {
        g_App.SetShadowMSAA(Mode->SampleDesc);
      } else {
        DXGI_SAMPLE_DESC NoMSAA = {1, 0};
        g_App.SetShadowMSAA(NoMSAA);
      }
      g_UpdateSelectedMultisamplingType = false;
    }
  }
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc(HWND Wnd, UINT Msg, WPARAM wParam, LPARAM lParam, bool* NoFurtherProcessing, void* UserContext)
{
  // Always allow dialog resource manager calls to handle global messages
  // so GUI state is updated correctly
  *NoFurtherProcessing = g_DialogResourceManager.MsgProc(Wnd, Msg, wParam, lParam);
  if (*NoFurtherProcessing) {
    return 0;
  }

  if (g_SettingsDlg.IsActive()) {
    g_SettingsDlg.MsgProc(Wnd, Msg, wParam, lParam);
    return 0;
  }

  // Any HUD input?
  if (!g_App.IsBenchmarking()) {
    // Do it in *reverse* draw order, so that anything on top will get input first!
    for (int i = (HUD_NUM-1); i >= 0; --i) {
      *NoFurtherProcessing = g_HUD[i].MsgProc(Wnd, Msg, wParam, lParam);
      if (*NoFurtherProcessing) {
        return 0;
      }
    }
  
    // Pass anything else to the application
    g_App.HandleMessages(Wnd, Msg, wParam, lParam);
  }

  return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard(UINT Char, bool KeyDown, bool AltDown, void* UserContext)
{
  // Exit on escape
  if (KeyDown && Char == VK_ESCAPE) {
    SendMessage(DXUTGetHWND(), WM_CLOSE, 0, 0);
  }

  // Begin a benchmark batch
  if (KeyDown && AltDown && Char == 'B') {
    g_App.BeginBenchmarking();
  }

  // Save a screenshot
  if (KeyDown && AltDown && Char == 'S') {
    // Format date/time for file name
    const unsigned int BufferSize = 256;
    wchar_t Time[BufferSize], Date[BufferSize];

    GetDateFormat(LOCALE_USER_DEFAULT, 0, NULL, L"yyyyMMdd", Date, BufferSize);
    GetTimeFormat(LOCALE_USER_DEFAULT, 0, NULL, L"HHmmss", Time, BufferSize);

    std::wostringstream FileName;
    FileName << Date << "_" << Time;
    g_App.SaveScreenshot(FileName.str());
  }
}


//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent(UINT Event, int ControlID, CDXUTControl* Control, void* UserContext)
{
  if (g_App.IsBenchmarking()) return;
  
  // I'm lazy and quite happy to just throw all of these into one callback bin :)
  switch (ControlID) {
  case UI_TOGGLEFULLSCREEN:
    DXUTToggleFullScreen();
    break;
  case UI_CHANGEDEVICE:
    g_SettingsDlg.SetActive(!g_SettingsDlg.IsActive());
    break;
  case UI_SCENE:
    {
      CDXUTComboBox* Combo = g_HUD[HUD_GENERIC].GetComboBox(UI_SCENE);
      g_App.SetScene(PtrToInt(Combo->GetSelectedData()));
      break;
    }
  case UI_CONTROLMODE:
    {
      CDXUTComboBox* Combo = g_HUD[HUD_GENERIC].GetComboBox(UI_CONTROLMODE);
      g_App.SetControlMode(PtrToInt(Combo->GetSelectedData()));
      break;
    }
  case UI_ANIMATELIGHT:
    {
      CDXUTCheckBox* Check = g_HUD[HUD_GENERIC].GetCheckBox(UI_ANIMATELIGHT);
      g_App.SetAnimateLight(Check->GetChecked());
      break;
    }
  case UI_LIGHTINGONLY:
    {
      CDXUTCheckBox* Check = g_HUD[HUD_SHADOW].GetCheckBox(UI_LIGHTINGONLY);
      g_App.SetLightingOnly(Check->GetChecked());
      break;
    }
  case UI_FILTERINGTECHNIQUE:
    {
      CDXUTComboBox* Combo = g_HUD[HUD_SHADOW].GetComboBox(UI_FILTERINGTECHNIQUE);
      int Technique = PtrToInt(Combo->GetSelectedData());
      g_App.SetFilteringTechnique(Technique);
      UpdateHUDVisibility();
      break;
    }
  case UI_DIMENSIONS:
    {
      CDXUTComboBox* Combo = g_HUD[HUD_SHADOW].GetComboBox(UI_DIMENSIONS);
      int Dim = PtrToInt(Combo->GetSelectedData());
      g_App.SetShadowDimensions(Dim, Dim);
      break;
    }
  case UI_SOFTNESS:
    {
      CDXUTSlider* Slider = g_HUD[HUD_SOFTNESS].GetSlider(UI_SOFTNESS);
      float p = Slider->GetValue() / static_cast<float>(c_FactorSliderRes);
      g_App.SetSoftness(p);
      break;
    }
  case UI_LBR:
    {
      CDXUTCheckBox* Check = g_HUD[HUD_VSM].GetCheckBox(UI_LBR);
      g_App.SetLBR(Check->GetChecked());

      // Update enabled
      CDXUTSlider* Slider = g_HUD[HUD_VSM].GetSlider(UI_LBRAMOUNT);
      Slider->SetEnabled(Check->GetChecked());
      break;
    }
  case UI_MULTISAMPLING:
      g_UpdateSelectedMultisamplingType = true;
      break;
  case UI_LBRAMOUNT:
    {
      CDXUTSlider* Slider = g_HUD[HUD_VSM].GetSlider(UI_LBRAMOUNT);
      float p = Slider->GetValue() / static_cast<float>(c_FactorSliderRes);
      g_App.SetLBRAmount(p);
      break;
    }
  case UI_PSSMSPLITS:
  {
      CDXUTSlider* Slider = g_HUD[HUD_PSSM].GetSlider(UI_PSSMSPLITS);
      g_App.SetPSSMSplits(Slider->GetValue());
      break;
    }
  case UI_PSSMSPLITLAMBDA:
    {
      CDXUTSlider* Slider = g_HUD[HUD_PSSM].GetSlider(UI_PSSMSPLITLAMBDA);
      float p = Slider->GetValue() / static_cast<float>(c_FactorSliderRes);
      g_App.SetPSSMSplitLambda(p);
      break;
    }
  case UI_PSSMVISUALIZESPLITS:
    {
      CDXUTCheckBox* Check = g_HUD[HUD_PSSM].GetCheckBox(UI_PSSMVISUALIZESPLITS);
      g_App.SetPSSMVisualizeSplits(Check->GetChecked());
      break;
    }
  case UI_SATDISTRIBUTEPRECISION:
    {
      CDXUTCheckBox* Check = g_HUD[HUD_SATFP].GetCheckBox(UI_SATDISTRIBUTEPRECISION);
      g_App.SetDistributePrecision(Check->GetChecked());
      break;
    }
  default:
    break;
  }
}

void UpdateHUDVisibility()
{
  int Technique = g_App.GetFilteringTechnique();

  // Display/hide relevant panels
  g_HUD[HUD_SOFTNESS].SetVisible(Technique == App::FT_PCF ||
                                 Technique == App::FT_VSM ||
                                 Technique == App::FT_PSVSM ||
                                 Technique == App::FT_SAVSMFP ||
                                 Technique == App::FT_SAVSMINT);

  g_HUD[HUD_VSM].SetVisible(Technique == App::FT_VSM ||
                            Technique == App::FT_PSVSM ||
                            Technique == App::FT_SAVSMFP ||
                            Technique == App::FT_SAVSMINT);

  g_HUD[HUD_PSSM].SetVisible(Technique == App::FT_PSVSM);

  g_HUD[HUD_SATFP].SetVisible(Technique == App::FT_SAVSMFP);
}
