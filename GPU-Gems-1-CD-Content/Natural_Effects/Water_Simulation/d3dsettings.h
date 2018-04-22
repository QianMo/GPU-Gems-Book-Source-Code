//-----------------------------------------------------------------------------
// File: D3DSettings.h
//
// Desc: Settings class and change-settings dialog class for the Direct3D 
//       samples framework library.
//-----------------------------------------------------------------------------
#ifndef D3DSETTINGS_H
#define D3DSETTINGS_H


//-----------------------------------------------------------------------------
// Name: class CD3DSettings
// Desc: Current D3D settings: adapter, device, mode, formats, etc.
//-----------------------------------------------------------------------------
class CD3DSettings 
{
public:
    bool IsWindowed;

    D3DAdapterInfo* pWindowed_AdapterInfo;
    D3DDeviceInfo* pWindowed_DeviceInfo;
    D3DDeviceCombo* pWindowed_DeviceCombo;

    D3DDISPLAYMODE Windowed_DisplayMode; // not changable by the user
    D3DFORMAT Windowed_DepthStencilBufferFormat;
    D3DMULTISAMPLE_TYPE Windowed_MultisampleType;
    DWORD Windowed_MultisampleQuality;
    VertexProcessingType Windowed_VertexProcessingType;
    UINT Windowed_PresentInterval;
    int Windowed_Width;
    int Windowed_Height;

    D3DAdapterInfo* pFullscreen_AdapterInfo;
    D3DDeviceInfo* pFullscreen_DeviceInfo;
    D3DDeviceCombo* pFullscreen_DeviceCombo;

    D3DDISPLAYMODE Fullscreen_DisplayMode; // changable by the user
    D3DFORMAT Fullscreen_DepthStencilBufferFormat;
    D3DMULTISAMPLE_TYPE Fullscreen_MultisampleType;
    DWORD Fullscreen_MultisampleQuality;
    VertexProcessingType Fullscreen_VertexProcessingType;
    UINT Fullscreen_PresentInterval;

    D3DAdapterInfo* PAdapterInfo() { return IsWindowed ? pWindowed_AdapterInfo : pFullscreen_AdapterInfo; }
    D3DDeviceInfo* PDeviceInfo() { return IsWindowed ? pWindowed_DeviceInfo : pFullscreen_DeviceInfo; }
    D3DDeviceCombo* PDeviceCombo() { return IsWindowed ? pWindowed_DeviceCombo : pFullscreen_DeviceCombo; }

    int AdapterOrdinal() { return PDeviceCombo()->AdapterOrdinal; }
    D3DDEVTYPE DevType() { return PDeviceCombo()->DevType; }
    D3DFORMAT BackBufferFormat() { return PDeviceCombo()->BackBufferFormat; }

    D3DDISPLAYMODE DisplayMode() { return IsWindowed ? Windowed_DisplayMode : Fullscreen_DisplayMode; }
    void SetDisplayMode(D3DDISPLAYMODE value) { if (IsWindowed) Windowed_DisplayMode = value; else Fullscreen_DisplayMode = value; }

    D3DFORMAT DepthStencilBufferFormat() { return IsWindowed ? Windowed_DepthStencilBufferFormat : Fullscreen_DepthStencilBufferFormat; }
    void SetDepthStencilBufferFormat(D3DFORMAT value) { if (IsWindowed) Windowed_DepthStencilBufferFormat = value; else Fullscreen_DepthStencilBufferFormat = value; }

    D3DMULTISAMPLE_TYPE MultisampleType() { return IsWindowed ? Windowed_MultisampleType : Fullscreen_MultisampleType; }
    void SetMultisampleType(D3DMULTISAMPLE_TYPE value) { if (IsWindowed) Windowed_MultisampleType = value; else Fullscreen_MultisampleType = value; }

    DWORD MultisampleQuality() { return IsWindowed ? Windowed_MultisampleQuality : Fullscreen_MultisampleQuality; }
    void SetMultisampleQuality(DWORD value) { if (IsWindowed) Windowed_MultisampleQuality = value; else Fullscreen_MultisampleQuality = value; }

    VertexProcessingType GetVertexProcessingType() { return IsWindowed ? Windowed_VertexProcessingType : Fullscreen_VertexProcessingType; }
    void SetVertexProcessingType(VertexProcessingType value) { if (IsWindowed) Windowed_VertexProcessingType = value; else Fullscreen_VertexProcessingType = value; }

    UINT PresentInterval() { return IsWindowed ? Windowed_PresentInterval : Fullscreen_PresentInterval; }
    void SetPresentInterval(UINT value) { if (IsWindowed) Windowed_PresentInterval = value; else Fullscreen_PresentInterval = value; }
};




//-----------------------------------------------------------------------------
// Name: class CD3DSettingsDialog
// Desc: Dialog box to allow the user to change the D3D settings
//-----------------------------------------------------------------------------
class CD3DSettingsDialog
{
private:
    HWND m_hDlg;
    CD3DEnumeration* m_pEnumeration;
    CD3DSettings m_d3dSettings;

private:
    // ComboBox helper functions
    void ComboBoxAdd( int id, void* pData, TCHAR* pstrDesc );
    void ComboBoxSelect( int id, void* pData );
    void* ComboBoxSelected( int id );
    bool ComboBoxSomethingSelected( int id );
    UINT ComboBoxCount( int id );
    void ComboBoxSelectIndex( int id, int index );
    void ComboBoxClear( int id );
    bool ComboBoxContainsText( int id, TCHAR* pstrText );

    void AdapterChanged( void );
    void DeviceChanged( void );
    void WindowedFullscreenChanged( void );
    void AdapterFormatChanged( void );
    void ResolutionChanged( void );
    void RefreshRateChanged( void );
    void BackBufferFormatChanged( void );
    void DepthStencilBufferFormatChanged( void );
    void MultisampleTypeChanged( void );
    void MultisampleQualityChanged( void );
    void VertexProcessingChanged( void );
    void PresentIntervalChanged( void );

public:
    CD3DSettingsDialog( CD3DEnumeration* pEnumeration, CD3DSettings* pSettings);
    INT_PTR ShowDialog( HWND hwndParent );
    INT_PTR DialogProc( HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam );
    void GetFinalSettings( CD3DSettings* pSettings ) { *pSettings = m_d3dSettings; }
};

#endif



