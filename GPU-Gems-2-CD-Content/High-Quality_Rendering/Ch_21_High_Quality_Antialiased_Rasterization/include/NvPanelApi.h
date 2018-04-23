#ifndef __NVPANELAPI_H__
#define __NVPANELAPI_H__

/*
//
//  Copyright (c) 2003-2004 NVIDIA Corporation. All rights reserved.
//
//  This software may not, in whole or in part, be copied through any means,
//  mechanical, electromechanical, or otherwise, without the express permission
//  of NVIDIA Corporation.
//
//  Information furnished is believed to be accurate and reliable. 
//  However, NVIDIA assumes no responsibility for the consequences of use of
//  such information nor for any infringement of patents or other rights of 
//  third parties, which may result from its use.
//
//  No License is granted by implication or otherwise under any patent or 
//  patent rights of NVIDIA Corporation.
//
//  
//  This header file contains declaratios for externally accessible 
//  NVIDIA CONTROL PANEL API methods. For detailed description of the API
//  see "NVIDIA Corporation NVCPL.DLL API Manual".
//
//  Revision History:
//
//  52.90  10/26/03  Created header file for function prototypes as described by the NVControlPanel_API
//                   document Note that the rev number matches the driver version which the modification
//                   was made in.
//  
*/

#ifdef __cplusplus
    extern "C" {
#endif

#ifndef PGAMMARAMP
// PGAMMARAMP is defined in Windows DDK winddi.h header:
//    typedef struct _GAMMARAMP 
//    {
//        WORD  Red[256];
//        WORD  Green[256];
//        WORD  Blue[256];
//    } GAMMARAMP, *PGAMMARAMP;
#   define PGAMMARAMP void*
#   define REDEFINED_PGAMMARAMP
#endif//PGAMMARAMP

#ifndef CDECL
#   define CDECL __cdecl
#endif //CDECL

#define NVAPI_OPERATION_SUCCEEDED                       0
#define NVAPI_ERROR_INVALID_INPUT                       1
#define NVAPI_ERROR_NO_TV                               2
#define NVAPI_ERROR_FAILED_INITIALIZATION               3
#define NVAPI_ERROR_HARDWARE_DOESNT_SUPPORT_FEATURE     4
#define NVAPI_ERROR_SETTING_INCONGRUENT_WITH_MODALITY   5
#define NVAPI_WARNING_WSS_INCONGRUENT_WITH_CP           6
#define NVAPI_ERROR_UNKNOWN                             7


// Display Information functions.

enum NVDISPLAYMODE
{
    NVDISPLAYMODE_NONE        = -1,         // No Display (or Display Mode Unknown)
    NVDISPLAYMODE_STANDARD    =  0,         // Single-Display Mode
    NVDISPLAYMODE_CLONE       =  1,         // Clone Mode
    NVDISPLAYMODE_HSPAN       =  2,         // Horizontal Span
    NVDISPLAYMODE_VSPAN       =  3,         // Vertical Span
    NVDISPLAYMODE_DUALVIEW    =  4,         // DualView
};

enum NVDISPLAYTYPE // upper nibble indicates device class (for example: 0x2 for DFPs)
{
    NVDISPLAYTYPE_NONE        =     -1,     // No Display (or Display Type Unknown)
    NVDISPLAYTYPE_CRT         = 0x1000,     // Cathode Ray Tube (CRT class from 0x1...)
    NVDISPLAYTYPE_DFP         = 0x2000,     // Digital Flat Panel (DFP class from 0x2...)
    NVDISPLAYTYPE_DFP_LAPTOP  = 0x2001,     //   Subtype: Laptop Display Panel
    NVDISPLAYTYPE_TV          = 0x3000,     // Television Set (TV class from 0x3...)
    NVDISPLAYTYPE_TV_HDTV     = 0x3001,     //   Subtype: High-Definition Television
};
#define NVDISPLAYTYPE_CLASS_MASK 0xF000     // Mask for device class checks

enum NVTVFORMAT
{
    NVTVFORMAT_NONE           =  -1,        // No Format (field does not apply to current device)
    NVTVFORMAT_NTSC_M         =   0,        // NTSC/M
    NVTVFORMAT_NTSC_J         =   1,        // NTSC/J
    NVTVFORMAT_PAL_M          =   2,        // PAL/M
    NVTVFORMAT_PAL_A          =   3,        // PAL/B, D, G, H, I
    NVTVFORMAT_PAL_N          =   4,        // PAL/N
    NVTVFORMAT_PAL_NC         =   5,        // PAL/NC
    NVTVFORMAT_HD576i         =   8,        // HDTV 576i
    NVTVFORMAT_HD480i         =   9,        // HDTV 480i
    NVTVFORMAT_HD480p         =  10,        // HDTV 480p
    NVTVFORMAT_HD576p         =  11,        // HDTV 576p
    NVTVFORMAT_HD720p         =  12,        // HDTV 720p
    NVTVFORMAT_HD1080i        =  13,        // HDTV 1080i
    NVTVFORMAT_HD1080p        =  14,        // HDTV 1080p
    NVTVFORMAT_HD720i         =  16,        // HDTV 720i
};

enum NVDFPSCALING
{
    NVDFPSCALING_NONE         =  -1,        // No Scaling (or Scaling Unknown)
    NVDFPSCALING_NATIVE       =   1,        // Monitor Scaling
    NVDFPSCALING_SCALED       =   2,        // Scaling
    NVDFPSCALING_CENTERED     =   3,        // Centering
    NVDFPSCALING_SCALED8BIT   =   4,        // Scaling (8-bit)
    NVDFPSCALING_SCALEDASPECT =   5,        // Scaling (Fixed Aspect Ratio)
};

enum NVBOARDTYPE
{
    NVBOARDTYPE_GEFORCE        =   0,        // Geforce board
    NVBOARDTYPE_QUADRO         =   1,        // Quadro board
    NVBOARDTYPE_NVS            =   2,        // NVS board
};

#define MAX_NVDISPLAYNAME  256              // Maximum length for user-friendly display names

// NVDISPLAYINFO.dwInputFields1 and .dwOutputFields1 bitfields
#define NVDISPLAYINFO1_ALL                  0xffffffff  // special: all fields valid
#define NVDISPLAYINFO1_WINDOWSDEVICENAME    0x00000001  // szWindowsDeviceName valid
#define NVDISPLAYINFO1_ADAPTERNAME          0x00000002  // szAdapterName valid
#define NVDISPLAYINFO1_DRIVERVERSION        0x00000004  // szDriverVersion valid
#define NVDISPLAYINFO1_DISPLAYMODE          0x00000008  // nDisplayMode valid
#define NVDISPLAYINFO1_WINDOWSMONITORNUMBER 0x00000010  // dwWindowsMonitorNumber valid
#define NVDISPLAYINFO1_DISPLAYHEADINDEX     0x00000020  // nDisplayHeadIndex valid
#define NVDISPLAYINFO1_DISPLAYISPRIMARY     0x00000040  // bDisplayIsPrimary valid
#define NVDISPLAYINFO1_DISPLAYNAME          0x00000080  // szDisplayName valid
#define NVDISPLAYINFO1_VENDORNAME           0x00000100  // szVendorName valid
#define NVDISPLAYINFO1_MODELNAME            0x00000200  // szModelName valid
#define NVDISPLAYINFO1_GENERICNAME          0x00000400  // szGenericName valid
#define NVDISPLAYINFO1_UNIQUEID             0x00000800  // dwUniqueId valid
#define NVDISPLAYINFO1_DISPLAYTYPE          0x00001000  // nDisplayType valid
#define NVDISPLAYINFO1_DISPLAYWIDTH         0x00002000  // mmDisplayWidth valid
#define NVDISPLAYINFO1_DISPLAYHEIGHT        0x00004000  // mmDisplayHeight valid
#define NVDISPLAYINFO1_GAMMACHARACTERISTIC  0x00008000  // fGammaCharacteristic valid
#define NVDISPLAYINFO1_OPTIMALMODE          0x00010000  // dwOptimal... fields valid
#define NVDISPLAYINFO1_MAXIMUMSAFEMODE      0x00020000  // dwMaximumSafe... fields valid
#define NVDISPLAYINFO1_BITSPERPEL           0x00040000  // dwBitsPerPel valid
#define NVDISPLAYINFO1_PELSWIDTH            0x00080000  // dwPelsWidth valid
#define NVDISPLAYINFO1_PELSHEIGHT           0x00100000  // dwPelsHeight valid
#define NVDISPLAYINFO1_DISPLAYFREQUENCY     0x00200000  // dwDisplayFrequency valid
#define NVDISPLAYINFO1_DISPLAYRECT          0x00400000  // rcDisplayRect valid (rcDisplayRect.TopLeft on write)
#define NVDISPLAYINFO1_VISIBLEPELSWIDTH     0x00800000  // dwVisiblePelsWidth valid
#define NVDISPLAYINFO1_VISIBLEPELSHEIGHT    0x01000000  // dwVisiblePelsHeight valid
#define NVDISPLAYINFO1_DEGREESROTATION      0x02000000  // dwDegreesRotation valid
#define NVDISPLAYINFO1_TVFORMAT             0x04000000  // nTvFormat valid
#define NVDISPLAYINFO1_DFPSCALING           0x08000000  // nDfpScaling valid
#define NVDISPLAYINFO1_TVCONNECTORTYPES     0x10000000  // dwTVConnectorTypes valid
#define NVDISPLAYINFO1_CURRENTCONNECTORTYPE 0x20000000  // dwCurrentConnectorType is valid
#define NVDISPLAYINFO1_BOARDTYPE            0x40000000  // dwBoardType is valid
#define NVDISPLAYINFO1_DISPLAYINSTANCECOUNT 0x80000000  // dwDisplayInstance and dwDisplayInstanceCount are valid

// NVDISPLAYINFO.dwInputFields2 and .dwOutputFields2 bitfields
#define NVDISPLAYINFO2_ALL                  0xffffffff  // special: all fields valid
#define NVDISPLAYINFO2_PRODUCTNAME          0x00000001  // szProductName valid

typedef struct tagNVDISPLAYINFO
{
    DWORD cbSize;                           // Size of the NVDISPLAYINFO structure (in bytes), 
                                            //  set this field to sizeof(NVDISPLAYINFO) to indicate version level of structure
    DWORD dwInputFields1;                   // Specifies which members of structure should be used on input to function (see NVDISPLAYINFO1_*)
    DWORD dwOutputFields1;                  // Specifies which members of structure were processed as result of the call
    DWORD dwInputFields2;                   // Specifies which members of structure should be used on input to function (see NVDISPLAYINFO2_*)
    DWORD dwOutputFields2;                  // Specifies which members of structure were processed as result of the call
    
    char  szWindowsDeviceName[_MAX_PATH];   // Device name for use with CreateDC (for example: ".\\DISPLAY1")
    char  szAdapterName[MAX_NVDISPLAYNAME]; // User friendly name for the associated NVIDIA graphics card (for example: GeForce FX 5200 Ultra)
    char  szDriverVersion[64];              // Display driver version string for device (for example: "6.14.10.6003")
    enum  NVDISPLAYMODE nDisplayMode;       // Display mode for head on adapter (for example: Clone, HSpan, DualView)
    DWORD dwWindowsMonitorNumber;           // Windows monitor number for adapter (numbers listed in Microsoft Display Panel)
    int   nDisplayHeadIndex;                // Head index for display on adapter
    BOOL  bDisplayIsPrimary;                // TRUE if display head is primary on adapter

    char  szDisplayName[MAX_NVDISPLAYNAME]; // User friendly name for the display device, may reflect user overrides (for example: "EIZO L685")
    char  szVendorName[MAX_NVDISPLAYNAME];  // Vendor name for display device if available (for example: "EIZO")
    char  szModelName[MAX_NVDISPLAYNAME];   // Model name for display device if available (for example: "EIZ1728")
    char  szGenericName[MAX_NVDISPLAYNAME]; // Generic name for display device type (for example: "Digital Display")
    DWORD dwUniqueId;                       // Unique identifier for display device, including serial number, zero if not available

    enum  NVDISPLAYTYPE nDisplayType;       // Type of the display device (for example: CRT, DFP, or TV)
    DWORD mmDisplayWidth;                   // Width of maximum visible display surface or zero if unknown (in millimeters)
    DWORD mmDisplayHeight;                  // Height of maximum visible display surface or zero if unknown (in millimeters)
    float fGammaCharacteristic;             // Gamma transfer characteristic for monitor (for example: 2.2)
    
    DWORD dwOptimalPelsWidth;               // Width of display surface in optimal display mode (not necessarily highest resolution)
    DWORD dwOptimalPelsHeight;              // Height of display surface in optimal display mode (not necessarily highest resolution)
    DWORD dwOptimalDisplayFrequency;        // Refresh frequency in optimal display mode (not necessarily highest resolution)

    DWORD dwMaximumSafePelsWidth;           // Width of display surface in maximum safe display mode (not necessarily highest resolution)
    DWORD dwMaximumSafePelsHeight;          // Height of display surface in maximum safe display mode (not necessarily highest resolution)
    DWORD dwMaximumSafeDisplayFrequency;    // Refresh frequency in maximum safe display mode (not necessarily highest resolution)

    DWORD dwBitsPerPel;                     // Color resolution of the display device (for example: 8 bits for 256 colors)
    DWORD dwPelsWidth;                      // Width of the available display surface, including any pannable area (in pixels)
    DWORD dwPelsHeight;                     // Height of the available display surface, including any pannable area (in pixels)
    DWORD dwDisplayFrequency;               // Refresh frequency of the display device (in hertz)

    RECT  rcDisplayRect;                    // Desktop rectangle for display surface (considers DualView and head offset)
    DWORD dwVisiblePelsWidth;               // Width of the visible display surface, excluding any pannable area (in pixels)
    DWORD dwVisiblePelsHeight;              // Height of the visible display surface, excluding any pannable area (in pixels)

    DWORD dwDegreesRotation;                // Rotation angle of display surface (in degrees)
    enum  NVTVFORMAT nTvFormat;             // Television video signal format (for example: NTSC/M or HDTV 1080i)
    enum  NVDFPSCALING nDfpScaling;         // Digital Flat Panel scaling mode (for example: Monitor Native)

    DWORD dwTVConnectorTypes;               // Television connectors (values not enumerated)
    DWORD dwCurrentConnectorType;           // Television active connector (values not enumerated)
    DWORD dwBoardType;                      // Type of graphics board (NVBOARDTYPE_* enumeration)

    DWORD dwDisplayInstance;                // Display instance number (instance of szDisplayName) or zero if indeterminant
    DWORD dwDisplayInstanceCount;           // Display instance count (instances of szDisplayName) or zero if indeterminant

    char  szProductName[MAX_NVDISPLAYNAME]; // Product name for display device if available, bypasses user customization of szDisplayName (for example: "EIZO L685")
} NVDISPLAYINFO;


BOOL APIENTRY NvGetDisplayInfo( LPCSTR pszDeviceMoniker, NVDISPLAYINFO* pDisplayInfo );
typedef BOOL (APIENTRY* fNvGetDisplayInfo)( LPCSTR pszDeviceMoniker, NVDISPLAYINFO* pDisplayInfo );

// Desktop Configuration.

DWORD APIENTRY dtcfgex( LPSTR lpszCmdLine );
typedef DWORD (APIENTRY* fdtcfgex)( LPSTR lpszCmdLine );

DWORD WINAPI GetdtcfgLastError( void );
typedef DWORD (WINAPI* fGetdtcfgLastError)( void );

DWORD WINAPI GetdtcfgLastErrorEx( LPSTR lpszCmdline, DWORD *PdwCmdLineSize );
typedef DWORD (WINAPI* fGetdtcfgLastErrorEx)( LPSTR lpszCmdline, DWORD *PdwCmdLineSize );


// Gamma Ramp Functions. 

enum NVCOLORAPPLY
{
    NVCOLORAPPLY_DESKTOP,          // Apply color settings to Desktop
    NVCOLORAPPLY_OVERLAYVMR,       // Apply color settings to Overlay/Video Mirroring
    NVCOLORAPPLY_FULLSCREENVIDEO,  // Apply color settings to Fullscreen Video

    NVCOLORAPPLY_COUNT             // Number of apply color settings targets
};

BOOL CDECL NvColorGetGammaRamp( LPCSTR szUserDisplay, PGAMMARAMP pGammaNew );
typedef BOOL (CDECL* fNvColorGetGammaRamp)( LPCSTR szUserDisplay, PGAMMARAMP pGammaNew );

BOOL CDECL NvColorSetGammaRamp( LPCSTR szUserDisplay, DWORD dwUserRotateFlag, PGAMMARAMP pGammaNew );
typedef BOOL (CDECL* fNvColorSetGammaRamp)( LPCSTR szUserDisplay, DWORD dwUserRotateFlag, PGAMMARAMP pGammaNew );

BOOL APIENTRY NvColorGetGammaRampEx( LPCSTR szUserDisplay, PGAMMARAMP pGammaNew, NVCOLORAPPLY applyFrom );
typedef BOOL (APIENTRY* fNvColorGetGammaRampEx)( LPCSTR szUserDisplay, PGAMMARAMP pGammaNew, NVCOLORAPPLY applyFrom );

BOOL APIENTRY NvColorSetGammaRampEx( LPCSTR szUserDisplay, const PGAMMARAMP pGammaNew, NVCOLORAPPLY applyTo );
typedef BOOL (APIENTRY* fNvColorSetGammaRampEx)( LPCSTR szUserDisplay, const PGAMMARAMP pGammaNew, NVCOLORAPPLY applyFrom );

// Multi-Display Modes.

BOOL CDECL  NvGetFullScreenVideoMirroringEnabled( LPCSTR szUserDisplay , BOOL* pbEnabled );
typedef BOOL (CDECL* fNvGetFullScreenVideoMirroringEnabled)( LPCSTR szUserDisplay , BOOL* pbEnabled );

BOOL CDECL  NvSetFullScreenVideoMirroringEnabled( LPCSTR szUserDisplay , BOOL* pbEnabled ); 
typedef BOOL (CDECL* fNvSetFullScreenVideoMirroringEnabled)( LPCSTR szUserDisplay , BOOL* pbEnabled ); 

int  APIENTRY  NvGetWindowsDisplayState( int iDisplayIndex );
typedef int (APIENTRY* fNvGetWindowsDisplayState)( int iDisplayIndex );


// Flat Panel Functions. 

BOOL APIENTRY  NvCplGetFlatPanelNativeRes( LPCSTR dfpMoniker , DWORD *width, DWORD *height );
typedef BOOL (APIENTRY* fNvCplGetFlatPanelNativeRes)( LPCSTR dfpMoniker , DWORD *width, DWORD *height );

BOOL CDECL  NvCplGetScalingStatus( LPCSTR szDeviceMoniker , DWORD* pdwStatus );
typedef BOOL (CDECL* fNvCplGetScalingStatus)( LPCSTR szDeviceMoniker , DWORD* pdwStatus );



// Device and Adapter management and configuration/control.
// supplemental to dtcfgex and dtcfg

BOOL CDECL NvSelectDisplayDevice(unsigned int uiDisplayDeviceNumber); // ==0 Will return Current default device
                                                                         // after reinitialization if reiniting found necessary
                                                                         // return FALSE on failure, TRUE otherwise
typedef BOOL (CDECL* fNvSelectDisplayDevice)(unsigned int uiDisplayDeviceNumber);

BOOL APIENTRY  NvCplGetConnectedDevicesString( LPSTR lpszTextBuffer, DWORD cbTextBuffer, BOOL bOnlyActive );
typedef BOOL (APIENTRY* fNvCplGetConnectedDevicesString)( LPSTR lpszTextBuffer, DWORD cbTextBuffer, BOOL bOnlyActive );


// PowerMizer Functions. 

BOOL APIENTRY  nvGetPwrMzrLevel( DWORD* dwBatteryLevel , DWORD* dwACLevel );
typedef BOOL (APIENTRY* fnvGetPwrMzrLevel)( DWORD* dwBatteryLevel , DWORD* dwACLevel );

BOOL APIENTRY  nvSetPwrMzrLevel( DWORD* dwBatteryLevel , DWORD* dwACLevel );
typedef BOOL (APIENTRY* fnvSetPwrMzrLevel)( DWORD* dwBatteryLevel , DWORD* dwACLevel );

// Power Management
void APIENTRY PowerManageHelper_Nvcpl( HWND hwnd, HINSTANCE hInst, LPSTR lpszCmdLine, int nCmdShow );
typedef void (APIENTRY* fPowerManageHelper_Nvcpl)( HWND hwnd, HINSTANCE hInstance, LPSTR lpszCmdLine, int nCmdShow );

// Data Control Functions and Definitions

#define NVCPL_API_AGP_BUS_MODE      1
#define NVCPL_API_VIDEO_RAM_SIZE    2
#define NVCPL_API_TX_RATE           3
#define NVCPL_API_CURRENT_AA_VALUE  4
#define NVCPL_API_AGP_LIMIT         5
#define NVCPL_API_FRAME_QUEUE_LIMIT 6

BOOL CDECL  nvCplGetDataInt( long lFlag, long* plInfo );
typedef BOOL (CDECL* fnvCplGetDataInt)( long lFlag, long* plInfo );

BOOL CDECL  nvCplSetDataInt( long lFlag, long plInfo );
typedef BOOL (CDECL* fnvCplSetDataInt)( long lFlag, long plInfo );



// Temperature Monitoring.

BOOL CDECL  nvCplGetThermalSettings( int iDisplayIndex , DWORD* dwCoreTemp, DWORD* dwAmbientTemp, DWORD* UpperLimit );
typedef BOOL (CDECL* fnvCplGetThermalSettings)( int iDisplayIndex , DWORD* dwCoreTemp, DWORD* dwAmbientTemp, DWORD* UpperLimit );

BOOL CDECL NvSetDVDOptimalEnabled( BOOL bEnable );
typedef BOOL (CDECL* fNvSetDVDOptimalEnabled)( BOOL bEnable );

BOOL CDECL  NvCplIsExternalPowerConnectorAttached( BOOL* p_bExternalPowerConnectorAttached );
typedef BOOL (CDECL* fNvCplIsExternalPowerConnectorAttached)( BOOL* p_bExternalPowerConnectorAttached );



// TV Format Functions and Definitions

#define NVAPI_TVFORMAT_NTSC_M       0
#define NVAPI_TVFORMAT_NTSC_J       1
#define NVAPI_TVFORMAT_PAL_M        2
#define NVAPI_TVFORMAT_PAL_ABDGHI   3
#define NVAPI_TVFORMAT_PAL_N        4
#define NVAPI_TVFORMAT_PAL_NC       5
#define NVAPI_TVFORMAT_HD480I       9
#define NVAPI_TVFORMAT_HD480P       10
#define NVAPI_TVFORMAT_HD576P       11
#define NVAPI_TVFORMAT_HD720P       12
#define NVAPI_TVFORMAT_HD1080I      13
#define NVAPI_TVFORMAT_HD1080P      14
#define NVAPI_TVFORMAT_HD576I       15
#define NVAPI_TVFORMAT_HD720I       16
#define NVAPI_TVFORMAT_D1           NVAPI_TVFORMAT_HD480I
#define NVAPI_TVFORMAT_D2           NVAPI_TVFORMAT_HD480P
#define NVAPI_TVFORMAT_D3           NVAPI_TVFORMAT_HD1080I
#define NVAPI_TVFORMAT_D4           NVAPI_TVFORMAT_HD720P
#define NVAPI_TVFORMAT_D5           NVAPI_TVFORMAT_HD1080P

BOOL CDECL NvGetCurrentTVFormat( DWORD* pdwFormat );
typedef BOOL (CDECL* fNvGetCurrentTVFormat)( DWORD* pdwFormat );

#define NVAPI_TV_ENCODER_CONNECTOR_UNKNOWN       0x0
#define NVAPI_TV_ENCODER_CONNECTOR_SDTV          0x1
#define NVAPI_TV_ENCODER_CONNECTOR_HDTV          0x2
#define NVAPI_TV_ENCODER_CONNECTOR_HDTV_AND_SDTV 0x3 

BOOL CDECL NvGetTVConnectedStatus( DWORD* pdwConnected );
typedef BOOL (CDECL* fNvGetTVConnectedStatus)( DWORD* pdwConnected );

#define NVCPL_API_OVERSCAN_SHIFT                            0x00000010
#define NVCPL_API_UNDERSCAN                                 0x00000020
#define NVCPL_API_NATIVEHD                                  0x00000080

BOOL CDECL  NVTVOutManageOverscanConfiguration( DWORD dwSelectedTVFormat,
                                                DWORD *pdwOverscanConfig,                  
                                                BOOL  bReadConfig );        //TRUE==read, FALSE==write                  
                                                                  
                                                                  
typedef BOOL (CDECL* fNVTVOutManageOverscanConfiguration)( DWORD dwSelectedTVFormat,
                                                           DWORD *pdwOverscanConfig,       
                                                           BOOL bReadConfig );    //TRUE==read, FALSE==write       
                                                                   
                                                                  
//#if 0
                                        #define NVAPI_ASPECT_FULLSCREEN     0  /* 4:3 aspect              */
                                        #define NVAPI_ASPECT_LETTERBOX      1  /* 4:3 aspect, letterbox'd */
                                        #define NVAPI_ASPECT_WIDESCREEN     2  /* 16:9 aspect             */

                                        BOOL CDECL NvSetHDAspect( DWORD* pdwAspect );
                                        typedef BOOL (CDECL* fNvSetHDAspect)( DWORD* pdwAspect );
//#endif

#define NVAPI_LICENSE_TYPE_OVERCLOCKING       1
#define NVAPI_LICENSE_TYPE_ADVANCED_TIMING    2

BOOL CDECL NvGetShowLicenseKeyAgreement( DWORD dwLicenseType, DWORD* pdwData );
typedef BOOL (CDECL* fNvGetShowLicenseKeyAgreement)( DWORD dwLicenseType, DWORD* pdwData );

BOOL CDECL NvSetShowLicenseKeyAgreement( DWORD dwLicenseType, DWORD dwData );
typedef BOOL (CDECL* fNvSetShowLicenseKeyAgreement)( DWORD dwLicenseType, DWORD dwData );

#ifdef __cplusplus
#define _cplusplus
} //extern "C" {
#endif

#endif  // __NVPANELAPI_H__
