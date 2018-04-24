//----------------------------------------------------------------------------------
// File:   GPUBlendShapes.cpp
// Author: Tristan Lorach
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

#pragma warning (disable: 4995) // remove warning for funcs "marked as #pragma deprecated"

#define MEDIAPATH "..\\media\\"
#define DBGINFO true

#include "DXUT.h"
#include "DXUTmisc.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "sdkmesh_old.h"
#include "resource.h"

#include "NVUTSkybox.h"

#include <fstream>

extern void PrintMessage(int level, LPCWSTR fmt, ...);
#define LOGMSG      PrintMessage // could be printf or whatever...
#define LOG_MSG     0
#define LOG_WARN    1
#define LOG_ERR     2

#include "NVD3D10RawMesh.h"
#include "curve/CurveReader.h"

/*----------------------------------------------
  --
  -- The object we use (Dawn) is made of many meshes. 
  -- It could have been only one but the original model is like this.
  --
  ----------------------------------------------*/
#define MESHNUM 5
static NVD3D10RawMesh       g_MeshTable[MESHNUM];
static LPWSTR               g_meshFiles[] = {
L"MediaFor_GPUBlendShapes/dawn_HeadShape.mesh",
L"MediaFor_GPUBlendShapes/dawn_LEyeShape.mesh",
L"MediaFor_GPUBlendShapes/dawn_REyeShape.mesh",
L"MediaFor_GPUBlendShapes/dawn_BodyShape.mesh",
L"MediaFor_GPUBlendShapes/dawn_teethUpperShape.mesh",
};

enum TRenderMode
{
    RENDER_BUFFERTEMPLATE,
    RENDER_STREAMOUT
};
TRenderMode                 g_renderMode = RENDER_BUFFERTEMPLATE;

/*----------------------------------------------
  --
  -- UI IDs
  --
  ----------------------------------------------*/
#define IDC_TOGGLEFULLSCREEN          1
#define IDC_TOGGLEREF                 2
#define IDC_CHANGEDEVICE              3
#define IDC_TIMELINE                  4
#define IDC_BSGROUP                   5
#define IDC_WEIGHT0                   6
#define IDC_WEIGHT1                   7
#define IDC_WEIGHT2                   8
#define IDC_WEIGHT3                   9
#define IDC_ANIMATE                   10
#define IDC_SHOWMORE                  11
#define IDC_GLOWEXP                   12
#define IDC_GLOWSTRENGH               13
#define IDC_BUMPSTRENGH               14
#define IDC_REFLSTRENGH               15
#define IDC_FRESEXP                   16
#define IDC_RESET                     17
#define IDC_MODE                      18

std::vector<float>          g_weightTable;
int                         g_curBSGroup = 0;
CurvePool                   g_CVPool;
struct BSCurve
{
    BSCurve(int i, CurveVector* p) : weighid(i), pcv(p) {}
    int             weighid;
    CurveVector*    pcv;
};
std::vector<BSCurve>        g_BSCurves; // to make match curves and Blendshapes

ID3D10Effect*               g_Effect                = NULL;
int                         g_curTechnique          = 0;
ID3D10EffectScalarVariable* g_Weight                = NULL;

bool                        g_Wireframe             = false;
int                         g_WarningCounter        = 10;

float                       g_Timeline              = 0;
bool                        g_bUpdateAnimOnce       = false;
int                         g_CurImage              = 0;
float                       g_TexFactor             = 1.0f;
float                       g_TexAlpha              = 0.2;

CModelViewerCamera          g_Camera;

float                       g_ClearColor[]          = {0.4f, 0.4f, 0.4f, 1.0f};

bool                        g_Anim                  = true;
bool                        g_Initialised           = false;
bool                        g_ShowEffectParams      = false;

CDXUTDialogResourceManager  g_DialogResourceManager;// manager for shared resources of dialogs
CD3DSettingsDlg             g_D3DSettingsDlg;       // Device settings dialog
CDXUTDialog                 g_HUD;                  // manages the 3D UI
CDXUTDialog                 g_SampleUI;             // dialog for sample specific controls

void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CreateWeightsUI();
/*----------------------------------------------
  --
  -- Message display
  --
  ----------------------------------------------*/
void PrintMessage(int level, LPCWSTR fmt, ...)
{
    static WCHAR dest[400];
    LPCWSTR *ppc = (LPCWSTR*)_ADDRESSOF(fmt);
    wvsprintf(dest, fmt, (va_list)&(ppc[1]));
    if(level == LOG_ERR) {
        OutputDebugString(dest);
        OutputDebugString(L"\n");
        #ifndef  NDEBUG
        _wassert(dest, _CRT_WIDE(__FILE__), __LINE__);
        #else
        int r = MessageBox(NULL, dest, L"Error. Continue ?",MB_OKCANCEL|MB_ICONQUESTION);
        if(r == 2) exit(0);
        #endif
    }
    if(level == LOG_WARN) OutputDebugString(L"WARNING: ");
    OutputDebugString(dest);
    OutputDebugString(L"\n");
}
/*----------------------------------------------
    --
    -- Misc helpers
    --
    ----------------------------------------------*/
namespace Tools 
{
    ID3DX10Font*               g_Font               = NULL;
    ID3DX10Sprite*             g_FontSprite         = NULL;
    WCHAR                      g_path[MAX_PATH+1];
    char                       g_pathb[MAX_PATH+1];
    inline const char * PathNameAsBS(const char * filename)
    {
        HRESULT hr;
        WCHAR wfname[MAX_PATH];
        if(!filename)
            return NULL;
        mbstowcs_s(NULL, wfname, MAX_PATH, filename, MAX_PATH);
        hr = ::DXUTFindDXSDKMediaFileCch(Tools::g_path, MAX_PATH, wfname);
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"couldn't find %S. Maybe wrong path...", filename);
            return NULL;
        }
        wcstombs_s(NULL, Tools::g_pathb, MAX_PATH, Tools::g_path, MAX_PATH);
        return Tools::g_pathb;
    }
    inline const char * PathNameAsBS(LPWSTR filename)
    {
        HRESULT hr;
        if(!filename)
            return NULL;
        hr = ::DXUTFindDXSDKMediaFileCch(Tools::g_path, MAX_PATH, filename);
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"couldn't find %S. Maybe wrong path...", filename);
            return NULL;
        }
        wcstombs_s(NULL, Tools::g_pathb, MAX_PATH, Tools::g_path, MAX_PATH);
        return Tools::g_pathb;
    }
    inline WCHAR * PathNameAsWS(const char * filename)
    {
        HRESULT hr;
        WCHAR wfname[MAX_PATH];
        if(!filename)
            return NULL;
        mbstowcs_s(NULL, wfname, MAX_PATH, filename, MAX_PATH-1);
        hr = ::DXUTFindDXSDKMediaFileCch(g_path, MAX_PATH, wfname);
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"couldn't find %S. Maybe wrong path...", filename);
            return NULL;
        }
        return g_path;
    }
    inline WCHAR * PathNameAsWS(LPWSTR filename)
    {
        HRESULT hr;
        if(!filename)
            return NULL;
        hr = ::DXUTFindDXSDKMediaFileCch(g_path, MAX_PATH, filename);
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"couldn't find %S. Maybe wrong path...", filename);
            return NULL;
        }
        return g_path;
    }

    //--------------------------------------------------------------------------------------
    // GetPassDesc
    //--------------------------------------------------------------------------------------
    D3D10_PASS_DESC g_tmpPassDesc;
    D3D10_PASS_DESC * GetPassDesc(ID3D10Effect *pEffect, LPCSTR technique, LPCSTR pass)
    {
        ID3D10EffectTechnique * m_Tech = NULL;
        if(HIWORD(technique))
        m_Tech = pEffect->GetTechniqueByName(technique);
        else
        m_Tech = pEffect->GetTechniqueByIndex((UINT)LOWORD(technique));
        if(HIWORD(pass))
        {
            if ( FAILED( m_Tech->GetPassByName(pass)->GetDesc(&Tools::g_tmpPassDesc) ) )
            {
                LOGMSG(LOG_ERR, L"Failed getting description\n" );
                return NULL;
            }
        } else
        {
            if ( FAILED( m_Tech->GetPassByIndex((UINT)LOWORD(pass))->GetDesc(&Tools::g_tmpPassDesc) ) )
            {
                LOGMSG(LOG_ERR, L"Failed getting description\n" );
                return NULL;
            }
        }
        return &Tools::g_tmpPassDesc;
    }
    //--------------------------------------------------------------------------------------
    // CreateTextureFromFile
    // NOTE: for now, using nv_dds_common.cpp : some formats don't work with D3DX helpers
    //--------------------------------------------------------------------------------------
    HRESULT CreateTexture2DFromFile(LPCSTR texfile, ID3D10Texture2D **ppTex, ID3D10ShaderResourceView **ppView, ID3D10Device* pd3dDevice)
    {
        HRESULT hr = S_OK;
        ID3D10Texture2D *pTex;
        ID3D10ShaderResourceView *pView;
        ID3D10Resource *pTexture;
        D3DX10_IMAGE_LOAD_INFO LoadInfo;
        D3DX10_IMAGE_INFO      SrcInfo;
        hr = D3DX10GetImageInfoFromFile(Tools::PathNameAsWS(texfile), NULL, &SrcInfo);

        LoadInfo.Width          = SrcInfo.Width;
        LoadInfo.Height         = SrcInfo.Height;
        LoadInfo.Depth          = SrcInfo.Depth;
        LoadInfo.FirstMipLevel  = 0;
        LoadInfo.MipLevels      = SrcInfo.MipLevels;
        LoadInfo.Usage          = D3D10_USAGE_DEFAULT;
        LoadInfo.BindFlags      = D3D10_BIND_SHADER_RESOURCE;
        LoadInfo.CpuAccessFlags = 0;
        LoadInfo.MiscFlags      = SrcInfo.MiscFlags;
        LoadInfo.Format         = SrcInfo.Format;
        LoadInfo.Filter         = D3DX10_FILTER_LINEAR;
        LoadInfo.MipFilter      = D3DX10_FILTER_LINEAR;
        LoadInfo.pSrcInfo       = &SrcInfo;

        hr = D3DX10CreateTextureFromFile( pd3dDevice, Tools::PathNameAsWS(texfile), &LoadInfo, NULL, &pTexture);
        if(FAILED(hr))
          return hr;
        D3D10_RESOURCE_DIMENSION d;
        pTexture->GetType(&d);
        hr = pTexture->QueryInterface(__uuidof(ID3D10Texture2D), (void**)&pTex);
        pTexture->Release();
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"Error in QueryInterface(ID3D10Texture2D) for %S", texfile);
            return hr;
        }
        D3D10_SHADER_RESOURCE_VIEW_DESC viewDesc;
        viewDesc.Format = LoadInfo.Format;
        viewDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
        viewDesc.Texture2D.MostDetailedMip = LoadInfo.FirstMipLevel;
        viewDesc.Texture2D.MipLevels = LoadInfo.MipLevels;
        if ( FAILED( pd3dDevice->CreateShaderResourceView( pTexture, &viewDesc, &pView) ) )
        {
            LOGMSG(LOG_ERR,L"Error in CreateShaderResourceView for %S", texfile);
            if(ppTex) *ppTex = pTex;
            else pTex->Release();
            return hr;
        }
        LOGMSG(LOG_MSG, L"Created texture and view for %S", texfile);
        if(ppTex) *ppTex = pTex;
        else pTex->Release();
        if(ppView) *ppView = pView;
        else pView->Release();
      return hr;
    }
    HRESULT CreateTexture3DFromFile(LPCSTR texfile, ID3D10Texture3D **ppTex, ID3D10ShaderResourceView **ppView, ID3D10Device* pd3dDevice)
    {
        HRESULT hr = S_OK;
        ID3D10Texture3D *pTex;
        ID3D10ShaderResourceView *pView;

        ID3D10Resource *pTexture;
        D3DX10_IMAGE_LOAD_INFO LoadInfo;
        D3DX10_IMAGE_INFO      SrcInfo;
        hr = D3DX10GetImageInfoFromFile(Tools::PathNameAsWS(texfile), NULL, &SrcInfo);

        LoadInfo.Width          = SrcInfo.Width;
        LoadInfo.Height         = SrcInfo.Height;
        LoadInfo.Depth          = SrcInfo.Depth;
        LoadInfo.FirstMipLevel  = 0;
        LoadInfo.MipLevels      = SrcInfo.MipLevels;
        LoadInfo.Usage          = D3D10_USAGE_DEFAULT;
        LoadInfo.BindFlags      = D3D10_BIND_SHADER_RESOURCE;
        LoadInfo.CpuAccessFlags = 0;
        LoadInfo.MiscFlags      = SrcInfo.MiscFlags;
        LoadInfo.Format         = SrcInfo.Format;
        LoadInfo.Filter         = D3DX10_FILTER_LINEAR;
        LoadInfo.MipFilter      = D3DX10_FILTER_LINEAR;
        LoadInfo.pSrcInfo       = &SrcInfo;

        hr = D3DX10CreateTextureFromFile( pd3dDevice, Tools::PathNameAsWS(texfile), &LoadInfo, NULL, &pTexture);
        if(FAILED(hr))
          return hr;
        D3D10_RESOURCE_DIMENSION d;
        pTexture->GetType(&d);
        hr = pTexture->QueryInterface(__uuidof(ID3D10Texture3D), (void**)&pTex);
        pTexture->Release();
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"Error in QueryInterface(ID3D10Texture3D) for %S", texfile);
            return hr;
        }
        D3D10_SHADER_RESOURCE_VIEW_DESC viewDesc;
        viewDesc.Format = LoadInfo.Format;
        viewDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE3D;
        viewDesc.Texture2D.MostDetailedMip = LoadInfo.FirstMipLevel;
        viewDesc.Texture2D.MipLevels = LoadInfo.MipLevels;
        if ( FAILED( pd3dDevice->CreateShaderResourceView( pTexture, &viewDesc, &pView) ) )
        {
            LOGMSG(LOG_ERR,L"Error in CreateShaderResourceView for %S", texfile);
            if(ppTex) *ppTex = pTex;
            else pTex->Release();
            return hr;
        }

        LOGMSG(LOG_MSG, L"Created texture and view for %S", texfile);
        if(ppTex) *ppTex = pTex;
        else pTex->Release();
        if(ppView) *ppView = pView;
        else pView->Release();
        return S_OK;
    }
    //
    // NOTE: For now using nv_dds_common.cpp because D3DX10CreateTextureFromFile() doesn't want to load the Cubemap dds file...
    //
    HRESULT CreateCubeTextureFromFile(LPCSTR texfile, ID3D10Texture2D **ppTex, ID3D10ShaderResourceView **ppView, ID3D10Device* pd3dDevice)
    {
        HRESULT hr = S_OK;
        ID3D10Texture2D *pTex;
        ID3D10ShaderResourceView *pView;
        ID3D10Resource *pTexture;
        D3DX10_IMAGE_LOAD_INFO LoadInfo;
        D3DX10_IMAGE_INFO      SrcInfo;
        hr = D3DX10GetImageInfoFromFile(Tools::PathNameAsWS(texfile), NULL, &SrcInfo);

        LoadInfo.Width          = SrcInfo.Width;
        LoadInfo.Height         = SrcInfo.Height;
        LoadInfo.Depth          = SrcInfo.Depth;
        LoadInfo.FirstMipLevel  = 0;
        LoadInfo.MipLevels      = SrcInfo.MipLevels;
        LoadInfo.Usage          = D3D10_USAGE_DEFAULT;
        LoadInfo.BindFlags      = D3D10_BIND_SHADER_RESOURCE;
        LoadInfo.CpuAccessFlags = 0;
        LoadInfo.MiscFlags      = SrcInfo.MiscFlags;
        LoadInfo.Format         = SrcInfo.Format;
        LoadInfo.Filter         = D3DX10_FILTER_LINEAR;
        LoadInfo.MipFilter      = D3DX10_FILTER_LINEAR;
        LoadInfo.pSrcInfo       = &SrcInfo;

        hr = D3DX10CreateTextureFromFile( pd3dDevice, Tools::PathNameAsWS(texfile), &LoadInfo, NULL, &pTexture);
        if(FAILED(hr))
          return hr;
        D3D10_RESOURCE_DIMENSION d;
        pTexture->GetType(&d);
        hr = pTexture->QueryInterface(__uuidof(ID3D10Texture2D), (void**)&pTex);
        pTexture->Release();
        if(FAILED(hr))
        {
            LOGMSG(LOG_ERR,L"Error in QueryInterface(ID3D10Texture2D) for %S", texfile);
            pTexture->Release();
            return hr;
        }
        D3D10_SHADER_RESOURCE_VIEW_DESC viewDesc;
        viewDesc.Format = LoadInfo.Format;
        viewDesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURECUBE;
        viewDesc.Texture2D.MostDetailedMip = LoadInfo.FirstMipLevel;
        viewDesc.Texture2D.MipLevels = LoadInfo.MipLevels;
        if ( FAILED( pd3dDevice->CreateShaderResourceView( pTexture, &viewDesc, &pView) ) )
        {
            LOGMSG(LOG_ERR,L"Error in CreateShaderResourceView for %S", texfile);
            if(ppTex) *ppTex = pTex;
            else pTex->Release();
            return hr;
        }
        LOGMSG(LOG_MSG, L"Created texture and view for %S", texfile);
        if(ppTex) *ppTex = pTex;
        else pTex->Release();
        if(ppView) *ppView = pView;
        else pView->Release();
        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Load texture that we found in the effect : semantic "name"
    //--------------------------------------------------------------------------------------
    HRESULT LoadEffectTextures(ID3D10Effect* pEffect, ID3D10Device* pd3dDevice)
    {
        HRESULT h     = S_OK;
        HRESULT hrRet = S_OK;
        for(UINT i=0, hr = S_OK; !FAILED(hr); i++)
        {
            ID3D10EffectVariable *v = pEffect->GetVariableByIndex(i);
            if(v == NULL)
                break;
            D3D10_EFFECT_TYPE_DESC d;
            hr = v->GetType()->GetDesc(&d);
            ID3D10EffectVariable *f = v->GetAnnotationByName("file");
            if(!f)
            {
                LOGMSG(LOG_WARN, L"no 'file' annotation for the texture");
                continue;
            }
            const char * pFname = NULL;
            ID3D10ShaderResourceView * texView = NULL;
            if(d.Type == D3D10_SVT_TEXTURE2D)
            {
                if( FAILED( h=f->AsString()->GetString(&pFname) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_WARN, L"cannot get the name of the 'file' semantic" );
                }
                else if( FAILED( h = Tools::CreateTexture2DFromFile(Tools::PathNameAsBS(pFname), NULL, &texView, pd3dDevice) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_ERR, L"failed to load texture %S", pFname );
                }
                else if( texView && FAILED( h=v->AsShaderResource()->SetResource(texView) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_ERR, L"failed to bind texture_face with DDS image" );
                }
                SAFE_RELEASE(texView);
            }
            else if(d.Type == D3D10_SVT_TEXTURECUBE)
            {
                if( FAILED( h=f->AsString()->GetString(&pFname) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_WARN, L"cannot get the name of the 'file' semantic" );
                }
                else if( FAILED( h=Tools::CreateCubeTextureFromFile(Tools::PathNameAsBS(pFname), NULL, &texView, pd3dDevice) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_ERR, L"failed to load texture %S", pFname );
                }
                else if( FAILED( h=v->AsShaderResource()->SetResource(texView) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_ERR, L"failed to bind texture_face with DDS image" );
                }
                SAFE_RELEASE(texView);
            }
            else if(d.Type == D3D10_SVT_TEXTURE3D)
            {
                if( FAILED( h=f->AsString()->GetString(&pFname) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_WARN, L"cannot get the name of the 'file' semantic" );
                }
                else if( FAILED( h = Tools::CreateTexture3DFromFile(Tools::PathNameAsBS(pFname), NULL, &texView, pd3dDevice) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_ERR, L"failed to load texture %S", pFname );
                }
                else if( FAILED( h=v->AsShaderResource()->SetResource(texView) ) )
                {
                    hrRet = h;
                    LOGMSG(LOG_ERR, L"failed to bind texture_face with DDS image" );
                }
                SAFE_RELEASE(texView);
            }
        }
        return hrRet;
    }
    //-----------------------------------------------------------------------------
    // D3D10 overlay for stats (fps...)
    //-----------------------------------------------------------------------------
    void DrawStats()
    {
        CDXUTTextHelper txtHelper( Tools::g_Font, Tools::g_FontSprite, 15 );
        txtHelper.Begin();
        txtHelper.SetInsertionPos( 2, 0 );
        txtHelper.SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
        txtHelper.DrawTextLine( DXUTGetFrameStats(true) );
        txtHelper.DrawTextLine( DXUTGetDeviceStats() );
        txtHelper.End();
    }
    void InitFonts(ID3D10Device* pd3dDevice)
    {
        D3DX10CreateFont( pd3dDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
                          OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                          L"Arial", &Tools::g_Font );
        D3DX10CreateSprite( pd3dDevice, 512, &Tools::g_FontSprite );
    }
    void ReleaseFonts()
    {
        SAFE_RELEASE(Tools::g_Font);
        SAFE_RELEASE(Tools::g_FontSprite);
    }
} //Tools

NVUTSkybox g_Skybox;    // skybox

// Environment map for the skybox
ID3D10Texture2D * g_EnvMap = NULL;                    
ID3D10ShaderResourceView * g_EnvMapSRV = NULL;

/*----------------------------------------------
  --
  -- DXUT GUI
  --
  ----------------------------------------------*/
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{    
    WCHAR sz[100];
    char  sz2[100];
    switch( nControlID )
    {
        case IDC_TOGGLEFULLSCREEN: DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:        DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:     g_D3DSettingsDlg.SetActive( !g_D3DSettingsDlg.IsActive() ); break;
        case IDC_ANIMATE:
        {
            g_Anim = g_SampleUI.GetCheckBox( IDC_ANIMATE )->GetChecked();
            /*bool enable = g_Anim ? false : true;
            g_SampleUI.GetSlider( IDC_WEIGHT0 )->SetEnabled(enable);
            g_SampleUI.GetSlider( IDC_WEIGHT1 )->SetEnabled(enable);
            g_SampleUI.GetSlider( IDC_WEIGHT2 )->SetEnabled(enable);
            g_SampleUI.GetSlider( IDC_WEIGHT3 )->SetEnabled(enable);*/
            break;
        }
        case IDC_TIMELINE:
        {
            g_Timeline = (float) (g_SampleUI.GetSlider( IDC_TIMELINE )->GetValue()* 0.1f);
            StringCchPrintf( sz, 100, L"Timeline : %0.2f", g_Timeline); 
            g_SampleUI.GetStatic( IDC_TIMELINE )->SetText( sz );
            g_bUpdateAnimOnce = true;
            break;
        }
        case IDC_BSGROUP:
        {
            float v;
            g_curBSGroup = g_SampleUI.GetComboBox(IDC_BSGROUP)->GetSelectedIndex();

            v = g_weightTable[g_curBSGroup*4 + 0];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 0].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT0 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT0 )->SetValue( (int)(v * 1000.0) );

            v = g_weightTable[g_curBSGroup*4 + 1];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 1].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f", sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT1 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT1 )->SetValue( (int)(v * 1000.0) );

            v = g_weightTable[g_curBSGroup*4 + 2];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 2].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f", sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT2 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT2 )->SetValue( (int)(v * 1000.0) );

            v = g_weightTable[g_curBSGroup*4 + 3];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 3].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f", sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT3 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT3 )->SetValue( (int)(v * 1000.0) );
        }
        break;
        case IDC_WEIGHT0:
        {
            float v;
            v = (float) (g_SampleUI.GetSlider( IDC_WEIGHT0 )->GetValue()* 0.001f);
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 0].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT0 )->SetText( sz );
            g_weightTable[(g_SampleUI.GetComboBox(IDC_BSGROUP)->GetSelectedIndex())*4 + 0] = v;
            break;
        }
        case IDC_WEIGHT1:
        {
            float v;
            v = (float) (g_SampleUI.GetSlider( IDC_WEIGHT1 )->GetValue()* 0.001f);
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 1].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT1 )->SetText( sz );
            g_weightTable[(g_SampleUI.GetComboBox(IDC_BSGROUP)->GetSelectedIndex())*4 + 1] = v;
            break;
        }
        case IDC_WEIGHT2:
        {
            float v;
            v = (float) (g_SampleUI.GetSlider( IDC_WEIGHT2 )->GetValue()* 0.001f);
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 2].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT2 )->SetText( sz );
            g_weightTable[(g_SampleUI.GetComboBox(IDC_BSGROUP)->GetSelectedIndex())*4 + 2] = v;
            break;
        }
        case IDC_WEIGHT3:
        {
            float v;
            v = (float) (g_SampleUI.GetSlider( IDC_WEIGHT3 )->GetValue()* 0.001f);
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 3].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT3 )->SetText( sz );
            g_weightTable[(g_SampleUI.GetComboBox(IDC_BSGROUP)->GetSelectedIndex())*4 + 3] = v;
            break;
        }
        case IDC_SHOWMORE:
        {
            bool bShow = g_SampleUI.GetCheckBox( IDC_SHOWMORE )->GetChecked();
            g_SampleUI.GetStatic( IDC_GLOWEXP )->SetVisible(bShow);
            g_SampleUI.GetStatic( IDC_GLOWSTRENGH )->SetVisible(bShow);
            g_SampleUI.GetStatic( IDC_BUMPSTRENGH )->SetVisible(bShow);
            g_SampleUI.GetStatic( IDC_REFLSTRENGH )->SetVisible(bShow);
            g_SampleUI.GetStatic( IDC_FRESEXP )->SetVisible(bShow);
            g_SampleUI.GetSlider( IDC_GLOWEXP )->SetVisible(bShow);
            g_SampleUI.GetSlider( IDC_GLOWSTRENGH )->SetVisible(bShow);
            g_SampleUI.GetSlider( IDC_BUMPSTRENGH )->SetVisible(bShow);
            g_SampleUI.GetSlider( IDC_REFLSTRENGH )->SetVisible(bShow);
            g_SampleUI.GetSlider( IDC_FRESEXP )->SetVisible(bShow);
        }
        case IDC_GLOWEXP:
        {
            float f = (float) (g_SampleUI.GetSlider( IDC_GLOWEXP )->GetValue())* 0.001f;
            StringCchPrintf( sz, 100, L"Glow Exp : %0.2f", f); 
            g_SampleUI.GetStatic( IDC_GLOWEXP )->SetText( sz );
            g_Effect->GetVariableByName("glowExp")->AsScalar()->SetFloat(f);
            break;
        }
        case IDC_GLOWSTRENGH:
        {
            float f = (float) (g_SampleUI.GetSlider( IDC_GLOWSTRENGH )->GetValue())* 0.001f;
            StringCchPrintf( sz, 100, L"Glow Strength : %0.2f", f); 
            g_SampleUI.GetStatic( IDC_GLOWSTRENGH )->SetText( sz );
            g_Effect->GetVariableByName("glowStrength")->AsScalar()->SetFloat(f);
            break;
        }
        case IDC_BUMPSTRENGH:
        {
            float f = (float) (g_SampleUI.GetSlider( IDC_BUMPSTRENGH )->GetValue())* 0.001f;
            StringCchPrintf( sz, 100, L"Bump Strength : %0.2f", f); 
            g_SampleUI.GetStatic( IDC_BUMPSTRENGH )->SetText( sz );
            g_Effect->GetVariableByName("bumpStrength")->AsScalar()->SetFloat(f);
            break;
        }
        case IDC_REFLSTRENGH:
        {
            float f = (float) (g_SampleUI.GetSlider( IDC_REFLSTRENGH )->GetValue())* 0.001f;
            StringCchPrintf( sz, 100, L"Refl. Strength : %0.2f", f); 
            g_SampleUI.GetStatic( IDC_REFLSTRENGH )->SetText( sz );
            g_Effect->GetVariableByName("reflStrength")->AsScalar()->SetFloat(f);
            break;
        }
        case IDC_FRESEXP:
        {
            float f = (float) (g_SampleUI.GetSlider( IDC_FRESEXP )->GetValue())* 0.001f;
            StringCchPrintf( sz, 100, L"Fresnel Exp : %0.2f", f); 
            g_SampleUI.GetStatic( IDC_FRESEXP )->SetText( sz );
            g_Effect->GetVariableByName("fresExp")->AsScalar()->SetFloat(f);
            break;
        }
        case IDC_MODE:
        {
            g_renderMode = (TRenderMode)g_SampleUI.GetComboBox(IDC_MODE)->GetSelectedIndex();
            break;
        }
        case IDC_RESET:
        {
            int n = g_MeshTable[0].pMesh->numBlendShapes;
            for(int i=0; i<n ;i++)
                g_weightTable[i] = 0.0;
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 0].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : 0",sz2); 
            g_SampleUI.GetStatic( IDC_WEIGHT0 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT0 )->SetValue( 0 );

            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 1].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : 0", sz2); 
            g_SampleUI.GetStatic( IDC_WEIGHT1 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT1 )->SetValue( 0 );

            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 2].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : 0", sz2); 
            g_SampleUI.GetStatic( IDC_WEIGHT2 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT2 )->SetValue( 0 );

            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 3].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : 0", sz2); 
            g_SampleUI.GetStatic( IDC_WEIGHT3 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT3 )->SetValue( 0 );

            g_Timeline = 0.0;
            g_SampleUI.GetSlider( IDC_TIMELINE )->SetValue(0);
            g_SampleUI.GetStatic( IDC_TIMELINE )->SetText( L"Timeline : 0" );
            g_Anim = false;
            g_SampleUI.GetCheckBox( IDC_ANIMATE )->SetChecked(false);

            g_SampleUI.GetStatic( IDC_GLOWEXP )->SetText( L"Glow Exp : 3.0" );
            g_SampleUI.GetSlider( IDC_GLOWEXP )->SetValue( 3000 );
            g_Effect->GetVariableByName("glowExp")->AsScalar()->SetFloat(3.0);

            g_SampleUI.GetStatic( IDC_GLOWSTRENGH )->SetText( L"Glow Strength : 0.5" );
            g_SampleUI.GetSlider( IDC_GLOWSTRENGH )->SetValue( 500 );
            g_Effect->GetVariableByName("glowStrength")->AsScalar()->SetFloat(0.5);

            g_SampleUI.GetStatic( IDC_BUMPSTRENGH )->SetText( L"Bump Strength : 2.0" );
            g_SampleUI.GetSlider( IDC_BUMPSTRENGH )->SetValue( 2000 );
            g_Effect->GetVariableByName("bumpStrength")->AsScalar()->SetFloat(2.0);

            g_SampleUI.GetStatic( IDC_REFLSTRENGH )->SetText( L"Refl. Strength : 1.4" );
            g_SampleUI.GetSlider( IDC_REFLSTRENGH )->SetValue( 1400 );
            g_Effect->GetVariableByName("reflStrength")->AsScalar()->SetFloat(1.4);

            g_SampleUI.GetStatic( IDC_FRESEXP )->SetText( L"Fresnel Exp : 3.0" );
            g_SampleUI.GetSlider( IDC_FRESEXP )->SetValue( 3000 );
            g_Effect->GetVariableByName("fresExp")->AsScalar()->SetFloat(3.0);
        }
        break;
    }
}
/*----------------------------------------------
    --
    --
    --
    ----------------------------------------------*/
HRESULT CreateUI(ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc)
{
    g_D3DSettingsDlg.GetDialogControl()->RemoveAllControls();
    g_D3DSettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.RemoveAllControls();
    g_SampleUI.RemoveAllControls();
    g_HUD.Init( &g_DialogResourceManager );
    g_SampleUI.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent ); 
    int iY = 10;
    int iX = pBackBufferSurfaceDesc->Width - 110;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen (F4)", iX-15, iY, 125, 22, VK_F4 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", iX-15, iY += 24, 125, 22, VK_F3 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", iX-15, iY += 24, 125, 22, VK_F2 );
    g_HUD.AddButton( IDC_RESET, L"Reset (F5)", iX-15, iY += 24, 125, 22, VK_F5 );

    g_SampleUI.AddComboBox( IDC_MODE, iX-15, iY += 24, 125, 22); 
        g_SampleUI.GetComboBox( IDC_MODE )->AddItem(L"Use Buffer Template", NULL);
        g_SampleUI.GetComboBox( IDC_MODE )->AddItem(L"Use Stream Output", NULL);

    g_SampleUI.SetCallback( OnGUIEvent ); 

    g_SampleUI.AddCheckBox( IDC_SHOWMORE, L"Surface tweaks", iX-15, iY += 24, 125, 22, false );
    g_SampleUI.AddCheckBox( IDC_ANIMATE, L"Animate", iX-15, iY += 24, 125, 22, g_Anim );

    WCHAR sz[100];
    StringCchPrintf( sz, 100, L"Timeline : %0.2f", g_Timeline); 
    g_SampleUI.AddStatic( IDC_TIMELINE, sz, iX-15, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_TIMELINE, iX, iY += 24, 100, 22, 0, 10000, 0 );

    StringCchPrintf( sz, 100, L"Blendshape Groups"); 
    g_SampleUI.AddStatic( IDC_BSGROUP, sz, iX-15, iY += 24, 125, 22 );
    g_SampleUI.AddComboBox(IDC_BSGROUP, iX, iY += 24, 100, 22);
    CreateWeightsUI();

    StringCchPrintf( sz, 100, L"Blendshape weight : 0"); 
    g_SampleUI.AddStatic( IDC_WEIGHT0, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_WEIGHT0, iX, iY += 24, 100, 22, 0, 1000, 0 );

    StringCchPrintf( sz, 100, L"Blendshape weight : 0"); 
    g_SampleUI.AddStatic( IDC_WEIGHT1, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_WEIGHT1, iX, iY += 24, 100, 22, 0, 1000, 0 );

    StringCchPrintf( sz, 100, L"Blendshape weight : 0"); 
    g_SampleUI.AddStatic( IDC_WEIGHT2, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_WEIGHT2, iX, iY += 24, 100, 22, 0, 1000, 0 );

    StringCchPrintf( sz, 100, L"Blendshape weight : 0"); 
    g_SampleUI.AddStatic( IDC_WEIGHT3, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_WEIGHT3, iX, iY += 24, 100, 22, 0, 1000, 0 );

    /*bool enable = g_Anim ? false : true;
    g_SampleUI.GetSlider( IDC_WEIGHT0 )->SetEnabled(enable);
    g_SampleUI.GetSlider( IDC_WEIGHT1 )->SetEnabled(enable);
    g_SampleUI.GetSlider( IDC_WEIGHT2 )->SetEnabled(enable);
    g_SampleUI.GetSlider( IDC_WEIGHT3 )->SetEnabled(enable);*/

    // Additional tweakables
    iY = 10;
    iX = 20;
    StringCchPrintf( sz, 100, L"Glow Exp : 3.0"); 
    g_SampleUI.AddStatic( IDC_GLOWEXP, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_GLOWEXP, iX, iY += 24, 100, 22, 0, 6000, 3000 );

    StringCchPrintf( sz, 100, L"Glow Strength : 0.5"); 
    g_SampleUI.AddStatic( IDC_GLOWSTRENGH, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_GLOWSTRENGH, iX, iY += 24, 100, 22, 0, 6000, 500 );

    StringCchPrintf( sz, 100, L"Bump Strength : 2.0"); 
    g_SampleUI.AddStatic( IDC_BUMPSTRENGH, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_BUMPSTRENGH, iX, iY += 24, 100, 22, 0, 6000, 2000 );

    StringCchPrintf( sz, 100, L"Refl. Strength : 1.4"); 
    g_SampleUI.AddStatic( IDC_REFLSTRENGH, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_REFLSTRENGH, iX, iY += 24, 100, 22, 0, 6000, 1400 );

    StringCchPrintf( sz, 100, L"Fresnel Exp : 3.0"); 
    g_SampleUI.AddStatic( IDC_FRESEXP, sz, iX-20, iY += 24, 125, 22 );
    g_SampleUI.AddSlider( IDC_FRESEXP, iX, iY += 24, 100, 22, 0, 6000, 3000 );

    g_SampleUI.GetStatic( IDC_GLOWEXP )->SetVisible(false);
    g_SampleUI.GetStatic( IDC_GLOWSTRENGH )->SetVisible(false);
    g_SampleUI.GetStatic( IDC_BUMPSTRENGH )->SetVisible(false);
    g_SampleUI.GetStatic( IDC_REFLSTRENGH )->SetVisible(false);
    g_SampleUI.GetStatic( IDC_FRESEXP )->SetVisible(false);
    g_SampleUI.GetSlider( IDC_GLOWEXP )->SetVisible(false);
    g_SampleUI.GetSlider( IDC_GLOWSTRENGH )->SetVisible(false);
    g_SampleUI.GetSlider( IDC_BUMPSTRENGH )->SetVisible(false);
    g_SampleUI.GetSlider( IDC_REFLSTRENGH )->SetVisible(false);
    g_SampleUI.GetSlider( IDC_FRESEXP )->SetVisible(false);
    return S_OK;
}
/*----------------------------------------------
  --
  --
  --
  ----------------------------------------------*/
bool InitEffect(ID3D10Device* pd3dDevice)
{
    ID3D10Blob *pErrors;
    HRESULT  hr= S_OK;
    hr = D3DX10CreateEffectFromFile(Tools::PathNameAsWS(L"GPUBlendShapes.fx"), NULL, NULL, D3D10_SHADER_NO_PRESHADER, 0, pd3dDevice, NULL, NULL, &g_Effect, &pErrors);

    if( FAILED( hr ) )
    {
        LPCSTR szErrors = NULL;
        if(pErrors)
        {
            szErrors = (LPCSTR)pErrors->GetBufferPointer();
            LOGMSG(LOG_ERR, L"Effect compiler message:\n%S", szErrors);
            pErrors->Release();
        }
        g_Initialised = false;
        return false;
    }
    g_Weight = g_Effect->GetVariableBySemantic("BSWEIGHT")->AsScalar();
    if(!g_Weight)
    {
        g_Initialised = false;
        return false;
    }
    //
    // Texture Load : using semantic 'file' in the effect...
    //
    Tools::LoadEffectTextures(g_Effect, pd3dDevice);
    return true;
}

/*----------------------------------------------
  --
  --
  --
  ----------------------------------------------*/
HRESULT GenerateInputLayouts(ID3D10Device* pd3dDevice)
{
    HRESULT hr;
    if(!g_Effect)
        return E_FAIL;
    for(int i=0; i<MESHNUM; i++)
    {
        ID3D10EffectTechnique * tech = NULL;
        if(g_MeshTable[i].pMesh->numBlendShapes > 0)
          tech = g_Effect->GetTechniqueByName("nv_f_head_frontSO");
        else
          tech = g_Effect->GetTechniqueByName(g_MeshTable[i].pMesh->primGroup[0].name);
        hr = g_MeshTable[i].createInputLayout(pd3dDevice, tech);
        if(FAILED(hr))
            return hr;
    }
    return S_OK;
}
/*----------------------------------------------
  --
  --
  --
  ----------------------------------------------*/
void CreateWeightsUI()
{
    if(!g_MeshTable[0].pMesh)
        return;
    //
    // setup the weights on the first one to have GPUBlendShapes (assuming all have the same amount)
    //
    int i=0; // simplification : in fact we know the first one has the blendshapes...
    g_weightTable.clear();
    {
      g_weightTable.resize(g_MeshTable[i].pMesh->numBlendShapes, 0);
      int nsliders = g_MeshTable[i].pMesh->numBlendShapes;
      for(int j=0; j< nsliders; j++)
      {
        char ctrlname[32];
        if(j%4 == 0)
        {
            WCHAR sz[100];
            StringCchPrintf( sz, 100, L"Group %d", j/4); 
            g_SampleUI.GetComboBox( IDC_BSGROUP )->AddItem(sz, NULL);
        }
        sprintf_s(ctrlname, 32, "%S", g_MeshTable[i].pMesh->bsSlots[j].name);
      }
    }
}
/*----------------------------------------------
  --
  --
  --
  ----------------------------------------------*/
bool InitInputGeometry(ID3D10Device* pd3dDevice)
{
    HRESULT  hr;
    //
    // Load all the Meshes
    //
    for(int i=0; i<MESHNUM; i++)
    {
        if(!g_MeshTable[i].loadMesh(Tools::PathNameAsBS(g_meshFiles[i])))
        {
          return false;
        }
        //
        // Buffers : create for vertex input and for resource (Buffer<> template in the effect)
        //
        hr = g_MeshTable[i].createVertexBuffers(pd3dDevice, true,false, true,true, DBGINFO);
        hr = g_MeshTable[i].createBSOffsetsAndWeightBuffers(pd3dDevice, DBGINFO);
        hr = g_MeshTable[i].createIndexBuffers(pd3dDevice, DBGINFO);
        //
        // Special case for #0 : the blendshape of the face exposed in 3 additional attributes
        //
        //
        //ask for Stream out
        //
        if(g_MeshTable[i].pMesh->numBlendShapes > 0)
        {
            g_MeshTable[i].generateLayoutDesc(4); // argument is for additional attributes for GPUBlendShapes
            hr = g_MeshTable[i].createStreamOutBuffers(pd3dDevice, 2,0, DBGINFO); // ask for 2 Streamout buffers (ping-pong) based on slot #0
        }
        else g_MeshTable[i].generateLayoutDesc();
        //
        // If you want to get the layout details dumped in VC8 window...
        //
#ifdef DBGINFO
        g_MeshTable[i].pMesh->debugDumpLayout();
#endif
    }
    //
    // Additional UI setup
    //
    CreateWeightsUI();
    //
    // Input layout. For every meshes, generate them
    //
    GenerateInputLayouts(pd3dDevice);
    //
    // init the curve system
    //
    g_CVPool.newCVFromFile(Tools::PathNameAsBS("MediaFor_GPUBlendShapes\\anim.txt"));
#if 0
    // WARNING: having a little bug in the way to make them match... using the other alternative
    // take 1st mesh (0) because all have the same BS names and same BS amount
    for(int i=0; i < g_MeshTable[0].pMesh->numBlendShapes; i++)
    {
        // extract from the pool the curves that match with each BS
        CurveVector *pcv = g_CVPool.getCV( g_MeshTable[0].pMesh->bsSlots[i].name );
        if(pcv)
        {
            g_BSCurves.push_back(BSCurve(i,pcv));
        } else {
            LOGMSG(LOG_WARN, L"Blendshape %S doesn't have any curve anim\n", g_MeshTable[0].pMesh->bsSlots[i].name);
        }
    }
#else
// alternate way to make the curve match the BS.
#define SHANIASMILE 0
#define SMILEOPENSQUINT2 1
#define ANDIESMILE 3
#define BESTOPENSMILE1 4
#define IRRITATED 6
#define EYEBLINKFIX1 7
#define EYELOOKUP1 9
#define MOUTHOPEN1 10
#define BROWSUP1 11
#define STUDY1 16
#define SMIRKRT 17
#define SMIRKLFT 19
#define SMIRKSLY1 20
#define DISGUST 23
#define EYEBLINKFORCEDLFT 28
#define MILDSQUINT 29
#define NOSTRILSIN 31
#define SMILECLOSED2 35
#define ANGRYEYES 38
#define SADNEW 39
#define EARSBACK 41
#define EARSFWD 42
#define BREATHBLOW 44
#define BROWUPRT 46
#define BROWDOWNLFT 49 

    g_BSCurves.clear();
#define MATCH_CV(n,i)\
    { CurveVector *pcv = g_CVPool.getCV(n); g_BSCurves.push_back(BSCurve(i,pcv)); }
    MATCH_CV("ShaniaSmile", SHANIASMILE);
    MATCH_CV("SmileOpenSquint2", SMILEOPENSQUINT2);
    MATCH_CV("AndieSmile", ANDIESMILE);
    MATCH_CV("BestOpenSmile1", BESTOPENSMILE1);
    MATCH_CV("Irritated", IRRITATED);
    MATCH_CV("EyeBlinkFix1", EYEBLINKFIX1);
    MATCH_CV("EyeLookUp1", EYELOOKUP1);
    MATCH_CV("MouthOpen1", MOUTHOPEN1);
    MATCH_CV("BrowsUp1", BROWSUP1);
    MATCH_CV("study1", STUDY1);
    MATCH_CV("SmirkRt", SMIRKRT);
    MATCH_CV("SmirkLft", SMIRKLFT);
    MATCH_CV("SmirkSly1", SMIRKSLY1);
    MATCH_CV("Disgust", DISGUST);
    MATCH_CV("EyeBlinkForcedLft", EYEBLINKFORCEDLFT);
    MATCH_CV("MildSquint", MILDSQUINT);
    MATCH_CV("NostrilsIn", NOSTRILSIN);
    MATCH_CV("SmileClosed2", SMILECLOSED2);
    MATCH_CV("AngryEyes", ANGRYEYES);
    MATCH_CV("SadNew", SADNEW);
    MATCH_CV("EarsBack", EARSBACK);
    MATCH_CV("EarsFwd", EARSFWD);
    MATCH_CV("BreathBlow", BREATHBLOW);
    MATCH_CV("BrowUpRt", BROWUPRT);
    MATCH_CV("BrowDownLft", BROWDOWNLFT);
#endif
    return true;
}


//--------------------------------------------------------------------------------------
// Reject any D3D10 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D10DeviceAcceptable( UINT Adapter, UINT Output, D3D10_DRIVER_TYPE DeviceType, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10CreateDevice( ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;
    g_Initialised = true;

    CreateUI(pd3dDevice, pBackBufferSurfaceDesc);
    V_RETURN( g_DialogResourceManager.OnD3D10CreateDevice( pd3dDevice ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D10CreateDevice( pd3dDevice ) );
    
    V_RETURN( g_Skybox.OnCreateDevice( pd3dDevice ) );

    Tools::InitFonts(pd3dDevice);
    //
    // Effect
    //
    if(!InitEffect(pd3dDevice))
    {
        g_Initialised = false;
    }
    //
    // Geometry
    //
    g_weightTable.clear();
    if(!InitInputGeometry(pd3dDevice) )
    {
        g_Initialised = false;
    }
    
    // Load envmap for the skybox
    hr = Tools::CreateCubeTextureFromFile(Tools::PathNameAsBS("..\\..\\media\\CM_Forest.dds"), &g_EnvMap, &g_EnvMapSRV, pd3dDevice);
    if(FAILED(hr))
    {
       LOGMSG(LOG_ERR, L"Failed to load Skybox map");
       return hr;
    }

    g_Skybox.SetTexture(g_EnvMapSRV);
    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10ResizedSwapChain( ID3D10Device* pd3dDevice, IDXGISwapChain *pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr = S_OK;
    V_RETURN( g_DialogResourceManager.OnD3D10ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D10ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    g_Skybox.OnResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc );

    float fAspectRatio = (float)pBackBufferSurfaceDesc->Width / (float)pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( 15.0*D3DX_PI/180.0, fAspectRatio, 0.1f, 100.0f );
    g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height);
    CreateUI(pd3dDevice, pBackBufferSurfaceDesc);
    return S_OK;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    g_Camera.FrameMove(fElapsedTime);
    if(g_Anim)
    {
        if(g_Timeline > 1000.0) 
            g_Timeline = 0;
        else
            g_Timeline += (fElapsedTime * 48.0f);
    }
}


//--------------------------------------------------------------------------------------
// Render the scene using Stream output
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender_StreamOut( ID3D10Device* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
    ID3D10Buffer *pNullBuffers[] = { NULL };
    UINT offsets[6] = {0, 0, 0, 0, 0, 0};
    const char *n;
    ID3D10EffectTechnique *tech;
    //
    // Process all the meshes with Blendshapes
    //
    for(int m=0; m<MESHNUM; m++)
    {
        if(g_MeshTable[m].pMesh->numBlendShapes == 0)
            continue;
        //
        // In our case, we assume 
        //
        ID3D10Buffer *BufferTable[5] = {
            g_MeshTable[m].pVertexBuffers[0], //interleaved data
            g_MeshTable[m].pVertexBuffers[1], //blendshape 0
            g_MeshTable[m].pVertexBuffers[2], //blendshape 1
            g_MeshTable[m].pVertexBuffers[3], //blendshape 2
            g_MeshTable[m].pVertexBuffers[4], //blendshape 3
        };
        pd3dDevice->IASetInputLayout( g_MeshTable[m].pLayout );
        //
        // here we take a commong technique for all... assume the IA Layout is compatible
        //
        tech = g_Effect->GetTechniqueByName("nv_f_head_frontSO");//n);
        D3D10_TECHNIQUE_DESC td; td.Name = NULL; tech->GetDesc(&td);
        if((!tech)||(!td.Name))
            continue;
        float W[4] = {g_weightTable[0], g_weightTable[1], g_weightTable[2], g_weightTable[3]};
        g_Weight->SetFloatArray(W, 0, 4);
        ID3D10EffectPass *pPass = tech->GetPassByIndex(0);
        pPass->Apply(0);
        pd3dDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_POINTLIST );
        //
        // Loop to compute progressive blendshape accumulation.
        // Ping-pong between 2 buffers
        //
        int nBS = g_MeshTable[m].pMesh->numBlendShapes/4;
        int nBSPass = 0;
        for(int i=0; i<nBS; i++)
        {
            if(nBSPass > 0) // after the first shot : alternate the 2 Stream buffers
            {
                W[0] = g_weightTable[i*4+0];
                W[1] = g_weightTable[i*4+1];
                W[2] = g_weightTable[i*4+2];
                W[3] = g_weightTable[i*4+3];
                // bypass these weights if all are set to 0
                // Note: we could do a better job here by packing != Blendshapes together...
                if((W[0]==0.0f)&&(W[1]==0.0f)&&(W[2]==0.0f)&&(W[3]==0.0f))
                    continue;
                BufferTable[0] = g_MeshTable[m].pStreamOutBuffers[1-(nBSPass&1)]; //ping-pong
                BufferTable[1] = g_MeshTable[m].pVertexBuffers[1 + i*4]; // Update the Blendshape[s]. We assume GPUBlendShapes are just after the primary VBuffer
                BufferTable[2] = g_MeshTable[m].pVertexBuffers[2 + i*4];
                BufferTable[3] = g_MeshTable[m].pVertexBuffers[3 + i*4];
                BufferTable[4] = g_MeshTable[m].pVertexBuffers[4 + i*4];
                g_Weight->SetFloatArray(W, 0, 4);
                pPass->Apply(0); // to propagate SetFloatArray() change
            }
            pd3dDevice->IASetVertexBuffers( 0, 5, BufferTable, g_MeshTable[m].pStrides, offsets );
            pd3dDevice->SOSetTargets( 1, &(g_MeshTable[m].pStreamOutBuffers[nBSPass&1]), offsets );
            pd3dDevice->Draw(g_MeshTable[m].pMesh->slots[0].vertexCount , 0);
            pd3dDevice->SOSetTargets( 1, pNullBuffers, offsets ); // Optional. Just to avoid the warning
            nBSPass++;
        }
        //
        // In the mesh m, loop into various primitive groups
        //
        for(int i=0; i<g_MeshTable[m].numIndexBuffers; i++)
        {
            //---------------------------------------------
            // Now we draw the resulting transformed buffer
            //
            n = g_MeshTable[m].pMesh->primGroup[i].name;
            tech = g_Effect->GetTechniqueByName(n);
            //tech returning a technique even when the tech name doesn't exist !! Argl
            D3D10_TECHNIQUE_DESC td; td.Name = NULL; 
            tech->GetDesc(&td);
            if((!tech)||(!td.Name)||(strcmp(td.Name, n)))
            {
              continue;
            }
            tech->GetPassByIndex(0)->Apply(0);
            pd3dDevice->SOSetTargets( 1, pNullBuffers, offsets );
            //
            // Use the Streamed buffer at Slot #0 : replacing the the original one.
            //
            pd3dDevice->IASetVertexBuffers( 0, 1, &(g_MeshTable[m].pStreamOutBuffers[1-(nBSPass&1)]), g_MeshTable[m].pStrides, offsets );
            pd3dDevice->IASetIndexBuffer( g_MeshTable[m].pIndexBuffers[i], g_MeshTable[m].pMesh->primGroup[i].indexFormatDX10, 0 );
            pd3dDevice->IASetPrimitiveTopology( g_MeshTable[m].pMesh->primGroup[i].topologyDX10 );
            pd3dDevice->DrawIndexed(g_MeshTable[m].pMesh->primGroup[i].indexCount, 0, 0);
        }
    }
}
//--------------------------------------------------------------------------------------
// Compute the list of Blendshapes to apply, depending on their weights
// Map the buffers and assign various weights and offsets
// The vertex shader will walk through this list and apply BS according to their weights
//--------------------------------------------------------------------------------------
//#define WAR_MAP // alternate method to copy data...
void buildBlendShapeBatchList(ID3D10Device* pd3dDevice, NVD3D10RawMesh &mesh)
{
    HRESULT hr;
    UINT  *pOffsets = NULL;
    float *pWeights = NULL;
    ID3D10Buffer *pBufW = NULL;
    ID3D10Buffer *pBufO = NULL;
    D3D10_MAP mapmode;
    if(!mesh.bsResource.pOffsetsResource)
        return;
#ifdef WAR_MAP
    mapmode = D3D10_MAP_WRITE;
    D3D10_BUFFER_DESC bufferDescMesh =
    {
        mesh.pMesh->numBlendShapes * sizeof(float),
        D3D10_USAGE_STAGING, 0, D3D10_CPU_ACCESS_WRITE, 0 
    };
    hr = pd3dDevice->CreateBuffer( &bufferDescMesh, NULL, &pBufW);
    if( FAILED( hr ) )
    {
        LOGMSG(LOG_ERR, L"Failed creating STAGING buffer" );
        return;
    }
    bufferDescMesh.ByteWidth = mesh.pMesh->numBlendShapes * sizeof(UINT);
    hr = pd3dDevice->CreateBuffer( &bufferDescMesh, NULL, &pBufO);
    if( FAILED( hr ) )
    {
        LOGMSG(LOG_ERR, L"Failed creating STAGING buffer" );
        return;
    }
#else
    mapmode = D3D10_MAP_WRITE_DISCARD;
    pBufW = mesh.bsResource.pWeightsResource;
    pBufO = mesh.bsResource.pOffsetsResource;
#endif
    //
    // Map resources
    //
    hr = pBufW->Map(mapmode, 0, (void**)&pWeights);
    if(FAILED(hr))
    {
        LOGMSG(LOG_ERR,L"Failed to Map bsResource.pWeightsResource");
        return;
    }
    hr = pBufO->Map(mapmode, 0, (void**)&pOffsets);
    if(FAILED(hr))
    {
        LOGMSG(LOG_ERR,L"Failed to Map bsResource.pOffsetsResource");
        return;
    }
    //
    // Walk through Blendshape weights
    //
    mesh.bsResource.numUsedBS = 0;
    for(unsigned int i=0; i<g_weightTable.size(); i++)
    {
        const float &w = g_weightTable[i];
        if(w > 0.0)
        {
            *pWeights++ = w;
            *pOffsets++ = i;
            mesh.bsResource.numUsedBS++;
        }
    }
    g_Effect->GetVariableByName( "numBS" )->AsScalar()->SetInt(mesh.bsResource.numUsedBS);
    //
    // Unmap resources
    //
    pBufW->Unmap();
    pBufO->Unmap();
#ifdef WAR_MAP
    pd3dDevice->CopyResource(mesh.bsResource.pWeightsResource, pBufW);
    pd3dDevice->CopyResource(mesh.bsResource.pOffsetsResource, pBufO);
    pBufW->Release();
    pBufO->Release();
#endif
}
//--------------------------------------------------------------------------------------
// Render the scene using Stream output
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender_BufferTemplate( ID3D10Device* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
    ID3D10Buffer *pNullBuffers[] = { NULL };
    UINT offsets[4] = {0, 0, 0, 0};
    ID3D10EffectTechnique *tech;
    for(int m=0; m<MESHNUM; m++)
    {
        if(g_MeshTable[m].pMesh->numBlendShapes == 0)
            continue;
        //
        // Compute the list of Blendshapes to apply, depending on their weights
        // (i.e. weights=0 are bypassed)
        //
        buildBlendShapeBatchList(pd3dDevice, g_MeshTable[m]);
        g_Effect->GetVariableByName( "bsVertices" )->AsShaderResource()->SetResource( g_MeshTable[m].bsResource.pVertexView );
        g_Effect->GetVariableByName( "bsOffsets" )->AsShaderResource()->SetResource( g_MeshTable[m].bsResource.pOffsetsView);
        g_Effect->GetVariableByName( "bsWeights" )->AsShaderResource()->SetResource( g_MeshTable[m].bsResource.pWeightsView);
        g_Effect->GetVariableByName( "bsPitch" )->AsScalar()->SetInt(g_MeshTable[m].pMesh->bsSlots[0].vertexCount * 3); // assume we have 3 float3 attribs
        pd3dDevice->SOSetTargets( 1, pNullBuffers, offsets );
//#pragma message("TODO : correct pMesh->numSlots+4")
        pd3dDevice->IASetVertexBuffers( 0, g_MeshTable[m].pMesh->numSlots+4, g_MeshTable[m].pVertexBuffers, g_MeshTable[m].pStrides, offsets );
        pd3dDevice->IASetInputLayout( g_MeshTable[m].pLayout );
        for(int i=0; i<g_MeshTable[m].numIndexBuffers; i++)
        {
            char techName[200]; // Yes... old school but enough for a sample demo like this one :)
            sprintf_s(techName, 200, "%s_buffertemplate", g_MeshTable[m].pMesh->primGroup[i].name);
            //
            // take the technique with the same name as primitive group's name
            //
            tech = g_Effect->GetTechniqueByName(techName);
            //'tech' returning a technique even when the tech name doesn't exist !
            D3D10_TECHNIQUE_DESC td; td.Name = NULL; 
            tech->GetDesc(&td);
            if((!tech)||(!td.Name)||(strcmp(td.Name, techName)))
            {
                LOGMSG(LOG_WARN, L"couldn't find the technique %S", techName);
                continue;
            }
            tech->GetPassByIndex(0)->Apply(0);
            pd3dDevice->IASetIndexBuffer( g_MeshTable[m].pIndexBuffers[i], g_MeshTable[m].pMesh->primGroup[i].indexFormatDX10, 0 );
            pd3dDevice->IASetPrimitiveTopology( g_MeshTable[m].pMesh->primGroup[i].topologyDX10 );
            pd3dDevice->DrawIndexed(g_MeshTable[m].pMesh->primGroup[i].indexCount, 0, 0);
        }
    }
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D10 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender( ID3D10Device* pd3dDevice, double fTime, float fElapsedTime, void* pUserContext )
{
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.OnRender( fElapsedTime );
        return;
    }
    D3DXMATRIX mProj = *g_Camera.GetProjMatrix();
    D3DXMATRIX mView = *g_Camera.GetViewMatrix();;
    D3DXMATRIX mWorld = *g_Camera.GetWorldMatrix();

    // Calculate the matrix world*view*proj
    D3DXMATRIX mWorldView, mWorldViewI, mWorldViewIT;
    D3DXMATRIX mWorldViewProj, mWorldViewProjI, mWorldI, mWorldIT, mViewProj, mViewProjI;
    D3DXMatrixMultiply( &mViewProj, &mView, &mProj );
    D3DXMatrixInverse( &mViewProjI, NULL, &mViewProj );
    D3DXMatrixMultiply( &mWorldView, &mWorld, &mView );
    D3DXMatrixInverse(&mWorldViewI, NULL, &mWorldView);
    D3DXMatrixTranspose(&mWorldViewIT, &mWorldViewI);
    D3DXMatrixInverse(&mWorldI, NULL, &mWorld);
    D3DXMatrixTranspose(&mWorldIT, &mWorldI);
    D3DXMatrixMultiply( &mWorldViewProj, &mWorldView, &mProj );
    D3DXMatrixInverse(&mWorldViewProjI, NULL, &mWorldViewProj);
    //
    //Clear the rendertarget
    //
    pd3dDevice->ClearDepthStencilView(DXUTGetD3D10DepthStencilView(), D3D10_CLEAR_DEPTH, 1.0f, 0);
    pd3dDevice->ClearRenderTargetView( DXUTGetD3D10RenderTargetView(), g_ClearColor);

    g_Skybox.OnFrameRender(mWorldViewProj); // render skybox

    // if something went wrong in the main effect, don't do anything
    if(!g_Initialised)
    {
        return;
    }
    g_Effect->GetVariableByName("WorldViewProj")->AsMatrix()->SetMatrix(mWorldViewProj);
    g_Effect->GetVariableByName("ViewProj")->AsMatrix()->SetMatrix(mViewProj);
    g_Effect->GetVariableByName("World")->AsMatrix()->SetMatrix(mWorld);
    g_Effect->GetVariableByName("WorldIT")->AsMatrix()->SetMatrix(mWorldIT);
    g_Effect->GetVariableByName("WorldView")->AsMatrix()->SetMatrix(mWorldView);
    g_Effect->GetVariableByName("WorldViewIT")->AsMatrix()->SetMatrix(mWorldViewIT);
    g_Effect->GetVariableByName("Proj")->AsMatrix()->SetMatrix(mProj);

    D3DXMATRIX mViewI;
    D3DXMatrixInverse(&mViewI, NULL, &mView);
    g_Effect->GetVariableByName("eyeWorld")->AsVector()->SetFloatVector(mViewI.m[3]);
    //
    // update the weights from the curve animations
    //
    if(g_Anim || g_bUpdateAnimOnce)
    {
        for(unsigned int i=0; i<g_BSCurves.size(); i++)
        {
            float v;
            CurveVector *pcv = g_BSCurves[i].pcv;
            int id = g_BSCurves[i].weighid;
            if(pcv)
            {
                pcv->evaluate(g_Timeline, &v);
                g_weightTable[id] = v;
            }
        }
        if(!g_bUpdateAnimOnce)
        {
            WCHAR sz[100];
            char sz2[100];
            float v;
            StringCchPrintf( sz, 100, L"Timeline : %0.2f", g_Timeline); 
            g_SampleUI.GetStatic( IDC_TIMELINE )->SetText( sz );
            g_SampleUI.GetSlider( IDC_TIMELINE )->SetValue( (int)(g_Timeline * 10.0) );
            v = g_weightTable[g_curBSGroup*4 + 0];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 0].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT0 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT0 )->SetValue( (int)(v * 1000.0) );

            v = g_weightTable[g_curBSGroup*4 + 1];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 1].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT1 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT1 )->SetValue( (int)(v * 1000.0) );

            v = g_weightTable[g_curBSGroup*4 + 2];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 2].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT2 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT2 )->SetValue( (int)(v * 1000.0) );

            v = g_weightTable[g_curBSGroup*4 + 3];
            strcpy_s(sz2, 100, g_MeshTable[0].pMesh->bsSlots[g_curBSGroup*4 + 3].name);
            sz2[strlen(sz2)-9] = '\0'; // removing 'HeadShape' suffix (exported by Maya...)
            StringCchPrintf( sz, 100, L"%S : %0.2f",sz2, v); 
            g_SampleUI.GetStatic( IDC_WEIGHT3 )->SetText( sz );
            g_SampleUI.GetSlider( IDC_WEIGHT3 )->SetValue( (int)(v * 1000.0) );
        }
        g_bUpdateAnimOnce = false;
    }
    //
    // Now do Meshes having Blendshapes, according to the method to expose
    //
    switch(g_renderMode)
    {
    case RENDER_STREAMOUT:
        OnD3D10FrameRender_StreamOut( pd3dDevice, fTime, fElapsedTime, pUserContext );
        break;
    case RENDER_BUFFERTEMPLATE:
        OnD3D10FrameRender_BufferTemplate( pd3dDevice, fTime, fElapsedTime, pUserContext );
        break;
    }
    //
    // Process all the 'static' meshes. Skip the ones with Blendshapes
    //
    ID3D10Buffer *pNullBuffers[] = { NULL };
    UINT offsets[4] = {0, 0, 0, 0};
    for(int m=0; m<MESHNUM; m++)
    {
        const char *n;
        ID3D10EffectTechnique *tech;
        if(g_MeshTable[m].pMesh->numBlendShapes == 0)
        {
            pd3dDevice->SOSetTargets( 1, pNullBuffers, offsets );
            pd3dDevice->IASetVertexBuffers( 0, g_MeshTable[m].pMesh->numSlots, g_MeshTable[m].pVertexBuffers, g_MeshTable[m].pStrides, offsets );
            pd3dDevice->IASetInputLayout( g_MeshTable[m].pLayout );
            for(int i=0; i<g_MeshTable[m].numIndexBuffers; i++)
            {
                n = g_MeshTable[m].pMesh->primGroup[i].name;
                //
                // take the technique with the same name as primitive group's name
                //
                tech = g_Effect->GetTechniqueByName(n);
                //'tech' returning a technique even when the tech name doesn't exist !
                D3D10_TECHNIQUE_DESC td; td.Name = NULL; 
                tech->GetDesc(&td);
                if((!tech)||(!td.Name)||(strcmp(td.Name, n)))
                {
                    continue;
                }
                tech->GetPassByIndex(0)->Apply(0);
                pd3dDevice->IASetIndexBuffer( g_MeshTable[m].pIndexBuffers[i], g_MeshTable[m].pMesh->primGroup[i].indexFormatDX10, 0 );
                pd3dDevice->IASetPrimitiveTopology( g_MeshTable[m].pMesh->primGroup[i].topologyDX10 );
                pd3dDevice->DrawIndexed(g_MeshTable[m].pMesh->primGroup[i].indexCount, 0, 0);
            }
        }
    }
    //
    // Stats + UI
    //
    Tools::DrawStats();
    g_SampleUI.OnRender( fElapsedTime );
    g_HUD.OnRender( fElapsedTime );
}
//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10ReleasingSwapChain( void* pUserContext )
{
    g_DialogResourceManager.OnD3D10ReleasingSwapChain();

    g_Skybox.OnReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D10DestroyDevice();
    g_D3DSettingsDlg.GetDialogControl()->RemoveAllControls();
    g_D3DSettingsDlg.OnD3D10DestroyDevice();
    g_Skybox.OnDestroyDevice();
    Tools::ReleaseFonts();
    SAFE_RELEASE(g_Effect);
    SAFE_RELEASE(g_EnvMapSRV);
    SAFE_RELEASE(g_EnvMap);
    for(int i=0; i<MESHNUM; i++)
    {
        g_MeshTable[i].destroy();
    }
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, 
                          bool* pbNoFurtherProcessing, void* pUserContext )
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    g_Camera.HandleMessages(hWnd, uMsg, wParam, lParam);
    return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
  if(bKeyDown)
    switch(nChar)
    {
    case VK_OEM_PLUS:
    case VK_ADD:
      break;
    case VK_ESCAPE:
      PostQuitMessage( 0 );
      break;
    case ' ':
      g_Anim = g_Anim ? false : true;
      break;
    }
}


//--------------------------------------------------------------------------------------
// Handle mouse button presses
//--------------------------------------------------------------------------------------
void CALLBACK OnMouse( bool bLeftButtonDown, bool bRightButtonDown, bool bMiddleButtonDown, 
                       bool bSideButton1Down, bool bSideButton2Down, int nMouseWheelDelta, 
                       int xPos, int yPos, void* pUserContext )
{
}


//--------------------------------------------------------------------------------------
// Call if device was removed.  Return true to find a new device, false to quit
//--------------------------------------------------------------------------------------
bool CALLBACK OnDeviceRemoved( void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Initialize everything and go into a render loop
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // DXUT will create and use the best device (either D3D9 or D3D10) 
    // that is available on the system depending on which D3D callbacks are set below

    // Set general DXUT callbacks
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackKeyboard( OnKeyboard );
    DXUTSetCallbackMouse( OnMouse );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackDeviceRemoved( OnDeviceRemoved );

   // Set the D3D10 DXUT callbacks. Remove these sets if the app doesn't need to support D3D10
    DXUTSetCallbackD3D10DeviceAcceptable( IsD3D10DeviceAcceptable );
    DXUTSetCallbackD3D10DeviceCreated( OnD3D10CreateDevice );
    DXUTSetCallbackD3D10SwapChainResized( OnD3D10ResizedSwapChain );
    DXUTSetCallbackD3D10FrameRender( OnD3D10FrameRender );
    DXUTSetCallbackD3D10SwapChainReleasing( OnD3D10ReleasingSwapChain );
    DXUTSetCallbackD3D10DeviceDestroyed( OnD3D10DestroyDevice );

    HRESULT hr;
    V_RETURN(DXUTSetMediaSearchPath(L"..\\Source\\GPUBlendShapes"));
    
    // Perform any application-level initialization here
    
    DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"GPUBlendShapes", 0, 0, 0, 200, 400);
    DXUTCreateDevice( true, 800, 600 );  

    //
    // Camera
    //
    D3DXVECTOR3 vFromPt   = D3DXVECTOR3(-3.5f, 0.5f, 8.0f);
    D3DXVECTOR3 vLookatPt = D3DXVECTOR3(0.0f, 0, 0.0f);
    g_Camera.SetViewParams( &vFromPt, &vLookatPt);
    g_Camera.SetModelCenter(D3DXVECTOR3(0.0f, 6.5f, 0.0f));
    float fAspectRatio = (float)800 / (float)600;
    g_Camera.SetProjParams( 15.0*D3DX_PI/180.0, fAspectRatio, 0.1f, 100.0f );
    g_Camera.SetWindow( 800, 600);

    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}


