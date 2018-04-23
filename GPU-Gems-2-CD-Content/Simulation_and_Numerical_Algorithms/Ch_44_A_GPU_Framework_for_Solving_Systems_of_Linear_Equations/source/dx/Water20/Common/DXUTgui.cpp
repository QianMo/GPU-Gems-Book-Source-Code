//--------------------------------------------------------------------------------------
// File: DXUTgui.cpp
//
// Desc: 
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "dxstdafx.h"

#ifndef WM_XBUTTONDOWN
#define WM_XBUTTONDOWN 0x020B // (not always defined)
#endif
#ifndef WM_XBUTTONUP
#define WM_XBUTTONUP 0x020C // (not always defined)
#endif
#ifndef WM_MOUSEWHEEL
#define WM_MOUSEWHEEL 0x020A // (not always defined)
#endif
#ifndef WHEEL_DELTA
#define WHEEL_DELTA 120 // (not always defined)
#endif

// Minimum scroll bar thumb size
#define SCROLLBAR_MINTHUMBSIZE 8

// Delay and repeat period when clicking on the scroll bar arrows
#define SCROLLBAR_ARROWCLICK_DELAY  0.33
#define SCROLLBAR_ARROWCLICK_REPEAT 0.05


// DXUT_MAX_EDITBOXLENGTH is the maximum string length allowed in edit boxes,
// including the NULL terminator.
// 
// Uniscribe does not support strings having bigger-than-16-bits length.
// This means that the string must be less than 65536 characters long,
// including the NULL terminator.
#define DXUT_MAX_EDITBOXLENGTH 0xFFFF


//--------------------------------------------------------------------------------------
// Global/Static Members
//--------------------------------------------------------------------------------------
CDXUTDialogResourceManager* DXUTGetGlobalDialogResourceManager()
{
    // Using an accessor function gives control of the construction order
    static CDXUTDialogResourceManager manager;
    return &manager;
}

double        CDXUTDialog::s_fTimeRefresh = 0.0f;
CDXUTControl* CDXUTDialog::s_pControlFocus = NULL;        // The control which has focus
CDXUTControl* CDXUTDialog::s_pControlPressed = NULL;      // The control currently pressed


struct DXUT_SCREEN_VERTEX
{
    float x, y, z, h;
    D3DCOLOR color;
    float tu, tv;

    static DWORD FVF;
};
DWORD DXUT_SCREEN_VERTEX::FVF = D3DFVF_XYZRHW | D3DFVF_DIFFUSE | D3DFVF_TEX1;


inline int RectWidth( RECT &rc ) { return ( (rc).right - (rc).left ); }
inline int RectHeight( RECT &rc ) { return ( (rc).bottom - (rc).top ); }


//--------------------------------------------------------------------------------------
// CDXUTDialog class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTDialog::CDXUTDialog()
{
    m_x = 0;
    m_y = 0;
    m_width = 0;
    m_height = 0;

    m_bCaption = false;
    m_bMinimized = false;
    m_wszCaption[0] = L'\0';
    m_nCaptionHeight = 18;

    m_colorTopLeft = 0;
    m_colorTopRight = 0;
    m_colorBottomLeft = 0;
    m_colorBottomRight = 0;

    m_pCallbackEvent = NULL;

    m_fTimeLastRefresh = 0;

    m_pControlMouseOver = NULL;

    m_pNextDialog = this;
    m_pPrevDialog = this;

    m_nDefaultControlID = 0xffff;
    m_bNonUserEvents = false;
    m_bKeyboardInput = false;
    m_bMouseInput = true;

    InitDefaultElements();
}


//--------------------------------------------------------------------------------------
CDXUTDialog::~CDXUTDialog()
{
    int i=0;

    RemoveAllControls();

    m_Fonts.RemoveAll();
    m_Textures.RemoveAll();

    for( i=0; i < m_DefaultElements.GetSize(); i++ )
    {
        DXUTElementHolder* pElementHolder = m_DefaultElements.GetAt( i );
        SAFE_DELETE( pElementHolder );
    }

    m_DefaultElements.RemoveAll();
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::RemoveControl( int ID )
{
    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt( i );
        if( pControl->GetID() == ID )
        {
            // Clean focus first
            ClearFocus();

            // Clear references to this control
            if( s_pControlFocus == pControl )
                s_pControlFocus = NULL;
            if( s_pControlPressed == pControl )
                s_pControlPressed = NULL;
            if( m_pControlMouseOver == pControl )
                m_pControlMouseOver = NULL;

            SAFE_DELETE( pControl );
            m_Controls.Remove( i );

            return;
        }
    }
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::RemoveAllControls()
{
    if( s_pControlFocus && s_pControlFocus->m_pDialog == this )
        s_pControlFocus = NULL;
    if( s_pControlPressed && s_pControlPressed->m_pDialog == this )
        s_pControlPressed = NULL;
    m_pControlMouseOver = NULL;

    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt( i );
        SAFE_DELETE( pControl );
    }

    m_Controls.RemoveAll();
}


//--------------------------------------------------------------------------------------
CDXUTDialogResourceManager::CDXUTDialogResourceManager()
{
    m_pd3dDevice = NULL;
    m_pStateBlock = NULL;
    m_pSprite = NULL;
}


//--------------------------------------------------------------------------------------
CDXUTDialogResourceManager::~CDXUTDialogResourceManager()
{
    int i;
    for( i=0; i < m_FontCache.GetSize(); i++ )
    {
        DXUTFontNode* pFontNode = m_FontCache.GetAt( i );
        SAFE_DELETE( pFontNode );
    }
    m_FontCache.RemoveAll();   

    for( i=0; i < m_TextureCache.GetSize(); i++ )
    {
        DXUTTextureNode* pTextureNode = m_TextureCache.GetAt( i );
        SAFE_DELETE( pTextureNode );
    }
    m_TextureCache.RemoveAll();   
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialogResourceManager::OnCreateDevice( LPDIRECT3DDEVICE9 pd3dDevice )
{
    HRESULT hr = S_OK;
    int i=0;

    m_pd3dDevice = pd3dDevice;
    
    for( i=0; i < m_FontCache.GetSize(); i++ )
    {
        hr = CreateFont( i );
        if( FAILED(hr) )
            return hr;
    }
    
    for( i=0; i < m_TextureCache.GetSize(); i++ )
    {
        hr = CreateTexture( i );
        if( FAILED(hr) )
            return hr;
    }

    hr = D3DXCreateSprite( pd3dDevice, &m_pSprite );
    if( FAILED(hr) )
        return DXUT_ERR( L"D3DXCreateSprite", hr );

    // Call CDXUTIMEEditBox's StaticOnCreateDevice()
    // to initialize certain window-dependent data.
    CDXUTIMEEditBox::StaticOnCreateDevice();

    return S_OK;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialogResourceManager::OnResetDevice()
{
    HRESULT hr = S_OK;

    for( int i=0; i < m_FontCache.GetSize(); i++ )
    {
        DXUTFontNode* pFontNode = m_FontCache.GetAt( i );

        if( pFontNode->pFont )
            pFontNode->pFont->OnResetDevice();
    }

    if( m_pSprite )
        m_pSprite->OnResetDevice();

    IDirect3DDevice9* pd3dDevice = DXUTGetD3DDevice();

    V_RETURN( pd3dDevice->CreateStateBlock( D3DSBT_ALL, &m_pStateBlock ) );

    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTDialogResourceManager::OnLostDevice()
{
    for( int i=0; i < m_FontCache.GetSize(); i++ )
    {
        DXUTFontNode* pFontNode = m_FontCache.GetAt( i );

        if( pFontNode->pFont )
            pFontNode->pFont->OnLostDevice();
    }

    if( m_pSprite )
        m_pSprite->OnLostDevice();

    SAFE_RELEASE( m_pStateBlock  );
}

    
//--------------------------------------------------------------------------------------
void CDXUTDialogResourceManager::OnDestroyDevice()
{
    int i=0; 

    m_pd3dDevice = NULL;

    // Release the resources but don't clear the cache, as these will need to be
    // recreated if the device is recreated
    for( i=0; i < m_FontCache.GetSize(); i++ )
    {
        DXUTFontNode* pFontNode = m_FontCache.GetAt( i );
        SAFE_RELEASE( pFontNode->pFont );
    }
    
    for( i=0; i < m_TextureCache.GetSize(); i++ )
    {
        DXUTTextureNode* pTextureNode = m_TextureCache.GetAt( i );
        SAFE_RELEASE( pTextureNode->pTexture );
    }

    SAFE_RELEASE( m_pSprite );
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::Refresh()
{
    if( s_pControlFocus )
        s_pControlFocus->OnFocusOut();

    if( m_pControlMouseOver )
        m_pControlMouseOver->OnMouseLeave();

    s_pControlFocus = NULL;
    s_pControlPressed = NULL;
    m_pControlMouseOver = NULL;

    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt(i);
        pControl->Refresh();
    }

    if( m_bKeyboardInput )
        FocusDefaultControl();
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::OnRender( float fElapsedTime )
{   
    // See if the dialog needs to be refreshed
    if( m_fTimeLastRefresh < s_fTimeRefresh )
    {
        m_fTimeLastRefresh = DXUTGetTime();
        Refresh();
    }

    DXUT_SCREEN_VERTEX vertices[4] =
    {
        (float)m_x,           (float)m_y,            0.5f, 1.0f, m_colorTopLeft, 0.0f, 0.5f, 
        (float)m_x + m_width, (float)m_y,            0.5f, 1.0f, m_colorTopRight, 1.0f, 0.5f,
        (float)m_x + m_width, (float)m_y + m_height, 0.5f, 1.0f, m_colorBottomRight, 1.0f, 1.0f, 
        (float)m_x,           (float)m_y + m_height, 0.5f, 1.0f, m_colorBottomLeft, 0.0f, 1.0f, 
    };

    IDirect3DDevice9* pd3dDevice = DXUTGetGlobalDialogResourceManager()->GetD3DDevice();     

    // Set up a state block here and restore it when finished drawing all the controls
    DXUTGetGlobalDialogResourceManager()->m_pStateBlock->Capture();

    pd3dDevice->SetRenderState( D3DRS_ALPHABLENDENABLE, TRUE );
    pd3dDevice->SetRenderState( D3DRS_SRCBLEND, D3DBLEND_SRCALPHA );
    pd3dDevice->SetRenderState( D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA );
    pd3dDevice->SetRenderState( D3DRS_ALPHATESTENABLE, FALSE );

    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP, D3DTOP_SELECTARG2 );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );

    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1 );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG1, D3DTA_DIFFUSE );

    pd3dDevice->SetVertexShader( NULL );
    pd3dDevice->SetPixelShader( NULL );

    //pd3dDevice->Clear( 0, NULL, D3DCLEAR_ZBUFFER, 0, 1.0f, 0 );
    pd3dDevice->SetRenderState( D3DRS_ZENABLE, FALSE );

    if( !m_bMinimized )
    {
        pd3dDevice->SetFVF( DXUT_SCREEN_VERTEX::FVF );
        pd3dDevice->DrawPrimitiveUP( D3DPT_TRIANGLEFAN, 2, vertices, sizeof(DXUT_SCREEN_VERTEX) );
    }


    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP, D3DTOP_MODULATE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );
    
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_MODULATE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAARG2, D3DTA_DIFFUSE );

    pd3dDevice->SetSamplerState( 0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR );

    DXUTTextureNode* pTextureNode = GetTexture( 0 );
    pd3dDevice->SetTexture( 0, pTextureNode->pTexture );

    DXUTGetGlobalDialogResourceManager()->m_pSprite->Begin( D3DXSPRITE_DONOTSAVESTATE );

    // Render the caption if it's enabled.
    if( m_bCaption )
    {
        // DrawSprite will offset the rect down by
        // m_nCaptionHeight, so adjust the rect higher
        // here to negate the effect.
        RECT rc = { 0, -m_nCaptionHeight, m_width, 0 };
        DrawSprite( &m_CapElement, &rc );
        rc.left += 5; // Make a left margin
        WCHAR wszOutput[256];
        wcsncpy( wszOutput, m_wszCaption, 256 );
        wszOutput[255] = 0;
        if( m_bMinimized )
            wcsncat( wszOutput, L" (Minimized)", 256 - lstrlenW( wszOutput ) );
        DrawText( wszOutput, &m_CapElement, &rc, true );
    }

    // If the dialog is minimized, skip rendering
    // its controls.
    if( !m_bMinimized )
    {
        for( int i=0; i < m_Controls.GetSize(); i++ )
        {
            CDXUTControl* pControl = m_Controls.GetAt(i);   

            // Focused control is drawn last
            if( pControl == s_pControlFocus )
                continue;

            pControl->Render( pd3dDevice, fElapsedTime );
        }

        if( s_pControlFocus != NULL && s_pControlFocus->m_pDialog == this )
            s_pControlFocus->Render( pd3dDevice, fElapsedTime );
    }

    DXUTGetGlobalDialogResourceManager()->m_pSprite->End();

    DXUTGetGlobalDialogResourceManager()->m_pStateBlock->Apply();

    return S_OK;
}


//--------------------------------------------------------------------------------------
VOID CDXUTDialog::SendEvent( UINT nEvent, bool bTriggeredByUser, CDXUTControl* pControl )
{
    // If no callback has been registered there's nowhere to send the event to
    if( m_pCallbackEvent == NULL )
        return;

    // Discard events triggered programatically if these types of events haven't been
    // enabled
    if( !bTriggeredByUser && !m_bNonUserEvents )
        return;

    m_pCallbackEvent( nEvent, pControl->GetID(), pControl );
}


//--------------------------------------------------------------------------------------
int CDXUTDialogResourceManager::AddFont( LPCWSTR strFaceName, LONG height, LONG weight )
{
    // See if this font already exists
    for( int i=0; i < m_FontCache.GetSize(); i++ )
    {
        DXUTFontNode* pFontNode = m_FontCache.GetAt(i);
        if( 0 == _wcsnicmp( pFontNode->strFace, strFaceName, MAX_PATH-1 ) &&
            pFontNode->nHeight == height &&
            pFontNode->nWeight == weight )
        {
            return i;
        }
    }

    // Add a new font and try to create it
    DXUTFontNode* pNewFontNode = new DXUTFontNode();
    if( pNewFontNode == NULL )
        return -1;

    ZeroMemory( pNewFontNode, sizeof(DXUTFontNode) );
    wcsncpy( pNewFontNode->strFace, strFaceName, MAX_PATH-1 );
    pNewFontNode->nHeight = height;
    pNewFontNode->nWeight = weight;
    m_FontCache.Add( pNewFontNode );
    
    int iFont = m_FontCache.GetSize()-1;

    // If a device is available, try to create immediately
    if( m_pd3dDevice )
        CreateFont( iFont );

    return iFont;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::SetFont( UINT index, LPCWSTR strFaceName, LONG height, LONG weight )
{
    // Make sure the list is at least as large as the index being set
    UINT i;
    for( i=m_Fonts.GetSize(); i <= index; i++ )
    {
        m_Fonts.Add( -1 );
    }

    int iFont = DXUTGetGlobalDialogResourceManager()->AddFont( strFaceName, height, weight );
    m_Fonts.SetAt( index, iFont );

    return S_OK;
}


//--------------------------------------------------------------------------------------
DXUTFontNode* CDXUTDialog::GetFont( UINT index )
{
    if( NULL == DXUTGetGlobalDialogResourceManager() )
        return NULL;
    return DXUTGetGlobalDialogResourceManager()->GetFontNode( m_Fonts.GetAt( index ) );
}


//--------------------------------------------------------------------------------------
int CDXUTDialogResourceManager::AddTexture( LPCWSTR strFilename )
{
    // See if this texture already exists
    for( int i=0; i < m_TextureCache.GetSize(); i++ )
    {
        DXUTTextureNode* pTextureNode = m_TextureCache.GetAt(i);
        if( 0 == _wcsnicmp( pTextureNode->strFilename, strFilename, MAX_PATH-1 ) )
        {
            return i;
        }
    }

    // Add a new texture and try to create it
    DXUTTextureNode* pNewTextureNode = new DXUTTextureNode();
    if( pNewTextureNode == NULL )
        return -1;

    ZeroMemory( pNewTextureNode, sizeof(DXUTTextureNode) );
    wcsncpy( pNewTextureNode->strFilename, strFilename, MAX_PATH-1 );
    m_TextureCache.Add( pNewTextureNode );
    
    int iTexture = m_TextureCache.GetSize()-1;

    // If a device is available, try to create immediately
    if( m_pd3dDevice )
        CreateTexture( iTexture );

    return iTexture;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::SetTexture( UINT index, LPCWSTR strFilename )
{
    // Make sure the list is at least as large as the index being set
    for( UINT i=m_Textures.GetSize(); i <= index; i++ )
    {
        m_Textures.Add( -1 );
    }

    int iTexture = DXUTGetGlobalDialogResourceManager()->AddTexture( strFilename );

    m_Textures.SetAt( index, iTexture );
    return S_OK;
}


//--------------------------------------------------------------------------------------
DXUTTextureNode* CDXUTDialog::GetTexture( UINT index )
{
    if( NULL == DXUTGetGlobalDialogResourceManager() )
        return NULL;
    return DXUTGetGlobalDialogResourceManager()->GetTextureNode( m_Textures.GetAt( index ) );
}



//--------------------------------------------------------------------------------------
bool CDXUTDialog::MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    bool bHandled = false;

    // If caption is enable, check for clicks in the caption area.
    if( m_bCaption )
    {
        static bool bDrag;

        if( uMsg == WM_LBUTTONDOWN || uMsg == WM_LBUTTONDBLCLK )
        {
            POINT mousePoint = { short(LOWORD(lParam)), short(HIWORD(lParam)) };

            if( mousePoint.x >= m_x && mousePoint.x < m_x + m_width &&
                mousePoint.y >= m_y && mousePoint.y < m_y + m_nCaptionHeight )
            {
                bDrag = true;
                SetCapture( DXUTGetHWND() );
                return true;
            }
        } else
        if( uMsg == WM_LBUTTONUP && bDrag )
        {
            POINT mousePoint = { short(LOWORD(lParam)), short(HIWORD(lParam)) };

            if( mousePoint.x >= m_x && mousePoint.x < m_x + m_width &&
                mousePoint.y >= m_y && mousePoint.y < m_y + m_nCaptionHeight )
            {
                ReleaseCapture();
                bDrag = false;
                m_bMinimized = !m_bMinimized;
                return true;
            }
        }
    }

    // If the dialog is minimized, don't send any messages to controls.
    if( m_bMinimized )
        return false;

    // If a control is in focus, it belongs to this dialog, and it's enabled, then give
    // it the first chance at handling the message.
    if( s_pControlFocus && 
        s_pControlFocus->m_pDialog == this && 
        s_pControlFocus->GetEnabled() )
    {
        // If the control MsgProc handles it, then we don't.
        if( s_pControlFocus->MsgProc( uMsg, wParam, lParam ) )
            return true;
    }

    switch( uMsg )
    {
        case WM_ACTIVATEAPP:
            // Call OnFocusIn()/OnFocusOut() of the control that currently has the focus
            // as the application is activated/deactivated.  This matches the Windows
            // behavior.
            if( s_pControlFocus && 
                s_pControlFocus->m_pDialog == this && 
                s_pControlFocus->GetEnabled() )
            {
                if( wParam )
                    s_pControlFocus->OnFocusIn();
                else
                    s_pControlFocus->OnFocusOut();
            }
            break;

        // Keyboard messages
        case WM_KEYDOWN:
        case WM_SYSKEYDOWN:
        case WM_KEYUP:
        case WM_SYSKEYUP:
        {
            // If a control is in focus, it belongs to this dialog, and it's enabled, then give
            // it the first chance at handling the message.
            if( s_pControlFocus && 
                s_pControlFocus->m_pDialog == this && 
                s_pControlFocus->GetEnabled() )
            {
                if( s_pControlFocus->HandleKeyboard( uMsg, wParam, lParam ) )
                    return true;
            }

            // Not yet handled, see if this matches a control's hotkey
            // Activate the hotkey if the focus doesn't belong to an
            // edit box.
            if( uMsg == WM_KEYUP && ( !s_pControlFocus ||
                                      ( s_pControlFocus->GetType() != DXUT_CONTROL_EDITBOX
                                     && s_pControlFocus->GetType() != DXUT_CONTROL_IMEEDITBOX ) ) )
            {
                for( int i=0; i < m_Controls.GetSize(); i++ )
                {
                    CDXUTControl* pControl = m_Controls.GetAt( i );
                    if( pControl->GetHotkey() == wParam )
                    {
                        pControl->OnHotkey();
                        return true;
                    }
                }
            }

            // Not yet handled, check for focus messages
            if( uMsg == WM_KEYDOWN )
            {
                // If keyboard input is not enabled, this message should be ignored
                if( !m_bKeyboardInput )
                    return false;

                switch( wParam )
                {
                    case VK_RIGHT:
                    case VK_DOWN:
                        if( s_pControlFocus != NULL )
                        {
                            OnCycleFocus( true );
                            return true;
                        }
                        break;

                    case VK_LEFT:
                    case VK_UP:
                        if( s_pControlFocus != NULL )
                        {
                            OnCycleFocus( false );
                            return true;
                        }
                        break;

                    case VK_TAB: 
                        if( s_pControlFocus == NULL )
                        {
                            FocusDefaultControl();
                        }
                        else
                        {
                            bool bShiftDown = ((GetAsyncKeyState( VK_SHIFT ) & 0x8000) != 0);
                            OnCycleFocus( !bShiftDown );
                        }
                        return true;
                }
            }

            break;
        }


        // Mouse messages
        case WM_MOUSEMOVE:
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
        case WM_XBUTTONDOWN:
        case WM_XBUTTONUP:
        case WM_LBUTTONDBLCLK:
        case WM_MBUTTONDBLCLK:
        case WM_RBUTTONDBLCLK:
        case WM_XBUTTONDBLCLK:
        case WM_MOUSEWHEEL:
        {
            // If not accepting mouse input, return false to indicate the message should still 
            // be handled by the application (usually to move the camera).
            if( !m_bMouseInput )
                return false;

            POINT mousePoint = { short(LOWORD(lParam)), short(HIWORD(lParam)) };
            mousePoint.x -= m_x;
            mousePoint.y -= m_y;

            // If caption is enabled, offset the Y coordinate by the negative of its height.
            if( m_bCaption )
                mousePoint.y -= m_nCaptionHeight;

            // If a control is in focus, it belongs to this dialog, and it's enabled, then give
            // it the first chance at handling the message.
            if( s_pControlFocus && 
                s_pControlFocus->m_pDialog == this && 
                s_pControlFocus->GetEnabled() )
            {
                if( s_pControlFocus->HandleMouse( uMsg, mousePoint, wParam, lParam ) )
                    return true;
            }

            // Not yet handled, see if the mouse is over any controls
            CDXUTControl* pControl = GetControlAtPoint( mousePoint );
            if( pControl != NULL && pControl->GetEnabled() )
            {
                bHandled = pControl->HandleMouse( uMsg, mousePoint, wParam, lParam );
                if( bHandled )
                    return true;
            }
            else
            {
                // Mouse not over any controls in this dialog, if there was a control
                // which had focus it just lost it
                if( uMsg == WM_LBUTTONDOWN && 
                    s_pControlFocus && 
                    s_pControlFocus->m_pDialog == this )
                {
                    s_pControlFocus->OnFocusOut();
                    s_pControlFocus = NULL;
                }
            }

            // Still not handled, hand this off to the dialog. Return false to indicate the
            // message should still be handled by the application (usually to move the camera).
            switch( uMsg )
            {
                case WM_MOUSEMOVE:
                    OnMouseMove( mousePoint );
                    return false;
            }

            break;
        }
    }

    return false;
}

//--------------------------------------------------------------------------------------
CDXUTControl* CDXUTDialog::GetControlAtPoint( POINT pt )
{
    // Search through all child controls for the first one which
    // contains the mouse point
    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt(i);

        if( pControl == NULL )
        {
            continue;
        }

        // We only return the current control if it is visible
        // and enabled.  Because GetControlAtPoint() is used to do mouse
        // hittest, it makes sense to perform this filtering.
        if( pControl->ContainsPoint( pt ) && pControl->GetEnabled() && pControl->GetVisible() )
        {
            return pControl;
        }
    }

    return NULL;
}


//--------------------------------------------------------------------------------------
bool CDXUTDialog::GetControlEnabled( int ID )
{
    CDXUTControl* pControl = GetControl( ID );
    if( pControl == NULL )
        return false;

    return pControl->GetEnabled();
}



//--------------------------------------------------------------------------------------
void CDXUTDialog::SetControlEnabled( int ID, bool bEnabled )
{
    CDXUTControl* pControl = GetControl( ID );
    if( pControl == NULL )
        return;

    pControl->SetEnabled( bEnabled );
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::OnMouseUp( POINT pt )
{
    s_pControlPressed = NULL;
    m_pControlMouseOver = NULL;
}

//--------------------------------------------------------------------------------------
void CDXUTDialog::OnMouseMove( POINT pt )
{
    // Figure out which control the mouse is over now
    CDXUTControl* pControl = GetControlAtPoint( pt );

    // If the mouse is still over the same control, nothing needs to be done
    if( pControl == m_pControlMouseOver )
        return;

    // Handle mouse leaving the old control
    if( m_pControlMouseOver )
        m_pControlMouseOver->OnMouseLeave();

    // Handle mouse entering the new control
    m_pControlMouseOver = pControl;
    if( pControl != NULL )
        m_pControlMouseOver->OnMouseEnter();
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::SetDefaultElement( UINT nControlType, UINT iElement, CDXUTElement* pElement )
{
    // If this Element type already exist in the list, simply update the stored Element
    for( int i=0; i < m_DefaultElements.GetSize(); i++ )
    {
        DXUTElementHolder* pElementHolder = m_DefaultElements.GetAt( i );
        
        if( pElementHolder->nControlType == nControlType &&
            pElementHolder->iElement == iElement )
        {
            pElementHolder->Element = *pElement;
            return S_OK;
        }
    }

    // Otherwise, add a new entry
    DXUTElementHolder* pNewHolder;
    pNewHolder = new DXUTElementHolder;
    if( pNewHolder == NULL )
        return E_OUTOFMEMORY;

    pNewHolder->nControlType = nControlType;
    pNewHolder->iElement = iElement;
    pNewHolder->Element = *pElement;

    m_DefaultElements.Add( pNewHolder );
    return S_OK;
}


//--------------------------------------------------------------------------------------
CDXUTElement* CDXUTDialog::GetDefaultElement( UINT nControlType, UINT iElement )
{
    for( int i=0; i < m_DefaultElements.GetSize(); i++ )
    {
        DXUTElementHolder* pElementHolder = m_DefaultElements.GetAt( i );
        
        if( pElementHolder->nControlType == nControlType &&
            pElementHolder->iElement == iElement )
        {
            return &pElementHolder->Element;
        }
    }
    
    return NULL;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddStatic( int ID, LPCWSTR strText, int x, int y, int width, int height, bool bIsDefault, CDXUTStatic** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTStatic* pStatic = new CDXUTStatic( this );

    if( ppCreated != NULL )
        *ppCreated = pStatic;

    if( pStatic == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pStatic );
    if( FAILED(hr) )
        return hr;

    // Set the ID and list index
    pStatic->SetID( ID ); 
    pStatic->SetText( strText );
    pStatic->SetLocation( x, y );
    pStatic->SetSize( width, height );
    pStatic->m_bIsDefault = bIsDefault;

    return S_OK;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddButton( int ID, LPCWSTR strText, int x, int y, int width, int height, UINT nHotkey, bool bIsDefault, CDXUTButton** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTButton* pButton = new CDXUTButton( this );

    if( ppCreated != NULL )
        *ppCreated = pButton;

    if( pButton == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pButton );
    if( FAILED(hr) )
        return hr;

    // Set the ID and list index
    pButton->SetID( ID ); 
    pButton->SetText( strText );
    pButton->SetLocation( x, y );
    pButton->SetSize( width, height );
    pButton->SetHotkey( nHotkey );
    pButton->m_bIsDefault = bIsDefault;

    return S_OK;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddCheckBox( int ID, LPCWSTR strText, int x, int y, int width, int height, bool bChecked, UINT nHotkey, bool bIsDefault, CDXUTCheckBox** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTCheckBox* pCheckBox = new CDXUTCheckBox( this );

    if( ppCreated != NULL )
        *ppCreated = pCheckBox;

    if( pCheckBox == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pCheckBox );
    if( FAILED(hr) )
        return hr;

    // Set the ID and list index
    pCheckBox->SetID( ID ); 
    pCheckBox->SetText( strText );
    pCheckBox->SetLocation( x, y );
    pCheckBox->SetSize( width, height );
    pCheckBox->SetHotkey( nHotkey );
    pCheckBox->m_bIsDefault = bIsDefault;
    pCheckBox->SetChecked( bChecked );
    
    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddRadioButton( int ID, UINT nButtonGroup, LPCWSTR strText, int x, int y, int width, int height, bool bChecked, UINT nHotkey, bool bIsDefault, CDXUTRadioButton** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTRadioButton* pRadioButton = new CDXUTRadioButton( this );

    if( ppCreated != NULL )
        *ppCreated = pRadioButton;

    if( pRadioButton == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pRadioButton );
    if( FAILED(hr) )
        return hr;

    // Set the ID and list index
    pRadioButton->SetID( ID ); 
    pRadioButton->SetText( strText );
    pRadioButton->SetButtonGroup( nButtonGroup );
    pRadioButton->SetLocation( x, y );
    pRadioButton->SetSize( width, height );
    pRadioButton->SetHotkey( nHotkey );
    pRadioButton->SetChecked( bChecked );
    pRadioButton->m_bIsDefault = bIsDefault;
    pRadioButton->SetChecked( bChecked );

    return S_OK;
}




//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddComboBox( int ID, int x, int y, int width, int height, UINT nHotkey, bool bIsDefault, CDXUTComboBox** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTComboBox* pComboBox = new CDXUTComboBox( this );

    if( ppCreated != NULL )
        *ppCreated = pComboBox;

    if( pComboBox == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pComboBox );
    if( FAILED(hr) )
        return hr;

    // Set the ID and list index
    pComboBox->SetID( ID ); 
    pComboBox->SetLocation( x, y );
    pComboBox->SetSize( width, height );
    pComboBox->SetHotkey( nHotkey );
    pComboBox->m_bIsDefault = bIsDefault;

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddSlider( int ID, int x, int y, int width, int height, int min, int max, int value, bool bIsDefault, CDXUTSlider** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTSlider* pSlider = new CDXUTSlider( this );

    if( ppCreated != NULL )
        *ppCreated = pSlider;

    if( pSlider == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pSlider );
    if( FAILED(hr) )
        return hr;

    // Set the ID and list index
    pSlider->SetID( ID ); 
    pSlider->SetLocation( x, y );
    pSlider->SetSize( width, height );
    pSlider->m_bIsDefault = bIsDefault;
    pSlider->SetRange( min, max );
    pSlider->SetValue( value );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddEditBox( int ID, LPCWSTR strText, int x, int y, int width, int height, bool bIsDefault, CDXUTEditBox** ppCreated )
{
    HRESULT hr = S_OK;

    CDXUTEditBox *pEditBox = new CDXUTEditBox( this );

    if( ppCreated != NULL )
        *ppCreated = pEditBox;

    if( pEditBox == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pEditBox );
    if( FAILED(hr) )
        return hr;

    // Set the ID and position
    pEditBox->SetID( ID ); 
    pEditBox->SetLocation( x, y );
    pEditBox->SetSize( width, height );
    pEditBox->m_bIsDefault = bIsDefault;

    if( strText )
        pEditBox->SetText( strText );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddIMEEditBox( int ID, LPCWSTR strText, int x, int y, int width, int height, bool bIsDefault, CDXUTIMEEditBox** ppCreated )
{
    HRESULT hr = S_OK;
    CDXUTIMEEditBox *pEditBox = new CDXUTIMEEditBox( this );

    if( ppCreated != NULL )
        *ppCreated = pEditBox;

    if( pEditBox == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pEditBox );
    if( FAILED(hr) )
        return hr;

    // Set the ID and position
    pEditBox->SetID( ID ); 
    pEditBox->SetLocation( x, y );
    pEditBox->SetSize( width, height );
    pEditBox->m_bIsDefault = bIsDefault;

    if( strText )
        pEditBox->SetText( strText );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddListBox( int ID, int x, int y, int width, int height, DWORD dwStyle, CDXUTListBox** ppCreated )
{
    HRESULT hr = S_OK;
    CDXUTListBox *pListBox = new CDXUTListBox( this );

    if( ppCreated != NULL )
        *ppCreated = pListBox;

    if( pListBox == NULL )
        return E_OUTOFMEMORY;

    hr = AddControl( pListBox );
    if( FAILED(hr) )
        return hr;

    // Set the ID and position
    pListBox->SetID( ID );
    pListBox->SetLocation( x, y );
    pListBox->SetSize( width, height );
    pListBox->SetStyle( dwStyle );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::InitControl( CDXUTControl* pControl )
{
    HRESULT hr;

    if( pControl == NULL )
        return E_INVALIDARG;

    pControl->m_Index = m_Controls.GetSize();
    
    // Look for a default Element entries
    for( int i=0; i < m_DefaultElements.GetSize(); i++ )
    {
        DXUTElementHolder* pElementHolder = m_DefaultElements.GetAt( i );
        if( pElementHolder->nControlType == pControl->GetType() )
            pControl->SetElement( pElementHolder->iElement, &pElementHolder->Element );
    }

    V_RETURN( pControl->OnInit() );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::AddControl( CDXUTControl* pControl )
{
    HRESULT hr = S_OK;

    hr = InitControl( pControl );
    if( FAILED(hr) )
        return DXTRACE_ERR( L"CDXUTDialog::InitControl", hr );

    // Add to the list
    hr = m_Controls.Add( pControl );
    if( FAILED(hr) )
    {
        return DXTRACE_ERR( L"CGrowableArray::Add", hr );
    }

    return S_OK;
}


//--------------------------------------------------------------------------------------
CDXUTControl* CDXUTDialog::GetControl( int ID )
{
    // Try to find the control with the given ID
    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt( i );

        if( pControl->GetID() == ID )
        {
            return pControl;
        }
    }

    // Not found
    return NULL;
}



//--------------------------------------------------------------------------------------
CDXUTControl* CDXUTDialog::GetControl( int ID, UINT nControlType )
{
    // Try to find the control with the given ID
    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt( i );

        if( pControl->GetID() == ID && pControl->GetType() == nControlType )
        {
            return pControl;
        }
    }

    // Not found
    return NULL;
}



//--------------------------------------------------------------------------------------
CDXUTControl* CDXUTDialog::GetNextControl( CDXUTControl* pControl )
{
    int index = pControl->m_Index + 1;

    CDXUTDialog* pDialog = pControl->m_pDialog;
    
    // Cycle through dialogs in the loop to find the next control. Note
    // that if only one control exists in all looped dialogs it will
    // be the returned 'next' control.
    while( index >= (int) pDialog->m_Controls.GetSize() )
    {
        pDialog = pDialog->m_pNextDialog;
        index = 0;
    }
    
    return pDialog->m_Controls.GetAt( index );    
}

//--------------------------------------------------------------------------------------
CDXUTControl* CDXUTDialog::GetPrevControl( CDXUTControl* pControl )
{
    int index = pControl->m_Index - 1;

    CDXUTDialog* pDialog = pControl->m_pDialog;
    
    // Cycle through dialogs in the loop to find the next control. Note
    // that if only one control exists in all looped dialogs it will
    // be the returned 'previous' control.
    while( index < 0 )
    {
        pDialog = pDialog->m_pPrevDialog;
        if( pDialog == NULL )
            pDialog = pControl->m_pDialog;

        index = pDialog->m_Controls.GetSize() - 1;
    }
    
    return pDialog->m_Controls.GetAt( index );    
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::ClearRadioButtonGroup( UINT nButtonGroup )
{
    // Find all radio buttons with the given group number
    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt( i );

        if( pControl->GetType() == DXUT_CONTROL_RADIOBUTTON )
        {
            CDXUTRadioButton* pRadioButton = (CDXUTRadioButton*) pControl;

            if( pRadioButton->GetButtonGroup() == nButtonGroup )
                pRadioButton->SetChecked( false, false );
        }
    }
}



//--------------------------------------------------------------------------------------
void CDXUTDialog::ClearComboBox( int ID )
{
    CDXUTComboBox* pComboBox = GetComboBox( ID );
    if( pComboBox == NULL )
        return;

    pComboBox->RemoveAllItems();
}




//--------------------------------------------------------------------------------------
void CDXUTDialog::RequestFocus( CDXUTControl* pControl )
{
    if( s_pControlFocus == pControl )
        return;

    if( !pControl->CanHaveFocus() )
        return;

    if( s_pControlFocus )
        s_pControlFocus->OnFocusOut();

    pControl->OnFocusIn();
    s_pControlFocus = pControl;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::DrawRect( RECT* pRect, D3DCOLOR color )
{
    RECT rcScreen = *pRect;
    OffsetRect( &rcScreen, m_x, m_y );

    // If caption is enabled, offset the Y position by its height.
    if( m_bCaption )
        OffsetRect( &rcScreen, 0, m_nCaptionHeight );

    DXUT_SCREEN_VERTEX vertices[4] =
    {
        (float) rcScreen.left -0.5f,  (float) rcScreen.top -0.5f,    0.5f, 1.0f, color, 0, 0,
        (float) rcScreen.right -0.5f, (float) rcScreen.top -0.5f,    0.5f, 1.0f, color, 0, 0, 
        (float) rcScreen.right -0.5f, (float) rcScreen.bottom -0.5f, 0.5f, 1.0f, color, 0, 0, 
        (float) rcScreen.left -0.5f,  (float) rcScreen.bottom -0.5f, 0.5f, 1.0f, color, 0, 0,
    };

    IDirect3DDevice9* pd3dDevice = DXUTGetGlobalDialogResourceManager()->GetD3DDevice();

    // Since we're doing our own drawing here we need to flush the sprites
    DXUTGetGlobalDialogResourceManager()->m_pSprite->Flush();
    IDirect3DVertexDeclaration9 *pDecl = NULL;
    pd3dDevice->GetVertexDeclaration( &pDecl );  // Preserve the sprite's current vertex decl
    pd3dDevice->SetFVF( DXUT_SCREEN_VERTEX::FVF );

    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP, D3DTOP_SELECTARG2 );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG2 );

    pd3dDevice->DrawPrimitiveUP( D3DPT_TRIANGLEFAN, 2, vertices, sizeof(DXUT_SCREEN_VERTEX) );

    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP, D3DTOP_MODULATE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_MODULATE );

    // Restore the vertex decl
    pd3dDevice->SetVertexDeclaration( pDecl );
    pDecl->Release();

    return S_OK;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::DrawPolyLine( POINT* apPoints, UINT nNumPoints, D3DCOLOR color )
{
    DXUT_SCREEN_VERTEX* vertices = new DXUT_SCREEN_VERTEX[ nNumPoints ];
    if( vertices == NULL )
        return E_OUTOFMEMORY;

    DXUT_SCREEN_VERTEX* pVertex = vertices;
    POINT* pt = apPoints;
    for( UINT i=0; i < nNumPoints; i++ )
    {
        pVertex->x = m_x + (float) pt->x;
        pVertex->y = m_y + (float) pt->y;
        pVertex->z = 0.5f;
        pVertex->h = 1.0f;
        pVertex->color = color;
        pVertex->tu = 0.0f;
        pVertex->tv = 0.0f;

        pVertex++;
        pt++;
    }

    IDirect3DDevice9* pd3dDevice = DXUTGetGlobalDialogResourceManager()->GetD3DDevice();

    // Since we're doing our own drawing here we need to flush the sprites
    DXUTGetGlobalDialogResourceManager()->m_pSprite->Flush();
    IDirect3DVertexDeclaration9 *pDecl = NULL;
    pd3dDevice->GetVertexDeclaration( &pDecl );  // Preserve the sprite's current vertex decl
    pd3dDevice->SetFVF( DXUT_SCREEN_VERTEX::FVF );

    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP, D3DTOP_SELECTARG2 );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG2 );

    pd3dDevice->DrawPrimitiveUP( D3DPT_LINESTRIP, nNumPoints - 1, vertices, sizeof(DXUT_SCREEN_VERTEX) );

    pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP, D3DTOP_MODULATE );
    pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP, D3DTOP_MODULATE );

    // Restore the vertex decl
    pd3dDevice->SetVertexDeclaration( pDecl );
    pDecl->Release();

    SAFE_DELETE_ARRAY( vertices );
    return S_OK;
}
 


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::DrawSprite( CDXUTElement* pElement, RECT* prcDest )
{
    // No need to draw fully transparent layers
    if( pElement->TextureColor.Current.a == 0 )
        return S_OK;

    RECT rcTexture = pElement->rcTexture;
    
    RECT rcScreen = *prcDest;
    OffsetRect( &rcScreen, m_x, m_y );

    // If caption is enabled, offset the Y position by its height.
    if( m_bCaption )
        OffsetRect( &rcScreen, 0, m_nCaptionHeight );

    DXUTTextureNode* pTextureNode = GetTexture( pElement->iTexture );
    
    float fScaleX = (float) RectWidth( rcScreen ) / RectWidth( rcTexture );
    float fScaleY = (float) RectHeight( rcScreen ) / RectHeight( rcTexture );

    D3DXMATRIXA16 matTransform;
    D3DXMatrixScaling( &matTransform, fScaleX, fScaleY, 1.0f );

    DXUTGetGlobalDialogResourceManager()->m_pSprite->SetTransform( &matTransform );
    
    D3DXVECTOR3 vPos( (float)rcScreen.left, (float)rcScreen.top, 0.0f );

    vPos.x /= fScaleX;
    vPos.y /= fScaleY;

    return DXUTGetGlobalDialogResourceManager()->m_pSprite->Draw( pTextureNode->pTexture, &rcTexture, NULL, &vPos, pElement->TextureColor.Current );
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::CalcTextRect( LPCWSTR strText, CDXUTElement* pElement, RECT* prcDest, int nCount )
{
    HRESULT hr = S_OK;

    DXUTFontNode* pFontNode = GetFont( pElement->iFont );
    DWORD dwTextFormat = pElement->dwTextFormat | DT_CALCRECT;
    // Since we are only computing the rectangle, we don't need a sprite.
    hr = pFontNode->pFont->DrawText( NULL, strText, nCount, prcDest, dwTextFormat, pElement->FontColor.Current );
    if( FAILED(hr) )
        return hr;

    return S_OK;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialog::DrawText( LPCWSTR strText, CDXUTElement* pElement, RECT* prcDest, bool bShadow, int nCount )
{
    HRESULT hr = S_OK;

    // No need to draw fully transparent layers
    if( pElement->FontColor.Current.a == 0 )
        return S_OK;

    RECT rcScreen = *prcDest;
    OffsetRect( &rcScreen, m_x, m_y );

    // If caption is enabled, offset the Y position by its height.
    if( m_bCaption )
        OffsetRect( &rcScreen, 0, m_nCaptionHeight );

    //debug
    //DrawRect( &rcScreen, D3DCOLOR_ARGB(100, 255, 0, 0) );

    D3DXMATRIXA16 matTransform;
    D3DXMatrixIdentity( &matTransform );
    DXUTGetGlobalDialogResourceManager()->m_pSprite->SetTransform( &matTransform );

    DXUTFontNode* pFontNode = GetFont( pElement->iFont );
    
    if( bShadow )
    {
        RECT rcShadow = rcScreen;
        OffsetRect( &rcShadow, 1, 1 );
        hr = pFontNode->pFont->DrawText( DXUTGetGlobalDialogResourceManager()->m_pSprite, strText, nCount, &rcShadow, pElement->dwTextFormat, D3DCOLOR_ARGB(DWORD(pElement->FontColor.Current.a * 255), 0, 0, 0) );
        if( FAILED(hr) )
            return hr;
    }

    hr = pFontNode->pFont->DrawText( DXUTGetGlobalDialogResourceManager()->m_pSprite, strText, nCount, &rcScreen, pElement->dwTextFormat, pElement->FontColor.Current );
    if( FAILED(hr) )
        return hr;

    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::SetBackgroundColors( D3DCOLOR colorTopLeft, D3DCOLOR colorTopRight, D3DCOLOR colorBottomLeft, D3DCOLOR colorBottomRight )
{
    m_colorTopLeft = colorTopLeft;
    m_colorTopRight = colorTopRight;
    m_colorBottomLeft = colorBottomLeft;
    m_colorBottomRight = colorBottomRight;
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::SetNextDialog( CDXUTDialog* pNextDialog )
{ 
    if( pNextDialog == NULL )
        pNextDialog = this;
    
    m_pNextDialog = pNextDialog;
    m_pNextDialog->m_pPrevDialog = this;
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::ClearFocus()
{
    if( s_pControlFocus )
    {
        s_pControlFocus->OnFocusOut();
        s_pControlFocus = NULL;
    }
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::FocusDefaultControl()
{
    // Check for default control in this dialog
    for( int i=0; i < m_Controls.GetSize(); i++ )
    {
        CDXUTControl* pControl = m_Controls.GetAt( i );
        if( pControl->m_bIsDefault )
        {
            // Remove focus from the current control
            ClearFocus();

            // Give focus to the default control
            s_pControlFocus = pControl;
            s_pControlFocus->OnFocusIn();
            return;
        }
    }
}



//--------------------------------------------------------------------------------------
void CDXUTDialog::OnCycleFocus( bool bForward )
{
    // This should only be handled by the dialog which owns the focused control, and 
    // only if a control currently has focus
    if( s_pControlFocus == NULL || s_pControlFocus->m_pDialog != this )
        return;

    CDXUTControl* pControl = s_pControlFocus;
    for( int i=0; i < 0xffff; i++ )
    {
        pControl = (bForward) ? GetNextControl( pControl ) : GetPrevControl( pControl );
        
        // If we've gone in a full circle then focus doesn't change
        if( pControl == s_pControlFocus )
            return;

        // If the dialog accepts keybord input and the control can have focus then
        // move focus
        if( pControl->m_pDialog->m_bKeyboardInput && pControl->CanHaveFocus() )
        {
            s_pControlFocus->OnFocusOut();
            s_pControlFocus = pControl;
            s_pControlFocus->OnFocusIn();
            return;
        }
    }

    // If we reached this point, the chain of dialogs didn't form a complete loop
    DXTRACE_ERR( L"CDXUTDialog: Multiple dialogs are improperly chained together", E_FAIL );
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::OnMouseEnter( CDXUTControl* pControl )
{
    if( pControl == NULL )
        return;

    //pControl->m_bMouseOver = true;
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::OnMouseLeave( CDXUTControl* pControl )
{
    if( pControl == NULL )
        return;

    //pControl->m_bMouseOver = false;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTDialogResourceManager::CreateFont( UINT iFont )
{
    HRESULT hr = S_OK;

    DXUTFontNode* pFontNode = m_FontCache.GetAt( iFont );

    SAFE_RELEASE( pFontNode->pFont );
    
    V_RETURN( D3DXCreateFont( m_pd3dDevice, pFontNode->nHeight, 0, pFontNode->nWeight, 1, FALSE, DEFAULT_CHARSET, 
                              OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                              pFontNode->strFace, &pFontNode->pFont ) );

    return S_OK;
}

//--------------------------------------------------------------------------------------
HRESULT CDXUTDialogResourceManager::CreateTexture( UINT iTexture )
{
    HRESULT hr = S_OK;

    DXUTTextureNode* pTextureNode = m_TextureCache.GetAt( iTexture );
    
    // Make sure there's a texture to create
    if( pTextureNode->strFilename[0] == 0 )
        return S_OK;
    
    // Find the texture on the hard drive
    WCHAR strPath[MAX_PATH];
    hr = DXUTFindDXSDKMediaFileCch( strPath, MAX_PATH, pTextureNode->strFilename );
    if( FAILED(hr) )
    {
        return DXTRACE_ERR( L"DXUTFindDXSDKMediaFileCch", hr );
    }

    // Create texture
    D3DXIMAGE_INFO info;
    hr =  D3DXCreateTextureFromFileEx( m_pd3dDevice, strPath, D3DX_DEFAULT, D3DX_DEFAULT, 
                                       D3DX_DEFAULT, 0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, 
                                       D3DX_DEFAULT, D3DX_DEFAULT, 0, 
                                       &info, NULL, &pTextureNode->pTexture );
    if( FAILED(hr) )
    {
        return DXTRACE_ERR( L"D3DXCreateTextureFromFileEx", hr );
    }

    // Store dimensions
    pTextureNode->dwWidth = info.Width;
    pTextureNode->dwHeight = info.Height;

    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTDialog::InitDefaultElements()
{
    SetTexture( 0, L"UI\\DXUTControls.dds" );
    SetFont( 0, L"Arial", 14, FW_NORMAL );
    
    CDXUTElement Element;
    RECT rcTexture;

    //-------------------------------------
    // Element for the caption
    //-------------------------------------
    m_CapElement.SetFont( 0 );
    SetRect( &rcTexture, 17, 269, 241, 287 );
    m_CapElement.SetTexture( 0, &rcTexture );
    m_CapElement.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(255, 255, 255, 255);
    m_CapElement.FontColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(255, 255, 255, 255);
    m_CapElement.SetFont( 0, D3DCOLOR_ARGB(255, 255, 255, 255), DT_LEFT | DT_VCENTER );
    // Pre-blend as we don't need to transition the state
    m_CapElement.TextureColor.Blend( DXUT_STATE_NORMAL, 10.0f );
    m_CapElement.FontColor.Blend( DXUT_STATE_NORMAL, 10.0f );

    //-------------------------------------
    // CDXUTStatic
    //-------------------------------------
    Element.SetFont( 0 );
    Element.FontColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB( 200, 200, 200, 200 );

    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_STATIC, 0, &Element );
    

    //-------------------------------------
    // CDXUTButton - Button
    //-------------------------------------
    SetRect( &rcTexture, 0, 0, 136, 54 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0 );
    Element.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(150, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_PRESSED ] = D3DCOLOR_ARGB(200, 255, 255, 255);
    Element.FontColor.States[ DXUT_STATE_MOUSEOVER ] = D3DCOLOR_ARGB(255, 0, 0, 0);
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_BUTTON, 0, &Element );
    

    //-------------------------------------
    // CDXUTButton - Fill layer
    //-------------------------------------
    SetRect( &rcTexture, 136, 0, 272, 54 );
    Element.SetTexture( 0, &rcTexture, D3DCOLOR_ARGB(0, 255, 255, 255) );
    Element.TextureColor.States[ DXUT_STATE_MOUSEOVER ] = D3DCOLOR_ARGB(160, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_PRESSED ] = D3DCOLOR_ARGB(60, 0, 0, 0);
    Element.TextureColor.States[ DXUT_STATE_FOCUS ] = D3DCOLOR_ARGB(30, 255, 255, 255);
    
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_BUTTON, 1, &Element );


    //-------------------------------------
    // CDXUTCheckBox - Box
    //-------------------------------------
    SetRect( &rcTexture, 0, 54, 27, 81 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB(255, 255, 255, 255), DT_LEFT | DT_VCENTER );
    Element.FontColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB( 200, 200, 200, 200 );
    Element.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(150, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_FOCUS ] = D3DCOLOR_ARGB(200, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_PRESSED ] = D3DCOLOR_ARGB(255, 255, 255, 255);
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_CHECKBOX, 0, &Element );


    //-------------------------------------
    // CDXUTCheckBox - Check
    //-------------------------------------
    SetRect( &rcTexture, 27, 54, 54, 81 );
    Element.SetTexture( 0, &rcTexture );
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_CHECKBOX, 1, &Element );


    //-------------------------------------
    // CDXUTRadioButton - Box
    //-------------------------------------
    SetRect( &rcTexture, 54, 54, 81, 81 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB(255, 255, 255, 255), DT_LEFT | DT_VCENTER );
    Element.FontColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB( 200, 200, 200, 200 );
    Element.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(150, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_FOCUS ] = D3DCOLOR_ARGB(200, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_PRESSED ] = D3DCOLOR_ARGB(255, 255, 255, 255);
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_RADIOBUTTON, 0, &Element );


    //-------------------------------------
    // CDXUTRadioButton - Check
    //-------------------------------------
    SetRect( &rcTexture, 81, 54, 108, 81 );
    Element.SetTexture( 0, &rcTexture );
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_RADIOBUTTON, 1, &Element );


    //-------------------------------------
    // CDXUTComboBox - Main
    //-------------------------------------
    SetRect( &rcTexture, 7, 81, 247, 123 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0 );
    Element.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(150, 200, 200, 200);
    Element.TextureColor.States[ DXUT_STATE_FOCUS ] = D3DCOLOR_ARGB(170, 230, 230, 230);
    Element.TextureColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(70, 200, 200, 200);
    Element.FontColor.States[ DXUT_STATE_MOUSEOVER ] = D3DCOLOR_ARGB(255, 0, 0, 0);
    Element.FontColor.States[ DXUT_STATE_PRESSED ] = D3DCOLOR_ARGB(255, 0, 0, 0);
    Element.FontColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(200, 200, 200, 200);
    
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_COMBOBOX, 0, &Element );


    //-------------------------------------
    // CDXUTComboBox - Button
    //-------------------------------------
    SetRect( &rcTexture, 272, 0, 325, 49 );
    Element.SetTexture( 0, &rcTexture );
    Element.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(150, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_PRESSED ] = D3DCOLOR_ARGB(255, 150, 150, 150);
    Element.TextureColor.States[ DXUT_STATE_FOCUS ] = D3DCOLOR_ARGB(200, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(70, 255, 255, 255);
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_COMBOBOX, 1, &Element );


    //-------------------------------------
    // CDXUTComboBox - Dropdown
    //-------------------------------------
    SetRect( &rcTexture, 7, 123, 241, 265 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB(255, 0, 0, 0), DT_LEFT | DT_TOP );
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_COMBOBOX, 2, &Element );


    //-------------------------------------
    // CDXUTComboBox - Selection
    //-------------------------------------
    SetRect( &rcTexture, 7, 266, 241, 289 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB(255, 255, 255, 255), DT_LEFT | DT_TOP );
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_COMBOBOX, 3, &Element );


    //-------------------------------------
    // CDXUTSlider - Track
    //-------------------------------------
    SetRect( &rcTexture, 1, 290, 280, 331 );
    Element.SetTexture( 0, &rcTexture );
    Element.TextureColor.States[ DXUT_STATE_NORMAL ] = D3DCOLOR_ARGB(150, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_FOCUS ] = D3DCOLOR_ARGB(200, 255, 255, 255);
    Element.TextureColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(70, 255, 255, 255);
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_SLIDER, 0, &Element );

    //-------------------------------------
    // CDXUTSlider - Button
    //-------------------------------------
    SetRect( &rcTexture, 248, 55, 289, 96 );
    Element.SetTexture( 0, &rcTexture );

    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_SLIDER, 1, &Element );

    //-------------------------------------
    // CDXUTScrollBar - Track
    //-------------------------------------
    SetRect( &rcTexture, 243, 144, 265, 155 );
    Element.SetTexture( 0, &rcTexture );
    Element.TextureColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(255, 200, 200, 200);
    
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_SCROLLBAR, 0, &Element );

    //-------------------------------------
    // CDXUTScrollBar - Up Arrow
    //-------------------------------------
    SetRect( &rcTexture, 243, 124, 265, 144 );
    Element.SetTexture( 0, &rcTexture );
    Element.TextureColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(255, 200, 200, 200);
    
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_SCROLLBAR, 1, &Element );

    //-------------------------------------
    // CDXUTScrollBar - Down Arrow
    //-------------------------------------
    SetRect( &rcTexture, 243, 155, 265, 176 );
    Element.SetTexture( 0, &rcTexture );
    Element.TextureColor.States[ DXUT_STATE_DISABLED ] = D3DCOLOR_ARGB(255, 200, 200, 200);
    
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_SCROLLBAR, 2, &Element );

    //-------------------------------------
    // CDXUTScrollBar - Button
    //-------------------------------------
    SetRect( &rcTexture, 266, 123, 286, 167 );
    Element.SetTexture( 0, &rcTexture );
    
    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_SCROLLBAR, 3, &Element );


    //-------------------------------------
    // CDXUTEditBox
    //-------------------------------------
    // Element assignment:
    //   0 - text area
    //   1 - top left border
    //   2 - top border
    //   3 - top right border
    //   4 - left border
    //   5 - right border
    //   6 - lower left border
    //   7 - lower border
    //   8 - lower right border

    Element.SetFont( 0, D3DCOLOR_ARGB( 255, 0, 0, 0 ), DT_LEFT | DT_TOP );

    // Assign the style
    SetRect( &rcTexture, 14, 90, 241, 113 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 0, &Element );
    SetRect( &rcTexture, 8, 82, 14, 90 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 1, &Element );
    SetRect( &rcTexture, 14, 82, 241, 90 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 2, &Element );
    SetRect( &rcTexture, 241, 82, 246, 90 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 3, &Element );
    SetRect( &rcTexture, 8, 90, 14, 113 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 4, &Element );
    SetRect( &rcTexture, 241, 90, 246, 113 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 5, &Element );
    SetRect( &rcTexture, 8, 113, 14, 121 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 6, &Element );
    SetRect( &rcTexture, 14, 113, 241, 121 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 7, &Element );
    SetRect( &rcTexture, 241, 113, 246, 121 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_EDITBOX, 8, &Element );


    //-------------------------------------
    // CDXUTIMEEditBox
    //-------------------------------------

    Element.SetFont( 0, D3DCOLOR_ARGB( 255, 0, 0, 0 ), DT_LEFT | DT_TOP );

    // Assign the style
    SetRect( &rcTexture, 14, 90, 241, 113 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 0, &Element );
    SetRect( &rcTexture, 8, 82, 14, 90 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 1, &Element );
    SetRect( &rcTexture, 14, 82, 241, 90 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 2, &Element );
    SetRect( &rcTexture, 241, 82, 246, 90 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 3, &Element );
    SetRect( &rcTexture, 8, 90, 14, 113 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 4, &Element );
    SetRect( &rcTexture, 241, 90, 246, 113 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 5, &Element );
    SetRect( &rcTexture, 8, 113, 14, 121 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 6, &Element );
    SetRect( &rcTexture, 14, 113, 241, 121 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 7, &Element );
    SetRect( &rcTexture, 241, 113, 246, 121 );
    Element.SetTexture( 0, &rcTexture );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 8, &Element );
    // Element 9 for IME text, and indicator button
    SetRect( &rcTexture, 0, 0, 136, 54 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB( 255, 0, 0, 0 ), DT_CENTER | DT_VCENTER );
    SetDefaultElement( DXUT_CONTROL_IMEEDITBOX, 9, &Element );

    //-------------------------------------
    // CDXUTListBox - Main
    //-------------------------------------

    SetRect( &rcTexture, 13, 124, 241, 265 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB(255, 0, 0, 0), DT_LEFT | DT_TOP );

    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_LISTBOX, 0, &Element );

    //-------------------------------------
    // CDXUTListBox - Selection
    //-------------------------------------

    SetRect( &rcTexture, 17, 269, 241, 287 );
    Element.SetTexture( 0, &rcTexture );
    Element.SetFont( 0, D3DCOLOR_ARGB(255, 255, 255, 255), DT_LEFT | DT_TOP );

    // Assign the Element
    SetDefaultElement( DXUT_CONTROL_LISTBOX, 1, &Element );
}



//--------------------------------------------------------------------------------------
// CDXUTControl class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTControl::CDXUTControl( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_BUTTON;
    m_pDialog = pDialog;
    m_ID = 0;
    m_Index = 0;
    m_pUserData = NULL;

    m_bEnabled = true;
    m_bVisible = true;
    m_bMouseOver = false;
    m_bHasFocus = false;
    m_bIsDefault = false;

    m_pDialog = NULL;

    m_x = 0;
    m_y = 0;
    m_width = 0;
    m_height = 0;

   ZeroMemory( &m_rcBoundingBox, sizeof( m_rcBoundingBox ) );
}


CDXUTControl::~CDXUTControl()
{
    for( int i = 0; i < m_Elements.GetSize(); ++i )
    {
        delete m_Elements[i];
    }
    m_Elements.RemoveAll();
}


//--------------------------------------------------------------------------------------
void CDXUTControl::SetTextColor( D3DCOLOR Color )
{
    CDXUTElement* pElement = m_Elements.GetAt( 0 );

    if( pElement )
        pElement->FontColor.States[DXUT_STATE_NORMAL] = Color;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTControl::SetElement( UINT iElement, CDXUTElement* pElement )
{
    HRESULT hr = S_OK;

    if( pElement == NULL )
        return E_INVALIDARG;

    // Make certain the array is this large
    for( UINT i=m_Elements.GetSize(); i <= iElement; i++ )
    {
        CDXUTElement* pNewElement = new CDXUTElement();
        if( pNewElement == NULL )
            return E_OUTOFMEMORY;

        hr = m_Elements.Add( pNewElement );
        if( FAILED(hr) )
            return hr;
    }

    // Update the data
    CDXUTElement* pCurElement = m_Elements.GetAt( iElement );
    *pCurElement = *pElement;
    
    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTControl::Refresh()
{
    m_bMouseOver = false;
    m_bHasFocus = false;

    for( int i=0; i < m_Elements.GetSize(); i++ )
    {
        CDXUTElement* pElement = m_Elements.GetAt( i );
        pElement->Refresh();
    }
}


//--------------------------------------------------------------------------------------
void CDXUTControl::UpdateRects()
{
    SetRect( &m_rcBoundingBox, m_x, m_y, m_x + m_width, m_y + m_height );
}


//--------------------------------------------------------------------------------------
// CDXUTStatic class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTStatic::CDXUTStatic( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_STATIC;
    m_pDialog = pDialog;

    ZeroMemory( &m_strText, sizeof(m_strText) );  

    for( int i=0; i < m_Elements.GetSize(); i++ )
    {
        CDXUTElement* pElement = m_Elements.GetAt( i );
        SAFE_DELETE( pElement );
    }

    m_Elements.RemoveAll();
}


//--------------------------------------------------------------------------------------
void CDXUTStatic::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{    
    if( m_bVisible == false )
        return;

    DXUT_CONTROL_STATE iState = DXUT_STATE_NORMAL;

    if( m_bEnabled == false )
        iState = DXUT_STATE_DISABLED;
        
    CDXUTElement* pElement = m_Elements.GetAt( 0 );

    pElement->FontColor.Blend( iState, fElapsedTime );
    
    m_pDialog->DrawText( m_strText, pElement, &m_rcBoundingBox, true );
}

//--------------------------------------------------------------------------------------
HRESULT CDXUTStatic::GetTextCopy( LPWSTR strDest, UINT bufferCount )
{
    // Validate incoming parameters
    if( strDest == NULL || bufferCount == 0 )
    {
        return E_INVALIDARG;
    }

    // Copy the window text
    wcsncpy( strDest, m_strText, bufferCount );
    strDest[bufferCount-1] = 0;

    return S_OK;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTStatic::SetText( LPCWSTR strText )
{
    if( strText == NULL )
    {
        m_strText[0] = 0;
        return S_OK;
    }
    
    wcsncpy( m_strText, strText, MAX_PATH-1 ); 
    return S_OK;
}


//--------------------------------------------------------------------------------------
// CDXUTButton class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTButton::CDXUTButton( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_BUTTON;
    m_pDialog = pDialog;

    m_bPressed = false;
    m_nHotkey = 0;
}

//--------------------------------------------------------------------------------------
bool CDXUTButton::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_SPACE:
                    m_bPressed = true;
                    return true;
            }
        }

        case WM_KEYUP:
        {
            switch( wParam )
            {
                case VK_SPACE:
                    if( m_bPressed == true )
                    {
                        m_bPressed = false;
                        m_pDialog->SendEvent( EVENT_BUTTON_CLICKED, true, this );
                    }
                    return true;
            }
        }
    }
    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTButton::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            if( ContainsPoint( pt ) )
            {
                // Pressed while inside the control
                m_bPressed = true;
                SetCapture( DXUTGetHWND() );

                if( !m_bHasFocus )
                    m_pDialog->RequestFocus( this );

                return true;
            }

            break;
        }

        case WM_LBUTTONUP:
        {
            if( m_bPressed )
            {
                m_bPressed = false;
                ReleaseCapture();

                if( !m_pDialog->m_bKeyboardInput )
                    m_pDialog->ClearFocus();

                // Button click
                if( ContainsPoint( pt ) )
                    m_pDialog->SendEvent( EVENT_BUTTON_CLICKED, true, this );

                return true;
            }

            break;
        }
    };
    
    return false;
}

//--------------------------------------------------------------------------------------
void CDXUTButton::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    int nOffsetX = 0;
    int nOffsetY = 0;

    DXUT_CONTROL_STATE iState = DXUT_STATE_NORMAL;

    if( m_bVisible == false )
    {
        iState = DXUT_STATE_HIDDEN;
    }
    else if( m_bEnabled == false )
    {
        iState = DXUT_STATE_DISABLED;
    }
    else if( m_bPressed )
    {
        iState = DXUT_STATE_PRESSED;

        nOffsetX = 1;
        nOffsetY = 2;
    }
    else if( m_bMouseOver )
    {
        iState = DXUT_STATE_MOUSEOVER;

        nOffsetX = -1;
        nOffsetY = -2;
    }
    else if( m_bHasFocus )
    {
        iState = DXUT_STATE_FOCUS;
    }
    
    // Background fill layer
    //TODO: remove magic numbers
    CDXUTElement* pElement = m_Elements.GetAt( 0 );
    
    float fBlendRate = ( iState == DXUT_STATE_PRESSED ) ? 0.0f : 0.8f;

    RECT rcWindow = m_rcBoundingBox;
    OffsetRect( &rcWindow, nOffsetX, nOffsetY );

 
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    pElement->FontColor.Blend( iState, fElapsedTime, fBlendRate );

    m_pDialog->DrawSprite( pElement, &rcWindow );
    m_pDialog->DrawText( m_strText, pElement, &rcWindow );

    // Main button
    pElement = m_Elements.GetAt( 1 );


    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    pElement->FontColor.Blend( iState, fElapsedTime, fBlendRate );

    m_pDialog->DrawSprite( pElement, &rcWindow );
    m_pDialog->DrawText( m_strText, pElement, &rcWindow );
}



//--------------------------------------------------------------------------------------
// CDXUTCheckBox class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTCheckBox::CDXUTCheckBox( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_CHECKBOX;
    m_pDialog = pDialog;

    m_bChecked = false;
}
    

//--------------------------------------------------------------------------------------
bool CDXUTCheckBox::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_SPACE:
                    m_bPressed = true;
                    return true;
            }
        }

        case WM_KEYUP:
        {
            switch( wParam )
            {
                case VK_SPACE:
                    if( m_bPressed == true )
                    {
                        m_bPressed = false;
                        SetCheckedInternal( !m_bChecked, true );
                    }
                    return true;
            }
        }
    }
    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTCheckBox::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            if( ContainsPoint( pt ) )
            {
                // Pressed while inside the control
                m_bPressed = true;
                SetCapture( DXUTGetHWND() );

                if( !m_bHasFocus && m_pDialog->m_bKeyboardInput )
                    m_pDialog->RequestFocus( this );

                return true;
            }

            break;
        }

        case WM_LBUTTONUP:
        {
            if( m_bPressed )
            {
                m_bPressed = false;
                ReleaseCapture();

                // Button click
                if( ContainsPoint( pt ) )
                    SetCheckedInternal( !m_bChecked, true );
                
                return true;
            }

            break;
        }
    };
    
    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTCheckBox::SetCheckedInternal( bool bChecked, bool bFromInput ) 
{ 
    m_bChecked = bChecked; 

    m_pDialog->SendEvent( EVENT_CHECKBOX_CHANGED, bFromInput, this ); 
}


//--------------------------------------------------------------------------------------
BOOL CDXUTCheckBox::ContainsPoint( POINT pt ) 
{ 
    return ( PtInRect( &m_rcBoundingBox, pt ) || 
             PtInRect( &m_rcButton, pt ) ); 
}



//--------------------------------------------------------------------------------------
void CDXUTCheckBox::UpdateRects()
{
    CDXUTButton::UpdateRects();

    m_rcButton = m_rcBoundingBox;
    m_rcButton.right = m_rcButton.left + RectHeight( m_rcButton );

    m_rcText = m_rcBoundingBox;
    m_rcText.left += (int) ( 1.25f * RectWidth( m_rcButton ) );
}



//--------------------------------------------------------------------------------------
void CDXUTCheckBox::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    DXUT_CONTROL_STATE iState = DXUT_STATE_NORMAL;

    if( m_bVisible == false )
        iState = DXUT_STATE_HIDDEN;
    else if( m_bEnabled == false )
        iState = DXUT_STATE_DISABLED;
    else if( m_bPressed )
        iState = DXUT_STATE_PRESSED;
    else if( m_bMouseOver )
        iState = DXUT_STATE_MOUSEOVER;
    else if( m_bHasFocus )
        iState = DXUT_STATE_FOCUS;

    //debug
    //m_pDialog->DrawRect( &m_rcBoundingBox, D3DCOLOR_ARGB(255, 255, 255, 0) );
    //m_pDialog->DrawRect( &m_rcButton, D3DCOLOR_ARGB(255, 0, 255, 255) );

    //TODO: remove magic numbers
    CDXUTElement* pElement = m_Elements.GetAt( 0 );
    
    float fBlendRate = ( iState == DXUT_STATE_PRESSED ) ? 0.0f : 0.8f;

    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    pElement->FontColor.Blend( iState, fElapsedTime, fBlendRate );

    m_pDialog->DrawSprite( pElement, &m_rcButton );
    m_pDialog->DrawText( m_strText, pElement, &m_rcText, true );

    if( !m_bChecked )
        iState = DXUT_STATE_HIDDEN;

    pElement = m_Elements.GetAt( 1 );

    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    m_pDialog->DrawSprite( pElement, &m_rcButton );
}




//--------------------------------------------------------------------------------------
// CDXUTRadioButton class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTRadioButton::CDXUTRadioButton( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_RADIOBUTTON;
    m_pDialog = pDialog;
}



//--------------------------------------------------------------------------------------
bool CDXUTRadioButton::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_SPACE:
                    m_bPressed = true;
                    return true;
            }
        }

        case WM_KEYUP:
        {
            switch( wParam )
            {
                case VK_SPACE:
                    if( m_bPressed == true )
                    {
                        m_bPressed = false;
                        
                        m_pDialog->ClearRadioButtonGroup( m_nButtonGroup );
                        m_bChecked = !m_bChecked;

                        m_pDialog->SendEvent( EVENT_RADIOBUTTON_CHANGED, true, this );
                    }
                    return true;
            }
        }
    }
    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTRadioButton::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            if( ContainsPoint( pt ) )
            {
                // Pressed while inside the control
                m_bPressed = true;
                SetCapture( DXUTGetHWND() );

                if( !m_bHasFocus && m_pDialog->m_bKeyboardInput )
                    m_pDialog->RequestFocus( this );

                return true;
            }

            break;
        }

        case WM_LBUTTONUP:
        {
            if( m_bPressed )
            {
                m_bPressed = false;
                ReleaseCapture();

                // Button click
                if( ContainsPoint( pt ) )
                {
                    m_pDialog->ClearRadioButtonGroup( m_nButtonGroup );
                    m_bChecked = !m_bChecked;

                    m_pDialog->SendEvent( EVENT_RADIOBUTTON_CHANGED, true, this );
                }

                return true;
            }

            break;
        }
    };
    
    return false;
}

//--------------------------------------------------------------------------------------
void CDXUTRadioButton::SetCheckedInternal( bool bChecked, bool bClearGroup, bool bFromInput )
{
    if( bChecked && bClearGroup )
        m_pDialog->ClearRadioButtonGroup( m_nButtonGroup );

    m_bChecked = bChecked;
    m_pDialog->SendEvent( EVENT_RADIOBUTTON_CHANGED, bFromInput, this );
}




//--------------------------------------------------------------------------------------
// CDXUTComboBox class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTComboBox::CDXUTComboBox( CDXUTDialog *pDialog ) :
    m_ScrollBar( pDialog )
{
    m_Type = DXUT_CONTROL_COMBOBOX;
    m_pDialog = pDialog;

    m_nDropHeight = 100;

    m_nSBWidth = 16;
    m_bOpened = false;
    m_iSelected = -1;
    m_iFocused = -1;
}


//--------------------------------------------------------------------------------------
CDXUTComboBox::~CDXUTComboBox()
{
    RemoveAllItems();
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::SetTextColor( D3DCOLOR Color )
{
    CDXUTElement* pElement = m_Elements.GetAt( 0 );

    if( pElement )
        pElement->FontColor.States[DXUT_STATE_NORMAL] = Color;

    pElement = m_Elements.GetAt( 2 );

    if( pElement )
        pElement->FontColor.States[DXUT_STATE_NORMAL] = Color;
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::UpdateRects()
{
    
    CDXUTButton::UpdateRects();

    m_rcButton = m_rcBoundingBox;
    m_rcButton.left = m_rcButton.right - RectHeight( m_rcButton );

    m_rcText = m_rcBoundingBox;
    m_rcText.right = m_rcButton.left;

    m_rcDropdown = m_rcText;
    OffsetRect( &m_rcDropdown, 0, (int) (0.90f * RectHeight( m_rcText )) );
    m_rcDropdown.bottom += m_nDropHeight;
    m_rcDropdown.right -= m_nSBWidth;

    m_rcDropdownText = m_rcDropdown;
    m_rcDropdownText.left += (int) (0.1f * RectWidth( m_rcDropdown ));
    m_rcDropdownText.right -= (int) (0.1f * RectWidth( m_rcDropdown ));
    m_rcDropdownText.top += (int) (0.1f * RectHeight( m_rcDropdown ));
    m_rcDropdownText.bottom -= (int) (0.1f * RectHeight( m_rcDropdown ));

    // Update the scrollbar's rects
    m_ScrollBar.SetLocation( m_rcDropdown.right, m_rcDropdown.top+2 );
    m_ScrollBar.SetSize( m_nSBWidth, RectHeight( m_rcDropdown )-2 );
    DXUTFontNode* pFontNode = DXUTGetGlobalDialogResourceManager()->GetFontNode( m_Elements.GetAt( 2 )->iFont );
    if( pFontNode && pFontNode->nHeight )
    {
        m_ScrollBar.SetPageSize( RectHeight( m_rcDropdownText ) / pFontNode->nHeight );

        // The selected item may have been scrolled off the page.
        // Ensure that it is in page again.
        m_ScrollBar.ShowItem( m_iSelected );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::OnFocusOut()
{
    CDXUTButton::OnFocusOut();

    m_bOpened = false;
}
    

//--------------------------------------------------------------------------------------
bool CDXUTComboBox::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    const DWORD	REPEAT_MASK = (0x40000000);

    if( !m_bEnabled || !m_bVisible )
        return false;

    // Let the scroll bar have a chance to handle it first
    if( m_ScrollBar.HandleKeyboard( uMsg, wParam, lParam ) )
        return true;

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_RETURN:
                    if( m_bOpened )
                    {
                        if( m_iSelected != m_iFocused )
                        {
                            m_iSelected = m_iFocused;
                            m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                        }
                        m_bOpened = false;
                        
                        if( !m_pDialog->m_bKeyboardInput )
                            m_pDialog->ClearFocus();

                        return true;
                    }
                    break;

                case VK_F4:
                    // Filter out auto-repeats
                    if( lParam & REPEAT_MASK )
                        return true;

                    m_bOpened = !m_bOpened;

                    if( !m_bOpened )
                    {
                        m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );

                        if( !m_pDialog->m_bKeyboardInput )
                            m_pDialog->ClearFocus();
                    }

                    return true;

                case VK_LEFT:
                case VK_UP:
                    if( m_iFocused > 0 )
                    {
                        m_iFocused--;
                        m_iSelected = m_iFocused;

                        if( !m_bOpened )
                            m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                    }
                    
                    return true;

                case VK_RIGHT:
                case VK_DOWN:
                    if( m_iFocused+1 < (int)GetNumItems() )
                    {
                        m_iFocused++;
                        m_iSelected = m_iFocused;

                        if( !m_bOpened )
                            m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                    }

                    return true;
            }
            break;
        }
    }

    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTComboBox::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    // Let the scroll bar handle it first.
    if( m_ScrollBar.HandleMouse( uMsg, pt, wParam, lParam ) )
        return true;

    switch( uMsg )
    {
        case WM_MOUSEMOVE:
        {
            if( m_bOpened && PtInRect( &m_rcDropdown, pt ) )
            {
                // Determine which item has been selected
                for( int i=0; i < m_Items.GetSize(); i++ )
                {
                    DXUTComboBoxItem* pItem = m_Items.GetAt( i );
                    if( pItem -> bVisible &&
                        PtInRect( &pItem->rcActive, pt ) )
                    {
                        m_iFocused = i;
                    }
                }
                return true;
            }
            break;
        }

        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            if( ContainsPoint( pt ) )
            {
                // Pressed while inside the control
                m_bPressed = true;
                SetCapture( DXUTGetHWND() );

                if( !m_bHasFocus )
                    m_pDialog->RequestFocus( this );

                // Toggle dropdown
                if( m_bHasFocus )
                {
                    m_bOpened = !m_bOpened;
                
                    if( !m_bOpened )
                    {
                        if( !m_pDialog->m_bKeyboardInput )
                            m_pDialog->ClearFocus();
                    }
                }

                return true;
            }

            // Perhaps this click is within the dropdown
            if( m_bOpened && PtInRect( &m_rcDropdown, pt ) )
            {
                // Determine which item has been selected
                for( int i=m_ScrollBar.GetTrackPos(); i < m_Items.GetSize(); i++ )
                {
                    DXUTComboBoxItem* pItem = m_Items.GetAt( i );
                    if( pItem -> bVisible &&
                        PtInRect( &pItem->rcActive, pt ) )
                    {
                        m_iFocused = m_iSelected = i;
                        m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                        m_bOpened = false;
                        
                        if( !m_pDialog->m_bKeyboardInput )
                            m_pDialog->ClearFocus();

                        break;
                    }
                }

                return true;
            }

            // Mouse click not on main control or in dropdown, fire an event if needed
            if( m_bOpened )
            {
                m_iFocused = m_iSelected;

                m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                m_bOpened = false;
            }

            // Make sure the control is no longer in a pressed state
            m_bPressed = false;

            // Release focus if appropriate
            if( !m_pDialog->m_bKeyboardInput )
            {
                m_pDialog->ClearFocus();
            }

            break;
        }

        case WM_LBUTTONUP:
        {
            if( m_bPressed && ContainsPoint( pt ) )
            {
                // Button click
                m_bPressed = false;
                ReleaseCapture();
                return true;
            }

            break;
        }

        case WM_MOUSEWHEEL:
        {
            int zDelta = (short) HIWORD(wParam) / WHEEL_DELTA;
            if( m_bOpened )
            {
                UINT uLines;
                SystemParametersInfo( SPI_GETWHEELSCROLLLINES, 0, &uLines, 0 );
                m_ScrollBar.Scroll( -zDelta * uLines );
            } else
            {
                if( zDelta > 0 )
                {
                    if( m_iFocused > 0 )
                    {
                        m_iFocused--;
                        m_iSelected = m_iFocused;     
                        
                        if( !m_bOpened )
                            m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                    }          
                }
                else
                {
                    if( m_iFocused+1 < (int)GetNumItems() )
                    {
                        m_iFocused++;
                        m_iSelected = m_iFocused;   

                        if( !m_bOpened )
                            m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
                    }
                }
            }
            return true;
        }
    };
    
    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::OnHotkey()
{
    if( m_bOpened )
        return;

    if( m_iSelected == -1 )
        return;

    m_iSelected++;
    
    if( m_iSelected >= (int) m_Items.GetSize() )
        m_iSelected = 0;

    m_iFocused = m_iSelected;
    m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, true, this );
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    DXUT_CONTROL_STATE iState = DXUT_STATE_NORMAL;
    
    if( !m_bOpened )
        iState = DXUT_STATE_HIDDEN;

    // Dropdown box
    CDXUTElement* pElement = m_Elements.GetAt( 2 );

    // If we have not initialized the scroll bar page size,
    // do that now.
    static bool bSBInit;
    if( !bSBInit )
    {
        // Update the page size of the scroll bar
        if( DXUTGetGlobalDialogResourceManager()->GetFontNode( pElement->iFont )->nHeight )
            m_ScrollBar.SetPageSize( RectHeight( m_rcDropdownText ) / DXUTGetGlobalDialogResourceManager()->GetFontNode( pElement->iFont )->nHeight );
        else
            m_ScrollBar.SetPageSize( RectHeight( m_rcDropdownText ) );
        bSBInit = true;
    }

    // Scroll bar
    if( m_bOpened )
        m_ScrollBar.Render( pd3dDevice, fElapsedTime );

    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime );
    pElement->FontColor.Blend( iState, fElapsedTime );

    m_pDialog->DrawSprite( pElement, &m_rcDropdown );

    // Selection outline
    CDXUTElement* pSelectionElement = m_Elements.GetAt( 3 );
    pSelectionElement->TextureColor.Current = pElement->TextureColor.Current;
    pSelectionElement->FontColor.Current = pSelectionElement->FontColor.States[ DXUT_STATE_NORMAL ];

    DXUTFontNode* pFont = m_pDialog->GetFont( pElement->iFont );
    int curY = m_rcDropdownText.top;
    int nRemainingHeight = RectHeight( m_rcDropdownText );
    //WCHAR strDropdown[4096] = {0};

    for( int i = m_ScrollBar.GetTrackPos(); i < m_Items.GetSize(); i++ )
    {
        DXUTComboBoxItem* pItem = m_Items.GetAt( i );

        // Make sure there's room left in the dropdown
        nRemainingHeight -= pFont->nHeight;
        if( nRemainingHeight < 0 )
        {
            pItem->bVisible = false;
            continue;
        }

        SetRect( &pItem->rcActive, m_rcDropdownText.left, curY, m_rcDropdownText.right, curY + pFont->nHeight );
        curY += pFont->nHeight;
        
        //debug
        //int blue = 50 * i;
        //m_pDialog->DrawRect( &pItem->rcActive, 0xFFFF0000 | blue );

        pItem->bVisible = true;

        if( m_bOpened )
        {
            if( (int)i == m_iFocused )
            {
                RECT rc;
                SetRect( &rc, m_rcDropdown.left, pItem->rcActive.top-2, m_rcDropdown.right, pItem->rcActive.bottom+2 );
                m_pDialog->DrawSprite( pSelectionElement, &rc );
                m_pDialog->DrawText( pItem->strText, pSelectionElement, &pItem->rcActive );
            }
            else
            {
                m_pDialog->DrawText( pItem->strText, pElement, &pItem->rcActive );
            }
        }
    }

    int nOffsetX = 0;
    int nOffsetY = 0;

    iState = DXUT_STATE_NORMAL;
    
    if( m_bVisible == false )
        iState = DXUT_STATE_HIDDEN;
    else if( m_bEnabled == false )
        iState = DXUT_STATE_DISABLED;
    else if( m_bPressed )
    {
        iState = DXUT_STATE_PRESSED;

        nOffsetX = 1;
        nOffsetY = 2;
    }
    else if( m_bMouseOver )
    {
        iState = DXUT_STATE_MOUSEOVER;

        nOffsetX = -1;
        nOffsetY = -2;
    }
    else if( m_bHasFocus )
        iState = DXUT_STATE_FOCUS;

    float fBlendRate = ( iState == DXUT_STATE_PRESSED ) ? 0.0f : 0.8f;
    
    // Button
    pElement = m_Elements.GetAt( 1 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    
    RECT rcWindow = m_rcButton;
    OffsetRect( &rcWindow, nOffsetX, nOffsetY );
    m_pDialog->DrawSprite( pElement, &rcWindow );

    if( m_bOpened )
        iState = DXUT_STATE_PRESSED;

    // Main text box
    //TODO: remove magic numbers
    pElement = m_Elements.GetAt( 0 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    pElement->FontColor.Blend( iState, fElapsedTime, fBlendRate );

    m_pDialog->DrawSprite( pElement, &m_rcText);
    
    if( m_iSelected >= 0 && m_iSelected < (int) m_Items.GetSize() )
    {
        DXUTComboBoxItem* pItem = m_Items.GetAt( m_iSelected );
        if( pItem != NULL )
        {
            m_pDialog->DrawText( pItem->strText, pElement, &m_rcText );
        
        }
    }
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTComboBox::AddItem( const WCHAR* strText, void* pData )
{
    // Validate parameters
    if( strText== NULL )
    {
        return E_INVALIDARG;
    }
    
    // Create a new item and set the data
    DXUTComboBoxItem* pItem = new DXUTComboBoxItem;
    if( pItem == NULL )
    {
        return DXTRACE_ERR_MSGBOX( L"new", E_OUTOFMEMORY );
    }
    
    ZeroMemory( pItem, sizeof(DXUTComboBoxItem) );
    wcsncpy( pItem->strText, strText, 255 );
    pItem->pData = pData;

    m_Items.Add( pItem );

    // Update the scroll bar with new range
    m_ScrollBar.SetTrackRange( 0, m_Items.GetSize() );

    // If this is the only item in the list, it's selected
    if( GetNumItems() == 1 )
    {
        m_iSelected = 0;
        m_iFocused = 0;
        m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, false, this );
    }

    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::RemoveItem( UINT index )
{
    DXUTComboBoxItem* pItem = m_Items.GetAt( index );
    SAFE_DELETE( pItem );
    m_Items.Remove( index );
    m_ScrollBar.SetTrackRange( 0, m_Items.GetSize() );
    if( m_iSelected >= m_Items.GetSize() )
        m_iSelected = m_Items.GetSize() - 1;
}


//--------------------------------------------------------------------------------------
void CDXUTComboBox::RemoveAllItems()
{
    for( int i=0; i < m_Items.GetSize(); i++ )
    {
        DXUTComboBoxItem* pItem = m_Items.GetAt( i );
        SAFE_DELETE( pItem );
    }

    m_Items.RemoveAll();
    m_ScrollBar.SetTrackRange( 0, 1 );
    m_iFocused = m_iSelected = -1;
}



//--------------------------------------------------------------------------------------
bool CDXUTComboBox::ContainsItem( const WCHAR* strText, UINT iStart )
{
    return ( -1 != FindItem( strText, iStart ) );
}


//--------------------------------------------------------------------------------------
int CDXUTComboBox::FindItem( const WCHAR* strText, UINT iStart )
{
    if( strText == NULL )
        return -1;

    for( int i = iStart; i < m_Items.GetSize(); i++ )
    {
        DXUTComboBoxItem* pItem = m_Items.GetAt(i);

        if( 0 == wcscmp( pItem->strText, strText ) )
        {
            return i;
        }
    }

    return -1;
}


//--------------------------------------------------------------------------------------
void* CDXUTComboBox::GetSelectedData()
{
    if( m_iSelected < 0 )
        return NULL;

    DXUTComboBoxItem* pItem = m_Items.GetAt( m_iSelected );
    return pItem->pData;
}


//--------------------------------------------------------------------------------------
DXUTComboBoxItem* CDXUTComboBox::GetSelectedItem()
{
    if( m_iSelected < 0 )
        return NULL;

    return m_Items.GetAt( m_iSelected );
}


//--------------------------------------------------------------------------------------
void* CDXUTComboBox::GetItemData( const WCHAR* strText )
{
    int index = FindItem( strText );
    if( index == -1 )
    {
        return NULL;
    }

    DXUTComboBoxItem* pItem = m_Items.GetAt(index);
    if( pItem == NULL )
    {
        DXTRACE_ERR( L"CGrowableArray::GetAt", E_FAIL );
        return NULL;
    }

    return pItem->pData;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTComboBox::SetSelectedByIndex( UINT index )
{
    if( index >= GetNumItems() )
        return E_INVALIDARG;

    m_iFocused = m_iSelected = index;
    m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, false, this );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTComboBox::SetSelectedByText( const WCHAR* strText )
{
    if( strText == NULL )
        return E_INVALIDARG;

    int index = FindItem( strText );
    if( index == -1 )
        return E_FAIL;

    m_iFocused = m_iSelected = index;
    m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, false, this );

    return S_OK;
}



//--------------------------------------------------------------------------------------
HRESULT CDXUTComboBox::SetSelectedByData( void* pData )
{
    for( int i=0; i < m_Items.GetSize(); i++ )
    {
        DXUTComboBoxItem* pItem = m_Items.GetAt(i);

        if( pItem->pData == pData )
        {
            m_iFocused = m_iSelected = i;
            m_pDialog->SendEvent( EVENT_COMBOBOX_SELECTION_CHANGED, false, this );
            return S_OK;
        }
    }

    return E_FAIL;
}



//--------------------------------------------------------------------------------------
CDXUTSlider::CDXUTSlider( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_SLIDER;
    m_pDialog = pDialog;

    m_nMin = 0;
    m_nMax = 100;
    m_nValue = 50;

    m_bPressed = false;
}


//--------------------------------------------------------------------------------------
BOOL CDXUTSlider::ContainsPoint( POINT pt ) 
{ 
    return ( PtInRect( &m_rcBoundingBox, pt ) || 
             PtInRect( &m_rcButton, pt ) ); 
}


//--------------------------------------------------------------------------------------
void CDXUTSlider::UpdateRects()
{
    CDXUTControl::UpdateRects();

    m_rcButton = m_rcBoundingBox;
    m_rcButton.right = m_rcButton.left + RectHeight( m_rcButton );
    OffsetRect( &m_rcButton, -RectWidth( m_rcButton )/2, 0 );

    m_nButtonX = (int) ( (m_nValue - m_nMin) * (float)RectWidth( m_rcBoundingBox ) / (m_nMax - m_nMin) );
    OffsetRect( &m_rcButton, m_nButtonX, 0 );
}

int CDXUTSlider::ValueFromPos( int x )
{ 
    float fValuePerPixel = (float)(m_nMax - m_nMin) / RectWidth( m_rcBoundingBox );
    return (int) (0.5f + m_nMin + fValuePerPixel * (x - m_rcBoundingBox.left)) ; 
}

//--------------------------------------------------------------------------------------
bool CDXUTSlider::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_HOME:
                    SetValueInternal( m_nMin, true );
                    return true;

                case VK_END:
                    SetValueInternal( m_nMax, true );
                    return true;

                case VK_PRIOR:
                case VK_LEFT:
                case VK_UP:
                    SetValueInternal( m_nValue - 1, true );
                    return true;

                case VK_NEXT:
                case VK_RIGHT:
                case VK_DOWN:
                    SetValueInternal( m_nValue + 1, true );
                    return true;
            }
            break;
        }
    }
    

    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTSlider::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            if( PtInRect( &m_rcButton, pt ) )
            {
                // Pressed while inside the control
                m_bPressed = true;
                SetCapture( DXUTGetHWND() );

                m_nDragX = pt.x;
                //m_nDragY = pt.y;
                m_nDragOffset = m_nButtonX - m_nDragX;

                //m_nDragValue = m_nValue;

                if( !m_bHasFocus )
                    m_pDialog->RequestFocus( this );

                return true;
            }

            if( PtInRect( &m_rcBoundingBox, pt ) )
            {
               if( pt.x > m_nButtonX + m_x )
               {
                   SetValueInternal( m_nValue + 1, true );
                   return true;
               }

               if( pt.x < m_nButtonX + m_x )
               {
                   SetValueInternal( m_nValue - 1, true );
                   return true;
               }
            }

            break;
        }

        case WM_LBUTTONUP:
        {
            if( m_bPressed )
            {
                m_bPressed = false;
                ReleaseCapture();
                m_pDialog->ClearFocus();
                m_pDialog->SendEvent( EVENT_SLIDER_VALUE_CHANGED, true, this );

                return true;
            }

            break;
        }

        case WM_MOUSEMOVE:
        {
            if( m_bPressed )
            {
                SetValueInternal( ValueFromPos( m_x + pt.x + m_nDragOffset ), true );
                return true;
            }

            break;
        }
    };
    
    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTSlider::SetRange( int nMin, int nMax ) 
{
    m_nMin = nMin; 
    m_nMax = nMax; 

    SetValueInternal( m_nValue, false );
}



//--------------------------------------------------------------------------------------
void CDXUTSlider::SetValueInternal( int nValue, bool bFromInput )
{
    // Clamp to range
    nValue = max( m_nMin, nValue );
    nValue = min( m_nMax, nValue );
    
    if( nValue == m_nValue )
        return;

    m_nValue = nValue;
    UpdateRects();

    m_pDialog->SendEvent( EVENT_SLIDER_VALUE_CHANGED, bFromInput, this );
}


//--------------------------------------------------------------------------------------
void CDXUTSlider::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    int nOffsetX = 0;
    int nOffsetY = 0;

    DXUT_CONTROL_STATE iState = DXUT_STATE_NORMAL;

    if( m_bVisible == false )
    {
        iState = DXUT_STATE_HIDDEN;
    }
    else if( m_bEnabled == false )
    {
        iState = DXUT_STATE_DISABLED;
    }
    else if( m_bPressed )
    {
        iState = DXUT_STATE_PRESSED;

        nOffsetX = 1;
        nOffsetY = 2;
    }
    else if( m_bMouseOver )
    {
        iState = DXUT_STATE_MOUSEOVER;
        
        nOffsetX = -1;
        nOffsetY = -2;
    }
    else if( m_bHasFocus )
    {
        iState = DXUT_STATE_FOCUS;
    }

    float fBlendRate = ( iState == DXUT_STATE_PRESSED ) ? 0.0f : 0.8f;

    CDXUTElement* pElement = m_Elements.GetAt( 0 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate ); 
    m_pDialog->DrawSprite( pElement, &m_rcBoundingBox );

    //TODO: remove magic numbers
    pElement = m_Elements.GetAt( 1 );
       
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    m_pDialog->DrawSprite( pElement, &m_rcButton );
}


//--------------------------------------------------------------------------------------
// CDXUTScrollBar class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTScrollBar::CDXUTScrollBar( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_SCROLLBAR;
    m_pDialog = pDialog;

    m_bShowThumb = true;

    SetRect( &m_rcUpButton, 0, 0, 0, 0 );
    SetRect( &m_rcDownButton, 0, 0, 0, 0 );
    SetRect( &m_rcTrack, 0, 0, 0, 0 );
    SetRect( &m_rcThumb, 0, 0, 0, 0 );
    m_nPosition = 0;
    m_nPageSize = 1;
    m_nStart = 0;
    m_nEnd = 1;
    m_Arrow = CLEAR;
    m_dArrowTS = 0.0;
}


//--------------------------------------------------------------------------------------
CDXUTScrollBar::~CDXUTScrollBar()
{
}


//--------------------------------------------------------------------------------------
void CDXUTScrollBar::UpdateRects()
{
    CDXUTControl::UpdateRects();

    // Make the buttons square

    SetRect( &m_rcUpButton, m_rcBoundingBox.left, m_rcBoundingBox.top,
                            m_rcBoundingBox.right, m_rcBoundingBox.top + RectWidth( m_rcBoundingBox ) );
    SetRect( &m_rcDownButton, m_rcBoundingBox.left, m_rcBoundingBox.bottom - RectWidth( m_rcBoundingBox ),
                              m_rcBoundingBox.right, m_rcBoundingBox.bottom );
    SetRect( &m_rcTrack, m_rcUpButton.left, m_rcUpButton.bottom,
                         m_rcDownButton.right, m_rcDownButton.top );
    m_rcThumb.left = m_rcUpButton.left;
    m_rcThumb.right = m_rcUpButton.right;

    UpdateThumbRect();
}


//--------------------------------------------------------------------------------------
// Compute the dimension of the scroll thumb
void CDXUTScrollBar::UpdateThumbRect()
{
    if( m_nEnd - m_nStart > m_nPageSize )
    {
        int nThumbHeight = max( RectHeight( m_rcTrack ) * m_nPageSize / ( m_nEnd - m_nStart ), SCROLLBAR_MINTHUMBSIZE );
        int nMaxPosition = m_nEnd - m_nStart - m_nPageSize;
        m_rcThumb.top = m_rcTrack.top + ( m_nPosition - m_nStart ) * ( RectHeight( m_rcTrack ) - nThumbHeight )
                        / nMaxPosition;
        m_rcThumb.bottom = m_rcThumb.top + nThumbHeight;
        m_bShowThumb = true;

    } 
    else
    {
        // No content to scroll
        m_rcThumb.bottom = m_rcThumb.top;
        m_bShowThumb = false;
    }
}


//--------------------------------------------------------------------------------------
// Scroll() scrolls by nDelta items.  A positive value scrolls down, while a negative
// value scrolls up.
void CDXUTScrollBar::Scroll( int nDelta )
{
    // Perform scroll
    m_nPosition += nDelta;

    // Cap position
    Cap();

    // Update thumb position
    UpdateThumbRect();
}


//--------------------------------------------------------------------------------------
void CDXUTScrollBar::ShowItem( int nIndex )
{
    // Cap the index

    if( nIndex < 0 )
        nIndex = 0;

    if( nIndex >= m_nEnd )
        nIndex = m_nEnd - 1;

    // Adjust position

    if( m_nPosition > nIndex )
        m_nPosition = nIndex;
    else
    if( m_nPosition + m_nPageSize <= nIndex )
        m_nPosition = nIndex - m_nPageSize + 1;

    UpdateThumbRect();
}


//--------------------------------------------------------------------------------------
bool CDXUTScrollBar::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTScrollBar::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    static int ThumbOffsetY;
    static bool bDrag;

    m_LastMouse = pt;
    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            SetCapture( DXUTGetHWND() );

            // Check for click on up button

            if( PtInRect( &m_rcUpButton, pt ) )
            {
                if( m_nPosition > m_nStart )
                    --m_nPosition;
                UpdateThumbRect();
                m_Arrow = CLICKED_UP;
                m_dArrowTS = DXUTGetTime();
                return true;
            }

            // Check for click on down button

            if( PtInRect( &m_rcDownButton, pt ) )
            {
                if( m_nPosition + m_nPageSize < m_nEnd )
                    ++m_nPosition;
                UpdateThumbRect();
                m_Arrow = CLICKED_DOWN;
                m_dArrowTS = DXUTGetTime();
                return true;
            }

            // Check for click on thumb

            if( PtInRect( &m_rcThumb, pt ) )
            {
                bDrag = true;
                ThumbOffsetY = pt.y - m_rcThumb.top;
                return true;
            }

            // Check for click on track

            if( m_rcThumb.left <= pt.x &&
                m_rcThumb.right > pt.x )
            {
                if( m_rcThumb.top > pt.y &&
                    m_rcTrack.top <= pt.y )
                {
                    Scroll( -( m_nPageSize - 1 ) );
                    return true;
                } else
                if( m_rcThumb.bottom <= pt.y &&
                    m_rcTrack.bottom > pt.y )
                {
                    Scroll( m_nPageSize - 1 );
                    return true;
                }
            }

            break;
        }

        case WM_LBUTTONUP:
        {
            bDrag = false;
            ReleaseCapture();
            UpdateThumbRect();
            m_Arrow = CLEAR;
            break;
        }

        case WM_MOUSEMOVE:
        {
            if( bDrag )
            {
                m_rcThumb.bottom += pt.y - ThumbOffsetY - m_rcThumb.top;
                m_rcThumb.top = pt.y - ThumbOffsetY;
                if( m_rcThumb.top < m_rcTrack.top )
                    OffsetRect( &m_rcThumb, 0, m_rcTrack.top - m_rcThumb.top );
                else
                if( m_rcThumb.bottom > m_rcTrack.bottom )
                    OffsetRect( &m_rcThumb, 0, m_rcTrack.bottom - m_rcThumb.bottom );

                // Compute first item index based on thumb position

                int nMaxFirstItem = m_nEnd - m_nStart - m_nPageSize;  // Largest possible index for first item
                int nMaxThumb = RectHeight( m_rcTrack ) - RectHeight( m_rcThumb );  // Largest possible thumb position from the top

                m_nPosition = m_nStart +
                              ( m_rcThumb.top - m_rcTrack.top +
                                nMaxThumb / ( nMaxFirstItem * 2 ) ) * // Shift by half a row to avoid last row covered by only one pixel
                              nMaxFirstItem  / nMaxThumb;

                return true;
            }

            break;
        }
    }

    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTScrollBar::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    // Check if the arrow button has been held for a while.
    // If so, update the thumb position to simulate repeated
    // scroll.
    if( m_Arrow != CLEAR )
    {
        double dCurrTime = DXUTGetTime();
        if( PtInRect( &m_rcUpButton, m_LastMouse ) )
        {
            switch( m_Arrow )
            {
                case CLICKED_UP:
                    if( SCROLLBAR_ARROWCLICK_DELAY < dCurrTime - m_dArrowTS )
                    {
                        Scroll( -1 );
                        m_Arrow = HELD_UP;
                        m_dArrowTS = dCurrTime;
                    }
                    break;
                case HELD_UP:
                    if( SCROLLBAR_ARROWCLICK_REPEAT < dCurrTime - m_dArrowTS )
                    {
                        Scroll( -1 );
                        m_dArrowTS = dCurrTime;
                    }
                    break;
            }
        } else
        if( PtInRect( &m_rcDownButton, m_LastMouse ) )
        {
            switch( m_Arrow )
            {
                case CLICKED_DOWN:
                    if( SCROLLBAR_ARROWCLICK_DELAY < dCurrTime - m_dArrowTS )
                    {
                        Scroll( 1 );
                        m_Arrow = HELD_DOWN;
                        m_dArrowTS = dCurrTime;
                    }
                    break;
                case HELD_DOWN:
                    if( SCROLLBAR_ARROWCLICK_REPEAT < dCurrTime - m_dArrowTS )
                    {
                        Scroll( 1 );
                        m_dArrowTS = dCurrTime;
                    }
                    break;
            }
        }
    }

    DXUT_CONTROL_STATE iState = DXUT_STATE_NORMAL;

    if( m_bVisible == false )
        iState = DXUT_STATE_HIDDEN;
    else if( m_bEnabled == false || m_bShowThumb == false )
        iState = DXUT_STATE_DISABLED;
    else if( m_bMouseOver )
        iState = DXUT_STATE_MOUSEOVER;
    else if( m_bHasFocus )
        iState = DXUT_STATE_FOCUS;


    float fBlendRate = ( iState == DXUT_STATE_PRESSED ) ? 0.0f : 0.8f;

    // Background track layer
    CDXUTElement* pElement = m_Elements.GetAt( 0 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    m_pDialog->DrawSprite( pElement, &m_rcTrack );

    // Up Arrow
    pElement = m_Elements.GetAt( 1 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    m_pDialog->DrawSprite( pElement, &m_rcUpButton );

    // Down Arrow
    pElement = m_Elements.GetAt( 2 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    m_pDialog->DrawSprite( pElement, &m_rcDownButton );

    // Thumb button
    pElement = m_Elements.GetAt( 3 );
    
    // Blend current color
    pElement->TextureColor.Blend( iState, fElapsedTime, fBlendRate );
    m_pDialog->DrawSprite( pElement, &m_rcThumb );
 
}


//--------------------------------------------------------------------------------------
void CDXUTScrollBar::SetTrackRange( int nStart, int nEnd )
{
    m_nStart = nStart; m_nEnd = nEnd;
    Cap();
    UpdateThumbRect();
}


//--------------------------------------------------------------------------------------
void CDXUTScrollBar::Cap()  // Clips position at boundaries. Ensures it stays within legal range.
{
    if( m_nPosition < m_nStart ||
        m_nEnd - m_nStart <= m_nPageSize )
    {
        m_nPosition = m_nStart;
    }
    else
    if( m_nPosition + m_nPageSize > m_nEnd )
        m_nPosition = m_nEnd - m_nPageSize;
}

//--------------------------------------------------------------------------------------
// CDXUTListBox class
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CDXUTListBox::CDXUTListBox( CDXUTDialog *pDialog ) :
    m_ScrollBar( pDialog )
{
    m_Type = DXUT_CONTROL_LISTBOX;
    m_pDialog = pDialog;

    m_dwStyle = 0;
    m_nSBWidth = 16;
    m_nSelected = -1;
    m_nSelStart = 0;
    m_bDrag = false;
    m_nBorder = 6;
    m_nMargin = 5;
    m_nTextHeight = 0;
}


//--------------------------------------------------------------------------------------
CDXUTListBox::~CDXUTListBox()
{
    RemoveAllItems();
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::UpdateRects()
{
    CDXUTControl::UpdateRects();

    m_rcSelection = m_rcBoundingBox;
    m_rcSelection.right -= m_nSBWidth;
    InflateRect( &m_rcSelection, -m_nBorder, -m_nBorder );
    m_rcText = m_rcSelection;
    InflateRect( &m_rcText, -m_nMargin, 0 );

    // Update the scrollbar's rects
    m_ScrollBar.SetLocation( m_rcBoundingBox.right - m_nSBWidth, m_rcBoundingBox.top );
    m_ScrollBar.SetSize( m_nSBWidth, m_height );
    DXUTFontNode* pFontNode = DXUTGetGlobalDialogResourceManager()->GetFontNode( m_Elements.GetAt( 0 )->iFont );
    if( pFontNode && pFontNode->nHeight )
    {
        m_ScrollBar.SetPageSize( RectHeight( m_rcText ) / pFontNode->nHeight );

        // The selected item may have been scrolled off the page.
        // Ensure that it is in page again.
        m_ScrollBar.ShowItem( m_nSelected );
    }
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTListBox::AddItem( const WCHAR *wszText, void *pData )
{
    DXUTListBoxItem *pNewItem = new DXUTListBoxItem;
    if( !pNewItem )
        return E_OUTOFMEMORY;

    wcsncpy( pNewItem->strText, wszText, 256 );
    pNewItem->strText[255] = L'\0';
    pNewItem->pData = pData;
    SetRect( &pNewItem->rcActive, 0, 0, 0, 0 );
    pNewItem->bSelected = false;

    HRESULT hr = m_Items.Add( pNewItem );
    if( SUCCEEDED( hr ) )
        m_ScrollBar.SetTrackRange( 0, m_Items.GetSize() );

    return hr;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTListBox::InsertItem( int nIndex, const WCHAR *wszText, void *pData )
{
    DXUTListBoxItem *pNewItem = new DXUTListBoxItem;
    if( !pNewItem )
        return E_OUTOFMEMORY;

    wcsncpy( pNewItem->strText, wszText, 256 );
    pNewItem->strText[255] = L'\0';
    pNewItem->pData = pData;
    SetRect( &pNewItem->rcActive, 0, 0, 0, 0 );
    pNewItem->bSelected = false;

    HRESULT hr = m_Items.Insert( nIndex, pNewItem );
    if( SUCCEEDED( hr ) )
        m_ScrollBar.SetTrackRange( 0, m_Items.GetSize() );

    return hr;
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::RemoveItem( int nIndex )
{
    if( nIndex < 0 || nIndex >= (int)m_Items.GetSize() )
        return;

    DXUTListBoxItem *pItem = m_Items.GetAt( nIndex );

    delete pItem;
    m_Items.Remove( nIndex );
    m_ScrollBar.SetTrackRange( 0, m_Items.GetSize() );
    if( m_nSelected >= (int)m_Items.GetSize() )
        m_nSelected = m_Items.GetSize() - 1;

    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::RemoveItemByText( WCHAR *wszText )
{
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::RemoveItemByData( void *pData )
{
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::RemoveAllItems()
{
    for( int i = 0; i < m_Items.GetSize(); ++i )
    {
        DXUTListBoxItem *pItem = m_Items.GetAt( i );
        delete pItem;
    }

    m_Items.RemoveAll();
    m_ScrollBar.SetTrackRange( 0, 1 );
}


//--------------------------------------------------------------------------------------
DXUTListBoxItem *CDXUTListBox::GetItem( int nIndex )
{
    if( nIndex < 0 || nIndex >= (int)m_Items.GetSize() )
        return NULL;

    return m_Items[nIndex];
}


//--------------------------------------------------------------------------------------
// For single-selection listbox, returns the index of the selected item.
// For multi-selection, returns the first selected item after the nPreviousSelected position.
// To search for the first selected item, the app passes -1 for nPreviousSelected.  For
// subsequent searches, the app passes the returned index back to GetSelectedIndex as.
// nPreviousSelected.
// Returns -1 on error or if no item is selected.
int CDXUTListBox::GetSelectedIndex( int nPreviousSelected )
{
    if( nPreviousSelected < -1 )
        return -1;

    if( m_dwStyle & MULTISELECTION )
    {
        // Multiple selection enabled. Search for the next item with the selected flag.
        for( int i = nPreviousSelected + 1; i < (int)m_Items.GetSize(); ++i )
        {
            DXUTListBoxItem *pItem = m_Items.GetAt( i );

            if( pItem->bSelected )
                return i;
        }

        return -1;
    }
    else
    {
        // Single selection
        return m_nSelected;
    }
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::SelectItem( int nNewIndex )
{
    // If no item exists, do nothing.
    if( m_Items.GetSize() == 0 )
        return;

    int nOldSelected = m_nSelected;

    // Adjust m_nSelected
    m_nSelected = nNewIndex;

    // Perform capping
    if( m_nSelected < 0 )
        m_nSelected = 0;
    if( m_nSelected >= (int)m_Items.GetSize() )
        m_nSelected = m_Items.GetSize() - 1;

    if( nOldSelected != m_nSelected )
    {
        if( m_dwStyle & MULTISELECTION )
        {
            m_Items[m_nSelected]->bSelected = true;
        }

        // Update selection start
        m_nSelStart = m_nSelected;

        // Adjust scroll bar
        m_ScrollBar.ShowItem( m_nSelected );
    }

    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
}


//--------------------------------------------------------------------------------------
bool CDXUTListBox::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    // Let the scroll bar have a chance to handle it first
    if( m_ScrollBar.HandleKeyboard( uMsg, wParam, lParam ) )
        return true;

    switch( uMsg )
    {
        case WM_KEYDOWN:
            switch( wParam )
            {
                case VK_UP:
                case VK_DOWN:
                case VK_NEXT:
                case VK_PRIOR:
                case VK_HOME:
                case VK_END:

                    // If no item exists, do nothing.
                    if( m_Items.GetSize() == 0 )
                        return true;

                    int nOldSelected = m_nSelected;

                    // Adjust m_nSelected
                    switch( wParam )
                    {
                        case VK_UP: --m_nSelected; break;
                        case VK_DOWN: ++m_nSelected; break;
                        case VK_NEXT: m_nSelected += m_ScrollBar.GetPageSize() - 1; break;
                        case VK_PRIOR: m_nSelected -= m_ScrollBar.GetPageSize() - 1; break;
                        case VK_HOME: m_nSelected = 0; break;
                        case VK_END: m_nSelected = m_Items.GetSize() - 1; break;
                    }

                    // Perform capping
                    if( m_nSelected < 0 )
                        m_nSelected = 0;
                    if( m_nSelected >= (int)m_Items.GetSize() )
                        m_nSelected = m_Items.GetSize() - 1;

                    if( nOldSelected != m_nSelected )
                    {
                        if( m_dwStyle & MULTISELECTION )
                        {
                            // Multiple selection

                            // Clear all selection
                            for( int i = 0; i < (int)m_Items.GetSize(); ++i )
                            {
                                DXUTListBoxItem *pItem = m_Items[i];
                                pItem->bSelected = false;
                            }

                            if( GetKeyState( VK_SHIFT ) < 0 )
                            {
                                // Select all items from m_nSelStart to
                                // m_nSelected
                                int nEnd = max( m_nSelStart, m_nSelected );

                                for( int n = min( m_nSelStart, m_nSelected ); n <= nEnd; ++n )
                                    m_Items[n]->bSelected = true;
                            }
                            else
                            {
                                m_Items[m_nSelected]->bSelected = true;

                                // Update selection start
                                m_nSelStart = m_nSelected;
                            }
                        } else
                            m_nSelStart = m_nSelected;

                        // Adjust scroll bar

                        m_ScrollBar.ShowItem( m_nSelected );

                        // Send notification

                        m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
                    }
                    return true;
            }
            break;
    }

    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTListBox::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    // First acquire focus
    if( WM_LBUTTONDOWN == uMsg )
        if( !m_bHasFocus )
            m_pDialog->RequestFocus( this );

    // Let the scroll bar handle it first.
    if( m_ScrollBar.HandleMouse( uMsg, pt, wParam, lParam ) )
        return true;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
            // Check for clicks in the text area
            if( m_Items.GetSize() > 0 && PtInRect( &m_rcSelection, pt ) )
            {
                // Compute the index of the clicked item

                int nClicked;
                if( m_nTextHeight )
                    nClicked = m_ScrollBar.GetTrackPos() + ( pt.y - m_rcText.top ) / m_nTextHeight;
                else
                    nClicked = -1;

                // Only proceed if the click falls on top of an item.

                if( nClicked >= m_ScrollBar.GetTrackPos() &&
                    nClicked < (int)m_Items.GetSize() &&
                    nClicked < m_ScrollBar.GetTrackPos() + m_ScrollBar.GetPageSize() )
                {
                    SetCapture( DXUTGetHWND() );
                    m_bDrag = true;

                    // If this is a double click, fire off an event and exit
                    // since the first click would have taken care of the selection
                    // updating.
                    if( uMsg == WM_LBUTTONDBLCLK )
                    {
                        m_pDialog->SendEvent( EVENT_LISTBOX_ITEM_DBLCLK, true, this );
                        return true;
                    }

                    m_nSelected = nClicked;
                    if( !( wParam & MK_SHIFT ) )
                        m_nSelStart = m_nSelected;

                    // If this is a multi-selection listbox, update per-item
                    // selection data.

                    if( m_dwStyle & MULTISELECTION )
                    {
                        // Determine behavior based on the state of Shift and Ctrl

                        DXUTListBoxItem *pSelItem = m_Items.GetAt( m_nSelected );
                        if( ( wParam & (MK_SHIFT|MK_CONTROL) ) == MK_CONTROL )
                        {
                            // Control click. Reverse the selection of this item.

                            pSelItem->bSelected = !pSelItem->bSelected;
                        } else
                        if( ( wParam & (MK_SHIFT|MK_CONTROL) ) == MK_SHIFT )
                        {
                            // Shift click. Set the selection for all items
                            // from last selected item to the current item.
                            // Clear everything else.

                            int nBegin = min( m_nSelStart, m_nSelected );
                            int nEnd = max( m_nSelStart, m_nSelected );

                            for( int i = 0; i < nBegin; ++i )
                            {
                                DXUTListBoxItem *pItem = m_Items.GetAt( i );
                                pItem->bSelected = false;
                            }

                            for( int i = nEnd + 1; i < (int)m_Items.GetSize(); ++i )
                            {
                                DXUTListBoxItem *pItem = m_Items.GetAt( i );
                                pItem->bSelected = false;
                            }

                            for( int i = nBegin; i <= nEnd; ++i )
                            {
                                DXUTListBoxItem *pItem = m_Items.GetAt( i );
                                pItem->bSelected = true;
                            }
                        } else
                        if( ( wParam & (MK_SHIFT|MK_CONTROL) ) == ( MK_SHIFT|MK_CONTROL ) )
                        {
                            // Control-Shift-click.

                            // The behavior is:
                            //   Set all items from m_nSelStart to m_nSelected to
                            //     the same state as m_nSelStart, not including m_nSelected.
                            //   Set m_nSelected to selected.

                            int nBegin = min( m_nSelStart, m_nSelected );
                            int nEnd = max( m_nSelStart, m_nSelected );

                            // The two ends do not need to be set here.

                            bool bLastSelected = m_Items.GetAt( m_nSelStart )->bSelected;
                            for( int i = nBegin + 1; i < nEnd; ++i )
                            {
                                DXUTListBoxItem *pItem = m_Items.GetAt( i );
                                pItem->bSelected = bLastSelected;
                            }

                            pSelItem->bSelected = true;

                            // Restore m_nSelected to the previous value
                            // This matches the Windows behavior

                            m_nSelected = m_nSelStart;
                        } else
                        {
                            // Simple click.  Clear all items and select the clicked
                            // item.


                            for( int i = 0; i < (int)m_Items.GetSize(); ++i )
                            {
                                DXUTListBoxItem *pItem = m_Items.GetAt( i );
                                pItem->bSelected = false;
                            }

                            pSelItem->bSelected = true;
                        }
                    }  // End of multi-selection case

                    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
                }

                return true;
            }
            break;

        case WM_LBUTTONUP:
        {
            ReleaseCapture();
            m_bDrag = false;

            if( m_nSelected != -1 )
            {
                // Set all items between m_nSelStart and m_nSelected to
                // the same state as m_nSelStart
                int nEnd = max( m_nSelStart, m_nSelected );

                for( int n = min( m_nSelStart, m_nSelected ) + 1; n < nEnd; ++n )
                    m_Items[n]->bSelected = m_Items[m_nSelStart]->bSelected;
                m_Items[m_nSelected]->bSelected = m_Items[m_nSelStart]->bSelected;

                // If m_nSelStart and m_nSelected are not the same,
                // the user has dragged the mouse to make a selection.
                // Notify the application of this.
                if( m_nSelStart != m_nSelected )
                    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
            }
            return false;
        }

        case WM_MOUSEMOVE:
            if( m_bDrag )
            {
                // Compute the index of the item below cursor

                int nItem;
                if( m_nTextHeight )
                    nItem = m_ScrollBar.GetTrackPos() + ( pt.y - m_rcText.top ) / m_nTextHeight;
                else
                    nItem = -1;

                // Only proceed if the cursor is on top of an item.

                if( nItem >= (int)m_ScrollBar.GetTrackPos() &&
                    nItem < (int)m_Items.GetSize() &&
                    nItem < m_ScrollBar.GetTrackPos() + m_ScrollBar.GetPageSize() )
                {
                    m_nSelected = nItem;
                    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
                } else
                if( nItem < (int)m_ScrollBar.GetTrackPos() )
                {
                    // User drags the mouse above window top
                    m_ScrollBar.Scroll( -1 );
                    m_nSelected = m_ScrollBar.GetTrackPos();
                    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
                } else
                if( nItem >= m_ScrollBar.GetTrackPos() + m_ScrollBar.GetPageSize() )
                {
                    // User drags the mouse below window bottom
                    m_ScrollBar.Scroll( 1 );
                    m_nSelected = min( (int)m_Items.GetSize(), m_ScrollBar.GetTrackPos() + m_ScrollBar.GetPageSize() ) - 1;
                    m_pDialog->SendEvent( EVENT_LISTBOX_SELECTION, true, this );
                }
            }
            break;

        case WM_MOUSEWHEEL:
        {
            UINT uLines;
            SystemParametersInfo( SPI_GETWHEELSCROLLLINES, 0, &uLines, 0 );
            int nScrollAmount = int((short)HIWORD(wParam)) / WHEEL_DELTA * uLines;
            m_ScrollBar.Scroll( -nScrollAmount );
            return true;
        }
    }

    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTListBox::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    if( m_bVisible == false )
        return;

    CDXUTElement* pElement = m_Elements.GetAt( 0 );
    pElement->TextureColor.Blend( DXUT_STATE_NORMAL, fElapsedTime );
    pElement->FontColor.Blend( DXUT_STATE_NORMAL, fElapsedTime );

    CDXUTElement* pSelElement = m_Elements.GetAt( 1 );
    pSelElement->TextureColor.Blend( DXUT_STATE_NORMAL, fElapsedTime );
    pSelElement->FontColor.Blend( DXUT_STATE_NORMAL, fElapsedTime );

    m_pDialog->DrawSprite( pElement, &m_rcBoundingBox );

    // Render the text
    if( m_Items.GetSize() > 0 )
    {
        // Find out the height of a single line of text
        RECT rc = m_rcText;
        RECT rcSel = m_rcSelection;
        rc.bottom = rc.top + DXUTGetGlobalDialogResourceManager()->GetFontNode( pElement->iFont )->nHeight;

        // Update the line height formation
        m_nTextHeight = rc.bottom - rc.top;

        static bool bSBInit;
        if( !bSBInit )
        {
            // Update the page size of the scroll bar
            if( m_nTextHeight )
                m_ScrollBar.SetPageSize( RectHeight( m_rcText ) / m_nTextHeight );
            else
                m_ScrollBar.SetPageSize( RectHeight( m_rcText ) );
            bSBInit = true;
        }

        rc.right = m_rcText.right;
        for( int i = m_ScrollBar.GetTrackPos(); i < (int)m_Items.GetSize(); ++i )
        {
            if( rc.bottom > m_rcText.bottom )
                break;

            DXUTListBoxItem *pItem = m_Items.GetAt( i );

            // Determine if we need to render this item with the
            // selected element.
            bool bSelectedStyle = false;

            if( !( m_dwStyle & MULTISELECTION ) && i == m_nSelected )
                bSelectedStyle = true;
            else
            if( m_dwStyle & MULTISELECTION )
            {
                if( m_bDrag &&
                    ( ( i >= m_nSelected && i < m_nSelStart ) ||
                      ( i <= m_nSelected && i > m_nSelStart ) ) )
                    bSelectedStyle = m_Items[m_nSelStart]->bSelected;
                else
                if( pItem->bSelected )
                    bSelectedStyle = true;
            }

            if( bSelectedStyle )
            {
                rcSel.top = rc.top; rcSel.bottom = rc.bottom;
                m_pDialog->DrawSprite( pSelElement, &rcSel );
                m_pDialog->DrawText( pItem->strText, pSelElement, &rc );
            }
            else
                m_pDialog->DrawText( pItem->strText, pElement, &rc );

            OffsetRect( &rc, 0, m_nTextHeight );
        }
    }

    // Render the scroll bar

    m_ScrollBar.Render( pd3dDevice, fElapsedTime );
}


// Helper class that help us automatically initialize and uninitialize external API.
// Important: C++ does not guaranteed the order global and static objects are
//            initialized in.  Therefore, do not use edit controls inside
//            a constructor.
class CExternalApiInitializer
{
public:
    CExternalApiInitializer()
    {
        CDXUTEditBox::CUniBuffer::InitializeUniscribe();
        CDXUTIMEEditBox::InitializeImm();
    }
    ~CExternalApiInitializer()
    {
        CDXUTEditBox::CUniBuffer::UninitializeUniscribe();
        CDXUTIMEEditBox::UninitializeImm();
    }
} EXTERNAL_API_INITIALIZER;

// Static member initialization
HINSTANCE CDXUTEditBox::CUniBuffer::s_hDll = NULL;
HRESULT (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptApplyDigitSubstitution)( const SCRIPT_DIGITSUBSTITUTE*, SCRIPT_CONTROL*, SCRIPT_STATE* )
    = Dummy_ScriptApplyDigitSubstitution;
HRESULT (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptStringAnalyse)( HDC, const void *, int, int, int, DWORD, int, SCRIPT_CONTROL*, SCRIPT_STATE*,
                                                    const int*, SCRIPT_TABDEF*, const BYTE*, SCRIPT_STRING_ANALYSIS* )
    = Dummy_ScriptStringAnalyse;
HRESULT (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptStringCPtoX)( SCRIPT_STRING_ANALYSIS, int, BOOL, int* )
    = Dummy_ScriptStringCPtoX;
HRESULT (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptStringXtoCP)( SCRIPT_STRING_ANALYSIS, int, int*, int* )
    = Dummy_ScriptStringXtoCP;
HRESULT (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptStringFree)( SCRIPT_STRING_ANALYSIS* )
    = Dummy_ScriptStringFree;
const SCRIPT_LOGATTR* (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptString_pLogAttr)( SCRIPT_STRING_ANALYSIS )
    = Dummy_ScriptString_pLogAttr;
const int* (WINAPI *CDXUTEditBox::CUniBuffer::_ScriptString_pcOutChars)( SCRIPT_STRING_ANALYSIS )
    = Dummy_ScriptString_pcOutChars;
bool CDXUTEditBox::s_bHideCaret;   // If true, we don't render the caret.



//--------------------------------------------------------------------------------------
// CDXUTEditBox class
//--------------------------------------------------------------------------------------

// When scrolling, EDITBOX_SCROLLEXTENT is reciprocal of the amount to scroll.
// If EDITBOX_SCROLLEXTENT = 4, then we scroll 1/4 of the control each time.
#define EDITBOX_SCROLLEXTENT 4

//--------------------------------------------------------------------------------------
CDXUTEditBox::CDXUTEditBox( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_EDITBOX;
    m_pDialog = pDialog;

    m_nBorder = 5;  // Default border width
    m_nSpacing = 4;  // Default spacing

    m_bCaretOn = true;
    m_dfBlink = GetCaretBlinkTime() * 0.001f;
    m_dfLastBlink = DXUTGetGlobalTimer()->GetAbsoluteTime();
    s_bHideCaret = false;
    m_nFirstVisible = 0;
    m_TextColor = D3DCOLOR_ARGB( 255, 16, 16, 16 );
    m_SelTextColor = D3DCOLOR_ARGB( 255, 255, 255, 255 );
    m_SelBkColor = D3DCOLOR_ARGB( 255, 40, 50, 92 );
    m_CaretColor = D3DCOLOR_ARGB( 255, 0, 0, 0 );
    m_nCaret = m_nSelStart = 0;
    m_bInsertMode = true;

    m_bMouseDrag = false;
}


//--------------------------------------------------------------------------------------
CDXUTEditBox::~CDXUTEditBox()
{
}


//--------------------------------------------------------------------------------------
// PlaceCaret: Set the caret to a character position, and adjust the scrolling if
//             necessary.
//--------------------------------------------------------------------------------------
void CDXUTEditBox::PlaceCaret( int nCP )
{
    assert( nCP >= 0 && nCP <= m_Buffer.GetTextSize() );
    m_nCaret = nCP;

    // Obtain the X offset of the character.
    int nX1st, nX, nX2, nXOffset;
    m_Buffer.CPtoX( m_nFirstVisible, FALSE, &nX1st );  // 1st visible char
    nXOffset = nX1st;
    m_Buffer.CPtoX( nCP, FALSE, &nX );  // LEAD
    // If nCP is the NULL terminator, get the leading edge instead of trailing.
    if( nCP == m_Buffer.GetTextSize() )
        nX2 = nX;
    else
        m_Buffer.CPtoX( nCP, TRUE, &nX2 );  // TRAIL

    // If the left edge of the char is smaller than the left edge of the 1st visible char,
    // we need to scroll left until this char is visible.
    if( nX < nX1st )
    {
        // Simply make the first visible character the char at the new caret position.
        m_nFirstVisible = nCP;
    }
    else
    // If the right of the character is bigger than the offset of the control's
    // right edge, we need to scroll right to this character.
    if( nX2 > nX1st + RectWidth( m_rcText ) )
    {
        // Compute the X of the new left-most pixel
        int nXNewLeft = nX2 - RectWidth( m_rcText );

        // Compute the char position of this character
        int nCPNew1st, nNewTrail;
        m_Buffer.XtoCP( nXNewLeft, &nCPNew1st, &nNewTrail );

        // If this coordinate is not on a character border,
        // start from the next character so that the caret
        // position does not fall outside the text rectangle.
        int nXNew1st;
        m_Buffer.CPtoX( nCPNew1st, FALSE, &nXNew1st );
        if( nXNew1st < nXNewLeft )
            ++nCPNew1st;

        m_nFirstVisible = nCPNew1st;
    }
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::ClearText()
{
    m_Buffer.Clear();
    PlaceCaret( 0 );
    m_nSelStart = 0;
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::SetText( LPCWSTR wszText, bool bSelected )
{
    assert( wszText != NULL );

    m_Buffer.SetText( wszText );
    // Move the caret to the end of the text
    PlaceCaret( m_Buffer.GetTextSize() );
    m_nSelStart = bSelected ? 0 : m_nCaret;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTEditBox::GetTextCopy( LPWSTR strDest, UINT bufferCount )
{
    assert( strDest );

    wcsncpy( strDest, m_Buffer.GetBuffer(), bufferCount );
    *(strDest + bufferCount - 1) = L'\0';

    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::DeleteSelectionText()
{
    int nFirst = min( m_nCaret, m_nSelStart );
    int nLast = max( m_nCaret, m_nSelStart );
    // Update caret and selection
    PlaceCaret( nFirst );
    m_nSelStart = m_nCaret;
    // Remove the characters
    for( int i = nFirst; i < nLast; ++i )
        m_Buffer.RemoveChar( nFirst );
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::UpdateRects()
{
    CDXUTControl::UpdateRects();

    // Update the text rectangle
    m_rcText = m_rcBoundingBox;
    // First inflate by m_nBorder to compute render rects
    InflateRect( &m_rcText, -m_nBorder, -m_nBorder );

    // Update the render rectangles
    m_rcRender[0] = m_rcText;
    SetRect( &m_rcRender[1], m_rcBoundingBox.left, m_rcBoundingBox.top, m_rcText.left, m_rcText.top );
    SetRect( &m_rcRender[2], m_rcText.left, m_rcBoundingBox.top, m_rcText.right, m_rcText.top );
    SetRect( &m_rcRender[3], m_rcText.right, m_rcBoundingBox.top, m_rcBoundingBox.right, m_rcText.top );
    SetRect( &m_rcRender[4], m_rcBoundingBox.left, m_rcText.top, m_rcText.left, m_rcText.bottom );
    SetRect( &m_rcRender[5], m_rcText.right, m_rcText.top, m_rcBoundingBox.right, m_rcText.bottom );
    SetRect( &m_rcRender[6], m_rcBoundingBox.left, m_rcText.bottom, m_rcText.left, m_rcBoundingBox.bottom );
    SetRect( &m_rcRender[7], m_rcText.left, m_rcText.bottom, m_rcText.right, m_rcBoundingBox.bottom );
    SetRect( &m_rcRender[8], m_rcText.right, m_rcText.bottom, m_rcBoundingBox.right, m_rcBoundingBox.bottom );

    // Inflate further by m_nSpacing
    InflateRect( &m_rcText, -m_nSpacing, -m_nSpacing );
}


void CDXUTEditBox::CopyToClipboard()
{
    // Copy the selection text to the clipboard
    if( m_nCaret != m_nSelStart && OpenClipboard( NULL ) )
    {
        EmptyClipboard();

        HGLOBAL hBlock = GlobalAlloc( GMEM_MOVEABLE, sizeof(WCHAR) * ( m_Buffer.GetTextSize() + 1 ) );
        if( hBlock )
        {
            WCHAR *pwszText = (WCHAR*)GlobalLock( hBlock );
            if( pwszText )
            {
                int nFirst = min( m_nCaret, m_nSelStart );
                int nLast = max( m_nCaret, m_nSelStart );
                if( nLast - nFirst > 0 )
                    CopyMemory( pwszText, m_Buffer.GetBuffer() + nFirst, (nLast - nFirst) * sizeof(WCHAR) );
                pwszText[nLast - nFirst] = L'\0';  // Terminate it
                GlobalUnlock( hBlock );
            }
            SetClipboardData( CF_UNICODETEXT, hBlock );
        }
        CloseClipboard();
        // We must not free the object until CloseClipboard is called.
        if( hBlock )
            GlobalFree( hBlock );
    }
}


void CDXUTEditBox::PasteFromClipboard()
{
    DeleteSelectionText();

    if( OpenClipboard( NULL ) )
    {
        HANDLE handle = GetClipboardData( CF_UNICODETEXT );
        if( handle )
        {
            // Convert the ANSI string to Unicode, then
            // insert to our buffer.
            WCHAR *pwszText = (WCHAR*)GlobalLock( handle );
            if( pwszText )
            {
                // Copy all characters up to null.
                if( m_Buffer.InsertString( m_nCaret, pwszText ) )
                    PlaceCaret( m_nCaret + lstrlenW( pwszText ) );
                m_nSelStart = m_nCaret;
                GlobalUnlock( handle );
            }
        }
        CloseClipboard();
    }
}


//--------------------------------------------------------------------------------------
bool CDXUTEditBox::HandleKeyboard( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    bool bHandled = false;

    switch( uMsg )
    {
        case WM_KEYDOWN:
        {
            switch( wParam )
            {
                case VK_HOME:
                    PlaceCaret( 0 );
                    if( GetKeyState( VK_SHIFT ) >= 0 )
                        // Shift is not down. Update selection
                        // start along with the caret.
                        m_nSelStart = m_nCaret;
                    ResetCaretBlink();
                    bHandled = true;
                    break;

                case VK_END:
                    PlaceCaret( m_Buffer.GetTextSize() );
                    if( GetKeyState( VK_SHIFT ) >= 0 )
                        // Shift is not down. Update selection
                        // start along with the caret.
                        m_nSelStart = m_nCaret;
                    ResetCaretBlink();
                    bHandled = true;
                    break;

                case VK_INSERT:
                    if( GetKeyState( VK_CONTROL ) < 0 )
                    {
                        // Control Insert. Copy to clipboard
                        CopyToClipboard();
                    } else
                    if( GetKeyState( VK_SHIFT ) < 0 )
                    {
                        // Shift Insert. Paste from clipboard
                        PasteFromClipboard();
                    } else
                    {
                        // Toggle caret insert mode
                        m_bInsertMode = !m_bInsertMode;
                    }
                    break;

                case VK_DELETE:
                    // Check if there is a text selection.
                    if( m_nCaret != m_nSelStart )
                    {
                        DeleteSelectionText();
                        m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                    }
                    else
                    {
                        // Deleting one character
                        if( m_Buffer.RemoveChar( m_nCaret ) )
                            m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                    }
                    ResetCaretBlink();
                    bHandled = true;
                    break;

                case VK_LEFT:
                    if( GetKeyState( VK_CONTROL ) < 0 )
                    {
                        // Control is down. Move the caret to a new item
                        // instead of a character.
                        m_Buffer.GetPriorItemPos( m_nCaret, &m_nCaret );
                        PlaceCaret( m_nCaret );
                    }
                    else
                    if( m_nCaret > 0 )
                        PlaceCaret( m_nCaret - 1 );
                    if( GetKeyState( VK_SHIFT ) >= 0 )
                        // Shift is not down. Update selection
                        // start along with the caret.
                        m_nSelStart = m_nCaret;
                    ResetCaretBlink();
                    bHandled = true;
                    break;

                case VK_RIGHT:
                    if( GetKeyState( VK_CONTROL ) < 0 )
                    {
                        // Control is down. Move the caret to a new item
                        // instead of a character.
                        m_Buffer.GetNextItemPos( m_nCaret, &m_nCaret );
                        PlaceCaret( m_nCaret );
                    }
                    else
                    if( m_nCaret < m_Buffer.GetTextSize() )
                        PlaceCaret( m_nCaret + 1 );
                    if( GetKeyState( VK_SHIFT ) >= 0 )
                        // Shift is not down. Update selection
                        // start along with the caret.
                        m_nSelStart = m_nCaret;
                    ResetCaretBlink();
                    bHandled = true;
                    break;

                case VK_UP:
                case VK_DOWN:
                    // Trap up and down arrows so that the dialog
                    // does not switch focus to another control.
                    bHandled = true;
                    break;

                default:
                    bHandled = wParam != VK_ESCAPE;  // Let the application handle Esc.
            }
        }
    }
    return bHandled;
}


//--------------------------------------------------------------------------------------
bool CDXUTEditBox::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            if( !m_bHasFocus )
                m_pDialog->RequestFocus( this );

            if( !ContainsPoint( pt ) )
                return false;

            m_bMouseDrag = true;
            SetCapture( DXUTGetHWND() );
            // Determine the character corresponding to the coordinates.
            int nCP, nTrail, nX1st;
            m_Buffer.CPtoX( m_nFirstVisible, FALSE, &nX1st );  // X offset of the 1st visible char
            if( SUCCEEDED( m_Buffer.XtoCP( pt.x - m_rcText.left + nX1st, &nCP, &nTrail ) ) )
            {
                // Cap at the NULL character.
                if( nTrail && nCP < m_Buffer.GetTextSize() )
                    PlaceCaret( nCP + 1 );
                else
                    PlaceCaret( nCP );
                m_nSelStart = m_nCaret;
                ResetCaretBlink();
            }
            return true;
        }

        case WM_LBUTTONUP:
            ReleaseCapture();
            m_bMouseDrag = false;
            break;

        case WM_MOUSEMOVE:
            if( m_bMouseDrag )
            {
                // Determine the character corresponding to the coordinates.
                int nCP, nTrail, nX1st;
                m_Buffer.CPtoX( m_nFirstVisible, FALSE, &nX1st );  // X offset of the 1st visible char
                if( SUCCEEDED( m_Buffer.XtoCP( pt.x - m_rcText.left + nX1st, &nCP, &nTrail ) ) )
                {
                    // Cap at the NULL character.
                    if( nTrail && nCP < m_Buffer.GetTextSize() )
                        PlaceCaret( nCP + 1 );
                    else
                        PlaceCaret( nCP );
                }
            }
            break;
    }

    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::OnFocusIn()
{
    CDXUTControl::OnFocusIn();

    ResetCaretBlink();
}


//--------------------------------------------------------------------------------------
bool CDXUTEditBox::MsgProc( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_CHAR:
        {
            switch( (WCHAR)wParam )
            {
                // Backspace
                case VK_BACK:
                {
                    // If there's a selection, treat this
                    // like a delete key.
                    if( m_nCaret != m_nSelStart )
                    {
                        DeleteSelectionText();
                        m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                    }
                    else
                    if( m_nCaret > 0 )
                    {
                        // Move the caret, then delete the char.
                        PlaceCaret( m_nCaret - 1 );
                        m_nSelStart = m_nCaret;
                        m_Buffer.RemoveChar( m_nCaret );
                        m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                    }
                    ResetCaretBlink();
                    break;
                }

                case 24:        // Ctrl-X Cut
                case VK_CANCEL: // Ctrl-C Copy
                {
                    CopyToClipboard();

                    // If the key is Ctrl-X, delete the selection too.
                    if( (WCHAR)wParam == 24 )
                    {
                        DeleteSelectionText();
                        m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                    }

                    break;
                }

                // Ctrl-V Paste
                case 22:
                {
                    PasteFromClipboard();
                    m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                    break;
                }

                // Ctrl-A Select All
                case 1:
                    if( m_nSelStart == m_nCaret )
                    {
                        m_nSelStart = 0;
                        PlaceCaret( m_Buffer.GetTextSize() );
                    }
                    break;

                case VK_RETURN:
                    // Invoke the callback when the user presses Enter.
                    m_pDialog->SendEvent( EVENT_EDITBOX_STRING, true, this );
                    break;

                // Junk characters we don't want in the string
                case 26:  // Ctrl Z
                case 2:   // Ctrl B
                case 14:  // Ctrl N
                case 19:  // Ctrl S
                case 4:   // Ctrl D
                case 6:   // Ctrl F
                case 7:   // Ctrl G
                case 10:  // Ctrl J
                case 11:  // Ctrl K
                case 12:  // Ctrl L
                case 17:  // Ctrl Q
                case 23:  // Ctrl W
                case 5:   // Ctrl E
                case 18:  // Ctrl R
                case 20:  // Ctrl T
                case 25:  // Ctrl Y
                case 21:  // Ctrl U
                case 9:   // Ctrl I
                case 15:  // Ctrl O
                case 16:  // Ctrl P
                case 27:  // Ctrl [
                case 29:  // Ctrl ]
                case 28:  // Ctrl \ 
                    break;

                default:
                {
                    // If there's a selection and the user
                    // starts to type, the selection should
                    // be deleted.
                    if( m_nCaret != m_nSelStart )
                        DeleteSelectionText();

                    // If we are in overwrite mode and there is already
                    // a char at the caret's position, simply replace it.
                    // Otherwise, we insert the char as normal.
                    if( !m_bInsertMode && m_nCaret < m_Buffer.GetTextSize() )
                    {
                        m_Buffer[m_nCaret] = (WCHAR)wParam;
                        PlaceCaret( m_nCaret + 1 );
                        m_nSelStart = m_nCaret;
                    } else
                    {
                        // Insert the char
                        if( m_Buffer.InsertChar( m_nCaret, (WCHAR)wParam ) )
                        {
                            PlaceCaret( m_nCaret + 1 );
                            m_nSelStart = m_nCaret;
                        }
                    }
                    ResetCaretBlink();
                    m_pDialog->SendEvent( EVENT_EDITBOX_CHANGE, true, this );
                }
            }
            return true;
        }
    }
    return false;
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    if( m_bVisible == false )
        return;

    HRESULT hr;
    int nSelStartX = 0, nCaretX = 0;  // Left and right X cordinates of the selection region

    CDXUTElement* pElement = GetElement( 0 );
    if( pElement )
    {
        m_Buffer.SetFontIndex( pElement->iFont );
        PlaceCaret( m_nCaret );  // Call PlaceCaret now that we have the DC,
                                    // so that scrolling can be handled.
    }

    // Render the control graphics
    for( int e = 0; e < 9; ++e )
    {
        CDXUTElement* pElement = m_Elements.GetAt( e );
        pElement->TextureColor.Blend( DXUT_STATE_NORMAL, fElapsedTime );

        m_pDialog->DrawSprite( pElement, &m_rcRender[e] );
    }

    //
    // Compute the X coordinates of the first visible character.
    //
    int nXFirst;
    m_Buffer.CPtoX( m_nFirstVisible, FALSE, &nXFirst );

    //
    // Compute the X coordinates of the selection rectangle
    //
    hr = m_Buffer.CPtoX( m_nCaret, FALSE, &nCaretX );
    if( m_nCaret != m_nSelStart )
        hr = m_Buffer.CPtoX( m_nSelStart, FALSE, &nSelStartX );
    else
        nSelStartX = nCaretX;

    //
    // Render the selection rectangle
    //
    RECT rcSelection;  // Make this available for rendering selected text
    if( m_nCaret != m_nSelStart )
    {
        int nSelLeftX = nCaretX, nSelRightX = nSelStartX;
        // Swap if left is bigger than right
        if( nSelLeftX > nSelRightX )
            { int nTemp = nSelLeftX; nSelLeftX = nSelRightX; nSelRightX = nTemp; }

        SetRect( &rcSelection, nSelLeftX, m_rcText.top, nSelRightX, m_rcText.bottom );
        OffsetRect( &rcSelection, m_rcText.left - nXFirst, 0 );
        IntersectRect( &rcSelection, &m_rcText, &rcSelection );
        m_pDialog->DrawRect( &rcSelection, m_SelBkColor );
    }

    //
    // Render the text
    //
    // Element 0 for text
    m_Elements.GetAt( 0 )->FontColor.Current = m_TextColor;
    m_pDialog->DrawText( m_Buffer.GetBuffer() + m_nFirstVisible, m_Elements.GetAt( 0 ), &m_rcText );

    // Render the selected text
    if( m_nCaret != m_nSelStart )
    {
        int nFirstToRender = max( m_nFirstVisible, min( m_nSelStart, m_nCaret ) );
        int nNumChatToRender = max( m_nSelStart, m_nCaret ) - nFirstToRender;
        m_Elements.GetAt( 0 )->FontColor.Current = m_SelTextColor;
        m_pDialog->DrawText( m_Buffer.GetBuffer() + nFirstToRender,
                             m_Elements.GetAt( 0 ), &rcSelection, false, nNumChatToRender );
    }

    //
    // Blink the caret
    //
    if( DXUTGetGlobalTimer()->GetAbsoluteTime() - m_dfLastBlink >= m_dfBlink )
    {
        m_bCaretOn = !m_bCaretOn;
        m_dfLastBlink = DXUTGetGlobalTimer()->GetAbsoluteTime();
    }

    //
    // Render the caret if this control has the focus
    //
    if( m_bHasFocus && m_bCaretOn && !s_bHideCaret )
    {
        // Start the rectangle with insert mode caret
        RECT rcCaret = { m_rcText.left - nXFirst + nCaretX - 1, m_rcText.top,
                         m_rcText.left - nXFirst + nCaretX + 1, m_rcText.bottom };

        // If we are in overwrite mode, adjust the caret rectangle
        // to fill the entire character.
        if( !m_bInsertMode )
        {
            // Obtain the right edge X coord of the current character
            int nRightEdgeX;
            m_Buffer.CPtoX( m_nCaret, TRUE, &nRightEdgeX );
            rcCaret.right = m_rcText.left - nXFirst + nRightEdgeX;
        }

        m_pDialog->DrawRect( &rcCaret, m_CaretColor );
    }
}


#define IN_FLOAT_CHARSET( c ) \
    ( (c) == L'-' || (c) == L'.' || ( (c) >= L'0' && (c) <= L'9' ) )

void CDXUTEditBox::ParseFloatArray( float *pNumbers, int nCount )
{
    int nWritten = 0;  // Number of floats written
    const WCHAR *pToken, *pEnd;
    WCHAR wszToken[60];

    pToken = m_Buffer.GetBuffer();
    while( nWritten < nCount && *pToken != L'\0' )
    {
        // Skip leading spaces
        while( *pToken == L' ' )
            ++pToken;

        if( *pToken == L'\0' )
            break;

        // Locate the end of number
        pEnd = pToken;
        while( IN_FLOAT_CHARSET( *pEnd ) )
            ++pEnd;

        // Copy the token to our buffer
        int nTokenLen = min( sizeof(wszToken) / sizeof(wszToken[0]) - 1, int(pEnd - pToken) );
        wcsncpy( wszToken, pToken, nTokenLen );
        wszToken[nTokenLen] = L'\0';
        *pNumbers = (float)wcstod( wszToken, NULL );
        ++nWritten;
        ++pNumbers;
        pToken = pEnd;
    }
}


void CDXUTEditBox::SetTextFloatArray( const float *pNumbers, int nCount )
{
    WCHAR wszBuffer[512];
    WCHAR *pNext = wszBuffer;

    for( int i = 0; i < nCount; ++i )
    {
        pNext += _snwprintf( pNext, 512 - (pNext - wszBuffer), L"%.4f ", pNumbers[i] );
    }

    // Don't want the last space
    if( nCount > 0 )
        *(pNext - 1) = L'\0';
    else
        *pNext = L'\0';

    SetText( wszBuffer );
}


//--------------------------------------------------------------------------------------
// CDXUTIMEEditBox class
//--------------------------------------------------------------------------------------
// IME constants
#define CHT_IMEFILENAME1    "TINTLGNT.IME" // New Phonetic
#define CHT_IMEFILENAME2    "CINTLGNT.IME" // New Chang Jie
#define CHT_IMEFILENAME3    "MSTCIPHA.IME" // Phonetic 5.1
#define CHS_IMEFILENAME1    "PINTLGNT.IME" // MSPY1.5/2/3
#define CHS_IMEFILENAME2    "MSSCIPYA.IME" // MSPY3 for OfficeXP

#define LANG_CHT            MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL)
#define LANG_CHS            MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED)
#define _CHT_HKL            ( (HKL)(INT_PTR)0xE0080404 ) // New Phonetic
#define _CHT_HKL2           ( (HKL)(INT_PTR)0xE0090404 ) // New Chang Jie
#define _CHS_HKL            ( (HKL)(INT_PTR)0xE00E0804 ) // MSPY
#define MAKEIMEVERSION( major, minor ) \
    ( (DWORD)( ( (BYTE)( major ) << 24 ) | ( (BYTE)( minor ) << 16 ) ) )

#define IMEID_CHT_VER42 ( LANG_CHT | MAKEIMEVERSION( 4, 2 ) )	// New(Phonetic/ChanJie)IME98  : 4.2.x.x // Win98
#define IMEID_CHT_VER43 ( LANG_CHT | MAKEIMEVERSION( 4, 3 ) )	// New(Phonetic/ChanJie)IME98a : 4.3.x.x // Win2k
#define IMEID_CHT_VER44 ( LANG_CHT | MAKEIMEVERSION( 4, 4 ) )	// New ChanJie IME98b          : 4.4.x.x // WinXP
#define IMEID_CHT_VER50 ( LANG_CHT | MAKEIMEVERSION( 5, 0 ) )	// New(Phonetic/ChanJie)IME5.0 : 5.0.x.x // WinME
#define IMEID_CHT_VER51 ( LANG_CHT | MAKEIMEVERSION( 5, 1 ) )	// New(Phonetic/ChanJie)IME5.1 : 5.1.x.x // IME2002(w/OfficeXP)
#define IMEID_CHT_VER52 ( LANG_CHT | MAKEIMEVERSION( 5, 2 ) )	// New(Phonetic/ChanJie)IME5.2 : 5.2.x.x // IME2002a(w/Whistler)
#define IMEID_CHT_VER60 ( LANG_CHT | MAKEIMEVERSION( 6, 0 ) )	// New(Phonetic/ChanJie)IME6.0 : 6.0.x.x // IME XP(w/WinXP SP1)
#define IMEID_CHS_VER41	( LANG_CHS | MAKEIMEVERSION( 4, 1 ) )	// MSPY1.5	// SCIME97 or MSPY1.5 (w/Win98, Office97)
#define IMEID_CHS_VER42	( LANG_CHS | MAKEIMEVERSION( 4, 2 ) )	// MSPY2	// Win2k/WinME
#define IMEID_CHS_VER53	( LANG_CHS | MAKEIMEVERSION( 5, 3 ) )	// MSPY3	// WinXP

// Function pointers
INPUTCONTEXT* (WINAPI * CDXUTIMEEditBox::_ImmLockIMC)( HIMC )
    = CDXUTIMEEditBox::Dummy_ImmLockIMC;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmUnlockIMC)( HIMC )
    = CDXUTIMEEditBox::Dummy_ImmUnlockIMC;
LPVOID (WINAPI * CDXUTIMEEditBox::_ImmLockIMCC)( HIMCC )
    = CDXUTIMEEditBox::Dummy_ImmLockIMCC;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmUnlockIMCC)( HIMCC )
    = CDXUTIMEEditBox::Dummy_ImmUnlockIMCC;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmDisableTextFrameService)( DWORD )
    = CDXUTIMEEditBox::Dummy_ImmDisableTextFrameService;
LONG (WINAPI * CDXUTIMEEditBox::_ImmGetCompositionStringW)( HIMC, DWORD, LPVOID, DWORD )
    = CDXUTIMEEditBox::Dummy_ImmGetCompositionStringW;
DWORD (WINAPI * CDXUTIMEEditBox::_ImmGetCandidateListW)( HIMC, DWORD, LPCANDIDATELIST, DWORD )
    = CDXUTIMEEditBox::Dummy_ImmGetCandidateListW;
HIMC (WINAPI * CDXUTIMEEditBox::_ImmGetContext)( HWND )
    = CDXUTIMEEditBox::Dummy_ImmGetContext;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmReleaseContext)( HWND, HIMC )
    = CDXUTIMEEditBox::Dummy_ImmReleaseContext;
HIMC (WINAPI * CDXUTIMEEditBox::_ImmAssociateContext)( HWND, HIMC )
    = CDXUTIMEEditBox::Dummy_ImmAssociateContext;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmGetOpenStatus)( HIMC )
    = CDXUTIMEEditBox::Dummy_ImmGetOpenStatus;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmSetOpenStatus)( HIMC, BOOL )
    = CDXUTIMEEditBox::Dummy_ImmSetOpenStatus;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmGetConversionStatus)( HIMC, LPDWORD, LPDWORD )
    = CDXUTIMEEditBox::Dummy_ImmGetConversionStatus;
HWND (WINAPI * CDXUTIMEEditBox::_ImmGetDefaultIMEWnd)( HWND )
    = CDXUTIMEEditBox::Dummy_ImmGetDefaultIMEWnd;
UINT (WINAPI * CDXUTIMEEditBox::_ImmGetIMEFileNameA)( HKL, LPSTR, UINT )
    = CDXUTIMEEditBox::Dummy_ImmGetIMEFileNameA;
UINT (WINAPI * CDXUTIMEEditBox::_ImmGetVirtualKey)( HWND )
    = CDXUTIMEEditBox::Dummy_ImmGetVirtualKey;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmNotifyIME)( HIMC, DWORD, DWORD, DWORD )
    = CDXUTIMEEditBox::Dummy_ImmNotifyIME;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmSetConversionStatus)( HIMC, DWORD, DWORD )
    = CDXUTIMEEditBox::Dummy_ImmSetConversionStatus;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmSimulateHotKey)( HWND, DWORD )
    = CDXUTIMEEditBox::Dummy_ImmSimulateHotKey;
BOOL (WINAPI * CDXUTIMEEditBox::_ImmIsIME)( HKL )
    = CDXUTIMEEditBox::Dummy_ImmIsIME;
// Traditional Chinese IME
UINT (WINAPI * CDXUTIMEEditBox::_GetReadingString)( HIMC, UINT, LPWSTR, PINT, BOOL*, PUINT )
    = CDXUTIMEEditBox::Dummy_GetReadingString;
BOOL (WINAPI * CDXUTIMEEditBox::_ShowReadingWindow)( HIMC, BOOL )
    = CDXUTIMEEditBox::Dummy_ShowReadingWindow;

BOOL (APIENTRY * CDXUTIMEEditBox::_VerQueryValueA)( const LPVOID, LPSTR, LPVOID *, PUINT )
    = CDXUTIMEEditBox::Dummy_VerQueryValueA;
BOOL (APIENTRY * CDXUTIMEEditBox::_GetFileVersionInfoA)( LPSTR, DWORD, DWORD, LPVOID )
    = CDXUTIMEEditBox::Dummy_GetFileVersionInfoA;
DWORD (APIENTRY * CDXUTIMEEditBox::_GetFileVersionInfoSizeA)( LPSTR, LPDWORD )
    = CDXUTIMEEditBox::Dummy_GetFileVersionInfoSizeA;

HINSTANCE CDXUTIMEEditBox::s_hDllImm32;      // IMM32 DLL handle
HINSTANCE CDXUTIMEEditBox::s_hDllVer;        // Version DLL handle
HKL       CDXUTIMEEditBox::s_hklCurrent;     // Current keyboard layout of the process
bool      CDXUTIMEEditBox::s_bVerticalCand;  // Indicates that the candidates are listed vertically
WCHAR     CDXUTIMEEditBox::s_aszIndicator[5][3] = // String to draw to indicate current input locale
            {
                L"En",
                L"\x7B80",
                L"\x7E41",
                L"\xAC00",
                L"\x3042",
            };
LPWSTR    CDXUTIMEEditBox::s_wszCurrIndicator   // Points to an indicator string that corresponds to current input locale
            = CDXUTIMEEditBox::s_aszIndicator[0];
bool      CDXUTIMEEditBox::s_bInsertOnType;     // Insert the character as soon as a key is pressed (Korean behavior)
HINSTANCE CDXUTIMEEditBox::s_hDllIme;           // Instance handle of the current IME module
HIMC      CDXUTIMEEditBox::s_hImcDef;           // Default input context
CDXUTIMEEditBox::IMESTATE  CDXUTIMEEditBox::s_ImeState = IMEUI_STATE_OFF;
bool      CDXUTIMEEditBox::s_bEnableImeSystem;  // Whether the IME system is active
POINT     CDXUTIMEEditBox::s_ptCompString;      // Composition string position. Updated every frame.
int       CDXUTIMEEditBox::s_nCompCaret;
int       CDXUTIMEEditBox::s_nFirstTargetConv;  // Index of the first target converted char in comp string.  If none, -1.
CDXUTEditBox::CUniBuffer CDXUTIMEEditBox::s_CompString = CDXUTEditBox::CUniBuffer( MAX_COMPSTRING_SIZE );
BYTE      CDXUTIMEEditBox::s_abCompStringAttr[MAX_COMPSTRING_SIZE];
DWORD     CDXUTIMEEditBox::s_adwCompStringClause[MAX_COMPSTRING_SIZE];
WCHAR     CDXUTIMEEditBox::s_wszReadingString[32];
CDXUTIMEEditBox::CCandList CDXUTIMEEditBox::s_CandList;       // Data relevant to the candidate list
bool      CDXUTIMEEditBox::s_bShowReadingWindow; // Indicates whether reading window is visible
bool      CDXUTIMEEditBox::s_bHorizontalReading; // Indicates whether the reading window is vertical or horizontal
bool      CDXUTIMEEditBox::s_bChineseIME;
CGrowableArray< CDXUTIMEEditBox::CInputLocale > CDXUTIMEEditBox::s_Locale; // Array of loaded keyboard layout on system


//--------------------------------------------------------------------------------------
CDXUTIMEEditBox::CDXUTIMEEditBox( CDXUTDialog *pDialog )
{
    m_Type = DXUT_CONTROL_IMEEDITBOX;
    m_pDialog = pDialog;

    s_bEnableImeSystem = true;
    m_nIndicatorWidth = 0;
    m_ReadingColor = D3DCOLOR_ARGB( 188, 255, 255, 255 );
    m_ReadingWinColor = D3DCOLOR_ARGB( 128, 0, 0, 0 );
    m_ReadingSelColor = D3DCOLOR_ARGB( 255, 255, 0, 0 );
    m_ReadingSelBkColor = D3DCOLOR_ARGB( 128, 80, 80, 80 );
    m_CandidateColor = D3DCOLOR_ARGB( 255, 200, 200, 200 );
    m_CandidateWinColor = D3DCOLOR_ARGB( 128, 0, 0, 0 );
    m_CandidateSelColor = D3DCOLOR_ARGB( 255, 255, 255, 255 );
    m_CandidateSelBkColor = D3DCOLOR_ARGB( 128, 158, 158, 158 );
    m_CompColor = D3DCOLOR_ARGB( 255, 200, 200, 255 );
    m_CompWinColor = D3DCOLOR_ARGB( 198, 0, 0, 0 );
    m_CompCaretColor = D3DCOLOR_ARGB( 255, 255, 255, 255 );
    m_CompTargetColor = D3DCOLOR_ARGB( 255, 255, 255, 255 );
    m_CompTargetBkColor = D3DCOLOR_ARGB( 255, 150, 150, 150 );
    m_CompTargetNonColor = D3DCOLOR_ARGB( 255, 255, 255, 0 );
    m_CompTargetNonBkColor = D3DCOLOR_ARGB( 255, 150, 150, 150 );
    m_IndicatorImeColor = D3DCOLOR_ARGB( 255, 255, 255, 255 );
    m_IndicatorEngColor = D3DCOLOR_ARGB( 255, 0, 0, 0 );
    m_IndicatorBkColor = D3DCOLOR_ARGB( 255, 128, 128, 128 );
}


//--------------------------------------------------------------------------------------
CDXUTIMEEditBox::~CDXUTIMEEditBox()
{
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::SendKey( BYTE nVirtKey )
{
    keybd_event( nVirtKey, 0, 0,               0 );
    keybd_event( nVirtKey, 0, KEYEVENTF_KEYUP, 0 );
}


//--------------------------------------------------------------------------------------
// Called by CDXUTResourceCache's OnCreateDevice.  This gives the class a
// chance to initialize its default input context associated with the app window.
HRESULT CDXUTIMEEditBox::StaticOnCreateDevice()
{
    // Save the default input context
    s_hImcDef = _ImmGetContext( DXUTGetHWND() );
    _ImmReleaseContext( DXUTGetHWND(), s_hImcDef );

    return S_OK;
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::UpdateRects()
{
    // Temporary adjust m_width so that CDXUTEditBox can compute
    // the correct rects for its rendering since we need to make space
    // for the indicator button
    int nWidth = m_width;
    m_width -= m_nIndicatorWidth + m_nBorder * 2; // Make room for the indicator button
    CDXUTEditBox::UpdateRects();
    m_width = nWidth;  // Restore

    // Compute the indicator button rectangle
    SetRect( &m_rcIndicator, m_rcBoundingBox.right, m_rcBoundingBox.top, m_x + m_width, m_rcBoundingBox.bottom );
//    InflateRect( &m_rcIndicator, -m_nBorder, -m_nBorder );
    m_rcBoundingBox.right = m_rcBoundingBox.left + m_width;
}


//--------------------------------------------------------------------------------------
//	GetImeId( UINT uIndex )
//		returns 
//	returned value:
//	0: In the following cases
//		- Non Chinese IME input locale
//		- Older Chinese IME
//		- Other error cases
//
//	Othewise:
//      When uIndex is 0 (default)
//			bit 31-24:	Major version
//			bit 23-16:	Minor version
//			bit 15-0:	Language ID
//		When uIndex is 1
//			pVerFixedInfo->dwFileVersionLS
//
//	Use IMEID_VER and IMEID_LANG macro to extract version and language information.
//	

// We define the locale-invariant ID ourselves since it doesn't exist prior to WinXP
// For more information, see the CompareString() reference.
#define LCID_INVARIANT MAKELCID(MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), SORT_DEFAULT)

DWORD CDXUTIMEEditBox::GetImeId( UINT uIndex )
{
    static HKL hklPrev = 0;
    static DWORD dwID[2] = { 0, 0 };  // Cache the result
    
    DWORD   dwVerSize;
    DWORD   dwVerHandle;
    LPVOID  lpVerBuffer;
    LPVOID  lpVerData;
    UINT    cbVerData;
    char    szTmp[1024];

    if( uIndex >= sizeof( dwID ) / sizeof( dwID[0] ) )
        return 0;

    if( hklPrev == s_hklCurrent )
        return dwID[uIndex];

    hklPrev = s_hklCurrent;  // Save for the next invocation

    // Check if we are using an older Chinese IME
    if( !( ( s_hklCurrent == _CHT_HKL ) || ( s_hklCurrent == _CHT_HKL2 ) || ( s_hklCurrent == _CHS_HKL ) ) )
    {
        dwID[0] = dwID[1] = 0;
        return dwID[uIndex];
    }

    // Obtain the IME file name
    if ( !_ImmGetIMEFileNameA( s_hklCurrent, szTmp, ( sizeof(szTmp) / sizeof(szTmp[0]) ) - 1 ) )
    {
        dwID[0] = dwID[1] = 0;
        return dwID[uIndex];
    }

    // Check for IME that doesn't implement reading string API
    if ( !_GetReadingString )
    {
        if( ( CompareStringA( LCID_INVARIANT, NORM_IGNORECASE, szTmp, -1, CHT_IMEFILENAME1, -1 ) != CSTR_EQUAL ) &&
            ( CompareStringA( LCID_INVARIANT, NORM_IGNORECASE, szTmp, -1, CHT_IMEFILENAME2, -1 ) != CSTR_EQUAL ) &&
            ( CompareStringA( LCID_INVARIANT, NORM_IGNORECASE, szTmp, -1, CHT_IMEFILENAME3, -1 ) != CSTR_EQUAL ) &&
            ( CompareStringA( LCID_INVARIANT, NORM_IGNORECASE, szTmp, -1, CHS_IMEFILENAME1, -1 ) != CSTR_EQUAL ) &&
            ( CompareStringA( LCID_INVARIANT, NORM_IGNORECASE, szTmp, -1, CHS_IMEFILENAME2, -1 ) != CSTR_EQUAL ) )
        {
            dwID[0] = dwID[1] = 0;
            return dwID[uIndex];
        }
    }

    dwVerSize = _GetFileVersionInfoSizeA( szTmp, &dwVerHandle );
    if( dwVerSize )
    {
        lpVerBuffer = HeapAlloc( GetProcessHeap(), 0, dwVerSize );
        if( lpVerBuffer )
        {
            if( _GetFileVersionInfoA( szTmp, dwVerHandle, dwVerSize, lpVerBuffer ) )
            {
                if( _VerQueryValueA( lpVerBuffer, "\\", &lpVerData, &cbVerData ) )
                {
                    DWORD dwVer = ( (VS_FIXEDFILEINFO*)lpVerData )->dwFileVersionMS;
                    dwVer = ( dwVer & 0x00ff0000 ) << 8 | ( dwVer & 0x000000ff ) << 16;
                    if( _GetReadingString
                        ||
                        ( GetLanguage() == LANG_CHT &&
                          ( dwVer == MAKEIMEVERSION(4, 2) || 
                            dwVer == MAKEIMEVERSION(4, 3) || 
                            dwVer == MAKEIMEVERSION(4, 4) || 
                            dwVer == MAKEIMEVERSION(5, 0) ||
                            dwVer == MAKEIMEVERSION(5, 1) ||
                            dwVer == MAKEIMEVERSION(5, 2) ||
                            dwVer == MAKEIMEVERSION(6, 0) ) )
                        ||
                        ( GetLanguage() == LANG_CHS &&
                          ( dwVer == MAKEIMEVERSION(4, 1) ||
                            dwVer == MAKEIMEVERSION(4, 2) ||
                            dwVer == MAKEIMEVERSION(5, 3) ) )
                      )
                    {
                        dwID[0] = dwVer | GetLanguage();
                        dwID[1] = ( (VS_FIXEDFILEINFO*)lpVerData )->dwFileVersionLS;
                    }
                }
            }
            HeapFree( GetProcessHeap(), 0, lpVerBuffer );
        }
    }

    return dwID[uIndex];
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::CheckInputLocale()
{
    static HKL hklPrev = 0;
    s_hklCurrent = GetKeyboardLayout( 0 );
    if ( hklPrev == s_hklCurrent )
        return;

    hklPrev = s_hklCurrent;
    switch ( GetPrimaryLanguage() )
    {
        // Simplified Chinese
        case LANG_CHINESE:
            s_bVerticalCand = true;
            switch ( GetSubLanguage() )
            {
                case SUBLANG_CHINESE_SIMPLIFIED:
                    s_wszCurrIndicator = s_aszIndicator[INDICATOR_CHS];
                    s_bVerticalCand = GetImeId() == 0;
                    break;
                case SUBLANG_CHINESE_TRADITIONAL:
                    s_wszCurrIndicator = s_aszIndicator[INDICATOR_CHT];
                    break;
                default:    // unsupported sub-language
                    s_wszCurrIndicator = s_aszIndicator[INDICATOR_NON_IME];
                    break;
            }
            break;
        // Korean
        case LANG_KOREAN:
            s_wszCurrIndicator = s_aszIndicator[INDICATOR_KOREAN];
            s_bVerticalCand = false;
            break;
        // Japanese
        case LANG_JAPANESE:
            s_wszCurrIndicator = s_aszIndicator[INDICATOR_JAPANESE];
            s_bVerticalCand = true;
            break;
        default:
            // A non-IME language.  Obtain the language abbreviation
            // and store it for rendering the indicator later.
            s_wszCurrIndicator = s_aszIndicator[INDICATOR_NON_IME];
    }

    // If non-IME, use the language abbreviation.
    if( s_wszCurrIndicator == s_aszIndicator[INDICATOR_NON_IME] )
    {
        WCHAR wszLang[5];
        GetLocaleInfoW( MAKELCID( LOWORD( s_hklCurrent ), SORT_DEFAULT ), LOCALE_SABBREVLANGNAME, wszLang, 5 );
        s_wszCurrIndicator[0] = wszLang[0];
        s_wszCurrIndicator[1] = towlower( wszLang[1] );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::CheckToggleState()
{
    CheckInputLocale();
    bool bIme = _ImmIsIME( s_hklCurrent ) != 0;
    s_bChineseIME = ( GetPrimaryLanguage() == LANG_CHINESE ) && bIme;

    HIMC hImc;
    if( NULL != ( hImc = _ImmGetContext( DXUTGetHWND() ) ) )
    {
        if( s_bChineseIME )
        {
            DWORD dwConvMode, dwSentMode;
            _ImmGetConversionStatus( hImc, &dwConvMode, &dwSentMode );
            s_ImeState = ( dwConvMode & IME_CMODE_NATIVE ) ? IMEUI_STATE_ON : IMEUI_STATE_ENGLISH;
        }
        else
        {
            s_ImeState = ( bIme && _ImmGetOpenStatus( hImc ) != 0 ) ? IMEUI_STATE_ON : IMEUI_STATE_OFF;
        }
        _ImmReleaseContext( DXUTGetHWND(), hImc );
    }
    else
        s_ImeState = IMEUI_STATE_OFF;
}


//--------------------------------------------------------------------------------------
// Enable/disable the entire IME system.  When disabled, the default IME handling
// kicks in.
void CDXUTIMEEditBox::EnableImeSystem( bool bEnable )
{
    s_bEnableImeSystem = bEnable;
}


//--------------------------------------------------------------------------------------
// Sets up IME-specific APIs for the IME edit controls.  This is called every time
// the input locale changes.
void CDXUTIMEEditBox::SetupImeApi()
{
    char szImeFile[MAX_PATH + 1];

    _GetReadingString = NULL;
    _ShowReadingWindow = NULL;
    if( _ImmGetIMEFileNameA( s_hklCurrent, szImeFile, sizeof(szImeFile)/sizeof(szImeFile[0]) - 1 ) == 0 )
        return;

    if( s_hDllIme ) FreeLibrary( s_hDllIme );
    s_hDllIme = LoadLibraryA( szImeFile );
    if ( !s_hDllIme )
        return;
    _GetReadingString = (UINT (WINAPI*)(HIMC, UINT, LPWSTR, PINT, BOOL*, PUINT))
        ( GetProcAddress( s_hDllIme, "GetReadingString" ) );
    _ShowReadingWindow =(BOOL (WINAPI*)(HIMC, BOOL))
        ( GetProcAddress( s_hDllIme, "ShowReadingWindow" ) );
}


//--------------------------------------------------------------------------------------
// Resets the composition string.
void CDXUTIMEEditBox::ResetCompositionString()
{
    s_nCompCaret = 0;
    s_CompString.SetText( L"" );
    ZeroMemory( s_abCompStringAttr, sizeof(s_abCompStringAttr) );
}


//--------------------------------------------------------------------------------------
// Truncate composition string by sending keystrokes to the window.
void CDXUTIMEEditBox::TruncateCompString( bool bUseBackSpace, int iNewStrLen )
{
    if( !s_bInsertOnType )
        return;

    int cc = (int) wcslen( s_CompString.GetBuffer() );
    // Send right arrow keystrokes to move the caret
    //   to the end of the composition string.
    for (int i = 0; i < cc - s_nCompCaret; ++i )
        SendMessage( DXUTGetHWND(), WM_KEYDOWN, VK_RIGHT, 0 );
    SendMessage( DXUTGetHWND(), WM_KEYUP, VK_RIGHT, 0 );

    if( bUseBackSpace || m_bInsertMode )
        iNewStrLen = 0;

    // The caller sets bUseBackSpace to false if there's possibility of sending
    // new composition string to the app right after this function call.
    // 
    // If the app is in overwriting mode and new comp string is 
    // shorter than current one, delete previous comp string 
    // till it's same long as the new one. Then move caret to the beginning of comp string.
    // New comp string will overwrite old one.
    if( iNewStrLen < cc )
    {
        for( int i = 0; i < cc - iNewStrLen; ++i )
        {
            SendMessage( DXUTGetHWND(), WM_KEYDOWN, VK_BACK, 0 );  // Backspace character
            SendMessageW( DXUTGetHWND(), WM_CHAR, VK_BACK, 0 );
        }
        SendMessage( DXUTGetHWND(), WM_KEYUP, VK_BACK, 0 );
    }
    else
        iNewStrLen = cc;

    // Move the caret to the beginning by sending left keystrokes
    for (int i = 0; i < iNewStrLen; ++i )
        SendMessage( DXUTGetHWND(), WM_KEYDOWN, VK_LEFT, 0 );
    SendMessage( DXUTGetHWND(), WM_KEYUP, VK_LEFT, 0 );
}


//--------------------------------------------------------------------------------------
// Sends the current composition string to the application by sending keystroke
// messages.
void CDXUTIMEEditBox::SendCompString()
{
    for( int i = 0; i < lstrlen( s_CompString.GetBuffer() ); ++i )
        MsgProc( WM_CHAR, (WPARAM)s_CompString[i], 0 );
}


//--------------------------------------------------------------------------------------
// Outputs current composition string then cleans up the composition task.
void CDXUTIMEEditBox::FinalizeString( bool bSend )
{
    HIMC hImc;
    if( NULL == ( hImc = _ImmGetContext( DXUTGetHWND() ) ) )
        return;

    static bool bProcessing = false;
    if( bProcessing )    // avoid infinite recursion
    {
        DXUTTRACE( L"CDXUTIMEEditBox::FinalizeString: Reentrant detected!\n" );
        _ImmReleaseContext( DXUTGetHWND(), hImc );
        return;
    }
    bProcessing = true;

    if( !s_bInsertOnType && bSend )
    {
        // Send composition string to app.
        LONG lLength = lstrlen( s_CompString.GetBuffer() );
        // In case of CHT IME, don't send the trailing double byte space, if it exists.
        if( GetLanguage() == LANG_CHT
            && s_CompString[lLength - 1] == 0x3000 )
        {
            --lLength;
        }
        SendCompString();
    }

    ResetCompositionString();
    // Clear composition string in IME
    _ImmNotifyIME( hImc, NI_COMPOSITIONSTR, CPS_CANCEL, 0 );
    // the following line is necessary as Korean IME doesn't close cand list
    // when comp string is cancelled.
    _ImmNotifyIME( hImc, NI_CLOSECANDIDATE, 0, 0 ); 
    _ImmReleaseContext( DXUTGetHWND(), hImc );
    bProcessing = false;
}


//--------------------------------------------------------------------------------------
// Determine whether the reading window should be vertical or horizontal.
void CDXUTIMEEditBox::GetReadingWindowOrientation( DWORD dwId )
{
    s_bHorizontalReading = ( s_hklCurrent == _CHS_HKL ) || ( s_hklCurrent == _CHT_HKL2 ) || ( dwId == 0 );
    if( !s_bHorizontalReading && ( dwId & 0x0000FFFF ) == LANG_CHT )
    {
        WCHAR wszRegPath[MAX_PATH];
        HKEY hKey;
        DWORD dwVer = dwId & 0xFFFF0000;
        lstrcpy( wszRegPath, L"software\\microsoft\\windows\\currentversion\\" );
        lstrcat( wszRegPath, ( dwVer >= MAKEIMEVERSION( 5, 1 ) ) ? L"MSTCIPH" : L"TINTLGNT" );
        LONG lRc = RegOpenKeyExW( HKEY_CURRENT_USER, wszRegPath, 0, KEY_READ, &hKey );
        if (lRc == ERROR_SUCCESS)
        {
            DWORD dwSize = sizeof(DWORD), dwMapping, dwType;
            lRc = RegQueryValueExW( hKey, L"Keyboard Mapping", NULL, &dwType, (PBYTE)&dwMapping, &dwSize );
            if (lRc == ERROR_SUCCESS)
            {
                if ( ( dwVer <= MAKEIMEVERSION( 5, 0 ) && 
                       ( (BYTE)dwMapping == 0x22 || (BYTE)dwMapping == 0x23 ) )
                     ||
                     ( ( dwVer == MAKEIMEVERSION( 5, 1 ) || dwVer == MAKEIMEVERSION( 5, 2 ) ) &&
                       (BYTE)dwMapping >= 0x22 && (BYTE)dwMapping <= 0x24 )
                   )
                {
                    s_bHorizontalReading = true;
                }
            }
            RegCloseKey( hKey );
        }
    }
}


//--------------------------------------------------------------------------------------
// Obtain the reading string upon WM_IME_NOTIFY/INM_PRIVATE notification.
void CDXUTIMEEditBox::GetPrivateReadingString()
{
    DWORD dwId = GetImeId();
    if( !dwId )
    {
        s_bShowReadingWindow = false;
        return;
    }

    HIMC hImc;
    hImc = _ImmGetContext( DXUTGetHWND() );
    if( !hImc )
    {
        s_bShowReadingWindow = false;
        return;
    }

    DWORD dwReadingStrLen = 0;
    DWORD dwErr = 0;
    WCHAR *pwszReadingStringBuffer = NULL;  // Buffer for when the IME supports GetReadingString()
    WCHAR *wstr = 0;
    bool bUnicodeIme = false;  // Whether the IME context component is Unicode.
    INPUTCONTEXT *lpIC = NULL;

    if( _GetReadingString )
    {
        UINT uMaxUiLen;
        BOOL bVertical;
        // Obtain the reading string size
        dwReadingStrLen = _GetReadingString( hImc, 0, NULL, (PINT)&dwErr, &bVertical, &uMaxUiLen );
        if( dwReadingStrLen )
        {
            wstr = pwszReadingStringBuffer = (LPWSTR)HeapAlloc( GetProcessHeap(), 0, sizeof(WCHAR) * dwReadingStrLen );
            if( !pwszReadingStringBuffer )
            {
                // Out of memory. Exit.
                _ImmReleaseContext( DXUTGetHWND(), hImc );
                return;
            }

            // Obtain the reading string
            dwReadingStrLen = _GetReadingString( hImc, dwReadingStrLen, wstr, (PINT)&dwErr, &bVertical, &uMaxUiLen );
        }

        s_bHorizontalReading = !bVertical;
        bUnicodeIme = true;
    }
    else
    {
        // IMEs that doesn't implement Reading String API

        lpIC = _ImmLockIMC( hImc );
        
        LPBYTE p = 0;
        switch( dwId )
        {
            case IMEID_CHT_VER42: // New(Phonetic/ChanJie)IME98  : 4.2.x.x // Win98
            case IMEID_CHT_VER43: // New(Phonetic/ChanJie)IME98a : 4.3.x.x // WinMe, Win2k
            case IMEID_CHT_VER44: // New ChanJie IME98b          : 4.4.x.x // WinXP
                p = *(LPBYTE *)((LPBYTE)_ImmLockIMCC( lpIC->hPrivate ) + 24 );
                if( !p ) break;
                dwReadingStrLen = *(DWORD *)( p + 7 * 4 + 32 * 4 );
                dwErr = *(DWORD *)( p + 8 * 4 + 32 * 4 );
                wstr = (WCHAR *)( p + 56 );
                bUnicodeIme = true;
                break;

            case IMEID_CHT_VER50: // 5.0.x.x // WinME
                p = *(LPBYTE *)( (LPBYTE)_ImmLockIMCC( lpIC->hPrivate ) + 3 * 4 );
                if( !p ) break;
                p = *(LPBYTE *)( (LPBYTE)p + 1*4 + 5*4 + 4*2 );
                if( !p ) break;
                dwReadingStrLen = *(DWORD *)(p + 1*4 + (16*2+2*4) + 5*4 + 16);
                dwErr = *(DWORD *)(p + 1*4 + (16*2+2*4) + 5*4 + 16 + 1*4);
                wstr = (WCHAR *)(p + 1*4 + (16*2+2*4) + 5*4);
                bUnicodeIme = false;
                break;

            case IMEID_CHT_VER51: // 5.1.x.x // IME2002(w/OfficeXP)
            case IMEID_CHT_VER52: // 5.2.x.x // (w/whistler)
            case IMEID_CHS_VER53: // 5.3.x.x // SCIME2k or MSPY3 (w/OfficeXP and Whistler)
                p = *(LPBYTE *)((LPBYTE)_ImmLockIMCC( lpIC->hPrivate ) + 4);
                if( !p ) break;
                p = *(LPBYTE *)((LPBYTE)p + 1*4 + 5*4);
                if( !p ) break;
                dwReadingStrLen = *(DWORD *)(p + 1*4 + (16*2+2*4) + 5*4 + 16 * 2);
                dwErr = *(DWORD *)(p + 1*4 + (16*2+2*4) + 5*4 + 16 * 2 + 1*4);
                wstr  = (WCHAR *) (p + 1*4 + (16*2+2*4) + 5*4);
                bUnicodeIme = true;
                break;

            // the code tested only with Win 98 SE (MSPY 1.5/ ver 4.1.0.21)
            case IMEID_CHS_VER41:
            {
                int nOffset;
                nOffset = ( GetImeId( 1 ) >= 0x00000002 ) ? 8 : 7;

                p = *(LPBYTE *)((LPBYTE)_ImmLockIMCC( lpIC->hPrivate ) + nOffset * 4);
                if( !p ) break;
                dwReadingStrLen = *(DWORD *)(p + 7*4 + 16*2*4);
                dwErr = *(DWORD *)(p + 8*4 + 16*2*4);
                dwErr = min( dwErr, dwReadingStrLen );
                wstr = (WCHAR *)(p + 6*4 + 16*2*1);
                bUnicodeIme = true;
                break;
            }

            case IMEID_CHS_VER42: // 4.2.x.x // SCIME98 or MSPY2 (w/Office2k, Win2k, WinME, etc)
            {
	            OSVERSIONINFOW osi;
                osi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOW);
	            GetVersionExW( &osi );

                int nTcharSize = ( osi.dwPlatformId == VER_PLATFORM_WIN32_NT ) ? sizeof(WCHAR) : sizeof(char);
                p = *(LPBYTE *)((LPBYTE)_ImmLockIMCC( lpIC->hPrivate ) + 1*4 + 1*4 + 6*4);
                if( !p ) break;
                dwReadingStrLen = *(DWORD *)(p + 1*4 + (16*2+2*4) + 5*4 + 16 * nTcharSize);
                dwErr = *(DWORD *)(p + 1*4 + (16*2+2*4) + 5*4 + 16 * nTcharSize + 1*4);
                wstr  = (WCHAR *) (p + 1*4 + (16*2+2*4) + 5*4);
                bUnicodeIme = ( osi.dwPlatformId == VER_PLATFORM_WIN32_NT ) ? true : false;
            }
        }   // switch
    }

    // Copy the reading string to the candidate list first
    s_CandList.awszCandidate[0][0] = 0;
    s_CandList.awszCandidate[1][0] = 0;
    s_CandList.awszCandidate[2][0] = 0;
    s_CandList.awszCandidate[3][0] = 0;
    s_CandList.dwCount = dwReadingStrLen;
    s_CandList.dwSelection = (DWORD)-1; // do not select any char
    if( bUnicodeIme )
    {
        UINT i;
        for( i = 0; i < dwReadingStrLen; ++i ) // dwlen > 0, if known IME
        {
            if( dwErr <= i && s_CandList.dwSelection == (DWORD)-1 )
            {
                // select error char
                s_CandList.dwSelection = i;
            }

            s_CandList.awszCandidate[i][0] = wstr[i];
            s_CandList.awszCandidate[i][1] = 0;
        }
        s_CandList.awszCandidate[i][0] = 0;
    }
    else
    {
        char *p = (char *)wstr;
        DWORD i, j;
        for( i = 0, j = 0; i < dwReadingStrLen; ++i, ++j ) // dwlen > 0, if known IME
        {
            if( dwErr <= i && s_CandList.dwSelection == (DWORD)-1 )
            {
                s_CandList.dwSelection = j;
            }
            // Obtain the current code page
	        WCHAR wszCodePage[8];
            UINT uCodePage = CP_ACP;  // Default code page
            if( GetLocaleInfoW( MAKELCID( GetLanguage(), SORT_DEFAULT ),
                                LOCALE_IDEFAULTANSICODEPAGE,
                                wszCodePage,
                                sizeof(wszCodePage)/sizeof(wszCodePage[0]) ) )
            {
                uCodePage = wcstoul( wszCodePage, NULL, 0 );
            }
            MultiByteToWideChar( uCodePage, 0, p + i, IsDBCSLeadByteEx( uCodePage, p[i] ) ? 2 : 1,
                                 s_CandList.awszCandidate[j], 1 );
            if( IsDBCSLeadByteEx( uCodePage, p[i] ) )
                ++i;
        }
        s_CandList.awszCandidate[j][0] = 0;
        s_CandList.dwCount = j;
    }
    if( !_GetReadingString )
    {
        _ImmUnlockIMCC( lpIC->hPrivate );
        _ImmUnlockIMC( hImc );
        GetReadingWindowOrientation( dwId );
    }
    _ImmReleaseContext( DXUTGetHWND(), hImc );

    if( pwszReadingStringBuffer )
        HeapFree( GetProcessHeap(), 0, pwszReadingStringBuffer );

    // Copy the string to the reading string buffer
    if( s_CandList.dwCount > 0 )
        s_bShowReadingWindow = true;
    else
        s_bShowReadingWindow = false;
    if( s_bHorizontalReading )
    {
        s_CandList.nReadingError = -1;
        s_wszReadingString[0] = 0;
        for( UINT i = 0; i < s_CandList.dwCount; ++i )
        {
            if( s_CandList.dwSelection == i )
                s_CandList.nReadingError = lstrlen( s_wszReadingString );
            wcsncat( s_wszReadingString, s_CandList.awszCandidate[i], 32 - lstrlenW( s_wszReadingString ) - 1 );
        }
    }

    s_CandList.dwPageSize = MAX_CANDLIST;
}


//--------------------------------------------------------------------------------------
// This function is used only briefly in CHT IME handling,
// so accelerator isn't processed.
void CDXUTIMEEditBox::PumpMessage()
{
    MSG msg;

    while( PeekMessageW( &msg, NULL, 0, 0, PM_NOREMOVE ) )
    {
        if( !GetMessageW( &msg, NULL, 0, 0 ) )
        {
            PostQuitMessage( (int)msg.wParam );
            return;
        }
        TranslateMessage( &msg );
        DispatchMessageA( &msg );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::OnFocusIn()
{
    CDXUTEditBox::OnFocusIn();

    if( s_bEnableImeSystem )
    {
        _ImmAssociateContext( DXUTGetHWND(), s_hImcDef );
        CheckToggleState();
    } else
        _ImmAssociateContext( DXUTGetHWND(), NULL );

    //
    // Set up the IME global state according to the current instance state
    //
    HIMC hImc;
    if( NULL != ( hImc = _ImmGetContext( DXUTGetHWND() ) ) ) 
    {
        if( !s_bEnableImeSystem )
            s_ImeState = IMEUI_STATE_OFF;

        _ImmReleaseContext( DXUTGetHWND(), hImc );
        CheckToggleState();
    }
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::OnFocusOut()
{
    CDXUTEditBox::OnFocusOut();

    FinalizeString( false );  // Don't send the comp string as to match RichEdit behavior

    _ImmAssociateContext( DXUTGetHWND(), NULL );
}


//--------------------------------------------------------------------------------------
bool CDXUTIMEEditBox::StaticMsgProc( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    HIMC hImc;

    if( !s_bEnableImeSystem )
        return false;

    switch( uMsg )
    {
        case WM_ACTIVATEAPP:
            if( wParam )
            {
                // Populate s_Locale with the list of keyboard layouts.
                UINT cKL = GetKeyboardLayoutList( 0, NULL );
                s_Locale.RemoveAll();
                HKL *phKL = new HKL[cKL];
                if( phKL )
                {
                    GetKeyboardLayoutList( cKL, phKL );
                    for( UINT i = 0; i < cKL; ++i )
                    {
                        CInputLocale Locale;

                        // Filter out East Asian languages that are not IME.
                        if( ( PRIMARYLANGID( LOWORD( phKL[i] ) ) == LANG_CHINESE ||
                              PRIMARYLANGID( LOWORD( phKL[i] ) ) == LANG_JAPANESE ||
                              PRIMARYLANGID( LOWORD( phKL[i] ) ) == LANG_KOREAN ) &&
                              !_ImmIsIME( phKL[i] ) )
                              continue;

                        // If this language is already in the list, don't add it again.
                        bool bBreak = false;
                        for( int e = 0; e < s_Locale.GetSize(); ++e )
                            if( LOWORD( s_Locale.GetAt( e ).m_hKL ) ==
                                LOWORD( phKL[i] ) )
                            {
                                bBreak = true;
                                break;
                            }
                        if( bBreak )
                            break;

                        Locale.m_hKL = phKL[i];
                        WCHAR wszDesc[128] = L"";
                        switch( PRIMARYLANGID( LOWORD( phKL[i] ) ) )
                        {
                            // Simplified Chinese
                            case LANG_CHINESE:
                                switch( SUBLANGID( LOWORD( phKL[i] ) ) )
                                {
                                    case SUBLANG_CHINESE_SIMPLIFIED:
                                        lstrcpy( Locale.m_wszLangAbb, s_aszIndicator[INDICATOR_CHS] );
                                        break;
                                    case SUBLANG_CHINESE_TRADITIONAL:
                                        lstrcpy( Locale.m_wszLangAbb, s_aszIndicator[INDICATOR_CHT] );
                                        break;
                                    default:    // unsupported sub-language
                                        GetLocaleInfoW( MAKELCID( LOWORD( phKL[i] ), SORT_DEFAULT ), LOCALE_SABBREVLANGNAME, wszDesc, 128 );
                                        Locale.m_wszLangAbb[0] = wszDesc[0];
                                        Locale.m_wszLangAbb[1] = towlower( wszDesc[1] );
                                        Locale.m_wszLangAbb[2] = L'\0';
                                        break;
                                }
                                break;
                            // Korean
                            case LANG_KOREAN:
                                lstrcpy( Locale.m_wszLangAbb, s_aszIndicator[INDICATOR_KOREAN] );
                                break;
                            // Japanese
                            case LANG_JAPANESE:
                                lstrcpy( Locale.m_wszLangAbb, s_aszIndicator[INDICATOR_JAPANESE] );
                                break;         
                            default:
                                // A non-IME language.  Obtain the language abbreviation
                                // and store it for rendering the indicator later.
                                GetLocaleInfoW( MAKELCID( LOWORD( phKL[i] ), SORT_DEFAULT ), LOCALE_SABBREVLANGNAME, wszDesc, 128 );
                                Locale.m_wszLangAbb[0] = wszDesc[0];
                                Locale.m_wszLangAbb[1] = towlower( wszDesc[1] );
                                Locale.m_wszLangAbb[2] = L'\0';
                                break;
                        }

                        GetLocaleInfoW( MAKELCID( LOWORD( phKL[i] ), SORT_DEFAULT ), LOCALE_SLANGUAGE, wszDesc, 128 );
                        wcsncpy( Locale.m_wszLang, wszDesc, sizeof(Locale.m_wszLang) / sizeof(Locale.m_wszLang[0]) );
                        Locale.m_wszLang[sizeof(Locale.m_wszLang) / sizeof(Locale.m_wszLang[0]) - 1] = L'\0';

                        s_Locale.Add( Locale );
                    }
                    delete[] phKL;
                }
            }
            break;

        case WM_INPUTLANGCHANGE:
            DXUTTRACE( L"WM_INPUTLANGCHANGE\n" );
            {
                UINT uLang = GetPrimaryLanguage();
                CheckToggleState();
                if ( uLang != GetPrimaryLanguage() )
                {
                    // Korean IME always inserts on keystroke.  Other IMEs do not.
                    s_bInsertOnType = ( GetPrimaryLanguage() == LANG_KOREAN );
                }

                // IME changed.  Setup the new IME.
                SetupImeApi();
                if( _ShowReadingWindow )
                {
                    if ( NULL != ( hImc = _ImmGetContext( DXUTGetHWND() ) ) )
                    {
                        _ShowReadingWindow( hImc, false );
                        _ImmReleaseContext( DXUTGetHWND(), hImc );
                    }
                }
            }
            return true;

        case WM_IME_SETCONTEXT:
            DXUTTRACE( L"WM_IME_SETCONTEXT\n" );
            //
            // We don't want anything to display, so we have to clear this
            //
            lParam = 0;
            return false;

        // Handle WM_IME_STARTCOMPOSITION here since
        // we do not want the default IME handler to see
        // this when our fullscreen app is running.
        case WM_IME_STARTCOMPOSITION:
            DXUTTRACE( L"WM_IME_STARTCOMPOSITION\n" );
            ResetCompositionString();
            // Since the composition string has its own caret, we don't render
            // the edit control's own caret to avoid double carets on screen.
            s_bHideCaret = true;
            return true;

        case WM_IME_COMPOSITION:
            DXUTTRACE( L"WM_IME_COMPOSITION\n" );
            return true;
    }

    return false;
}


//--------------------------------------------------------------------------------------
bool CDXUTIMEEditBox::HandleMouse( UINT uMsg, POINT pt, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    switch( uMsg )
    {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK:
        {
            DXUTFontNode* pFont = m_pDialog->GetFont( m_Elements.GetAt( 9 )->iFont );

            // Check if this click is on top of the composition string
            int nCompStrWidth;
            s_CompString.CPtoX( s_CompString.GetTextSize(), FALSE, &nCompStrWidth );

            if( s_ptCompString.x <= pt.x &&
                s_ptCompString.y <= pt.y &&
                s_ptCompString.x + nCompStrWidth > pt.x &&
                s_ptCompString.y + pFont->nHeight > pt.y )
            {
                int nCharBodyHit, nCharHit;
                int nTrail;

                // Determine the character clicked on.
                s_CompString.XtoCP( pt.x - s_ptCompString.x, &nCharBodyHit, &nTrail );
                if( nTrail && nCharBodyHit < s_CompString.GetTextSize() )
                    nCharHit = nCharBodyHit + 1;
                else
                    nCharHit = nCharBodyHit;

                // Now generate keypress events to move the comp string cursor
                // to the click point.  First, if the candidate window is displayed,
                // send Esc to close it.
                HIMC hImc = _ImmGetContext( DXUTGetHWND() );
                if( !hImc )
                    return true;

                _ImmNotifyIME( hImc, NI_CLOSECANDIDATE, 0, 0 );
                _ImmReleaseContext( DXUTGetHWND(), hImc );

                switch( GetPrimaryLanguage() )
                {
                    case LANG_JAPANESE:
                        // For Japanese, there are two cases.  If s_nFirstTargetConv is
                        // -1, the comp string hasn't been converted yet, and we use
                        // s_nCompCaret.  For any other value of s_nFirstTargetConv,
                        // the string has been converted, so we use clause information.

                        if( s_nFirstTargetConv != -1 )
                        {
                            int nClauseClicked = 0;
                            while( (int)s_adwCompStringClause[nClauseClicked + 1] <= nCharBodyHit )
                                ++nClauseClicked;

                            int nClauseSelected = 0;
                            while( (int)s_adwCompStringClause[nClauseSelected + 1] <= s_nFirstTargetConv )
                                ++nClauseSelected;

                            BYTE nVirtKey = nClauseClicked > nClauseSelected ? VK_RIGHT : VK_LEFT;
                            int nSendCount = abs( nClauseClicked - nClauseSelected );
                            while( nSendCount-- > 0 )
                                SendKey( nVirtKey );

                            return true;
                        }

                        // Not converted case. Fall thru to Chinese case.

                    case LANG_CHINESE:
                    {
                        // For Chinese, use s_nCompCaret.
                        BYTE nVirtKey = nCharHit > s_nCompCaret ? VK_RIGHT : VK_LEFT;
                        int nSendCount = abs( nCharHit - s_nCompCaret );
                        while( nSendCount-- > 0 )
                            SendKey( nVirtKey );
                        break;
                    }
                }

                return true;
            }

            // Check if the click is on top of the candidate window
            if( s_CandList.bShowWindow && PtInRect( &s_CandList.rcCandidate, pt ) )
            {
                if( s_bVerticalCand )
                {
                    // Vertical candidate window

                    // Compute the row the click is on
                    int nRow = ( pt.y - s_CandList.rcCandidate.top ) / pFont->nHeight;

                    if( nRow < (int)s_CandList.dwCount )
                    {
                        // nRow is a valid entry.
                        // Now emulate keystrokes to select the candidate at this row.
                        switch( GetPrimaryLanguage() )
                        {
                            case LANG_CHINESE:
                            case LANG_KOREAN:
                                // For Chinese and Korean, simply send the number keystroke.
                                SendKey( '0' + nRow + 1 );
                                break;

                            case LANG_JAPANESE:
                                // For Japanese, move the selection to the target row,
                                // then send Right, then send Left.

                                BYTE nVirtKey;
                                if( nRow > (int)s_CandList.dwSelection )
                                    nVirtKey = VK_DOWN;
                                else
                                    nVirtKey = VK_UP;
                                int nNumToHit = abs( int( nRow - s_CandList.dwSelection ) );
                                for( int nStrike = 0; nStrike < nNumToHit; ++nStrike )
                                    SendKey( nVirtKey );

                                // Do this to close the candidate window without ending composition.
                                SendKey( VK_RIGHT );
                                SendKey( VK_LEFT );

                                break;
                        }
                    }
                } else
                {
                    // Horizontal candidate window

                    // Determine which the character the click has hit.
                    int nCharHit;
                    int nTrail;
                    s_CandList.HoriCand.XtoCP( pt.x - s_CandList.rcCandidate.left, &nCharHit, &nTrail );

                    // Determine which candidate string the character belongs to.
                    int nCandidate = s_CandList.dwCount - 1;

                    int nEntryStart = 0;
                    for( UINT i = 0; i < s_CandList.dwCount; ++i )
                    {
                        if( nCharHit >= nEntryStart )
                        {
                            // Haven't found it.
                            nEntryStart += lstrlenW( s_CandList.awszCandidate[i] ) + 1;  // plus space separator
                        } else
                        {
                            // Found it.  This entry starts at the right side of the click point,
                            // so the char belongs to the previous entry.
                            nCandidate = i - 1;
                            break;
                        }
                    }

                    // Now emulate keystrokes to select the candidate entry.
                    switch( GetPrimaryLanguage() )
                    {
                        case LANG_CHINESE:
                        case LANG_KOREAN:
                            // For Chinese and Korean, simply send the number keystroke.
                            SendKey( '0' + nCandidate + 1 );
                            break;
                    }
                }

                return true;
            }
        }
    }

    // If we didn't care for the msg, let the parent process it.
    return CDXUTEditBox::HandleMouse( uMsg, pt, wParam, lParam );
}


//--------------------------------------------------------------------------------------
bool CDXUTIMEEditBox::MsgProc( UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    if( !m_bEnabled || !m_bVisible )
        return false;

    // TEMPORARY!!! TEMPORARY!!! TEMPORARY!!!
    // See how I can get rid of trapped later.
    bool trappedData;
    bool *trapped = &trappedData;

    HIMC hImc;
    static LPARAM lAlt = 0x80000000, lCtrl = 0x80000000, lShift = 0x80000000;

    *trapped = false;
    if( !s_bEnableImeSystem )
        return CDXUTEditBox::MsgProc( uMsg, wParam, lParam );

    switch( uMsg )
    {
        //
        //  IME Handling
        //
        case WM_IME_COMPOSITION:
            DXUTTRACE( L"WM_IME_COMPOSITION\n" );
            {
                LONG lRet;  // Returned count in CHARACTERS
                WCHAR wszCompStr[MAX_COMPSTRING_SIZE];

                *trapped = true;
                if( NULL == ( hImc = _ImmGetContext( DXUTGetHWND() ) ) )
                {
                    break;
                }

                // ResultStr must be processed before composition string.
                if ( lParam & GCS_RESULTSTR )
                {
                    DXUTTRACE( L"  GCS_RESULTSTR\n" );
                    lRet = _ImmGetCompositionStringW( hImc, GCS_RESULTSTR, wszCompStr, sizeof( wszCompStr ) );
                    if( lRet > 0 )
                    {
                        lRet /= sizeof(WCHAR);
                        wszCompStr[lRet] = 0;  // Force terminate
                        TruncateCompString( false, (int)wcslen( wszCompStr ) );
                        s_CompString.SetText( wszCompStr );
                        SendCompString();
                        ResetCompositionString();
                    }
                }

                //
                // Reads in the composition string.
                //
                if ( lParam & GCS_COMPSTR )
                {
                    DXUTTRACE( L"  GCS_COMPSTR\n" );
                    //////////////////////////////////////////////////////
                    // Retrieve the latest user-selected IME candidates
                    lRet = _ImmGetCompositionStringW( hImc, GCS_COMPSTR, wszCompStr, sizeof( wszCompStr ) );
                    if( lRet > 0 )
                    {
                        lRet /= sizeof(WCHAR);  // Convert size in byte to size in char
                        wszCompStr[lRet] = 0;  // Force terminate
                        //
                        // Remove the whole of the string
                        //
                        TruncateCompString( false, (int)wcslen( wszCompStr ) );

                        s_CompString.SetText( wszCompStr );
                        lRet = _ImmGetCompositionStringW( hImc, GCS_COMPATTR, s_abCompStringAttr, sizeof( s_abCompStringAttr ) );
                        if( lRet > 0 )
                            s_abCompStringAttr[lRet] = 0;  // ??? Is this needed for attributes?

                        // Older CHT IME uses composition string for reading string
                        if ( GetLanguage() == LANG_CHT && !GetImeId() )
                        {
                            if( lstrlen( s_CompString.GetBuffer() ) )
                            {
                                s_CandList.dwCount = 4;             // Maximum possible length for reading string is 4
                                s_CandList.dwSelection = (DWORD)-1; // don't select any candidate

                                // Copy the reading string to the candidate list
                                for( int i = 3; i >= 0; --i )
                                {
                                    if( i > lstrlen( s_CompString.GetBuffer() ) - 1 )
                                        s_CandList.awszCandidate[i][0] = 0;  // Doesn't exist
                                    else
                                    {
                                        s_CandList.awszCandidate[i][0] = s_CompString[i];
                                        s_CandList.awszCandidate[i][1] = 0;
                                    }
                                }
                                s_CandList.dwPageSize = MAX_CANDLIST;
                                // Clear comp string after we are done copying
                                ZeroMemory( (LPVOID)s_CompString.GetBuffer(), 4 * sizeof(WCHAR) );
                                s_bShowReadingWindow = true;
                                GetReadingWindowOrientation( 0 );
                                if( s_bHorizontalReading )
                                {
                                    s_CandList.nReadingError = -1;  // Clear error

                                    // Create a string that consists of the current
                                    // reading string.  Since horizontal reading window
                                    // is used, we take advantage of this by rendering
                                    // one string instead of several.
                                    //
                                    // Copy the reading string from the candidate list
                                    // to the reading string buffer.
                                    s_wszReadingString[0] = 0;
                                    for( UINT i = 0; i < s_CandList.dwCount; ++i )
                                    {
                                        if( s_CandList.dwSelection == i )
                                            s_CandList.nReadingError = lstrlen( s_wszReadingString );
                                        wcsncat( s_wszReadingString, s_CandList.awszCandidate[i], 32 - lstrlenW( s_wszReadingString ) - 1 );
                                    }
                                }
                            }
                            else
                            {
                                s_CandList.dwCount = 0;
                                s_bShowReadingWindow = false;
                            }
                        }

                        // Get the caret position in composition string
                        s_nCompCaret = _ImmGetCompositionStringW( hImc, GCS_CURSORPOS, NULL, 0 );
                        if( s_nCompCaret < 0 )
                            s_nCompCaret = 0; // On error, set caret to pos 0.

                        if( s_bInsertOnType )
                        {
                            // Send composition string to the edit control via WM_CHAR
                            SendCompString();
                            // Restore the caret to the correct location.
                            // It's at the end right now, so compute the number
                            // of times left arrow should be pressed to
                            // send it to the original position.
                            int nCount = lstrlen( s_CompString.GetBuffer() + s_nCompCaret );
                            // Send left keystrokes
                            for( int i = 0; i < nCount; ++i )
                                SendMessage( DXUTGetHWND(), WM_KEYDOWN, VK_LEFT, 0 );
                            SendMessage( DXUTGetHWND(), WM_KEYUP, VK_LEFT, 0 );
                        }
                    }

                    ResetCaretBlink();
                }

                // Retrieve clause information
                if( lParam & GCS_COMPCLAUSE )
                {
                    lRet = _ImmGetCompositionStringW(hImc, GCS_COMPCLAUSE, s_adwCompStringClause, sizeof( s_adwCompStringClause ) );
                    s_adwCompStringClause[lRet / sizeof(DWORD)] = 0;  // Terminate
                }

                _ImmReleaseContext( DXUTGetHWND(), hImc );
            }
            break;

        case WM_IME_ENDCOMPOSITION:
            DXUTTRACE( L"WM_IME_ENDCOMPOSITION\n" );
            TruncateCompString();
            ResetCompositionString();
            // We can show the edit control's caret again.
            s_bHideCaret = false;
            // Hide reading window
            s_bShowReadingWindow = false;
            break;

        case WM_IME_NOTIFY:
            DXUTTRACE( L"WM_IME_NOTIFY %u\n", wParam );
            switch( wParam )
            {
                case IMN_SETCONVERSIONMODE:
                    DXUTTRACE( L"  IMN_SETCONVERSIONMODE\n" );
                case IMN_SETOPENSTATUS:
                    DXUTTRACE( L"  IMN_SETOPENSTATUS\n" );
                    CheckToggleState();
                    break;

                case IMN_OPENCANDIDATE:
                case IMN_CHANGECANDIDATE:
                {
                    DXUTTRACE( wParam == IMN_CHANGECANDIDATE ? L"  IMN_CHANGECANDIDATE\n" : L"  IMN_OPENCANDIDATE\n" );

                    s_CandList.bShowWindow = true;
                    *trapped = true;
                    if( NULL == ( hImc = _ImmGetContext( DXUTGetHWND() ) ) )
                        break;

                    LPCANDIDATELIST lpCandList = NULL;
                    DWORD dwLenRequired;

                    s_bShowReadingWindow = false;
                    // Retrieve the candidate list
                    dwLenRequired = _ImmGetCandidateListW( hImc, 0, NULL, 0 );
                    if( dwLenRequired )
                    {
                        lpCandList = (LPCANDIDATELIST)HeapAlloc( GetProcessHeap(), 0, dwLenRequired );
                        dwLenRequired = _ImmGetCandidateListW( hImc, 0, lpCandList, dwLenRequired );
                    }

                    if( lpCandList )
                    {
                        // Update candidate list data
                        s_CandList.dwSelection = lpCandList->dwSelection;
                        s_CandList.dwCount = lpCandList->dwCount;

                        int nPageTopIndex = 0;
                        s_CandList.dwPageSize = min( lpCandList->dwPageSize, MAX_CANDLIST );
                        if( GetPrimaryLanguage() == LANG_JAPANESE )
                        {
                            // Japanese IME organizes its candidate list a little
                            // differently from the other IMEs.
                            nPageTopIndex = ( s_CandList.dwSelection / s_CandList.dwPageSize ) * s_CandList.dwPageSize;
                        }
                        else
                            nPageTopIndex = lpCandList->dwPageStart;

                        // Make selection index relative to first entry of page
                        s_CandList.dwSelection = ( GetLanguage() == LANG_CHS && !GetImeId() ) ? (DWORD)-1
                                                 : s_CandList.dwSelection - nPageTopIndex;

                        ZeroMemory( s_CandList.awszCandidate, sizeof(s_CandList.awszCandidate) );
                        for( UINT i = nPageTopIndex, j = 0;
                            (DWORD)i < lpCandList->dwCount && j < s_CandList.dwPageSize;
                            i++, j++ )
                        {
                            // Initialize the candidate list strings
                            LPWSTR pwsz = s_CandList.awszCandidate[j];
                            // For every candidate string entry,
                            // write [index] + Space + [String] if vertical,
                            // write [index] + [String] + Space if horizontal.
	                        *pwsz++ = (WCHAR)( L'0' + ( (j + 1) % 10 ) );  // Index displayed is 1 based
	                        if( s_bVerticalCand )
		                        *pwsz++ = L' ';
                            WCHAR *pwszNewCand = (LPWSTR)( (LPBYTE)lpCandList + lpCandList->dwOffset[i] );
	                        while ( *pwszNewCand )
		                        *pwsz++ = *pwszNewCand++;
	                        if( !s_bVerticalCand )
		                        *pwsz++ = L' ';
	                        *pwsz = 0;  // Terminate
                        }

                        // Make dwCount in s_CandList be number of valid entries in the page.
                        s_CandList.dwCount = lpCandList->dwCount - lpCandList->dwPageStart;
                        if( s_CandList.dwCount > lpCandList->dwPageSize )
                            s_CandList.dwCount = lpCandList->dwPageSize;

                        HeapFree( GetProcessHeap(), 0, lpCandList );
                        _ImmReleaseContext( DXUTGetHWND(), hImc );

                        // Korean and old Chinese IME can't have selection.
                        // User must use the number hotkey or Enter to select
                        // a candidate.
                        if( GetPrimaryLanguage() == LANG_KOREAN ||
                            GetLanguage() == LANG_CHT && !GetImeId() )
                        {
                            s_CandList.dwSelection = (DWORD)-1;
                        }

                        // Initialize s_CandList.HoriCand if we have a
                        // horizontal candidate window.
                        if( !s_bVerticalCand )
                        {
                            WCHAR wszCand[256] = L"";

                            s_CandList.nFirstSelected = 0;
                            s_CandList.nHoriSelectedLen = 0;
                            for( UINT i = 0; i < MAX_CANDLIST; ++i )
                            {
                                if( s_CandList.awszCandidate[i][0] == L'\0' )
                                    break;

                                WCHAR wszEntry[32];
                                swprintf( wszEntry, L"%s ", s_CandList.awszCandidate[i] );
                                // If this is the selected entry, mark its char position.
                                if( s_CandList.dwSelection == i )
                                {
                                    s_CandList.nFirstSelected = lstrlen( wszCand );
                                    s_CandList.nHoriSelectedLen = lstrlen( wszEntry ) - 1;  // Minus space
                                }
                                lstrcat( wszCand, wszEntry );
                            }
                            wszCand[lstrlen(wszCand) - 1] = L'\0';  // Remove the last space
                            s_CandList.HoriCand.SetText( wszCand );
                        }
                    }
                    break;
                }
                
                case IMN_CLOSECANDIDATE:
                {
                    DXUTTRACE( L"  IMN_CLOSECANDIDATE\n" );
                    s_CandList.bShowWindow = false;
                    if( !s_bShowReadingWindow )
                    {
                        s_CandList.dwCount = 0;
                        ZeroMemory( s_CandList.awszCandidate, sizeof(s_CandList.awszCandidate) );
                    }
                    *trapped = true;
                    break;
                }

                case IMN_PRIVATE:
                    DXUTTRACE( L"  IMN_PRIVATE\n" );
                    {
                        if( !s_CandList.bShowWindow )
                            GetPrivateReadingString();

                        // Trap some messages to hide reading window
                        DWORD dwId = GetImeId();
                        switch( dwId )
                        {
                            case IMEID_CHT_VER42:
                            case IMEID_CHT_VER43:
                            case IMEID_CHT_VER44:
                            case IMEID_CHS_VER41:
                            case IMEID_CHS_VER42:
                                if( ( lParam == 1 ) || ( lParam == 2 ) )
                                {
                                    *trapped = true;
                                }
                                break;

                            case IMEID_CHT_VER50:
                            case IMEID_CHT_VER51:
                            case IMEID_CHT_VER52:
                            case IMEID_CHT_VER60:
                            case IMEID_CHS_VER53:
                                if( (lParam == 16) || (lParam == 17) || (lParam == 26) || (lParam == 27) || (lParam == 28) )
                                {
                                    *trapped = true;
                                }
                                break;
                        }
                    }
                    break;

                default:
                    *trapped = true;
                    break;
            }
            break;

        // fix for #15386 - When Text Service Framework is installed in Win2K, Alt+Shift and Ctrl+Shift combination (to switch 
        // input locale / keyboard layout) doesn't send WM_KEYUP message for the key that is released first. We need to check
        // if these keys are actually up whenever we receive key up message for other keys.
        case WM_KEYUP:
        case WM_SYSKEYUP:
            if ( !( lAlt & 0x80000000 ) && wParam != VK_MENU && ( GetAsyncKeyState( VK_MENU ) & 0x8000 ) == 0 )
            {
                PostMessageW( GetFocus(), WM_KEYUP, (WPARAM)VK_MENU, ( lAlt & 0x01ff0000 ) | 0xC0000001 );
            }   
            else if ( !( lCtrl & 0x80000000 ) && wParam != VK_CONTROL && ( GetAsyncKeyState( VK_CONTROL ) & 0x8000 ) == 0 )
            {
                PostMessageW( GetFocus(), WM_KEYUP, (WPARAM)VK_CONTROL, ( lCtrl & 0x01ff0000 ) | 0xC0000001 );
            }
            else if ( !( lShift & 0x80000000 ) && wParam != VK_SHIFT && ( GetAsyncKeyState( VK_SHIFT ) & 0x8000 ) == 0 )
            {
                PostMessageW( GetFocus(), WM_KEYUP, (WPARAM)VK_SHIFT, ( lShift & 0x01ff0000 ) | 0xC0000001 );
            }
            // fall through WM_KEYDOWN / WM_SYSKEYDOWN
        case WM_KEYDOWN:
        case WM_SYSKEYDOWN:
            switch ( wParam )
            {
                case VK_MENU:
                    lAlt = lParam;
                    break;
                case VK_SHIFT:
                    lShift = lParam;
                    break;
                case VK_CONTROL:
                    lCtrl = lParam;
                    break;
            }
            //break;
            // Fall through to default case
            // so we invoke the parent.

        default:
            // Let the parent handle the message that we
            // don't handle.
            return CDXUTEditBox::MsgProc( uMsg, wParam, lParam );

    }  // switch

    return *trapped;
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::RenderCandidateReadingWindow( IDirect3DDevice9* pd3dDevice, float fElapsedTime, bool bReading )
{
    RECT rc;
    UINT nNumEntries = bReading ? 4 : MAX_CANDLIST;
    D3DCOLOR TextColor, TextBkColor, SelTextColor, SelBkColor;
    int nX, nXFirst, nXComp;
    m_Buffer.CPtoX( m_nCaret, FALSE, &nX );
    m_Buffer.CPtoX( m_nFirstVisible, FALSE, &nXFirst );

    if( bReading )
    {
        TextColor = m_ReadingColor;
        TextBkColor = m_ReadingWinColor;
        SelTextColor = m_ReadingSelColor;
        SelBkColor = m_ReadingSelBkColor;
    } else
    {
        TextColor = m_CandidateColor;
        TextBkColor = m_CandidateWinColor;
        SelTextColor = m_CandidateSelColor;
        SelBkColor = m_CandidateSelBkColor;
    }

    // For Japanese IME, align the window with the first target converted character.
    // For all other IMEs, align with the caret.  This is because the caret
    // does not move for Japanese IME.
    if ( GetLanguage() == LANG_CHT && !GetImeId() )
        nXComp = 0;
    else
    if( GetPrimaryLanguage() == LANG_JAPANESE )
        s_CompString.CPtoX( s_nFirstTargetConv, FALSE, &nXComp );
    else
        s_CompString.CPtoX( s_nCompCaret, FALSE, &nXComp );

    // Compute the size of the candidate window
    int nWidthRequired = 0;
    int nHeightRequired = 0;
    int nSingleLineHeight = 0;

    if( ( s_bVerticalCand && !bReading ) ||
        ( !s_bHorizontalReading && bReading ) )
    {
        // Vertical window
        for( UINT i = 0; i < nNumEntries; ++i )
        {
            if( s_CandList.awszCandidate[i][0] == L'\0' )
                break;
            SetRect( &rc, 0, 0, 0, 0 );
            m_pDialog->CalcTextRect( s_CandList.awszCandidate[i], m_Elements.GetAt( 1 ), &rc );
            nWidthRequired = max( nWidthRequired, rc.right - rc.left );
            nSingleLineHeight = max( nSingleLineHeight, rc.bottom - rc.top );
        }
        nHeightRequired = nSingleLineHeight * nNumEntries;
    } else
    {
        // Horizontal window
        SetRect( &rc, 0, 0, 0, 0 );
        if( bReading )
            m_pDialog->CalcTextRect( s_wszReadingString, m_Elements.GetAt( 1 ), &rc );
        else
            m_pDialog->CalcTextRect( s_CandList.HoriCand.GetBuffer(), m_Elements.GetAt( 1 ), &rc );
        nWidthRequired = rc.right - rc.left;
        nSingleLineHeight = nHeightRequired = rc.bottom - rc.top;
    }

    // Now that we have the dimension, calculate the location for the candidate window.
    // We attempt to fit the window in this order:
    // bottom, top, right, left.

    bool bHasPosition = false;

    // Bottom
    SetRect( &rc, s_ptCompString.x + nXComp, s_ptCompString.y + m_rcText.bottom - m_rcText.top,
                  s_ptCompString.x + nXComp + nWidthRequired, s_ptCompString.y + m_rcText.bottom - m_rcText.top + nHeightRequired );
    // if the right edge is cut off, move it left.
    if( rc.right > m_pDialog->GetWidth() )
    {
        rc.left -= rc.right - m_pDialog->GetWidth();
        rc.right = m_pDialog->GetWidth();
    }
    if( rc.bottom <= m_pDialog->GetHeight() )
        bHasPosition = true;

    // Top
    if( !bHasPosition )
    {
        SetRect( &rc, s_ptCompString.x + nXComp, s_ptCompString.y - nHeightRequired,
                      s_ptCompString.x + nXComp + nWidthRequired, s_ptCompString.y );
        // if the right edge is cut off, move it left.
        if( rc.right > m_pDialog->GetWidth() )
        {
            rc.left -= rc.right - m_pDialog->GetWidth();
            rc.right = m_pDialog->GetWidth();
        }
        if( rc.top >= 0 )
            bHasPosition = true;
    }

    // Right
    if( !bHasPosition )
    {
        int nXCompTrail;
        s_CompString.CPtoX( s_nCompCaret, TRUE, &nXCompTrail );
        SetRect( &rc, s_ptCompString.x + nXCompTrail, 0,
                      s_ptCompString.x + nXCompTrail + nWidthRequired, nHeightRequired );
        if( rc.right <= m_pDialog->GetWidth() )
            bHasPosition = true;
    }

    // Left
    if( !bHasPosition )
    {
        SetRect( &rc, s_ptCompString.x + nXComp - nWidthRequired, 0,
                      s_ptCompString.x + nXComp, nHeightRequired );
        if( rc.right >= 0 )
            bHasPosition = true;
    }

    if( !bHasPosition )
    {
        // The dialog is too small for the candidate window.
        // Fall back to render at 0, 0.  Some part of the window
        // will be cut off.
        rc.left = 0;
        rc.right = nWidthRequired;
    }

    // If we are rendering the candidate window, save the position
    // so that mouse clicks are checked properly.
    if( !bReading )
        s_CandList.rcCandidate = rc;

    // Render the elements
    m_pDialog->DrawRect( &rc, TextBkColor );
    if( ( s_bVerticalCand && !bReading ) ||
        ( !s_bHorizontalReading && bReading ) )
    {
        // Vertical candidate window
        for( UINT i = 0; i < nNumEntries; ++i )
        {
            // Here we are rendering one line at a time
            rc.bottom = rc.top + nSingleLineHeight;
            // Use a different color for the selected string
            if( s_CandList.dwSelection == i )
            {
                m_pDialog->DrawRect( &rc, SelBkColor );
                m_Elements.GetAt( 1 )->FontColor.Current = SelTextColor;
            } else
                m_Elements.GetAt( 1 )->FontColor.Current = TextColor;

            m_pDialog->DrawText( s_CandList.awszCandidate[i], m_Elements.GetAt( 1 ), &rc );

            rc.top += nSingleLineHeight;
        }
    } else
    {
        // Horizontal candidate window
        m_Elements.GetAt( 1 )->FontColor.Current = TextColor;
        if( bReading )
            m_pDialog->DrawText( s_wszReadingString, m_Elements.GetAt( 1 ), &rc );
        else
            m_pDialog->DrawText( s_CandList.HoriCand.GetBuffer(), m_Elements.GetAt( 1 ), &rc );

        // Render the selected entry differently
        if( !bReading )
        {
            int nXLeft, nXRight;
            s_CandList.HoriCand.CPtoX( s_CandList.nFirstSelected, FALSE, &nXLeft );
            s_CandList.HoriCand.CPtoX( s_CandList.nFirstSelected + s_CandList.nHoriSelectedLen, FALSE, &nXRight );

            rc.right = rc.left + nXRight;
            rc.left += nXLeft;
            m_pDialog->DrawRect( &rc, SelBkColor );
            m_Elements.GetAt( 1 )->FontColor.Current = SelTextColor;
            m_pDialog->DrawText( s_CandList.HoriCand.GetBuffer() + s_CandList.nFirstSelected,
                                m_Elements.GetAt( 1 ), &rc, false, s_CandList.nHoriSelectedLen );
        }
    }
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::RenderComposition( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    RECT rcCaret = { 0, 0, 0, 0 };
    int nX, nXFirst;
    m_Buffer.CPtoX( m_nCaret, FALSE, &nX );
    m_Buffer.CPtoX( m_nFirstVisible, FALSE, &nXFirst );
    CDXUTElement *pElement = m_Elements.GetAt( 1 );

    // Get the required width
    RECT rc = { m_rcText.left + nX - nXFirst, m_rcText.top,
                m_rcText.left + nX - nXFirst, m_rcText.bottom };
    m_pDialog->CalcTextRect( s_CompString.GetBuffer(), pElement, &rc );

    // If the composition string is too long to fit within
    // the text area, move it to below the current line.
    // This matches the behavior of the default IME.
    if( rc.right > m_rcText.right )
        OffsetRect( &rc, m_rcText.left - rc.left, rc.bottom - rc.top );

    // Save the rectangle position for processing highlighted text.
    RECT rcFirst = rc;

    // Update s_ptCompString for RenderCandidateReadingWindow().
    s_ptCompString.x = rc.left; s_ptCompString.y = rc.top;

    D3DCOLOR TextColor = m_CompColor;
    // Render the window and string.
    // If the string is too long, we must wrap the line.
    pElement->FontColor.Current = TextColor;
    const WCHAR *pwszComp = s_CompString.GetBuffer();
    int nCharLeft = s_CompString.GetTextSize();
    for( ; ; )
    {
        // Find the last character that can be drawn on the same line.
        int nLastInLine;
        int bTrail;
        s_CompString.XtoCP( m_rcText.right - rc.left, &nLastInLine, &bTrail );
        int nNumCharToDraw = min( nCharLeft, nLastInLine );
        m_pDialog->CalcTextRect( pwszComp, pElement, &rc, nNumCharToDraw );

        // Draw the background
        // For Korean IME, blink the composition window background as if it
        // is a cursor.
        if( GetPrimaryLanguage() == LANG_KOREAN )
        {
            if( m_bCaretOn )
            {
                m_pDialog->DrawRect( &rc, m_CompWinColor );
            }
            else
            {
                // Not drawing composition string background. We
                // use the editbox's text color for composition
                // string text.
                TextColor = m_Elements.GetAt(0)->FontColor.States[DXUT_STATE_NORMAL];
            }
        } else
        {
            // Non-Korean IME. Always draw composition background.
            m_pDialog->DrawRect( &rc, m_CompWinColor );
        }

        // Draw the text
        pElement->FontColor.Current = TextColor;
        m_pDialog->DrawText( pwszComp, pElement, &rc, false, nNumCharToDraw );

        // Advance pointer and counter
        nCharLeft -= nNumCharToDraw;
        pwszComp += nNumCharToDraw;
        if( nCharLeft <= 0 )
            break;

        // Advance rectangle coordinates to beginning of next line
        OffsetRect( &rc, m_rcText.left - rc.left, rc.bottom - rc.top );
    }

    // Load the rect for the first line again.
    rc = rcFirst;

    // Inspect each character in the comp string.
    // For target-converted and target-non-converted characters,
    // we display a different background color so they appear highlighted.
    int nCharFirst = 0;
    nXFirst = 0;
    s_nFirstTargetConv = -1;
    BYTE *pAttr;
    const WCHAR *pcComp;
    for( pcComp = s_CompString.GetBuffer(), pAttr = s_abCompStringAttr;
          *pcComp != L'\0'; ++pcComp, ++pAttr )
    {
        D3DCOLOR bkColor;

        // Render a different background for this character
        int nXLeft, nXRight;
        s_CompString.CPtoX( int(pcComp - s_CompString.GetBuffer()), FALSE, &nXLeft );
        s_CompString.CPtoX( int(pcComp - s_CompString.GetBuffer()), TRUE, &nXRight );

        // Check if this character is off the right edge and should
        // be wrapped to the next line.
        if( nXRight - nXFirst > m_rcText.right - rc.left )
        {
            // Advance rectangle coordinates to beginning of next line
            OffsetRect( &rc, m_rcText.left - rc.left, rc.bottom - rc.top );

            // Update the line's first character information
            nCharFirst = int(pcComp - s_CompString.GetBuffer());
            s_CompString.CPtoX( nCharFirst, FALSE, &nXFirst );
        }

        // If the caret is on this character, save the coordinates
        // for drawing the caret later.
        if( s_nCompCaret == int(pcComp - s_CompString.GetBuffer()) )
        {
            rcCaret = rc;
            rcCaret.left += nXLeft - nXFirst - 1;
            rcCaret.right = rcCaret.left + 2;
        }

        // Set up color based on the character attribute
        if( *pAttr == ATTR_TARGET_CONVERTED )
        {
            pElement->FontColor.Current = m_CompTargetColor;
            bkColor = m_CompTargetBkColor;
        }
        else
        if( *pAttr == ATTR_TARGET_NOTCONVERTED )
        {
            pElement->FontColor.Current = m_CompTargetNonColor;
            bkColor = m_CompTargetNonBkColor;
        }
        else
        {
            continue;
        }

        RECT rcTarget = { rc.left + nXLeft - nXFirst, rc.top, rc.left + nXRight - nXFirst, rc.bottom };
        m_pDialog->DrawRect( &rcTarget, bkColor );
        m_pDialog->DrawText( pcComp, pElement, &rcTarget, false, 1 );

        // Record the first target converted character's index
        if( -1 == s_nFirstTargetConv )
            s_nFirstTargetConv = int(pAttr - s_abCompStringAttr);
    }

    // Render the composition caret
    if( m_bCaretOn )
    {
        // If the caret is at the very end, its position would not have
        // been computed in the above loop.  We compute it here.
        if( s_nCompCaret == s_CompString.GetTextSize() )
        {
            s_CompString.CPtoX( s_nCompCaret, FALSE, &nX );
            rcCaret = rc;
            rcCaret.left += nX - nXFirst - 1;
            rcCaret.right = rcCaret.left + 2;
        }

        m_pDialog->DrawRect( &rcCaret, m_CompCaretColor );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::RenderIndicator( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    CDXUTElement *pElement = m_Elements.GetAt( 9 );
    pElement->TextureColor.Blend( DXUT_STATE_NORMAL, fElapsedTime );

    m_pDialog->DrawSprite( pElement, &m_rcIndicator );
    RECT rc = m_rcIndicator;
    InflateRect( &rc, -m_nSpacing, -m_nSpacing );
    pElement->FontColor.Current = s_ImeState == IMEUI_STATE_ON && s_bEnableImeSystem ? m_IndicatorImeColor : m_IndicatorEngColor;
    RECT rcCalc = { 0, 0, 0, 0 };
    // If IME system is off, draw English indicator.
    WCHAR *pwszIndicator = s_bEnableImeSystem ? s_wszCurrIndicator : s_aszIndicator[0];

    m_pDialog->CalcTextRect( pwszIndicator, pElement, &rcCalc );
    m_pDialog->DrawText( pwszIndicator, pElement, &rc );
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::Render( IDirect3DDevice9* pd3dDevice, float fElapsedTime )
{
    if( m_bVisible == false )
        return;

    // If we have not computed the indicator symbol width,
    // do it.
    if( !m_nIndicatorWidth )
    {
        for( int i = 0; i < 5; ++i )
        {
            RECT rc = { 0, 0, 0, 0 };
            m_pDialog->CalcTextRect( s_aszIndicator[i], m_Elements.GetAt( 9 ), &rc );
            m_nIndicatorWidth = max( m_nIndicatorWidth, rc.right - rc.left );
        }
        // Update the rectangles now that we have the indicator's width
        UpdateRects();
    }

    // Let the parent render first (edit control)
    CDXUTEditBox::Render( pd3dDevice, fElapsedTime );

    CDXUTElement* pElement = GetElement( 1 );
    if( pElement )
    {
        s_CompString.SetFontIndex( pElement->iFont );
        s_CandList.HoriCand.SetFontIndex( pElement->iFont );
    }

    //
    // Now render the IME elements
    //
    if( m_bHasFocus )
    {
        // Render the input locale indicator
        RenderIndicator( pd3dDevice, fElapsedTime );

        // Display the composition string.
        // This method should also update s_ptCompString
        // for RenderCandidateReadingWindow.
        RenderComposition( pd3dDevice, fElapsedTime );

        // Display the reading/candidate window. RenderCandidateReadingWindow()
        // uses s_ptCompString to position itself.  s_ptCompString must have
        // been filled in by RenderComposition().
        if( s_bShowReadingWindow )
            // Reading window
            RenderCandidateReadingWindow( pd3dDevice, fElapsedTime, true );
        else
        if( s_CandList.bShowWindow )
            // Candidate list window
            RenderCandidateReadingWindow( pd3dDevice, fElapsedTime, false );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::CUniBuffer::InitializeUniscribe()
{
    WCHAR wszPath[MAX_PATH+1];
    if( !::GetSystemDirectory( wszPath, MAX_PATH+1 ) )
        return;
    lstrcatW( wszPath, UNISCRIBE_DLLNAME );
    s_hDll = LoadLibrary( wszPath );
    if( s_hDll )
    {
        FARPROC Temp;
        GETPROCADDRESS( s_hDll, ScriptApplyDigitSubstitution, Temp );
        GETPROCADDRESS( s_hDll, ScriptStringAnalyse, Temp );
        GETPROCADDRESS( s_hDll, ScriptStringCPtoX, Temp );
        GETPROCADDRESS( s_hDll, ScriptStringXtoCP, Temp );
        GETPROCADDRESS( s_hDll, ScriptStringFree, Temp );
        GETPROCADDRESS( s_hDll, ScriptString_pLogAttr, Temp );
        GETPROCADDRESS( s_hDll, ScriptString_pcOutChars, Temp );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::CUniBuffer::UninitializeUniscribe()
{
    if( s_hDll )
    {
        PLACEHOLDERPROC( ScriptApplyDigitSubstitution );
        PLACEHOLDERPROC( ScriptStringAnalyse );
        PLACEHOLDERPROC( ScriptStringCPtoX );
        PLACEHOLDERPROC( ScriptStringXtoCP );
        PLACEHOLDERPROC( ScriptStringFree );
        PLACEHOLDERPROC( ScriptString_pLogAttr );
        PLACEHOLDERPROC( ScriptString_pcOutChars );

        FreeLibrary( s_hDll );
        s_hDll = NULL;
    }
}


//--------------------------------------------------------------------------------------
bool CDXUTEditBox::CUniBuffer::Grow( int nNewSize )
{
    // If the current size is already the maximum allowed,
    // we can't possibly allocate more.
    if( m_nBufferSize == DXUT_MAX_EDITBOXLENGTH )
        return false;

    int nAllocateSize = ( nNewSize == -1 || nNewSize < m_nBufferSize * 2 ) ? ( m_nBufferSize ? m_nBufferSize * 2 : 256 ) : nNewSize * 2;

    // Cap the buffer size at the maximum allowed.
    if( nAllocateSize > DXUT_MAX_EDITBOXLENGTH )
        nAllocateSize = DXUT_MAX_EDITBOXLENGTH;

    WCHAR *pTempBuffer = new WCHAR[nAllocateSize];
    if( !pTempBuffer )
        return false;
    if( m_pwszBuffer )
        CopyMemory( pTempBuffer, m_pwszBuffer, (m_nTextSize + 1) * sizeof(WCHAR) );
    delete[] m_pwszBuffer;
    m_pwszBuffer = pTempBuffer;
    m_nBufferSize = nAllocateSize;
    return true;
}


//--------------------------------------------------------------------------------------
// Uniscribe -- Analyse() analyses the string in the buffer
//--------------------------------------------------------------------------------------
HRESULT CDXUTEditBox::CUniBuffer::Analyse()
{
    if( m_Analysis )
        _ScriptStringFree( &m_Analysis );

    SCRIPT_CONTROL ScriptControl; // For uniscribe
    SCRIPT_STATE   ScriptState;   // For uniscribe
    ZeroMemory( &ScriptControl, sizeof(ScriptControl) );
    ZeroMemory( &ScriptState, sizeof(ScriptState) );
    _ScriptApplyDigitSubstitution ( NULL, &ScriptControl, &ScriptState );

    DXUTFontNode* pFontNode = DXUTGetGlobalDialogResourceManager()->GetFontNode( m_iFont );

    HRESULT hr = _ScriptStringAnalyse( pFontNode->pFont ? pFontNode->pFont->GetDC() : NULL,
                                       m_pwszBuffer,
                                       m_nTextSize + 1,  // NULL is also analyzed.
                                       m_nTextSize * 3 / 2 + 16,
                                       -1,
                                       SSA_BREAK | SSA_GLYPHS | SSA_FALLBACK | SSA_LINK,
                                       0,
                                       &ScriptControl,
                                       &ScriptState,
                                       NULL,
                                       NULL,
                                       NULL,
                                       &m_Analysis );
    if( SUCCEEDED( hr ) )
        m_bAnalyseRequired = false;  // Analysis is up-to-date
    return hr;
}


//--------------------------------------------------------------------------------------
CDXUTEditBox::CUniBuffer::CUniBuffer( int nInitialSize )
{
    m_pwszBuffer = new WCHAR[nInitialSize];
    *m_pwszBuffer = 0;
    m_nBufferSize = nInitialSize;
    m_nTextSize = 0;
    m_bAnalyseRequired = true;
    m_Analysis = NULL;
    m_iFont = 0;
}


//--------------------------------------------------------------------------------------
CDXUTEditBox::CUniBuffer::~CUniBuffer()
{
    delete[] m_pwszBuffer;
    if( m_Analysis )
        _ScriptStringFree( &m_Analysis );
}


//--------------------------------------------------------------------------------------
bool CDXUTEditBox::CUniBuffer::SetBufferSize( int nSize )
{
    while( m_nBufferSize < nSize )
    {
        if( !Grow() )
            return false;
    }
    return true;
}


//--------------------------------------------------------------------------------------
WCHAR& CDXUTEditBox::CUniBuffer::operator[]( int n )  // No param checking
{
    // This version of operator[] is called only
    // if we are asking for write access, so
    // re-analysis is required.
    m_bAnalyseRequired = true;
    return m_pwszBuffer[n];
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::CUniBuffer::Clear()
{
    *m_pwszBuffer = L'\0';
    m_nTextSize = 0;
    m_bAnalyseRequired = true;
}


//--------------------------------------------------------------------------------------
// Inserts the char at specified index.
// If nIndex == -1, insert to the end.
//--------------------------------------------------------------------------------------
bool CDXUTEditBox::CUniBuffer::InsertChar( int nIndex, WCHAR wChar )
{
    assert( nIndex >= 0 );

    if( nIndex > m_nTextSize )
        return false;  // invalid index

    // Check for maximum length allowed
    if( GetTextSize() + 1 >= DXUT_MAX_EDITBOXLENGTH )
        return false;

    if( m_nTextSize + 1 >= m_nBufferSize )
    {
        if( !Grow() )
            return false;  // out of memory
    }

    MoveMemory( m_pwszBuffer + nIndex + 1, m_pwszBuffer + nIndex, sizeof(WCHAR) * ( m_nTextSize - nIndex + 1 ) );
    m_pwszBuffer[ nIndex ] = wChar;
    ++m_nTextSize;
    m_bAnalyseRequired = true;

    return true;
}


//--------------------------------------------------------------------------------------
// Removes the char at specified index.
// If nIndex == -1, remove the last char.
//--------------------------------------------------------------------------------------
bool CDXUTEditBox::CUniBuffer::RemoveChar( int nIndex )
{
    if( !m_nTextSize || nIndex < 0 || nIndex >= m_nTextSize )
        return false;  // Invalid index

    MoveMemory( m_pwszBuffer + nIndex, m_pwszBuffer + nIndex + 1, sizeof(WCHAR) * ( m_nTextSize - nIndex ) );
    --m_nTextSize;
    m_bAnalyseRequired = true;
    return true;
}


//--------------------------------------------------------------------------------------
// Inserts the first nCount characters of the string pStr at specified index.
// If nCount == -1, the entire string is inserted.
// If nIndex == -1, insert to the end.
//--------------------------------------------------------------------------------------
bool CDXUTEditBox::CUniBuffer::InsertString( int nIndex, const WCHAR *pStr, int nCount )
{
    assert( nIndex >= 0 );

    if( nIndex > m_nTextSize )
        return false;  // invalid index

    if( -1 == nCount )
        nCount = lstrlenW( pStr );

    // Check for maximum length allowed
    if( GetTextSize() + nCount >= DXUT_MAX_EDITBOXLENGTH )
        return false;

    if( m_nTextSize + nCount >= m_nBufferSize )
    {
        if( !Grow( m_nTextSize + nCount + 1 ) )
            return false;  // out of memory
    }

    MoveMemory( m_pwszBuffer + nIndex + nCount, m_pwszBuffer + nIndex, sizeof(WCHAR) * ( m_nTextSize - nIndex + 1 ) );
    CopyMemory( m_pwszBuffer + nIndex, pStr, nCount * sizeof(WCHAR) );
    m_nTextSize += nCount;
    m_bAnalyseRequired = true;

    return true;
}


//--------------------------------------------------------------------------------------
bool CDXUTEditBox::CUniBuffer::SetText( LPCWSTR wszText )
{
    assert( wszText != NULL );

    int nRequired = int(wcslen( wszText ) + 1);

    // Check for maximum length allowed
    if( nRequired >= DXUT_MAX_EDITBOXLENGTH )
        return false;

    while( GetBufferSize() < nRequired )
        if( !Grow() )
                break;
    // Check again in case out of memory occurred inside while loop.
    if( GetBufferSize() >= nRequired )
    {
        wcscpy( m_pwszBuffer, wszText );
        m_nTextSize = nRequired - 1;
        m_bAnalyseRequired = true;
        return true;
    }
    else
        return false;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTEditBox::CUniBuffer::CPtoX( int nCP, BOOL bTrail, int *pX )
{
    assert( pX );
    *pX = 0;  // Default

    HRESULT hr = S_OK;
    if( m_bAnalyseRequired )
        hr = Analyse();

    if( SUCCEEDED( hr ) )
        hr = _ScriptStringCPtoX( m_Analysis, nCP, bTrail, pX );

    return hr;
}


//--------------------------------------------------------------------------------------
HRESULT CDXUTEditBox::CUniBuffer::XtoCP( int nX, int *pCP, int *pnTrail )
{
    assert( pCP && pnTrail );
    *pCP = 0; *pnTrail = FALSE;  // Default

    HRESULT hr = S_OK;
    if( m_bAnalyseRequired )
        hr = Analyse();

    if( SUCCEEDED( hr ) )
        hr = _ScriptStringXtoCP( m_Analysis, nX, pCP, pnTrail );

    // If the coordinate falls outside the text region, we
    // can get character positions that don't exist.  We must
    // filter them here and convert them to those that do exist.
    if( *pCP == -1 && *pnTrail == TRUE )
    {
        *pCP = 0; *pnTrail = FALSE;
    } else
    if( *pCP > m_nTextSize && *pnTrail == FALSE )
    {
        *pCP = m_nTextSize; *pnTrail = TRUE;
    }

    return hr;
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::CUniBuffer::GetPriorItemPos( int nCP, int *pPrior )
{
    *pPrior = nCP;  // Default is the char itself

    if( m_bAnalyseRequired )
        if( FAILED( Analyse() ) )
            return;

    const SCRIPT_LOGATTR *pLogAttr = _ScriptString_pLogAttr( m_Analysis );
    if( !pLogAttr )
        return;

    if( !_ScriptString_pcOutChars( m_Analysis ) )
        return;
    int nInitial = *_ScriptString_pcOutChars( m_Analysis );
    if( nCP - 1 < nInitial )
        nInitial = nCP - 1;
    for( int i = nInitial; i > 0; --i )
        if( pLogAttr[i].fWordStop ||       // Either the fWordStop flag is set
            ( !pLogAttr[i].fWhiteSpace &&  // Or the previous char is whitespace but this isn't.
                pLogAttr[i-1].fWhiteSpace ) )
        {
            *pPrior = i;
            return;
        }
    // We have reached index 0.  0 is always a break point, so simply return it.
    *pPrior = 0;
}
    

//--------------------------------------------------------------------------------------
void CDXUTEditBox::CUniBuffer::GetNextItemPos( int nCP, int *pPrior )
{
    *pPrior = nCP;  // Default is the char itself

    HRESULT hr = S_OK;
    if( m_bAnalyseRequired )
        hr = Analyse();
    if( FAILED( hr ) )
        return;

    const SCRIPT_LOGATTR *pLogAttr = _ScriptString_pLogAttr( m_Analysis );
    if( !pLogAttr )
        return;

    if( !_ScriptString_pcOutChars( m_Analysis ) )
        return;
    int nInitial = *_ScriptString_pcOutChars( m_Analysis );
    if( nCP + 1 < nInitial )
        nInitial = nCP + 1;
    for( int i = nInitial; i < *_ScriptString_pcOutChars( m_Analysis ) - 1; ++i )
    {
        if( pLogAttr[i].fWordStop )      // Either the fWordStop flag is set
        {
            *pPrior = i;
            return;
        }
        else
        if( pLogAttr[i].fWhiteSpace &&  // Or this whitespace but the next char isn't.
            !pLogAttr[i+1].fWhiteSpace )
        {
            *pPrior = i+1;  // The next char is a word stop
            return;
        }
    }
    // We have reached the end. It's always a word stop, so simply return it.
    *pPrior = *_ScriptString_pcOutChars( m_Analysis ) - 1;
}


//--------------------------------------------------------------------------------------
void CDXUTEditBox::ResetCaretBlink()
{
    m_bCaretOn = true;
    m_dfLastBlink = DXUTGetGlobalTimer()->GetAbsoluteTime();
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::InitializeImm()
{
    FARPROC Temp;

    WCHAR wszPath[MAX_PATH+1];
    if( !::GetSystemDirectory( wszPath, MAX_PATH+1 ) )
        return;
    lstrcatW( wszPath, IMM32_DLLNAME );
    s_hDllImm32 = LoadLibrary( wszPath );
    if( s_hDllImm32 )
    {
        GETPROCADDRESS( s_hDllImm32, ImmLockIMC, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmUnlockIMC, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmLockIMCC, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmUnlockIMCC, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmDisableTextFrameService, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetCompositionStringW, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetCandidateListW, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetContext, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmReleaseContext, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmAssociateContext, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetOpenStatus, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmSetOpenStatus, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetConversionStatus, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetDefaultIMEWnd, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetIMEFileNameA, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmGetVirtualKey, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmNotifyIME, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmSetConversionStatus, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmSimulateHotKey, Temp );
        GETPROCADDRESS( s_hDllImm32, ImmIsIME, Temp );
    }

    if( !::GetSystemDirectory( wszPath, MAX_PATH+1 ) )
        return;
    lstrcatW( wszPath, VER_DLLNAME );
    s_hDllVer = LoadLibrary( wszPath );
    if( s_hDllVer )
    {
        GETPROCADDRESS( s_hDllVer, VerQueryValueA, Temp );
        GETPROCADDRESS( s_hDllVer, GetFileVersionInfoA, Temp );
        GETPROCADDRESS( s_hDllVer, GetFileVersionInfoSizeA, Temp );
    }
}


//--------------------------------------------------------------------------------------
void CDXUTIMEEditBox::UninitializeImm()
{
    if( s_hDllImm32 )
    {
        PLACEHOLDERPROC( ImmLockIMC );
        PLACEHOLDERPROC( ImmUnlockIMC );
        PLACEHOLDERPROC( ImmLockIMCC );
        PLACEHOLDERPROC( ImmUnlockIMCC );
        PLACEHOLDERPROC( ImmDisableTextFrameService );
        PLACEHOLDERPROC( ImmGetCompositionStringW );
        PLACEHOLDERPROC( ImmGetCandidateListW );
        PLACEHOLDERPROC( ImmGetContext );
        PLACEHOLDERPROC( ImmReleaseContext );
        PLACEHOLDERPROC( ImmAssociateContext );
        PLACEHOLDERPROC( ImmGetOpenStatus );
        PLACEHOLDERPROC( ImmSetOpenStatus );
        PLACEHOLDERPROC( ImmGetConversionStatus );
        PLACEHOLDERPROC( ImmGetDefaultIMEWnd );
        PLACEHOLDERPROC( ImmGetIMEFileNameA );
        PLACEHOLDERPROC( ImmGetVirtualKey );
        PLACEHOLDERPROC( ImmNotifyIME );
        PLACEHOLDERPROC( ImmSetConversionStatus );
        PLACEHOLDERPROC( ImmSimulateHotKey );
        PLACEHOLDERPROC( ImmIsIME );

        FreeLibrary( s_hDllImm32 );
        s_hDllImm32 = NULL;
    }
    if( s_hDllIme )
    {
        PLACEHOLDERPROC( GetReadingString );
        PLACEHOLDERPROC( ShowReadingWindow );

        FreeLibrary( s_hDllIme );
        s_hDllIme = NULL;
    }
    if( s_hDllVer )
    {
        PLACEHOLDERPROC( VerQueryValueA );
        PLACEHOLDERPROC( GetFileVersionInfoA );
        PLACEHOLDERPROC( GetFileVersionInfoSizeA );

        FreeLibrary( s_hDllVer );
        s_hDllVer = NULL;
    }
}


//--------------------------------------------------------------------------------------
void DXUTBlendColor::Init( D3DCOLOR defaultColor, D3DCOLOR disabledColor, D3DCOLOR hiddenColor )
{
    for( int i=0; i < MAX_CONTROL_STATES; i++ )
    {
        States[ i ] = defaultColor;
    }

    States[ DXUT_STATE_DISABLED ] = disabledColor;
    States[ DXUT_STATE_HIDDEN ] = hiddenColor;
    Current = hiddenColor;
}


//--------------------------------------------------------------------------------------
void DXUTBlendColor::Blend( UINT iState, float fElapsedTime, float fRate )
{
    D3DXCOLOR destColor = States[ iState ];
    D3DXColorLerp( &Current, &Current, &destColor, 1.0f - powf( fRate, 30 * fElapsedTime ) );
}



//--------------------------------------------------------------------------------------
void CDXUTElement::SetTexture( UINT iTexture, RECT* prcTexture, D3DCOLOR defaultTextureColor )
{
    this->iTexture = iTexture;
    
    if( prcTexture )
        rcTexture = *prcTexture;
    else
        SetRectEmpty( &rcTexture );
    
    TextureColor.Init( defaultTextureColor );
}
    

//--------------------------------------------------------------------------------------
void CDXUTElement::SetFont( UINT iFont, D3DCOLOR defaultFontColor, DWORD dwTextFormat )
{
    this->iFont = iFont;
    this->dwTextFormat = dwTextFormat;

    FontColor.Init( defaultFontColor );
}

//--------------------------------------------------------------------------------------
void CDXUTElement::Refresh()
{
    TextureColor.Current = TextureColor.States[ DXUT_STATE_HIDDEN ];
    FontColor.Current = FontColor.States[ DXUT_STATE_HIDDEN ];
}







