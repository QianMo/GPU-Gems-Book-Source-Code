//-----------------------------------------------------------------------------
// File: D3DUtil.cpp
//
// Desc: Shortcut macros and functions for using DX objects
//-----------------------------------------------------------------------------
#define STRICT
#include <Windows.h>
#include <WindowsX.h>
#include <tchar.h>
#include <stdio.h>
#include "D3DUtil.h"
#include "DXUtil.h"
#include "D3DX9.h"




//-----------------------------------------------------------------------------
// Name: D3DUtil_InitMaterial()
// Desc: Initializes a D3DMATERIAL9 structure, setting the diffuse and ambient
//       colors. It does not set emissive or specular colors.
//-----------------------------------------------------------------------------
VOID D3DUtil_InitMaterial( D3DMATERIAL9& mtrl, FLOAT r, FLOAT g, FLOAT b,
                           FLOAT a )
{
    ZeroMemory( &mtrl, sizeof(D3DMATERIAL9) );
    mtrl.Diffuse.r = mtrl.Ambient.r = r;
    mtrl.Diffuse.g = mtrl.Ambient.g = g;
    mtrl.Diffuse.b = mtrl.Ambient.b = b;
    mtrl.Diffuse.a = mtrl.Ambient.a = a;
}




//-----------------------------------------------------------------------------
// Name: D3DUtil_InitLight()
// Desc: Initializes a D3DLIGHT structure, setting the light position. The
//       diffuse color is set to white; specular and ambient are left as black.
//-----------------------------------------------------------------------------
VOID D3DUtil_InitLight( D3DLIGHT9& light, D3DLIGHTTYPE ltType,
                        FLOAT x, FLOAT y, FLOAT z )
{
    D3DXVECTOR3 vecLightDirUnnormalized(x, y, z);
    ZeroMemory( &light, sizeof(D3DLIGHT9) );
    light.Type        = ltType;
    light.Diffuse.r   = 1.0f;
    light.Diffuse.g   = 1.0f;
    light.Diffuse.b   = 1.0f;
    D3DXVec3Normalize( (D3DXVECTOR3*)&light.Direction, &vecLightDirUnnormalized );
    light.Position.x   = x;
    light.Position.y   = y;
    light.Position.z   = z;
    light.Range        = 1000.0f;
}




//-----------------------------------------------------------------------------
// Name: D3DUtil_CreateTexture()
// Desc: Helper function to create a texture. It checks the root path first,
//       then tries the DXSDK media path (as specified in the system registry).
//-----------------------------------------------------------------------------
HRESULT D3DUtil_CreateTexture( LPDIRECT3DDEVICE9 pd3dDevice, TCHAR* strTexture,
                               LPDIRECT3DTEXTURE9* ppTexture, D3DFORMAT d3dFormat )
{
    HRESULT hr;
    TCHAR strPath[MAX_PATH];

    // Get the path to the texture
    if( FAILED( hr = DXUtil_FindMediaFileCb( strPath, sizeof(strPath), strTexture ) ) )
        return hr;

    // Create the texture using D3DX
    return D3DXCreateTextureFromFileEx( pd3dDevice, strPath, 
                D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, d3dFormat, 
                D3DPOOL_MANAGED, D3DX_FILTER_TRIANGLE|D3DX_FILTER_MIRROR, 
                D3DX_FILTER_TRIANGLE|D3DX_FILTER_MIRROR, 0, NULL, NULL, ppTexture );
}




//-----------------------------------------------------------------------------
// Name: D3DUtil_GetCubeMapViewMatrix()
// Desc: Returns a view matrix for rendering to a face of a cubemap.
//-----------------------------------------------------------------------------
D3DXMATRIX D3DUtil_GetCubeMapViewMatrix( DWORD dwFace )
{
    D3DXVECTOR3 vEyePt   = D3DXVECTOR3( 0.0f, 0.0f, 0.0f );
    D3DXVECTOR3 vLookDir;
    D3DXVECTOR3 vUpDir;

    switch( dwFace )
    {
        case D3DCUBEMAP_FACE_POSITIVE_X:
            vLookDir = D3DXVECTOR3( 1.0f, 0.0f, 0.0f );
            vUpDir   = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_X:
            vLookDir = D3DXVECTOR3(-1.0f, 0.0f, 0.0f );
            vUpDir   = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
        case D3DCUBEMAP_FACE_POSITIVE_Y:
            vLookDir = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            vUpDir   = D3DXVECTOR3( 0.0f, 0.0f,-1.0f );
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_Y:
            vLookDir = D3DXVECTOR3( 0.0f,-1.0f, 0.0f );
            vUpDir   = D3DXVECTOR3( 0.0f, 0.0f, 1.0f );
            break;
        case D3DCUBEMAP_FACE_POSITIVE_Z:
            vLookDir = D3DXVECTOR3( 0.0f, 0.0f, 1.0f );
            vUpDir   = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_Z:
            vLookDir = D3DXVECTOR3( 0.0f, 0.0f,-1.0f );
            vUpDir   = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
    }

    // Set the view transform for this cubemap surface
    D3DXMATRIXA16 matView;
    D3DXMatrixLookAtLH( &matView, &vEyePt, &vLookDir, &vUpDir );
    return matView;
}




//-----------------------------------------------------------------------------
// Name: D3DUtil_GetRotationFromCursor()
// Desc: Returns a quaternion for the rotation implied by the window's cursor
//       position.
//-----------------------------------------------------------------------------
D3DXQUATERNION D3DUtil_GetRotationFromCursor( HWND hWnd,
                                              FLOAT fTrackBallRadius )
{
    POINT pt;
    RECT  rc;
    GetCursorPos( &pt );
    GetClientRect( hWnd, &rc );
    ScreenToClient( hWnd, &pt );
    FLOAT sx = ( ( ( 2.0f * pt.x ) / (rc.right-rc.left) ) - 1 );
    FLOAT sy = ( ( ( 2.0f * pt.y ) / (rc.bottom-rc.top) ) - 1 );
    FLOAT sz;

    if( sx == 0.0f && sy == 0.0f )
        return D3DXQUATERNION( 0.0f, 0.0f, 0.0f, 1.0f );

    FLOAT d2 = sqrtf( sx*sx + sy*sy );

    if( d2 < fTrackBallRadius * 0.70710678118654752440 ) // Inside sphere
        sz = sqrtf( fTrackBallRadius*fTrackBallRadius - d2*d2 );
    else                                                 // On hyperbola
        sz = (fTrackBallRadius*fTrackBallRadius) / (2.0f*d2);

    // Get two points on trackball's sphere
    D3DXVECTOR3 p1( sx, sy, sz );
    D3DXVECTOR3 p2( 0.0f, 0.0f, fTrackBallRadius );

    // Get axis of rotation, which is cross product of p1 and p2
    D3DXVECTOR3 vAxis;
    D3DXVec3Cross( &vAxis, &p1, &p2);

    // Calculate angle for the rotation about that axis
    D3DXVECTOR3 vecDiff = p2-p1;
    FLOAT t = D3DXVec3Length( &vecDiff ) / ( 2.0f*fTrackBallRadius );
    if( t > +1.0f) t = +1.0f;
    if( t < -1.0f) t = -1.0f;
    FLOAT fAngle = 2.0f * asinf( t );

    // Convert axis to quaternion
    D3DXQUATERNION quat;
    D3DXQuaternionRotationAxis( &quat, &vAxis, fAngle );
    return quat;
}




//-----------------------------------------------------------------------------
// Name: D3DUtil_SetDeviceCursor
// Desc: Gives the D3D device a cursor with image and hotspot from hCursor.
//-----------------------------------------------------------------------------
HRESULT D3DUtil_SetDeviceCursor( LPDIRECT3DDEVICE9 pd3dDevice, HCURSOR hCursor,
                                 BOOL bAddWatermark )
{
    HRESULT hr = E_FAIL;
    ICONINFO iconinfo;
    BOOL bBWCursor;
    LPDIRECT3DSURFACE9 pCursorSurface = NULL;
    HDC hdcColor = NULL;
    HDC hdcMask = NULL;
    HDC hdcScreen = NULL;
    BITMAP bm;
    DWORD dwWidth;
    DWORD dwHeightSrc;
    DWORD dwHeightDest;
    COLORREF crColor;
    COLORREF crMask;
    UINT x;
    UINT y;
    BITMAPINFO bmi;
    COLORREF* pcrArrayColor = NULL;
    COLORREF* pcrArrayMask = NULL;
    DWORD* pBitmap;
    HGDIOBJ hgdiobjOld;

    ZeroMemory( &iconinfo, sizeof(iconinfo) );
    if( !GetIconInfo( hCursor, &iconinfo ) )
        goto End;

    if (0 == GetObject((HGDIOBJ)iconinfo.hbmMask, sizeof(BITMAP), (LPVOID)&bm))
        goto End;
    dwWidth = bm.bmWidth;
    dwHeightSrc = bm.bmHeight;

    if( iconinfo.hbmColor == NULL )
    {
        bBWCursor = TRUE;
        dwHeightDest = dwHeightSrc / 2;
    }
    else 
    {
        bBWCursor = FALSE;
        dwHeightDest = dwHeightSrc;
    }

    // Create a surface for the fullscreen cursor
    if( FAILED( hr = pd3dDevice->CreateOffscreenPlainSurface( dwWidth, dwHeightDest, 
        D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH, &pCursorSurface, NULL ) ) )
    {
        goto End;
    }

    pcrArrayMask = new DWORD[dwWidth * dwHeightSrc];

    ZeroMemory(&bmi, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
    bmi.bmiHeader.biWidth = dwWidth;
    bmi.bmiHeader.biHeight = dwHeightSrc;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    hdcScreen = GetDC( NULL );
    hdcMask = CreateCompatibleDC( hdcScreen );
    if( hdcMask == NULL )
    {
        hr = E_FAIL;
        goto End;
    }
    hgdiobjOld = SelectObject(hdcMask, iconinfo.hbmMask);
    GetDIBits(hdcMask, iconinfo.hbmMask, 0, dwHeightSrc, 
        pcrArrayMask, &bmi, DIB_RGB_COLORS);
    SelectObject(hdcMask, hgdiobjOld);

    if (!bBWCursor)
    {
        pcrArrayColor = new DWORD[dwWidth * dwHeightDest];
        hdcColor = CreateCompatibleDC( hdcScreen );
        if( hdcColor == NULL )
        {
            hr = E_FAIL;
            goto End;
        }
        SelectObject(hdcColor, iconinfo.hbmColor);
        GetDIBits(hdcColor, iconinfo.hbmColor, 0, dwHeightDest, 
            pcrArrayColor, &bmi, DIB_RGB_COLORS);
    }

    // Transfer cursor image into the surface
    D3DLOCKED_RECT lr;
    pCursorSurface->LockRect( &lr, NULL, 0 );
    pBitmap = (DWORD*)lr.pBits;
    for( y = 0; y < dwHeightDest; y++ )
    {
        for( x = 0; x < dwWidth; x++ )
        {
            if (bBWCursor)
            {
                crColor = pcrArrayMask[dwWidth*(dwHeightDest-1-y) + x];
                crMask = pcrArrayMask[dwWidth*(dwHeightSrc-1-y) + x];
            }
            else
            {
                crColor = pcrArrayColor[dwWidth*(dwHeightDest-1-y) + x];
                crMask = pcrArrayMask[dwWidth*(dwHeightDest-1-y) + x];
            }
            if (crMask == 0)
                pBitmap[dwWidth*y + x] = 0xff000000 | crColor;
            else
                pBitmap[dwWidth*y + x] = 0x00000000;

            // It may be helpful to make the D3D cursor look slightly 
            // different from the Windows cursor so you can distinguish 
            // between the two when developing/testing code.  When
            // bAddWatermark is TRUE, the following code adds some
            // small grey "D3D" characters to the upper-left corner of
            // the D3D cursor image.
            if( bAddWatermark && x < 12 && y < 5 )
            {
                // 11.. 11.. 11.. .... CCC0
                // 1.1. ..1. 1.1. .... A2A0
                // 1.1. .1.. 1.1. .... A4A0
                // 1.1. ..1. 1.1. .... A2A0
                // 11.. 11.. 11.. .... CCC0

                const WORD wMask[5] = { 0xccc0, 0xa2a0, 0xa4a0, 0xa2a0, 0xccc0 };
                if( wMask[y] & (1 << (15 - x)) )
                {
                    pBitmap[dwWidth*y + x] |= 0xff808080;
                }
            }
        }
    }
    pCursorSurface->UnlockRect();

    // Set the device cursor
    if( FAILED( hr = pd3dDevice->SetCursorProperties( iconinfo.xHotspot, 
        iconinfo.yHotspot, pCursorSurface ) ) )
    {
        goto End;
    }

    hr = S_OK;

End:
    if( iconinfo.hbmMask != NULL )
        DeleteObject( iconinfo.hbmMask );
    if( iconinfo.hbmColor != NULL )
        DeleteObject( iconinfo.hbmColor );
    if( hdcScreen != NULL )
        ReleaseDC( NULL, hdcScreen );
    if( hdcColor != NULL )
        DeleteDC( hdcColor );
    if( hdcMask != NULL )
        DeleteDC( hdcMask );
    SAFE_DELETE_ARRAY( pcrArrayColor );
    SAFE_DELETE_ARRAY( pcrArrayMask );
    SAFE_RELEASE( pCursorSurface );
    return hr;
}



//-----------------------------------------------------------------------------
// Name: D3DFormatToString
// Desc: Returns the string for the given D3DFORMAT.
//-----------------------------------------------------------------------------
TCHAR* D3DUtil_D3DFormatToString( D3DFORMAT format, bool bWithPrefix )
{
    TCHAR* pstr = NULL;
    switch( format )
    {
    case D3DFMT_UNKNOWN:         pstr = TEXT("D3DFMT_UNKNOWN"); break;
    case D3DFMT_R8G8B8:          pstr = TEXT("D3DFMT_R8G8B8"); break;
    case D3DFMT_A8R8G8B8:        pstr = TEXT("D3DFMT_A8R8G8B8"); break;
    case D3DFMT_X8R8G8B8:        pstr = TEXT("D3DFMT_X8R8G8B8"); break;
    case D3DFMT_R5G6B5:          pstr = TEXT("D3DFMT_R5G6B5"); break;
    case D3DFMT_X1R5G5B5:        pstr = TEXT("D3DFMT_X1R5G5B5"); break;
    case D3DFMT_A1R5G5B5:        pstr = TEXT("D3DFMT_A1R5G5B5"); break;
    case D3DFMT_A4R4G4B4:        pstr = TEXT("D3DFMT_A4R4G4B4"); break;
    case D3DFMT_R3G3B2:          pstr = TEXT("D3DFMT_R3G3B2"); break;
    case D3DFMT_A8:              pstr = TEXT("D3DFMT_A8"); break;
    case D3DFMT_A8R3G3B2:        pstr = TEXT("D3DFMT_A8R3G3B2"); break;
    case D3DFMT_X4R4G4B4:        pstr = TEXT("D3DFMT_X4R4G4B4"); break;
    case D3DFMT_A2B10G10R10:     pstr = TEXT("D3DFMT_A2B10G10R10"); break;
    case D3DFMT_A8B8G8R8:        pstr = TEXT("D3DFMT_A8B8G8R8"); break;
    case D3DFMT_X8B8G8R8:        pstr = TEXT("D3DFMT_X8B8G8R8"); break;
    case D3DFMT_G16R16:          pstr = TEXT("D3DFMT_G16R16"); break;
    case D3DFMT_A2R10G10B10:     pstr = TEXT("D3DFMT_A2R10G10B10"); break;
    case D3DFMT_A16B16G16R16:    pstr = TEXT("D3DFMT_A16B16G16R16"); break;
    case D3DFMT_A8P8:            pstr = TEXT("D3DFMT_A8P8"); break;
    case D3DFMT_P8:              pstr = TEXT("D3DFMT_P8"); break;
    case D3DFMT_L8:              pstr = TEXT("D3DFMT_L8"); break;
    case D3DFMT_A8L8:            pstr = TEXT("D3DFMT_A8L8"); break;
    case D3DFMT_A4L4:            pstr = TEXT("D3DFMT_A4L4"); break;
    case D3DFMT_V8U8:            pstr = TEXT("D3DFMT_V8U8"); break;
    case D3DFMT_L6V5U5:          pstr = TEXT("D3DFMT_L6V5U5"); break;
    case D3DFMT_X8L8V8U8:        pstr = TEXT("D3DFMT_X8L8V8U8"); break;
    case D3DFMT_Q8W8V8U8:        pstr = TEXT("D3DFMT_Q8W8V8U8"); break;
    case D3DFMT_V16U16:          pstr = TEXT("D3DFMT_V16U16"); break;
    case D3DFMT_A2W10V10U10:     pstr = TEXT("D3DFMT_A2W10V10U10"); break;
    case D3DFMT_UYVY:            pstr = TEXT("D3DFMT_UYVY"); break;
    case D3DFMT_YUY2:            pstr = TEXT("D3DFMT_YUY2"); break;
    case D3DFMT_DXT1:            pstr = TEXT("D3DFMT_DXT1"); break;
    case D3DFMT_DXT2:            pstr = TEXT("D3DFMT_DXT2"); break;
    case D3DFMT_DXT3:            pstr = TEXT("D3DFMT_DXT3"); break;
    case D3DFMT_DXT4:            pstr = TEXT("D3DFMT_DXT4"); break;
    case D3DFMT_DXT5:            pstr = TEXT("D3DFMT_DXT5"); break;
    case D3DFMT_D16_LOCKABLE:    pstr = TEXT("D3DFMT_D16_LOCKABLE"); break;
    case D3DFMT_D32:             pstr = TEXT("D3DFMT_D32"); break;
    case D3DFMT_D15S1:           pstr = TEXT("D3DFMT_D15S1"); break;
    case D3DFMT_D24S8:           pstr = TEXT("D3DFMT_D24S8"); break;
    case D3DFMT_D24X8:           pstr = TEXT("D3DFMT_D24X8"); break;
    case D3DFMT_D24X4S4:         pstr = TEXT("D3DFMT_D24X4S4"); break;
    case D3DFMT_D16:             pstr = TEXT("D3DFMT_D16"); break;
    case D3DFMT_L16:             pstr = TEXT("D3DFMT_L16"); break;
    case D3DFMT_VERTEXDATA:      pstr = TEXT("D3DFMT_VERTEXDATA"); break;
    case D3DFMT_INDEX16:         pstr = TEXT("D3DFMT_INDEX16"); break;
    case D3DFMT_INDEX32:         pstr = TEXT("D3DFMT_INDEX32"); break;
    case D3DFMT_Q16W16V16U16:    pstr = TEXT("D3DFMT_Q16W16V16U16"); break;
    case D3DFMT_MULTI2_ARGB8:    pstr = TEXT("D3DFMT_MULTI2_ARGB8"); break;
    case D3DFMT_R16F:            pstr = TEXT("D3DFMT_R16F"); break;
    case D3DFMT_G16R16F:         pstr = TEXT("D3DFMT_G16R16F"); break;
    case D3DFMT_A16B16G16R16F:   pstr = TEXT("D3DFMT_A16B16G16R16F"); break;
    case D3DFMT_R32F:            pstr = TEXT("D3DFMT_R32F"); break;
    case D3DFMT_G32R32F:         pstr = TEXT("D3DFMT_G32R32F"); break;
    case D3DFMT_A32B32G32R32F:   pstr = TEXT("D3DFMT_A32B32G32R32F"); break;
    case D3DFMT_CxV8U8:          pstr = TEXT("D3DFMT_CxV8U8"); break;
    default:                     pstr = TEXT("Unknown format"); break;
    }
    if( bWithPrefix || _tcsstr( pstr, TEXT("D3DFMT_") )== NULL )
        return pstr;
    else
        return pstr + lstrlen( TEXT("D3DFMT_") );
}


//-----------------------------------------------------------------------------
// Name: D3DXQuaternionUnitAxisToUnitAxis2
// Desc: Axis to axis quaternion double angle (no normalization)
//       Takes two points on unit sphere an angle THETA apart, returns
//       quaternion that represents a rotation around cross product by 2*THETA.
//-----------------------------------------------------------------------------
inline D3DXQUATERNION* WINAPI D3DXQuaternionUnitAxisToUnitAxis2
( D3DXQUATERNION *pOut, const D3DXVECTOR3 *pvFrom, const D3DXVECTOR3 *pvTo)
{
    D3DXVECTOR3 vAxis;
    D3DXVec3Cross(&vAxis, pvFrom, pvTo);    // proportional to sin(theta)
    pOut->x = vAxis.x;
    pOut->y = vAxis.y;
    pOut->z = vAxis.z;
    pOut->w = D3DXVec3Dot( pvFrom, pvTo );
    return pOut;
}




//-----------------------------------------------------------------------------
// Name: D3DXQuaternionAxisToAxis
// Desc: Axis to axis quaternion 
//       Takes two points on unit sphere an angle THETA apart, returns
//       quaternion that represents a rotation around cross product by theta.
//-----------------------------------------------------------------------------
inline D3DXQUATERNION* WINAPI D3DXQuaternionAxisToAxis
( D3DXQUATERNION *pOut, const D3DXVECTOR3 *pvFrom, const D3DXVECTOR3 *pvTo)
{
    D3DXVECTOR3 vA, vB;
    D3DXVec3Normalize(&vA, pvFrom);
    D3DXVec3Normalize(&vB, pvTo);
    D3DXVECTOR3 vHalf(vA + vB);
    D3DXVec3Normalize(&vHalf, &vHalf);
    return D3DXQuaternionUnitAxisToUnitAxis2(pOut, &vA, &vHalf);
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DArcBall::CD3DArcBall()
{
    Init();
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
void CD3DArcBall::Init()
{
    D3DXQuaternionIdentity( &m_qDown );
    D3DXQuaternionIdentity( &m_qNow );
    D3DXMatrixIdentity( &m_matRotation );
    D3DXMatrixIdentity( &m_matRotationDelta );
    D3DXMatrixIdentity( &m_matTranslation );
    D3DXMatrixIdentity( &m_matTranslationDelta );
    m_bDrag = FALSE;
    m_fRadiusTranslation = 1.0f;
    m_bRightHanded = FALSE;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
VOID CD3DArcBall::SetWindow( int iWidth, int iHeight, float fRadius )
{
    // Set ArcBall info
    m_iWidth  = iWidth;
    m_iHeight = iHeight;
    m_fRadius = fRadius;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
D3DXVECTOR3 CD3DArcBall::ScreenToVector( int sx, int sy )
{
    // Scale to screen
    FLOAT x   = -(sx - m_iWidth/2)  / (m_fRadius*m_iWidth/2);
    FLOAT y   =  (sy - m_iHeight/2) / (m_fRadius*m_iHeight/2);

    if( m_bRightHanded )
    {
        x = -x;
        y = -y;
    }

    FLOAT z   = 0.0f;
    FLOAT mag = x*x + y*y;

    if( mag > 1.0f )
    {
        FLOAT scale = 1.0f/sqrtf(mag);
        x *= scale;
        y *= scale;
    }
    else
        z = sqrtf( 1.0f - mag );

    // Return vector
    return D3DXVECTOR3( x, y, z );
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
VOID CD3DArcBall::SetRadius( FLOAT fRadius )
{
    m_fRadiusTranslation = fRadius;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
LRESULT CD3DArcBall::HandleMouseMessages( HWND hWnd, UINT uMsg, WPARAM wParam,
                                          LPARAM lParam )
{
    UNREFERENCED_PARAMETER( hWnd );

    static int         iCurMouseX;      // Saved mouse position
    static int         iCurMouseY;
    static D3DXVECTOR3 s_vDown;         // Button down vector

    // Current mouse position
    int iMouseX = GET_X_LPARAM(lParam);
    int iMouseY = GET_Y_LPARAM(lParam);

    switch( uMsg )
    {
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
            // Store off the position of the cursor when the button is pressed
            iCurMouseX = iMouseX;
            iCurMouseY = iMouseY;
            return TRUE;

        case WM_LBUTTONDOWN:
            // Start drag mode
            m_bDrag = TRUE;
            s_vDown = ScreenToVector( iMouseX, iMouseY );
            m_qDown = m_qNow;
            return TRUE;

        case WM_LBUTTONUP:
            // End drag mode
            m_bDrag = FALSE;
            return TRUE;

        case WM_MOUSEMOVE:
            // Drag object
            if( MK_LBUTTON&wParam )
            {
                if( m_bDrag )
                {
                    // recompute m_qNow
                    D3DXVECTOR3 vCur = ScreenToVector( iMouseX, iMouseY );
                    D3DXQUATERNION qAxisToAxis;
                    D3DXQuaternionAxisToAxis(&qAxisToAxis, &s_vDown, &vCur);
                    m_qNow = m_qDown;
                    m_qNow *= qAxisToAxis;
                    D3DXMatrixRotationQuaternion(&m_matRotationDelta, &qAxisToAxis);
                }
                else
                    D3DXMatrixIdentity(&m_matRotationDelta);
                D3DXMatrixRotationQuaternion(&m_matRotation, &m_qNow);
                m_bDrag = TRUE;
            }
            else if( (MK_RBUTTON&wParam) || (MK_MBUTTON&wParam) )
            {
                // Normalize based on size of window and bounding sphere radius
                FLOAT fDeltaX = ( iCurMouseX-iMouseX ) * m_fRadiusTranslation / m_iWidth;
                FLOAT fDeltaY = ( iCurMouseY-iMouseY ) * m_fRadiusTranslation / m_iHeight;

                if( wParam & MK_RBUTTON )
                {
                    D3DXMatrixTranslation( &m_matTranslationDelta, -2*fDeltaX, 2*fDeltaY, 0.0f );
                    D3DXMatrixMultiply( &m_matTranslation, &m_matTranslation, &m_matTranslationDelta );
                }
                else  // wParam & MK_MBUTTON
                {
                    D3DXMatrixTranslation( &m_matTranslationDelta, 0.0f, 0.0f, 5*fDeltaY );
                    D3DXMatrixMultiply( &m_matTranslation, &m_matTranslation, &m_matTranslationDelta );
                }

                // Store mouse coordinate
                iCurMouseX = iMouseX;
                iCurMouseY = iMouseY;
            }
            return TRUE;
    }

    return FALSE;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DCamera::CD3DCamera()
{
    // Set attributes for the view matrix
    D3DXVECTOR3 vEyePt(0.0f,0.0f,0.0f);
    D3DXVECTOR3 vLookatPt(0.0f,0.0f,1.0f);
    D3DXVECTOR3 vUpVec(0.0f,1.0f,0.0f);
    SetViewParams( vEyePt, vLookatPt, vUpVec );

    // Set attributes for the projection matrix
    SetProjParams( D3DX_PI/4, 1.0f, 1.0f, 1000.0f );
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
VOID CD3DCamera::SetViewParams( D3DXVECTOR3 &vEyePt, D3DXVECTOR3& vLookatPt,
                                D3DXVECTOR3& vUpVec )
{
    // Set attributes for the view matrix
    m_vEyePt    = vEyePt;
    m_vLookatPt = vLookatPt;
    m_vUpVec    = vUpVec;
    D3DXVECTOR3 vDir = m_vLookatPt - m_vEyePt;
    D3DXVec3Normalize( &m_vView, &vDir );
    D3DXVec3Cross( &m_vCross, &m_vView, &m_vUpVec );

    D3DXMatrixLookAtLH( &m_matView, &m_vEyePt, &m_vLookatPt, &m_vUpVec );
    D3DXMatrixInverse( &m_matBillboard, NULL, &m_matView );
    m_matBillboard._41 = 0.0f;
    m_matBillboard._42 = 0.0f;
    m_matBillboard._43 = 0.0f;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
VOID CD3DCamera::SetProjParams( FLOAT fFOV, FLOAT fAspect, FLOAT fNearPlane,
                                FLOAT fFarPlane )
{
    // Set attributes for the projection matrix
    m_fFOV        = fFOV;
    m_fAspect     = fAspect;
    m_fNearPlane  = fNearPlane;
    m_fFarPlane   = fFarPlane;

    D3DXMatrixPerspectiveFovLH( &m_matProj, fFOV, fAspect, fNearPlane, fFarPlane );
}
