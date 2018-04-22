//-----------------------------------------------------------------------------
// File: D3DUtil.h
//
// Desc: Helper functions and typing shortcuts for Direct3D programming.
//-----------------------------------------------------------------------------
#ifndef D3DUTIL_H
#define D3DUTIL_H
#include <D3D9.h>
#include <D3DX9Math.h>




//-----------------------------------------------------------------------------
// Name: D3DUtil_InitMaterial()
// Desc: Initializes a D3DMATERIAL9 structure, setting the diffuse and ambient
//       colors. It does not set emissive or specular colors.
//-----------------------------------------------------------------------------
VOID D3DUtil_InitMaterial( D3DMATERIAL9& mtrl, FLOAT r=0.0f, FLOAT g=0.0f,
                                               FLOAT b=0.0f, FLOAT a=1.0f );




//-----------------------------------------------------------------------------
// Name: D3DUtil_InitLight()
// Desc: Initializes a D3DLIGHT structure, setting the light position. The
//       diffuse color is set to white, specular and ambient left as black.
//-----------------------------------------------------------------------------
VOID D3DUtil_InitLight( D3DLIGHT9& light, D3DLIGHTTYPE ltType,
                        FLOAT x=0.0f, FLOAT y=0.0f, FLOAT z=0.0f );




//-----------------------------------------------------------------------------
// Name: D3DUtil_CreateTexture()
// Desc: Helper function to create a texture. It checks the root path first,
//       then tries the DXSDK media path (as specified in the system registry).
//-----------------------------------------------------------------------------
HRESULT D3DUtil_CreateTexture( LPDIRECT3DDEVICE9 pd3dDevice, TCHAR* strTexture,
                               LPDIRECT3DTEXTURE9* ppTexture,
                               D3DFORMAT d3dFormat = D3DFMT_UNKNOWN );




//-----------------------------------------------------------------------------
// Name: D3DUtil_GetCubeMapViewMatrix()
// Desc: Returns a view matrix for rendering to a face of a cubemap.
//-----------------------------------------------------------------------------
D3DXMATRIX D3DUtil_GetCubeMapViewMatrix( DWORD dwFace );




//-----------------------------------------------------------------------------
// Name: D3DUtil_GetRotationFromCursor()
// Desc: Returns a quaternion for the rotation implied by the window's cursor
//       position.
//-----------------------------------------------------------------------------
D3DXQUATERNION D3DUtil_GetRotationFromCursor( HWND hWnd,
                                              FLOAT fTrackBallRadius=1.0f );




//-----------------------------------------------------------------------------
// Name: D3DUtil_SetDeviceCursor
// Desc: Builds and sets a cursor for the D3D device based on hCursor.
//-----------------------------------------------------------------------------
HRESULT D3DUtil_SetDeviceCursor( LPDIRECT3DDEVICE9 pd3dDevice, HCURSOR hCursor,
                                 BOOL bAddWatermark );


//-----------------------------------------------------------------------------
// Name: D3DUtil_D3DFormatToString
// Desc: Returns the string for the given D3DFORMAT.
//       bWithPrefix determines whether the string should include the "D3DFMT_"
//-----------------------------------------------------------------------------
TCHAR* D3DUtil_D3DFormatToString( D3DFORMAT format, bool bWithPrefix = true );


//-----------------------------------------------------------------------------
// Name: class CD3DArcBall
// Desc:
//-----------------------------------------------------------------------------
class CD3DArcBall
{
    INT            m_iWidth;   // ArcBall's window width
    INT            m_iHeight;  // ArcBall's window height
    FLOAT          m_fRadius;  // ArcBall's radius in screen coords
    FLOAT          m_fRadiusTranslation; // ArcBall's radius for translating the target

    D3DXQUATERNION m_qDown;               // Quaternion before button down
    D3DXQUATERNION m_qNow;                // Composite quaternion for current drag
    D3DXMATRIXA16  m_matRotation;         // Matrix for arcball's orientation
    D3DXMATRIXA16  m_matRotationDelta;    // Matrix for arcball's orientation
    D3DXMATRIXA16  m_matTranslation;      // Matrix for arcball's position
    D3DXMATRIXA16  m_matTranslationDelta; // Matrix for arcball's position
    BOOL           m_bDrag;               // Whether user is dragging arcball
    BOOL           m_bRightHanded;        // Whether to use RH coordinate system

    D3DXVECTOR3 ScreenToVector( int sx, int sy );

public:
    LRESULT     HandleMouseMessages( HWND, UINT, WPARAM, LPARAM );

    D3DXMATRIX* GetRotationMatrix()         { return &m_matRotation; }
    D3DXMATRIX* GetRotationDeltaMatrix()    { return &m_matRotationDelta; }
    D3DXMATRIX* GetTranslationMatrix()      { return &m_matTranslation; }
    D3DXMATRIX* GetTranslationDeltaMatrix() { return &m_matTranslationDelta; }
    BOOL        IsBeingDragged()            { return m_bDrag; }

    VOID        SetRadius( FLOAT fRadius );
    VOID        SetWindow( INT w, INT h, FLOAT r=0.9 );
    VOID        SetRightHanded( BOOL bRightHanded ) { m_bRightHanded = bRightHanded; }

                CD3DArcBall();
    VOID        Init();
};




//-----------------------------------------------------------------------------
// Name: class CD3DCamera
// Desc:
//-----------------------------------------------------------------------------
class CD3DCamera
{
    D3DXVECTOR3 m_vEyePt;       // Attributes for view matrix
    D3DXVECTOR3 m_vLookatPt;
    D3DXVECTOR3 m_vUpVec;

    D3DXVECTOR3 m_vView;
    D3DXVECTOR3 m_vCross;

    D3DXMATRIXA16  m_matView;
    D3DXMATRIXA16  m_matBillboard; // Special matrix for billboarding effects

    FLOAT       m_fFOV;         // Attributes for projection matrix
    FLOAT       m_fAspect;
    FLOAT       m_fNearPlane;
    FLOAT       m_fFarPlane;
    D3DXMATRIXA16  m_matProj;

public:
    // Access functions
    D3DXVECTOR3 GetEyePt()           { return m_vEyePt; }
    D3DXVECTOR3 GetLookatPt()        { return m_vLookatPt; }
    D3DXVECTOR3 GetUpVec()           { return m_vUpVec; }
    D3DXVECTOR3 GetViewDir()         { return m_vView; }
    D3DXVECTOR3 GetCross()           { return m_vCross; }

    FLOAT       GetFOV()             { return m_fFOV; }
    FLOAT       GetAspect()          { return m_fAspect; }
    FLOAT       GetNearPlane()       { return m_fNearPlane; }
    FLOAT       GetFarPlane()        { return m_fFarPlane; }

    D3DXMATRIX  GetViewMatrix()      { return m_matView; }
    D3DXMATRIX  GetBillboardMatrix() { return m_matBillboard; }
    D3DXMATRIX  GetProjMatrix()      { return m_matProj; }

    VOID SetViewParams( D3DXVECTOR3 &vEyePt, D3DXVECTOR3& vLookatPt,
                        D3DXVECTOR3& vUpVec );
    VOID SetProjParams( FLOAT fFOV, FLOAT fAspect, FLOAT fNearPlane,
                        FLOAT fFarPlane );

    CD3DCamera();
};




#endif // D3DUTIL_H
