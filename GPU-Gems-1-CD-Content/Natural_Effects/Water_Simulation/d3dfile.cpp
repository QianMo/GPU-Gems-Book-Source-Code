//-----------------------------------------------------------------------------
// File: D3DFile.cpp
//
// Desc: Support code for loading DirectX .X files.
//-----------------------------------------------------------------------------
#define STRICT
#include <tchar.h>
#include <stdio.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <dxfile.h>
#include <rmxfguid.h>
#include <rmxftmpl.h>
#include "D3DFile.h"
#include "DXUtil.h"




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DMesh::CD3DMesh( TCHAR* strName )
{
    _tcsncpy( m_strName, strName, sizeof(m_strName) / sizeof(TCHAR) );
    m_strName[sizeof(m_strName) / sizeof(TCHAR) - 1] = _T('\0');
    m_pSysMemMesh        = NULL;
    m_pLocalMesh         = NULL;
    m_dwNumMaterials     = 0L;
    m_pMaterials         = NULL;
    m_pTextures          = NULL;
    m_bUseMaterials      = TRUE;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DMesh::~CD3DMesh()
{
    Destroy();
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::Create( LPDIRECT3DDEVICE9 pd3dDevice, TCHAR* strFilename )
{
    TCHAR        strPath[MAX_PATH];
    LPD3DXBUFFER pAdjacencyBuffer = NULL;
    LPD3DXBUFFER pMtrlBuffer = NULL;
    HRESULT      hr;

    // Find the path for the file, and convert it to ANSI (for the D3DX API)
    DXUtil_FindMediaFileCb( strPath, sizeof(strPath), strFilename );

    // Load the mesh
    if( FAILED( hr = D3DXLoadMeshFromX( strPath, D3DXMESH_SYSTEMMEM, pd3dDevice, 
                                        &pAdjacencyBuffer, &pMtrlBuffer, NULL,
                                        &m_dwNumMaterials, &m_pSysMemMesh ) ) )
    {
        return hr;
    }

    // Optimize the mesh for performance
    if( FAILED( hr = m_pSysMemMesh->OptimizeInplace(
                        D3DXMESHOPT_COMPACT | D3DXMESHOPT_ATTRSORT | D3DXMESHOPT_VERTEXCACHE,
                        (DWORD*)pAdjacencyBuffer->GetBufferPointer(), NULL, NULL, NULL ) ) )
    {
        SAFE_RELEASE( pAdjacencyBuffer );
        SAFE_RELEASE( pMtrlBuffer );
        return hr;
    }

    // Get material info for the mesh
    // Get the array of materials out of the buffer
    if( pMtrlBuffer && m_dwNumMaterials > 0 )
    {
        // Allocate memory for the materials and textures
        D3DXMATERIAL* d3dxMtrls = (D3DXMATERIAL*)pMtrlBuffer->GetBufferPointer();
        m_pMaterials = new D3DMATERIAL9[m_dwNumMaterials];
        if( m_pMaterials == NULL )
        {
            hr = E_OUTOFMEMORY;
            goto LEnd;
        }
        m_pTextures  = new LPDIRECT3DTEXTURE9[m_dwNumMaterials];
        if( m_pTextures == NULL )
        {
            hr = E_OUTOFMEMORY;
            goto LEnd;
        }

        // Copy each material and create its texture
        for( DWORD i=0; i<m_dwNumMaterials; i++ )
        {
            // Copy the material
            m_pMaterials[i]         = d3dxMtrls[i].MatD3D;
            m_pTextures[i]          = NULL;

            // Create a texture
            if( d3dxMtrls[i].pTextureFilename )
            {
                TCHAR strTexture[MAX_PATH];
                TCHAR strTextureTemp[MAX_PATH];
                DXUtil_ConvertAnsiStringToGenericCb( strTextureTemp, d3dxMtrls[i].pTextureFilename, sizeof(strTextureTemp) );
                DXUtil_FindMediaFileCb( strTexture, sizeof(strTexture), strTextureTemp );

                if( FAILED( D3DXCreateTextureFromFile( pd3dDevice, strTexture, 
                                                       &m_pTextures[i] ) ) )
                    m_pTextures[i] = NULL;
            }
        }
    }
    hr = S_OK;

LEnd:
    SAFE_RELEASE( pAdjacencyBuffer );
    SAFE_RELEASE( pMtrlBuffer );

    return hr;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::Create( LPDIRECT3DDEVICE9 pd3dDevice,
                          LPDIRECTXFILEDATA pFileData )
{
    LPD3DXBUFFER pMtrlBuffer = NULL;
    LPD3DXBUFFER pAdjacencyBuffer = NULL;
    HRESULT      hr;

    // Load the mesh from the DXFILEDATA object
    if( FAILED( hr = D3DXLoadMeshFromXof( pFileData, D3DXMESH_SYSTEMMEM, pd3dDevice,
                                          &pAdjacencyBuffer, &pMtrlBuffer, NULL,
                                          &m_dwNumMaterials, &m_pSysMemMesh ) ) )
    {
        return hr;
    }

    // Optimize the mesh for performance
    if( FAILED( hr = m_pSysMemMesh->OptimizeInplace(
                        D3DXMESHOPT_COMPACT | D3DXMESHOPT_ATTRSORT | D3DXMESHOPT_VERTEXCACHE,
                        (DWORD*)pAdjacencyBuffer->GetBufferPointer(), NULL, NULL, NULL ) ) )
    {
        SAFE_RELEASE( pAdjacencyBuffer );
        SAFE_RELEASE( pMtrlBuffer );
        return hr;
    }

    // Get material info for the mesh
    // Get the array of materials out of the buffer
    if( pMtrlBuffer && m_dwNumMaterials > 0 )
    {
        // Allocate memory for the materials and textures
        D3DXMATERIAL* d3dxMtrls = (D3DXMATERIAL*)pMtrlBuffer->GetBufferPointer();
        m_pMaterials = new D3DMATERIAL9[m_dwNumMaterials];
        if( m_pMaterials == NULL )
        {
            hr = E_OUTOFMEMORY;
            goto LEnd;
        }
        m_pTextures  = new LPDIRECT3DTEXTURE9[m_dwNumMaterials];
        if( m_pTextures == NULL )
        {
            hr = E_OUTOFMEMORY;
            goto LEnd;
        }

        // Copy each material and create its texture
        for( DWORD i=0; i<m_dwNumMaterials; i++ )
        {
            // Copy the material
            m_pMaterials[i]         = d3dxMtrls[i].MatD3D;
            m_pTextures[i]          = NULL;

            // Create a texture
            if( d3dxMtrls[i].pTextureFilename )
            {
                TCHAR strTexture[MAX_PATH];
                TCHAR strTextureTemp[MAX_PATH];
                DXUtil_ConvertAnsiStringToGenericCb( strTextureTemp, d3dxMtrls[i].pTextureFilename, sizeof(strTextureTemp) );
                DXUtil_FindMediaFileCb( strTexture, sizeof(strTexture), strTextureTemp );

                if( FAILED( D3DXCreateTextureFromFile( pd3dDevice, strTexture, 
                                                       &m_pTextures[i] ) ) )
                    m_pTextures[i] = NULL;
            }
        }
    }
    hr = S_OK;

LEnd:
    SAFE_RELEASE( pAdjacencyBuffer );
    SAFE_RELEASE( pMtrlBuffer );

    return hr;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::SetFVF( LPDIRECT3DDEVICE9 pd3dDevice, DWORD dwFVF )
{
    LPD3DXMESH pTempSysMemMesh = NULL;
    LPD3DXMESH pTempLocalMesh  = NULL;

    if( m_pSysMemMesh )
    {
        if( FAILED( m_pSysMemMesh->CloneMeshFVF( D3DXMESH_SYSTEMMEM, dwFVF,
                                                 pd3dDevice, &pTempSysMemMesh ) ) )
            return E_FAIL;
    }
    if( m_pLocalMesh )
    {
        if( FAILED( m_pLocalMesh->CloneMeshFVF( 0L, dwFVF, pd3dDevice,
                                                &pTempLocalMesh ) ) )
        {
            SAFE_RELEASE( pTempSysMemMesh );
            return E_FAIL;
        }
    }

    SAFE_RELEASE( m_pSysMemMesh );
    SAFE_RELEASE( m_pLocalMesh );

    if( pTempSysMemMesh ) m_pSysMemMesh = pTempSysMemMesh;
    if( pTempLocalMesh )  m_pLocalMesh  = pTempLocalMesh;

    // Compute normals in case the meshes have them
    if( m_pSysMemMesh )
        D3DXComputeNormals( m_pSysMemMesh, NULL );
    if( m_pLocalMesh )
        D3DXComputeNormals( m_pLocalMesh, NULL );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::RestoreDeviceObjects( LPDIRECT3DDEVICE9 pd3dDevice )
{
    if( NULL == m_pSysMemMesh )
        return E_FAIL;

    // Make a local memory version of the mesh. Note: because we are passing in
    // no flags, the default behavior is to clone into local memory.
    if( FAILED( m_pSysMemMesh->CloneMeshFVF( 0L, m_pSysMemMesh->GetFVF(),
                                             pd3dDevice, &m_pLocalMesh ) ) )
        return E_FAIL;

    return S_OK;

}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::InvalidateDeviceObjects()
{
    SAFE_RELEASE( m_pLocalMesh );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::Destroy()
{
    InvalidateDeviceObjects();
    for( UINT i=0; i<m_dwNumMaterials; i++ )
        SAFE_RELEASE( m_pTextures[i] );
    SAFE_DELETE_ARRAY( m_pTextures );
    SAFE_DELETE_ARRAY( m_pMaterials );

    SAFE_RELEASE( m_pSysMemMesh );

    m_dwNumMaterials = 0L;

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DMesh::Render( LPDIRECT3DDEVICE9 pd3dDevice, bool bDrawOpaqueSubsets,
                          bool bDrawAlphaSubsets )
{
    if( NULL == m_pLocalMesh )
        return E_FAIL;

    // Frist, draw the subsets without alpha
    if( bDrawOpaqueSubsets )
    {
        for( DWORD i=0; i<m_dwNumMaterials; i++ )
        {
            if( m_bUseMaterials )
            {
                if( m_pMaterials[i].Diffuse.a < 1.0f )
                    continue;
                pd3dDevice->SetMaterial( &m_pMaterials[i] );
                pd3dDevice->SetTexture( 0, m_pTextures[i] );
            }
            m_pLocalMesh->DrawSubset( i );
        }
    }

    // Then, draw the subsets with alpha
    if( bDrawAlphaSubsets && m_bUseMaterials )
    {
        for( DWORD i=0; i<m_dwNumMaterials; i++ )
        {
            if( m_pMaterials[i].Diffuse.a == 1.0f )
                continue;

            // Set the material and texture
            pd3dDevice->SetMaterial( &m_pMaterials[i] );
            pd3dDevice->SetTexture( 0, m_pTextures[i] );
            m_pLocalMesh->DrawSubset( i );
        }
    }

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DFrame::CD3DFrame( TCHAR* strName )
{
    _tcsncpy( m_strName, strName, sizeof(m_strName) / sizeof(TCHAR) );
    m_strName[sizeof(m_strName) / sizeof(TCHAR) - 1] = _T('\0');
    D3DXMatrixIdentity( &m_mat );
    m_pMesh  = NULL;

    m_pChild = NULL;
    m_pNext  = NULL;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DFrame::~CD3DFrame()
{
    SAFE_DELETE( m_pChild );
    SAFE_DELETE( m_pNext );
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
bool CD3DFrame::EnumMeshes( bool (*EnumMeshCB)(CD3DMesh*,void*),
                            void* pContext )
{
    if( m_pMesh )
        EnumMeshCB( m_pMesh, pContext );
    if( m_pChild )
        m_pChild->EnumMeshes( EnumMeshCB, pContext );
    if( m_pNext )
        m_pNext->EnumMeshes( EnumMeshCB, pContext );

    return TRUE;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DMesh* CD3DFrame::FindMesh( TCHAR* strMeshName )
{
    CD3DMesh* pMesh;

    if( m_pMesh )
        if( !lstrcmpi( m_pMesh->m_strName, strMeshName ) )
            return m_pMesh;

    if( m_pChild )
        if( NULL != ( pMesh = m_pChild->FindMesh( strMeshName ) ) )
            return pMesh;

    if( m_pNext )
        if( NULL != ( pMesh = m_pNext->FindMesh( strMeshName ) ) )
            return pMesh;

    return NULL;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
CD3DFrame* CD3DFrame::FindFrame( TCHAR* strFrameName )
{
    CD3DFrame* pFrame;

    if( !lstrcmpi( m_strName, strFrameName ) )
        return this;

    if( m_pChild )
        if( NULL != ( pFrame = m_pChild->FindFrame( strFrameName ) ) )
            return pFrame;

    if( m_pNext )
        if( NULL != ( pFrame = m_pNext->FindFrame( strFrameName ) ) )
            return pFrame;

    return NULL;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFrame::Destroy()
{
    if( m_pMesh )  m_pMesh->Destroy();
    if( m_pChild ) m_pChild->Destroy();
    if( m_pNext )  m_pNext->Destroy();

    SAFE_DELETE( m_pMesh );
    SAFE_DELETE( m_pNext );
    SAFE_DELETE( m_pChild );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFrame::RestoreDeviceObjects( LPDIRECT3DDEVICE9 pd3dDevice )
{
    if( m_pMesh )  m_pMesh->RestoreDeviceObjects( pd3dDevice );
    if( m_pChild ) m_pChild->RestoreDeviceObjects( pd3dDevice );
    if( m_pNext )  m_pNext->RestoreDeviceObjects( pd3dDevice );
    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFrame::InvalidateDeviceObjects()
{
    if( m_pMesh )  m_pMesh->InvalidateDeviceObjects();
    if( m_pChild ) m_pChild->InvalidateDeviceObjects();
    if( m_pNext )  m_pNext->InvalidateDeviceObjects();
    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFrame::Render( LPDIRECT3DDEVICE9 pd3dDevice, bool bDrawOpaqueSubsets,
                           bool bDrawAlphaSubsets, D3DXMATRIX* pmatWorldMatrix )
{
    // For pure devices, specify the world transform. If the world transform is not
    // specified on pure devices, this function will fail.

    D3DXMATRIXA16 matSavedWorld, matWorld;

    if ( NULL == pmatWorldMatrix )
        pd3dDevice->GetTransform( D3DTS_WORLD, &matSavedWorld );
    else
        matSavedWorld = *pmatWorldMatrix;

    D3DXMatrixMultiply( &matWorld, &m_mat, &matSavedWorld );
    pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );

    if( m_pMesh )
        m_pMesh->Render( pd3dDevice, bDrawOpaqueSubsets, bDrawAlphaSubsets );

    if( m_pChild )
        m_pChild->Render( pd3dDevice, bDrawOpaqueSubsets, bDrawAlphaSubsets, &matWorld );

    pd3dDevice->SetTransform( D3DTS_WORLD, &matSavedWorld );

    if( m_pNext )
        m_pNext->Render( pd3dDevice, bDrawOpaqueSubsets, bDrawAlphaSubsets, &matSavedWorld );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFile::LoadFrame( LPDIRECT3DDEVICE9 pd3dDevice,
                             LPDIRECTXFILEDATA pFileData,
                             CD3DFrame* pParentFrame )
{
    LPDIRECTXFILEDATA   pChildData = NULL;
    LPDIRECTXFILEOBJECT pChildObj = NULL;
    const GUID* pGUID;
    DWORD       cbSize;
    CD3DFrame*  pCurrentFrame;
    HRESULT     hr;

    // Get the type of the object
    if( FAILED( hr = pFileData->GetType( &pGUID ) ) )
        return hr;

    if( *pGUID == TID_D3DRMMesh )
    {
        hr = LoadMesh( pd3dDevice, pFileData, pParentFrame );
        if( FAILED(hr) )
            return hr;
    }
    if( *pGUID == TID_D3DRMFrameTransformMatrix )
    {
        D3DXMATRIX* pmatMatrix;
        hr = pFileData->GetData( NULL, &cbSize, (void**)&pmatMatrix );
        if( FAILED(hr) )
            return hr;

        // Update the parent's matrix with the new one
        pParentFrame->SetMatrix( pmatMatrix );
    }
    if( *pGUID == TID_D3DRMFrame )
    {
        // Get the frame name
        CHAR  strAnsiName[512] = "";
        TCHAR strName[512];
        DWORD dwNameLength = 512;
        if( FAILED( hr = pFileData->GetName( strAnsiName, &dwNameLength ) ) )
            return hr;
        DXUtil_ConvertAnsiStringToGenericCb( strName, strAnsiName, sizeof(strName) );

        // Create the frame
        pCurrentFrame = new CD3DFrame( strName );
        if( pCurrentFrame == NULL )
            return E_OUTOFMEMORY;

        pCurrentFrame->m_pNext = pParentFrame->m_pChild;
        pParentFrame->m_pChild = pCurrentFrame;

        // Enumerate child objects
        while( SUCCEEDED( pFileData->GetNextObject( &pChildObj ) ) )
        {
            // Query the child for its FileData
            hr = pChildObj->QueryInterface( IID_IDirectXFileData,
                                            (void**)&pChildData );
            if( SUCCEEDED(hr) )
            {
                hr = LoadFrame( pd3dDevice, pChildData, pCurrentFrame );
                pChildData->Release();
            }

            pChildObj->Release();

            if( FAILED(hr) )
                return hr;
        }
    }

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFile::LoadMesh( LPDIRECT3DDEVICE9 pd3dDevice,
                            LPDIRECTXFILEDATA pFileData,
                            CD3DFrame* pParentFrame )
{
    // Currently only allowing one mesh per frame
    if( pParentFrame->m_pMesh )
        return E_FAIL;

    // Get the mesh name
    CHAR  strAnsiName[512] = {0};
    TCHAR strName[512];
    DWORD dwNameLength = 512;
    HRESULT hr;
    if( FAILED( hr = pFileData->GetName( strAnsiName, &dwNameLength ) ) )
        return hr;
    DXUtil_ConvertAnsiStringToGenericCb( strName, strAnsiName, sizeof(strName) );

    // Create the mesh
    pParentFrame->m_pMesh = new CD3DMesh( strName );
    if( pParentFrame->m_pMesh == NULL )
        return E_OUTOFMEMORY;
    pParentFrame->m_pMesh->Create( pd3dDevice, pFileData );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFile::CreateFromResource( LPDIRECT3DDEVICE9 pd3dDevice, TCHAR* strResource, TCHAR* strType )
{
    LPDIRECTXFILE           pDXFile   = NULL;
    LPDIRECTXFILEENUMOBJECT pEnumObj  = NULL;
    LPDIRECTXFILEDATA       pFileData = NULL;
    HRESULT hr;

    // Create a x file object
    if( FAILED( hr = DirectXFileCreate( &pDXFile ) ) )
        return E_FAIL;

    // Register templates for d3drm and patch extensions.
    if( FAILED( hr = pDXFile->RegisterTemplates( (void*)D3DRM_XTEMPLATES,
                                                 D3DRM_XTEMPLATE_BYTES ) ) )
    {
        pDXFile->Release();
        return E_FAIL;
    }
    
    CHAR strTypeAnsi[MAX_PATH];
    DXUtil_ConvertGenericStringToAnsiCb( strTypeAnsi, strType, sizeof(strTypeAnsi) );

    DXFILELOADRESOURCE dxlr;
    dxlr.hModule = NULL;
    dxlr.lpName = strResource;
    dxlr.lpType = (TCHAR*) strTypeAnsi;

    // Create enum object
    hr = pDXFile->CreateEnumObject( (void*)&dxlr, DXFILELOAD_FROMRESOURCE, 
                                    &pEnumObj );
    if( FAILED(hr) )
    {
        pDXFile->Release();
        return hr;
    }

    // Enumerate top level objects (which are always frames)
    while( SUCCEEDED( pEnumObj->GetNextDataObject( &pFileData ) ) )
    {
        hr = LoadFrame( pd3dDevice, pFileData, this );
        pFileData->Release();
        if( FAILED(hr) )
        {
            pEnumObj->Release();
            pDXFile->Release();
            return E_FAIL;
        }
    }

    SAFE_RELEASE( pFileData );
    SAFE_RELEASE( pEnumObj );
    SAFE_RELEASE( pDXFile );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFile::Create( LPDIRECT3DDEVICE9 pd3dDevice, TCHAR* strFilename )
{
    LPDIRECTXFILE           pDXFile   = NULL;
    LPDIRECTXFILEENUMOBJECT pEnumObj  = NULL;
    LPDIRECTXFILEDATA       pFileData = NULL;
    HRESULT hr;

    // Create a x file object
    if( FAILED( hr = DirectXFileCreate( &pDXFile ) ) )
        return E_FAIL;

    // Register templates for d3drm and patch extensions.
    if( FAILED( hr = pDXFile->RegisterTemplates( (void*)D3DRM_XTEMPLATES,
                                                 D3DRM_XTEMPLATE_BYTES ) ) )
    {
        pDXFile->Release();
        return E_FAIL;
    }

    // Find the path to the file, and convert it to ANSI (for the D3DXOF API)
    TCHAR strPath[MAX_PATH];
    CHAR  strPathANSI[MAX_PATH];
    DXUtil_FindMediaFileCb( strPath, sizeof(strPath), strFilename );
    DXUtil_ConvertGenericStringToAnsiCb( strPathANSI, strPath, sizeof(strPathANSI) );
    
    // Create enum object
    hr = pDXFile->CreateEnumObject( (void*)strPathANSI, DXFILELOAD_FROMFILE, 
                                    &pEnumObj );
    if( FAILED(hr) )
    {
        pDXFile->Release();
        return hr;
    }

    // Enumerate top level objects (which are always frames)
    while( SUCCEEDED( pEnumObj->GetNextDataObject( &pFileData ) ) )
    {
        hr = LoadFrame( pd3dDevice, pFileData, this );
        pFileData->Release();
        if( FAILED(hr) )
        {
            pEnumObj->Release();
            pDXFile->Release();
            return E_FAIL;
        }
    }

    SAFE_RELEASE( pFileData );
    SAFE_RELEASE( pEnumObj );
    SAFE_RELEASE( pDXFile );

    return S_OK;
}




//-----------------------------------------------------------------------------
// Name:
// Desc:
//-----------------------------------------------------------------------------
HRESULT CD3DFile::Render( LPDIRECT3DDEVICE9 pd3dDevice, D3DXMATRIX* pmatWorldMatrix )
{

    // For pure devices, specify the world transform. If the world transform is not
    // specified on pure devices, this function will fail.

    // Set up the world transformation
    D3DXMATRIX matSavedWorld, matWorld;

    if ( NULL == pmatWorldMatrix )
        pd3dDevice->GetTransform( D3DTS_WORLD, &matSavedWorld );
    else
        matSavedWorld = *pmatWorldMatrix;

    D3DXMatrixMultiply( &matWorld, &matSavedWorld, &m_mat );
    pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );

    // Render opaque subsets in the meshes
    if( m_pChild )
        m_pChild->Render( pd3dDevice, TRUE, FALSE, &matWorld );

    // Enable alpha blending
    pd3dDevice->SetRenderState( D3DRS_ALPHABLENDENABLE, TRUE );
    pd3dDevice->SetRenderState( D3DRS_SRCBLEND,  D3DBLEND_SRCALPHA );
    pd3dDevice->SetRenderState( D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA );

    // Render alpha subsets in the meshes
    if( m_pChild )
        m_pChild->Render( pd3dDevice, FALSE, TRUE, &matWorld );

    // Restore state
    pd3dDevice->SetRenderState( D3DRS_ALPHABLENDENABLE, FALSE );
    pd3dDevice->SetTransform( D3DTS_WORLD, &matSavedWorld );

    return S_OK;
}




