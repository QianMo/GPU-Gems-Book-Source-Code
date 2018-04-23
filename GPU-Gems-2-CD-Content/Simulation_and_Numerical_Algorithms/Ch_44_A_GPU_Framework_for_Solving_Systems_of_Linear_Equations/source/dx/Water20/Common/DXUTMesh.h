//-----------------------------------------------------------------------------
// File: DXUTMesh.h
//
// Desc: Support code for loading DirectX .X files.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once
#ifndef DXUTMESH_H
#define DXUTMESH_H





//-----------------------------------------------------------------------------
// Name: class CDXUTMesh
// Desc: Class for loading and rendering file-based meshes
//-----------------------------------------------------------------------------
class CDXUTMesh
{
public:
    WCHAR                   m_strName[512];

    LPD3DXMESH              m_pSysMemMesh;    // SysMem mesh, lives through resize
    LPD3DXMESH              m_pLocalMesh;     // Local mesh, rebuilt on resize
    
    DWORD                   m_dwNumMaterials; // Materials for the mesh
    D3DMATERIAL9*           m_pMaterials;
    IDirect3DBaseTexture9** m_pTextures;
    bool                    m_bUseMaterials;

public:
    // Rendering
    HRESULT Render( LPDIRECT3DDEVICE9 pd3dDevice, 
                    bool bDrawOpaqueSubsets = true,
                    bool bDrawAlphaSubsets = true );
    HRESULT Render( ID3DXEffect *pEffect,
                    D3DXHANDLE hTexture = NULL,
                    D3DXHANDLE hDiffuse = NULL,
                    D3DXHANDLE hAmbient = NULL,
                    D3DXHANDLE hSpecular = NULL,
                    D3DXHANDLE hEmissive = NULL,
                    bool bDrawOpaqueSubsets = true,
                    bool bDrawAlphaSubsets = true );

    // Mesh access
    LPD3DXMESH GetSysMemMesh() { return m_pSysMemMesh; }
    LPD3DXMESH GetLocalMesh()  { return m_pLocalMesh; }

    // Rendering options
    void    UseMeshMaterials( bool bFlag ) { m_bUseMaterials = bFlag; }
    HRESULT SetFVF( LPDIRECT3DDEVICE9 pd3dDevice, DWORD dwFVF );
    HRESULT SetVertexDecl( LPDIRECT3DDEVICE9 pd3dDevice, const D3DVERTEXELEMENT9 *pDecl );

    // Initializing
    HRESULT RestoreDeviceObjects( LPDIRECT3DDEVICE9 pd3dDevice );
    HRESULT InvalidateDeviceObjects();

    // Creation/destruction
    HRESULT Create( LPDIRECT3DDEVICE9 pd3dDevice, LPCWSTR strFilename );
    HRESULT Create( LPDIRECT3DDEVICE9 pd3dDevice, LPD3DXFILEDATA pFileData );
    HRESULT CreateMaterials( LPCWSTR strPath, IDirect3DDevice9 *pd3dDevice, ID3DXBuffer *pAdjacencyBuffer, ID3DXBuffer *pMtrlBuffer );
    HRESULT Destroy();

    CDXUTMesh( LPCWSTR strName = L"CDXUTMeshFile_Mesh" );
    virtual ~CDXUTMesh();
};




//-----------------------------------------------------------------------------
// Name: class CDXUTMeshFrame
// Desc: Class for loading and rendering file-based meshes
//-----------------------------------------------------------------------------
class CDXUTMeshFrame
{
public:
    WCHAR      m_strName[512];
    D3DXMATRIX m_mat;
    CDXUTMesh*  m_pMesh;

    CDXUTMeshFrame* m_pNext;
    CDXUTMeshFrame* m_pChild;

public:
    // Matrix access
    void        SetMatrix( D3DXMATRIX* pmat ) { m_mat = *pmat; }
    D3DXMATRIX* GetMatrix()                   { return &m_mat; }

    CDXUTMesh*   FindMesh( LPCWSTR strMeshName );
    CDXUTMeshFrame*  FindFrame( LPCWSTR strFrameName );
    bool        EnumMeshes( bool (*EnumMeshCB)(CDXUTMesh*,void*), 
                            void* pContext );

    HRESULT Destroy();
    HRESULT RestoreDeviceObjects( LPDIRECT3DDEVICE9 pd3dDevice );
    HRESULT InvalidateDeviceObjects();
    HRESULT Render( LPDIRECT3DDEVICE9 pd3dDevice, 
                    bool bDrawOpaqueSubsets = true,
                    bool bDrawAlphaSubsets = true,
                    D3DXMATRIX* pmatWorldMartix = NULL);
    
    CDXUTMeshFrame( LPCWSTR strName = L"CDXUTMeshFile_Frame" );
    virtual ~CDXUTMeshFrame();
};




//-----------------------------------------------------------------------------
// Name: class CDXUTMeshFile
// Desc: Class for loading and rendering file-based meshes
//-----------------------------------------------------------------------------
class CDXUTMeshFile : public CDXUTMeshFrame
{
    HRESULT LoadMesh( LPDIRECT3DDEVICE9 pd3dDevice, LPD3DXFILEDATA pFileData, 
                      CDXUTMeshFrame* pParentFrame );
    HRESULT LoadFrame( LPDIRECT3DDEVICE9 pd3dDevice, LPD3DXFILEDATA pFileData, 
                       CDXUTMeshFrame* pParentFrame );
public:
    HRESULT Create( LPDIRECT3DDEVICE9 pd3dDevice, LPCWSTR strFilename );
    HRESULT CreateFromResource( LPDIRECT3DDEVICE9 pd3dDevice, LPCWSTR strResource, LPCWSTR strType );
    // For pure devices, specify the world transform. If the world transform is not
    // specified on pure devices, this function will fail.
    HRESULT Render( LPDIRECT3DDEVICE9 pd3dDevice, D3DXMATRIX* pmatWorldMatrix = NULL );

    CDXUTMeshFile() : CDXUTMeshFrame( L"CDXUTMeshFile_Root" ) {}
};



#endif



