//-----------------------------------------------------------------------------
// File: SDKMesh.h
//
// Desc: Support code for loading DirectX .X files.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once
#ifndef SDKMESH_H
#define SDKMESH_H


//-----------------------------------------------------------------------------
// Class for loading and rendering file-based meshes using D3D10
//-----------------------------------------------------------------------------
class CDXUTMesh10
{
public:
    ID3DX10Mesh*  m_pMesh10;
    DWORD m_dwNumVerts;
    DWORD m_dwNumIndices;
    DWORD m_dwNumIndicesAdj;
    UINT m_uStride;
    D3DXMATERIAL *m_pMats;
    ID3D10Texture2D **m_ppTexture;
    ID3D10ShaderResourceView **m_ppSRV;
    DXGI_FORMAT m_IBFormat;
    D3DX10_ATTRIBUTE_RANGE *m_pAttr;
    UINT m_dwNumAttr;
    bool m_bDrawAdj;

protected:
    void RenderSubset( ID3D10EffectTechnique *pTechnique, 
                       UINT pass, 
                       ID3D10EffectShaderResourceVariable* ptxDiffuse,
                       ID3D10EffectVectorVariable* pvDiffuse, 
                       ID3D10EffectVectorVariable* pvSpecular, 
                       DWORD dwSubset );
    void SetResources( ID3D10EffectShaderResourceVariable* ptxDiffuse,
                       ID3D10EffectVectorVariable* pvDiffuse, 
                       ID3D10EffectVectorVariable* pvSpecular,
                       DWORD dwSubset );
    HRESULT CreateMesh( ID3D10Device *pDev10, LPCTSTR szFileName, VOID* pData, DWORD dwDataSizeInBytes, D3D10_INPUT_ELEMENT_DESC* playout, UINT cElements, bool bOptimize );

public:
    CDXUTMesh10();
    ~CDXUTMesh10();

    void ConvertToAdjacencyIndices();
    HRESULT Create( ID3D10Device *pDev10, LPCTSTR szFileName, D3D10_INPUT_ELEMENT_DESC* playout, UINT cElements, bool bOptimize=true );
    HRESULT CreateFromFileInMemory( ID3D10Device* pDev10, VOID* pData, DWORD dwDataSizeInBytes, D3D10_INPUT_ELEMENT_DESC* playout, UINT cElements, bool bOptimize=true );
    void Destroy();
    void Render( ID3D10Device *pDev );
    void Render( ID3D10Device *pDev, 
                 ID3D10EffectTechnique *pTechnique, 
                 ID3D10EffectShaderResourceVariable* ptxDiffuse = NULL,
                 ID3D10EffectVectorVariable* pvDiffuse = NULL, 
                 ID3D10EffectVectorVariable* pvSpecular = NULL, 
                 DWORD dwSubset = (DWORD)-1 );
    void RenderInstanced( ID3D10Device *pDev, 
                          ID3D10EffectTechnique *pTechnique, 
                          UINT uiInstanceCount,
                          ID3D10EffectShaderResourceVariable* ptxDiffuse = NULL,
                          ID3D10EffectVectorVariable* pvDiffuse = NULL, 
                          ID3D10EffectVectorVariable* pvSpecular = NULL );
};


//-----------------------------------------------------------------------------
// Class for loading and rendering file-based meshes using D3D9
//-----------------------------------------------------------------------------
class CDXUTMesh
{
public:
    WCHAR                   m_strName[512];
    LPD3DXMESH              m_pMesh;   // Managed mesh
    
    // Cache of data in m_pMesh for easy access
    IDirect3DVertexBuffer9* m_pVB;
    IDirect3DIndexBuffer9*  m_pIB;
    IDirect3DVertexDeclaration9* m_pDecl;
    DWORD                   m_dwNumVertices;
    DWORD                   m_dwNumFaces;
    DWORD                   m_dwBytesPerVertex;

    DWORD                   m_dwNumMaterials; // Materials for the mesh
    D3DMATERIAL9*           m_pMaterials;
    CHAR                    (*m_strMaterials)[MAX_PATH];
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
                    D3DXHANDLE hPower = NULL,
                    bool bDrawOpaqueSubsets = true,
                    bool bDrawAlphaSubsets = true );

    // Mesh access
    LPD3DXMESH GetMesh() { return m_pMesh; }

    // Rendering options
    void    UseMeshMaterials( bool bFlag ) { m_bUseMaterials = bFlag; }
    HRESULT SetFVF( LPDIRECT3DDEVICE9 pd3dDevice, DWORD dwFVF );
    HRESULT SetVertexDecl( LPDIRECT3DDEVICE9 pd3dDevice, const D3DVERTEXELEMENT9 *pDecl, 
                           bool bAutoComputeNormals = true, bool bAutoComputeTangents = true, 
                           bool bSplitVertexForOptimalTangents = false );

    // Initializing
    HRESULT RestoreDeviceObjects( LPDIRECT3DDEVICE9 pd3dDevice );
    HRESULT InvalidateDeviceObjects();

    // Creation/destruction
    HRESULT Create( LPDIRECT3DDEVICE9 pd3dDevice, LPCWSTR strFilename );
    HRESULT Create( LPDIRECT3DDEVICE9 pd3dDevice, LPD3DXFILEDATA pFileData );
    HRESULT Create( LPDIRECT3DDEVICE9 pd3dDevice, ID3DXMesh* pInMesh, D3DXMATERIAL* pd3dxMaterials, DWORD dwMaterials );
    HRESULT CreateMaterials( LPCWSTR strPath, IDirect3DDevice9 *pd3dDevice, D3DXMATERIAL* d3dxMtrls, DWORD dwNumMaterials );
    HRESULT Destroy();

    CDXUTMesh( LPCWSTR strName = L"CDXUTMeshFile_Mesh" );
    virtual ~CDXUTMesh();
};




//-----------------------------------------------------------------------------
// Class for loading and rendering file-based meshes
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
                    D3DXMATRIX* pmatWorldMatrix = NULL);

    CDXUTMeshFrame( LPCWSTR strName = L"CDXUTMeshFile_Frame" );
    virtual ~CDXUTMeshFrame();
};




//-----------------------------------------------------------------------------
// Class for loading and rendering file-based meshes
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



