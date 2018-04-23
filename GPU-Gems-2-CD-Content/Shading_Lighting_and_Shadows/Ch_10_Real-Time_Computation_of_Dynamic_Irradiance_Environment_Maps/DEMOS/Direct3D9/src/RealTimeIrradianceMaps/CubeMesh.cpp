#include "nvafx.h"

const DWORD CubeMesh::cubeFVF = D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE3(0);

const CubeMesh::Vertex CubeMesh::cubeVertices[8] = 
{
    { D3DXVECTOR3(-1.f, -1.f, -1.f), D3DXVECTOR3(-1.f, -1.f, -1.f) },
    { D3DXVECTOR3(-1.f, -1.f,  1.f), D3DXVECTOR3(-1.f, -1.f,  1.f) },
    { D3DXVECTOR3(-1.f,  1.f, -1.f), D3DXVECTOR3(-1.f,  1.f, -1.f) },
    { D3DXVECTOR3(-1.f,  1.f,  1.f), D3DXVECTOR3(-1.f,  1.f,  1.f) },
    { D3DXVECTOR3( 1.f, -1.f, -1.f), D3DXVECTOR3( 1.f, -1.f, -1.f) },
    { D3DXVECTOR3( 1.f, -1.f,  1.f), D3DXVECTOR3( 1.f, -1.f,  1.f) },
    { D3DXVECTOR3( 1.f,  1.f, -1.f), D3DXVECTOR3( 1.f,  1.f, -1.f) },
    { D3DXVECTOR3( 1.f,  1.f,  1.f), D3DXVECTOR3( 1.f,  1.f,  1.f) }
};

const SHORT CubeMesh::cubeIndices[36] =
{
    0, 3, 1, 3, 0, 2,  // x-
    4, 5, 7, 7, 6, 4,  // x+
    0, 1, 5, 5, 4, 0,  // y-
    2, 7, 3, 7, 2, 6,  // y+
    0, 6, 2, 6, 0, 4,  // z-
    1, 3, 7, 7, 5, 1   // z+
};

HRESULT CubeMesh::RestoreDeviceObjects( LPDIRECT3DDEVICE9 lpDevice )
{
    HRESULT hr = S_OK;
    if ( cubeIB!=NULL || cubeVB!=NULL )
        return D3DERR_DEVICENOTRESET;

    V_RETURN(lpDevice->CreateVertexBuffer(8*sizeof(CubeMesh::Vertex), D3DUSAGE_WRITEONLY, CubeMesh::cubeFVF, D3DPOOL_MANAGED, &cubeVB, NULL));
    V_RETURN(lpDevice->CreateIndexBuffer(3*2*6*sizeof(SHORT), D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &cubeIB, NULL));

    LPVOID vbData = NULL;
    LPVOID ibData = NULL;

    V_RETURN(cubeVB->Lock(0, 0, &vbData, 0));
    V_RETURN(cubeIB->Lock(0, 0, &ibData, 0));

    memcpy( ibData, CubeMesh::cubeIndices, sizeof(CubeMesh::cubeIndices) );
    memcpy( vbData, CubeMesh::cubeVertices, sizeof(CubeMesh::cubeVertices) );

    V_RETURN(cubeVB->Unlock());
    V_RETURN(cubeIB->Unlock());

    return S_OK;
}

HRESULT CubeMesh::InvalidateDeviceObjects()
{
    SAFE_RELEASE(cubeVB);
    SAFE_RELEASE(cubeIB);

    return S_OK;
}

HRESULT CubeMesh::Draw( LPDIRECT3DDEVICE9 lpDevice )
{
    HRESULT hr = S_OK;

    V_RETURN(lpDevice->SetStreamSource(0, cubeVB, 0, sizeof(CubeMesh::Vertex)));
    V_RETURN(lpDevice->SetIndices(cubeIB));
    V_RETURN(lpDevice->SetFVF(CubeMesh::cubeFVF));  
    V_RETURN(lpDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, 8, 0, 6*2));

    return S_OK;
}