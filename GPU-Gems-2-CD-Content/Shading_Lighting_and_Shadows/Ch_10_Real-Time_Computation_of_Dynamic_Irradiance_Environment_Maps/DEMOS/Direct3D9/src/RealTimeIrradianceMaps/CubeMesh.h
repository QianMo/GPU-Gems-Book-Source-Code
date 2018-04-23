#ifndef __CUBEMESH_H_included_
#define __CUBEMESH_H_included_

struct CubeMesh
{
    CubeMesh(): cubeVB(NULL), cubeIB(NULL) { }
    HRESULT RestoreDeviceObjects( LPDIRECT3DDEVICE9 );
    HRESULT InvalidateDeviceObjects( );
    HRESULT Draw( LPDIRECT3DDEVICE9 );

    struct Vertex
    {
        D3DXVECTOR3 position;
        D3DXVECTOR3 texCoord;
    };

    static const Vertex cubeVertices[8];
    static const SHORT  cubeIndices[36];

    static const DWORD cubeFVF;

    LPDIRECT3DVERTEXBUFFER9 cubeVB;
    LPDIRECT3DINDEXBUFFER9  cubeIB;
};

#endif