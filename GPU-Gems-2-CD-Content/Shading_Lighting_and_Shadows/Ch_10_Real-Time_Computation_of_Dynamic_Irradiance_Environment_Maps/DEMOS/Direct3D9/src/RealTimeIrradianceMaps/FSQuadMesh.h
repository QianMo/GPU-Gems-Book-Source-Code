#ifndef __FSQUADMESH_H_included_
#define __FSQUADMESH_H_included_

struct FSQuadMesh
{
    FSQuadMesh() { }
    HRESULT RestoreDeviceObjects( LPDIRECT3DDEVICE9 );
    HRESULT InvalidateDeviceObjects( );
    HRESULT Draw( LPDIRECT3DDEVICE9, BOOL bTexture );
};

#endif