#include "nvafx.h"

HRESULT FSQuadMesh::RestoreDeviceObjects( LPDIRECT3DDEVICE9 lpDevice )
{
    return S_OK;
}

HRESULT FSQuadMesh::InvalidateDeviceObjects()
{
    return S_OK;
}

HRESULT FSQuadMesh::Draw( LPDIRECT3DDEVICE9 lpDevice, BOOL bTexture )
{
    HRESULT hr = S_OK;

    LPDIRECT3DSURFACE9 pSurfRT;
    D3DSURFACE_DESC surfDesc;
    lpDevice->GetRenderTarget( 0, &pSurfRT );
    pSurfRT->GetDesc( &surfDesc );
    pSurfRT->Release();
    lpDevice->SetRenderState(D3DRS_ZENABLE, FALSE);

    if ( bTexture )
    {
        struct vtx
        {
            D3DXVECTOR4 position;
            D3DXVECTOR2 texcoord;
        };
        vtx fsQuad[3];
        fsQuad[0].position = D3DXVECTOR4( -0.5f, -0.5f, 0.5f, 1.f );
        fsQuad[1].position = D3DXVECTOR4( 2.f*(FLOAT)surfDesc.Width - 0.5f, -0.5f, 0.5f, 1.f );
        fsQuad[2].position = D3DXVECTOR4( -0.5f, 2.f*(FLOAT)surfDesc.Height - 0.5f, 0.5f, 1.f );
        fsQuad[0].texcoord = D3DXVECTOR2(0.f, 0.f);
        fsQuad[1].texcoord = D3DXVECTOR2(2.f, 0.f);
        fsQuad[2].texcoord = D3DXVECTOR2(0.f, 2.f);
        lpDevice->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE2(0));
        lpDevice->DrawPrimitiveUP(D3DPT_TRIANGLELIST, 1, fsQuad, sizeof(vtx));
    }
    else
    {
        D3DXVECTOR4 fsQuad[3];
        fsQuad[0] = D3DXVECTOR4( -0.5f, -0.5f, 0.5f, 1.f );
        fsQuad[1] = D3DXVECTOR4( 2.f*(FLOAT)surfDesc.Width - 0.5f, -0.5f, 0.5f, 1.f );
        fsQuad[2] = D3DXVECTOR4( -0.5f, 2.f*(FLOAT)surfDesc.Height - 0.5f, 0.5f, 1.f );
        lpDevice->SetFVF(D3DFVF_XYZRHW);
        lpDevice->DrawPrimitiveUP(D3DPT_TRIANGLELIST, 1, fsQuad, sizeof(D3DXVECTOR4));
    }

    lpDevice->SetRenderState(D3DRS_ZENABLE, TRUE);
    return S_OK;
}