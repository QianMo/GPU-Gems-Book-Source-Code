#pragma once
#include "dxstdafx.h"

/**
	\brief Uploads the specified world/view/projection transformation matrices to the GPU.
*/
void setWorldViewProj(D3DXMATRIXA16& mWorld, D3DXMATRIXA16& mView, D3DXMATRIXA16& mProj, ID3DXEffect* effect )
{
	HRESULT hr;
	D3DXMATRIXA16 mWorldView = mWorld * mView;
	D3DXMATRIXA16 mWorldViewProjection = mWorldView * mProj;

	D3DXMATRIXA16 mWorldViewI, mWorldViewIT, mWorldI, mWorldIT;
	D3DXMatrixInverse(&mWorldViewI, NULL, &mWorldView); 
	D3DXMatrixTranspose(&mWorldViewIT, &mWorldViewI);

	D3DXMatrixInverse(&mWorldI, NULL, &mWorld); 
	D3DXMatrixTranspose(&mWorldIT, &mWorldI);

	D3DXMATRIXA16 mViewI, mViewIT;
	D3DXMatrixInverse(&mViewI, NULL, &mView); 
	D3DXMatrixTranspose(&mViewIT, &mViewI);

	V( effect->SetMatrix( "View", &mView ) );
	V( effect->SetMatrix( "View", &mViewIT ) );
	V( effect->SetMatrix( "World", &mWorld ) );
	V( effect->SetMatrix( "WorldIT", &mWorldIT ) );
	V( effect->SetMatrix( "WorldView", &mWorldView ) );
	V( effect->SetMatrix( "WorldViewIT", &mWorldViewIT ) );
	V( effect->SetMatrix( "WorldViewProj", &mWorldViewProjection ) );
	V( effect->CommitChanges() );
}

/**
	Sets the given scaling and offset. 
	@return The resulting world transformation matrix.
*/
D3DXMATRIXA16 ScaleAndOffset(D3DXVECTOR3 vScale, D3DXVECTOR3 vOffset)
{
	D3DXMATRIXA16 mScale, mOffset;
	D3DXMatrixIdentity(&mScale);
	D3DXMatrixIdentity(&mOffset);

	D3DXMatrixTranslation( &mOffset, vOffset.x, vOffset.y, vOffset.z );
	D3DXMatrixScaling( &mScale, vScale.x, vScale.y, vScale.z );

	return mScale * mOffset;
}

/**
	Sets the given uniform scaling and an offset.
	@return The resulting world transformation matrix.
*/
D3DXMATRIXA16 ScaleAndOffset(float fScale, D3DXVECTOR3 vOffset)
{
	return ScaleAndOffset( D3DXVECTOR3(fScale,fScale,fScale), vOffset );
}

void getViewProjForCubeFace(D3DXMATRIXA16* mView, D3DXMATRIXA16* mProj, int cubeFace, D3DXVECTOR3 centerPos)
{
	D3DXVECTOR3 vecEye(0,0,0);
	D3DXVECTOR3 vecAt(0.0f, 0.0f, 0.0f);
	D3DXVECTOR3 vecUp(0.0f, 0.0f, 0.0f);

	switch( cubeFace )
	{
		case D3DCUBEMAP_FACE_POSITIVE_X:
			vecAt = D3DXVECTOR3( 1, 0, 0 ); 
			vecUp = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
			break;
        case D3DCUBEMAP_FACE_NEGATIVE_X:
            vecAt = D3DXVECTOR3(-1.0f, 0.0f, 0.0f );
            vecUp = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
        case D3DCUBEMAP_FACE_POSITIVE_Y:
            vecAt = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            vecUp = D3DXVECTOR3( 0.0f, 0.0f,-1.0f );
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_Y:
            vecAt = D3DXVECTOR3( 0.0f,-1.0f, 0.0f );
            vecUp = D3DXVECTOR3( 0.0f, 0.0f, 1.0f );
            break;
        case D3DCUBEMAP_FACE_POSITIVE_Z:
            vecAt = D3DXVECTOR3( 0.0f, 0.0f, 1.0f );
            vecUp = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
        case D3DCUBEMAP_FACE_NEGATIVE_Z: 
            vecAt = D3DXVECTOR3( 0.0f, 0.0f,-1.0f );	
            vecUp = D3DXVECTOR3( 0.0f, 1.0f, 0.0f );
            break;
     }
	 
	 vecEye += centerPos;
	 vecAt += centerPos;

	D3DXMatrixLookAtLH(mView, &vecEye, &vecAt, &vecUp);
	D3DXMatrixPerspectiveFovLH( mProj, D3DX_PI/2, 1, 0.001, 2.0 );
}

IDirect3DCubeTexture9* CreateCubeTexture( int size, D3DFORMAT Format, IDirect3DDevice9* pd3dDevice)
{
	HRESULT hr;
	IDirect3DCubeTexture9* pCubeTexture;

	if( FAILED(hr = D3DXCreateCubeTexture( pd3dDevice, 
			size, 1,
			D3DUSAGE_RENDERTARGET, 
			Format,
			D3DPOOL_DEFAULT,
			&pCubeTexture )))
	{
			MessageBox(NULL, L"Cube texture creation failed!", L"Error", MB_OK|MB_TOPMOST);
			exit(-1);
	}
	return pCubeTexture;
}

IDirect3DTexture9* CreateTexture( int resX, int resY, D3DFORMAT Format, IDirect3DDevice9* pd3dDevice)
{
	HRESULT hr;
	IDirect3DTexture9* pTexture;

	if( FAILED(hr = D3DXCreateTexture( pd3dDevice, 
			resX, resY, 1,
			D3DUSAGE_RENDERTARGET, 
			Format,
			D3DPOOL_DEFAULT,
			&pTexture )))
	{
			MessageBox(NULL, L"Texture creation failed!", L"Error", MB_OK|MB_TOPMOST);
			exit(-1);
	}
	return pTexture;
}