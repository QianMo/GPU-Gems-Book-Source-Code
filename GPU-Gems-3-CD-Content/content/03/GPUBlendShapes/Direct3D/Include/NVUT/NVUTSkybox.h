//----------------------------------------------------------------------------------
// File:   NVUTSkybox.h
// Email:  sdkfeedback@nvidia.com
// 
// Copyright (c) 2007 NVIDIA Corporation. All rights reserved.
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
//
//----------------------------------------------------------------------------------

// Desc: Encapsulation of skybox geometry and textures

#pragma once
#ifndef NVUTSKYBOX_H
#define NVUTSKYBOX_H

#include <DXUT.h>
#include <SDKmisc.h>

class NVUTSkybox
{
public:
    NVUTSkybox();
    ~NVUTSkybox();

    HRESULT OnCreateDevice( ID3D10Device* pd3dDevice );
    void OnResizedSwapChain( ID3D10Device* pd3dDevice, const DXGI_SURFACE_DESC *pBackBufferSurfaceDesc );
    void OnReleasingSwapChain();

    void OnFrameRender( const D3DXMATRIX &mWVP );
    void OnFrameRender( const D3DXMATRIX &mWVP, float SkyboxIntensity, const D3DXVECTOR4& FloorColor );
    void OnDestroyDevice();

    void SetTexture( ID3D10ShaderResourceView* ppEnvCubeRV ) { m_EnvMap = ppEnvCubeRV; };

protected:
    // Texture resource views
    ID3D10ShaderResourceView * m_EnvTex;

    // The device
    ID3D10Device * m_d3dDevice;

    // Vertex layout
    ID3D10InputLayout * m_VertexLayout;
    ID3D10Buffer * m_VertexBuffer, * m_IndexBuffer;

    // The effects and rendering techniques
    ID3D10Effect * m_Effect;
    ID3D10EffectTechnique * m_Technique, *m_TechniqueWithColor;
    ID3D10ShaderResourceView * m_EnvMap;

    // Effect variable pointers
    ID3D10EffectMatrixVariable * m_VarInvWVP;
    ID3D10EffectShaderResourceVariable * m_VarEnvMap;
    ID3D10EffectVectorVariable * m_VarFloorColor;
    ID3D10EffectScalarVariable * m_VarSkyboxIntensity;
};

#endif

