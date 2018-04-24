//----------------------------------------------------------------------------------
// File:   NVUTSkybox.fx
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

// Parameters

float4x4 g_mInvWorldViewProjection;
TextureCube g_EnvironmentTexture;
float4 g_FloorColor;
float g_SkyboxIntensity = 1.0;

SamplerState BilinearSampler
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

DepthStencilState DisableDepth
{
    DepthEnable = false;
    DepthWriteMask = 0;
};

RasterizerState DisableCulling
{
    CullMode = None; //Back
};

DepthStencilState EnableDepth
{
    DepthEnable = true;
    DepthWriteMask = All;
    DepthFunc = Less_Equal;
};

BlendState DisableBlend
{
    BlendEnable[0] = false;
};

//-----------------------------------------------------------------------------
// Skybox stuff
//-----------------------------------------------------------------------------
struct SkyboxVS_Input
{
    float2 Pos : POSITION;
};

struct SkyboxVS_Output
{
    float4 Pos : SV_POSITION;
    float3 Tex : TEXCOORD0;
};

//
// SHADERS
SkyboxVS_Output SkyboxVS( SkyboxVS_Input Input )
{
    SkyboxVS_Output Output;

    Output.Pos = float4( Input.Pos.xy, 1.0, 1.0 );
    Output.Tex = normalize( mul( float4( Input.Pos.xy, 1.0, 1.0 ), g_mInvWorldViewProjection ) );

    return Output;
}

float4 SkyboxPS( SkyboxVS_Output Input ) : SV_Target
{
    float4 Color = g_EnvironmentTexture.Sample( BilinearSampler, Input.Tex );

    return Color;
}

float4 SkyboxWithFloorColorPS(SkyboxVS_Output Input) : SV_Target
{
    float t =  saturate( -10.0 * Input.Tex.y );

    float4 Color = lerp( g_SkyboxIntensity * g_EnvironmentTexture.Sample( BilinearSampler, Input.Tex ), g_FloorColor, t );

    return Color;
}


//
// techniques
//
technique10 RenderSkybox
{
    pass p0
    {
        SetDepthStencilState( DisableDepth, 0 );

        SetVertexShader( CompileShader( vs_4_0, SkyboxVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, SkyboxPS() ) );
    }
}

technique10 RenderSkyboxWithFloorColor
{
    pass p0
    {
        SetDepthStencilState( DisableDepth, 0 );

        SetVertexShader( CompileShader( vs_4_0, SkyboxVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, SkyboxWithFloorColorPS() ) );
    }
}

