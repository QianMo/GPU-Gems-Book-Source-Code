#include "Common.fx"
#include "Depth.fx"


//--------------------------------------------------------------------------------------
// Textures and sampling
//--------------------------------------------------------------------------------------
Texture2D texShadow;

// Full anisotropic trilinear filtering
SamplerState sampVSM
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = ANISOTROPIC;
    MaxAnisotropy = 16;
};


//--------------------------------------------------------------------------------------
// VSM using hardware filtering
//--------------------------------------------------------------------------------------
float3 SpotLightShaderVSM(float3 SurfacePosition,
                          float3 SurfaceNormal,
                          float2 LightTexCoord,
                          out float DistToLight,
                          out float3 DirToLight)
{
    // Call parent
    float3 LightContrib = SpotLightShader(SurfacePosition, SurfaceNormal,
                                          DistToLight, DirToLight);
    float RescaledDist = RescaleDistToLight(DistToLight);

    float2 Moments = texShadow.Sample(sampVSM, LightTexCoord) + GetFPBias();
    float ShadowContrib = ChebyshevUpperBound(Moments, RescaledDist, g_VSMMinVariance);
    
    [flatten] if (g_LBR) {
        ShadowContrib = LBR(ShadowContrib);
    }
    
    return LightContrib * ShadowContrib;
}

float4 ShadingVSM_PS(Shading_PSIn Input) : SV_Target
{
    // Renormalize per-pixel
    float3 Normal = normalize(Input.Normal);
    
    // Project
    float2 LightTexCoord = (Input.PosLight.xy / Input.PosLight.w) * float2(0.5, -0.5) + 0.5;
    
    // Execute light shader
    float DistToLight;
    float3 DirToLight;
    float3 LightContrib = SpotLightShaderVSM(Input.PosWorld, Normal, LightTexCoord,
                                             DistToLight, DirToLight);
        
    // Execute surface shader
    float3 Color = PerPixelLighting(Input.PosWorld, Normal, Input.TexCoord,
                                    DirToLight, LightContrib);
    return float4(Color, 1);
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique10 Depth
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, Depth_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, MomentsFP_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}

technique10 Shading
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, Shading_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, ShadingVSM_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}
