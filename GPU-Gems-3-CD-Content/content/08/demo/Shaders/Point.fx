#include "Common.fx"
#include "Depth.fx"


//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
Texture2D texShadow;


//--------------------------------------------------------------------------------------
// Point sampling (standard shadow mapping)
//--------------------------------------------------------------------------------------
float3 SpotLightShaderSM(float3 SurfacePosition,
                         float3 SurfaceNormal,
                         float2 LightTexCoord,
                         out float DistToLight,
                         out float3 DirToLight)
{
    // Call parent
    float3 LightContrib = SpotLightShader(SurfacePosition, SurfaceNormal,
                                          DistToLight, DirToLight);
    float RescaledDist = (RescaleDistToLight(DistToLight) - g_DepthBias);
    
    // Sample shadow map
    float OccluderDepth = texShadow.Sample(sampPoint, LightTexCoord);
    float ShadowContrib = (OccluderDepth >= RescaledDist);
    
    return LightContrib * ShadowContrib;
}

float4 ShadingSM_PS(Shading_PSIn Input) : SV_Target
{
    // Renormalize per-pixel
    float3 Normal = normalize(Input.Normal);
    
    // Project
    float2 LightTexCoord = (Input.PosLight.xy / Input.PosLight.w) * float2(0.5, -0.5) + 0.5;

    // Execute light shader
    float DistToLight;
    float3 DirToLight;
    float3 LightContrib = SpotLightShaderSM(Input.PosWorld, Normal, LightTexCoord,
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
        SetPixelShader(CompileShader(ps_4_0, Depth_PS()));
        
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
        SetPixelShader(CompileShader(ps_4_0, ShadingSM_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}
