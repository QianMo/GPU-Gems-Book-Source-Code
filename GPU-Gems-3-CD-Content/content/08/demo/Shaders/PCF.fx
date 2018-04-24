#include "Common.fx"
#include "Depth.fx"


//--------------------------------------------------------------------------------------
// Uniforms
//--------------------------------------------------------------------------------------
float    g_ConeBias;                      // PCF cone slope bias (in [0, 1] units)


//--------------------------------------------------------------------------------------
// Textures and sampling
//--------------------------------------------------------------------------------------
Texture2D texShadow;

// Hardware bilinear PCF
SamplerComparisonState sampPCF
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    ComparisonFunc = Less_Equal;
};


//--------------------------------------------------------------------------------------
// Percentage closer filtering
//--------------------------------------------------------------------------------------
float ShadowContribPCF(float2 tc,
                       float2 dx,
                       float2 dy,
                       float Distance)
{
    // Preshader
    float2 TexelSize = 1 / g_ShadowTextureSize;
    
    // Tough choice here to make regarding clamping the tile area.
    // Note that we run into an issue with derivatives here, namely that they are wildly
    // incorrect sometimes (and other times we just want really long anisotropic filter
    // kernels). This poses a problem for PCF which scales linearly with the number of
    // samples taken... hence why we have to clamp filter sizes fairly aggressively.
    // Do note though that this will certainly DECREASE THE QUALITY in comparison
    // to SAT and other methods, but we are optionless!
    const float2 MaxSizeDerivatives = float2(32, 32);
    
    // Compute the filter tile information
    float2 Size;
    float2 CoordsUL = GetFilterTile(tc, dx, dy, g_ShadowTextureSize,
                                    g_MinFilterWidth, MaxSizeDerivatives, Size);
    
    float Contrib = 0;
    int2 PixelSize = round(Size * g_ShadowTextureSize);
    for (int y = 0; y < PixelSize.y; ++y) {
        for (int x = 0; x < PixelSize.x; ++x) {
            float2 TexCoord = CoordsUL + float2(x, y) * TexelSize;
            
            // Cone bias (disabled for now, but probably required in "real" scenes)
            //float KernelCenterDistance = length(TexCoord - tc);
            //float ConeBiasedDistance = Distance - KernelCenterDistance * g_ConeBias;
            
            Contrib += texShadow.SampleCmpLevelZero(sampPCF, TexCoord, Distance);
        }
    }
    
    // Average (this works fine with bilinear weights too)
    return Contrib / float(PixelSize.x * PixelSize.y);
}

float3 SpotLightShaderPCF(float3 SurfacePosition,
                          float3 SurfaceNormal,
                          float2 tc,
                          float2 dx,
                          float2 dy,
                          out float DistToLight,
                          out float3 DirToLight)
{
    // Call parent
    float3 LightContrib = SpotLightShader(SurfacePosition, SurfaceNormal,
                                          DistToLight, DirToLight);
    float RescaledDist = (RescaleDistToLight(DistToLight) - g_DepthBias);
    
    // It is *extremely* important here not to sample the shadow map when we're
    // outside of the light extents, since that will often be *tons* of samples!
    // Early out for backfaces, etc.
    float NdotL = dot(SurfaceNormal, DirToLight);
    [branch] if (NdotL < 0 || all(LightContrib <= 0)) {
        return float3(0, 0, 0);
    } else {
        return LightContrib * ShadowContribPCF(tc, dx, dy, RescaledDist);
    }
}

float4 ShadingPCF_PS(Shading_PSIn Input) : SV_Target
{
    // Renormalize per-pixel
    float3 Normal = normalize(Input.Normal);
    
    // Project
    float2 LightTexCoord = (Input.PosLight.xy / Input.PosLight.w) * float2(0.5, -0.5) + 0.5;

    // Compute derivatives (outside of all control flow)
    float2 dtdx = ddx(LightTexCoord);
    float2 dtdy = ddy(LightTexCoord);

    // Execute light shader
    float DistToLight;
    float3 DirToLight;
    float3 LightContrib = SpotLightShaderPCF(Input.PosWorld, Normal,
                                             LightTexCoord, dtdx, dtdy,
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
        SetPixelShader(CompileShader(ps_4_0, ShadingPCF_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}
