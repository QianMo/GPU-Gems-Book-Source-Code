#include "Common.fx"
#include "Depth.fx"


//--------------------------------------------------------------------------------------
// Constants
// NOTE: Making these static buggers up the effect file constant buffer...
//--------------------------------------------------------------------------------------
cbuffer perminent
{
    const int SplitPowLookup[8] = {0, 1, 1, 2, 2, 2, 2, 3};
    const float3 SplitColors[4] = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0} };
}


//--------------------------------------------------------------------------------------
// Uniforms
//--------------------------------------------------------------------------------------
float4 g_Splits;
float4x4 g_SplitMatrices[4];

// Debugging and UI
int g_DisplayArrayIndex;
bool g_VisualizeSplits;


//--------------------------------------------------------------------------------------
// Textures and sampling
//--------------------------------------------------------------------------------------
Texture2DArray texShadow;

// Full anisotropic trilinear filtering
SamplerState sampPSVSM
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = ANISOTROPIC;
    MaxAnisotropy = 16;
};


//--------------------------------------------------------------------------------------
// PSVSM using hardware filtering
//--------------------------------------------------------------------------------------
float3 SpotLightShaderPSVSM(float3 SurfacePosition,
                            float3 SurfaceNormal,
                            float2 LightTexCoord,
                            int Split,
                            out float DistToLight,
                            out float3 DirToLight)
{
    // Call parent
    float3 LightContrib = SpotLightShader(SurfacePosition, SurfaceNormal,
                                          DistToLight, DirToLight);
    float RescaledDist = RescaleDistToLight(DistToLight);
    
    float2 Moments = texShadow.Sample(sampPSVSM, float3(LightTexCoord, Split)) + GetFPBias();
    float ShadowContrib = ChebyshevUpperBound(Moments, RescaledDist, g_VSMMinVariance);
    
    [flatten] if (g_LBR) {
        ShadowContrib = LBR(ShadowContrib);
    }
    
    return LightContrib * ShadowContrib;
}

struct ShadingPSVSM_PSIn
{
     float4 Position     : SV_Position;  // Homogenious position
     float2 TexCoord     : TEXCOORD;     // Model texture coordinates
     float3 PosWorld     : PosWorld;     // World space position
     float3 Normal       : NORMAL;       // World space normal
     float  SliceDepth   : SliceDepth;   // Depth of fragment in screen space (pre-divide)
};

// Need a custom different vertex shader for this technique
ShadingPSVSM_PSIn ShadingPSVSM_VS(Shading_VSIn Input)
{
    ShadingPSVSM_PSIn Output;
    
    // Transform
    Output.Position   = mul(float4(Input.Position, 1), g_WorldViewProjMatrix);
    Output.PosWorld   = mul(float4(Input.Position, 1), g_WorldMatrix);
    Output.Normal     = mul(float4(Input.Normal, 0), g_WorldMatrix);   // Assume orthogonal
    
    Output.TexCoord   = Input.TexCoord;
    Output.SliceDepth = Output.Position.z;
    
    return Output;
}

float4 ShadingPSVSM_PS(ShadingPSVSM_PSIn Input) : SV_Target
{
    // Renormalize per-pixel
    float3 Normal = normalize(Input.Normal);
    
    // Compute which split we're in
    int Split = dot(1, Input.SliceDepth > g_Splits);
    
    // Ensure that every fragment in the quad choses the same split so that derivatives
    // will be meaningful for proper texture filtering and LOD selection.
    // Hehe... best trick/hack ever :)
    int SplitPow = 1 << Split;
    int SplitX = abs(ddx(SplitPow));
    int SplitY = abs(ddy(SplitPow));
    int SplitXY = abs(ddx(SplitY));
    int SplitMax = max(SplitXY, max(SplitX, SplitY));
    //SplitMax = min(SplitMax, 8);
    Split = SplitMax > 0 ? SplitPowLookup[SplitMax-1] : Split;
    
    // Project using the associated matrix
    float4 PosLight = mul(float4(Input.PosWorld, 1), g_SplitMatrices[Split]);
    float2 LightTexCoord = (PosLight.xy / PosLight.w) * float2(0.5, -0.5) + 0.5;
    
    // Execute light shader
    float DistToLight;
    float3 DirToLight;
    float3 LightContrib = SpotLightShaderPSVSM(Input.PosWorld, Normal,
                                               LightTexCoord, Split,
                                               DistToLight, DirToLight);
      
    
    // Execute surface shader
    float3 Color = PerPixelLighting(Input.PosWorld, Normal, Input.TexCoord,
                                    DirToLight, LightContrib);

    [flatten] if (g_VisualizeSplits) {
        Color = lerp(Color, SplitColors[Split], 0.15);
    }
    
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
        SetVertexShader(CompileShader(vs_4_0, ShadingPSVSM_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, ShadingPSVSM_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}




//--------------------------------------------------------------------------------------
// Display a shadow map slice
//--------------------------------------------------------------------------------------
float4 DisplayShadowMap_PS(PostProcess_PSIn Input) : SV_Target
{
    float2 Moments = texShadow.Sample(sampPSVSM,
                                      float3(Input.TexCoord, g_DisplayArrayIndex)) +
                     GetFPBias();
    
    return float4(Moments.xy, 0, 1);
}

technique10 DisplayShadowMap
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, DisplayShadowMap_PS()));
        
        SetRasterizerState(rsPostProcessNoScissor);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}