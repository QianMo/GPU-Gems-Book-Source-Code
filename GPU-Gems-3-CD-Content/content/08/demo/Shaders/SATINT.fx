#include "Common.fx"
#include "Depth.fx"
#include "SATSampleINT.fx"


//--------------------------------------------------------------------------------------
// Include common shading code
// Clamp the maximum filter width based on how much precision we're reserved for
// accumulation (see Common.fx)
static const float g_MaxFilterWidth = g_SATUINTMaxFilterWidth;
#include "SATShading.fx"


//--------------------------------------------------------------------------------------
// Convert a fp32 shadow map into an int32 one
//--------------------------------------------------------------------------------------
uint2 ConvertToIntShadowMap_PS(PostProcess_PSIn Input) : SV_Target
{
    // Read moments and unbias
    float2 Moments = texPPSource.SampleLevel(sampPoint, Input.TexCoord, 0);
    Moments += GetFPBias();
    
    // Convert to int
    uint2 MomentsINT = uint2(round(Moments * g_NormalizedFloatToSATUINT));
    return MomentsINT;
}

technique10 ConvertToIntShadowMap
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, ConvertToIntShadowMap_PS()));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
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
        SetPixelShader(CompileShader(ps_4_0, MomentsINT_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}

// Multisampled we render to a fp32 render target
technique10 DepthMSAA
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
