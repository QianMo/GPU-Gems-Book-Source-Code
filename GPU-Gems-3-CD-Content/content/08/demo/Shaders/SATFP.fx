#include "Common.fx"
#include "Depth.fx"
#include "SATSampleFP.fx"


//--------------------------------------------------------------------------------------
// Include common shading code
// No need to clamp maximum filter width
static const float g_MaxFilterWidth = g_ShadowTextureSize;
#include "SATShading.fx"


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

technique10 DepthDistribute
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, Depth_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, MomentsDistributeFP_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}
