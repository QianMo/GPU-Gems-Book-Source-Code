//--------------------------------------------------------------------------------------
// Uniforms
//--------------------------------------------------------------------------------------
float2 g_PPSourceSize;            // TODO: Replace with GetDimensions, and/or INT

// Clearing to constant
float4 g_ConstValue = 0;

// Blurring
float2 g_BlurDim;
int    g_BlurSamples;

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
typedef Texture2D<uint4> UintTex;

Texture2D texPPSource;
Texture2DArray texPPSourceArray;
UintTex texPPSourceUint;


//--------------------------------------------------------------------------------------
// State
//--------------------------------------------------------------------------------------
RasterizerState rsPostProcess
{
    CullMode = None;
    FillMode = Solid;
    ScissorEnable = true;
    MultisampleEnable = false;
};

RasterizerState rsPostProcessNoScissor
{
    CullMode = None;
    FillMode = Solid;
    ScissorEnable = false;
    MultisampleEnable = false;
};

shared DepthStencilState dsPostProcess
{
    DepthEnable = false;
    DepthWriteMask = 0;
    StencilEnable = false;
};


//--------------------------------------------------------------------------------------
// Post processing vertex shader
//--------------------------------------------------------------------------------------
struct PostProcess_VSIn
{
    uint VertexID   : SV_VertexID;
};

struct PostProcess_PSIn
{
    float4 Position : SV_Position;
    float2 TexCoord : TEXCOORD;
};

// Generate a full-screen triangle from vertex ID's
PostProcess_PSIn PostProcess_VS(PostProcess_VSIn Input)
{
    PostProcess_PSIn Output;
    
    Output.TexCoord = float2((Input.VertexID << 1) & 2, Input.VertexID & 2);
    Output.Position = float4(Output.TexCoord * float2(2, -2) + float2(-1, 1), 0, 1);
    
    return Output;
}


//--------------------------------------------------------------------------------------
// Return a constant
//--------------------------------------------------------------------------------------
float4 Const_PS() : SV_Target
{
    return g_ConstValue;
}

technique10 Const
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, Const_PS()));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}


//--------------------------------------------------------------------------------------
// Box-filtered blur
//--------------------------------------------------------------------------------------
float4 BoxBlur_PS(PostProcess_PSIn Input) : SV_Target
{
    // Preshader
    float2 TexelSize = 1 / g_PPSourceSize;
    float  BlurSizeInv = 1 / float(g_BlurSamples);
    float2 SampleOffset = TexelSize * g_BlurDim;
    float2 Offset = 0.5 * float(g_BlurSamples - 1) * SampleOffset;
    
    float2 BaseTexCoord = Input.TexCoord - Offset;
    
    // NOTE: This loop can potentially be optimized to use bilinear filtering to take
    // two samples at a time, rather than handle even/odd filters nicely. However the
    // resulting special-casing required for different filter sizes will probably
    // negate any benefit of fewer samples being taken. Besides, this method is already
    // supidly-fast even for gigantic kernels.
    float4 Sum = float4(0, 0, 0, 0);
    for (int i = 0; i < g_BlurSamples; ++i) {
        Sum += texPPSource.SampleLevel(sampBilinear, BaseTexCoord + i * SampleOffset, 0);
    }
    
    return Sum * BlurSizeInv;
}

technique10 BoxBlur
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, BoxBlur_PS()));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}


//--------------------------------------------------------------------------------------
// Now the same thing as above, except the texture array version!
// Don't even get me started...
//--------------------------------------------------------------------------------------
float4 BoxBlurArray_PS(PostProcess_PSIn Input) : SV_Target
{
    // Preshader
    float2 TexelSize = 1 / g_PPSourceSize;
    float  BlurSizeInv = 1 / float(g_BlurSamples);
    float2 SampleOffset = TexelSize * g_BlurDim;
    float2 Offset = 0.5 * float(g_BlurSamples - 1) * SampleOffset;
    
    float2 BaseTexCoord = Input.TexCoord - Offset;
    
    // NOTE: This loop can potentially be optimized to use bilinear filtering to take
    // two samples at a time, rather than handle even/odd filters nicely. However the
    // resulting special-casing required for different filter sizes will probably
    // negate any benefit of fewer samples being taken. Besides, this method is already
    // supidly-fast even for gigantic kernels.
    float4 Sum = float4(0, 0, 0, 0);
    for (int i = 0; i < g_BlurSamples; ++i) {
        Sum += texPPSourceArray.SampleLevel(sampBilinear,
                                            float3(BaseTexCoord + i * SampleOffset, 0),
                                            0);
    }
    
    return Sum * BlurSizeInv;
}

technique10 BoxBlurArray
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, BoxBlurArray_PS()));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}
