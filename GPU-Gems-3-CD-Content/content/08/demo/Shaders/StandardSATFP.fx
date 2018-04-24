//--------------------------------------------------------------------------------------
// Implements a standard summed area table
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Uniforms
//--------------------------------------------------------------------------------------
int2 g_SATPassOffset;


//--------------------------------------------------------------------------------------
// Samplers
//--------------------------------------------------------------------------------------
SamplerState sampSATGen
{
    AddressU = Border;
    AddressV = Border;
    Filter = MIN_MAG_MIP_POINT;
    BorderColor = float4(0, 0, 0, 0);
};

SamplerState sampSATPoint
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = MIN_MAG_MIP_POINT;
};

SamplerState sampSATLinear
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = MIN_MAG_LINEAR_MIP_POINT;
};


//--------------------------------------------------------------------------------------
// SAT Generation via Recursive Doubling
//--------------------------------------------------------------------------------------
float4 GenerateSATRD_PS(uniform int Samples, PostProcess_PSIn Input) : SV_Target
{
    float2 PassOffset = float2(g_SATPassOffset) / g_PPSourceSize;
    
    float4 t = 0;
    [unroll] for (int i = 0; i < Samples; ++i) {
        t += texPPSource.SampleLevel(sampSATGen, Input.TexCoord - i * PassOffset, 0);
    }
    
    return t;
}

technique10 GenerateSATRD2
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, GenerateSATRD_PS(2)));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}

technique10 GenerateSATRD4
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, GenerateSATRD_PS(4)));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}

technique10 GenerateSATRD8
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, PostProcess_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, GenerateSATRD_PS(8)));
        
        SetRasterizerState(rsPostProcess);
        SetDepthStencilState(dsPostProcess, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}






//--------------------------------------------------------------------------------------
// Point Sampling Functions
//--------------------------------------------------------------------------------------
// Point sample using two corners in the SAT
// Coords.xy is upper left, Coords.zw is lower right.
// Inclusive on upper left, not on lower right.
// Returns the sum (not average) - mostly an internal function
float4 SampleSATSum(Texture2D Texture, float4 Coords, int2 CoordsOffset)
{
    // Sample four SAT corners
    float4 nn = Texture.SampleLevel(sampSATPoint, Coords.xy, 0, CoordsOffset);
    float4 np = Texture.SampleLevel(sampSATPoint, Coords.xw, 0, CoordsOffset);
    float4 pn = Texture.SampleLevel(sampSATPoint, Coords.zy, 0, CoordsOffset);
    float4 pp = Texture.SampleLevel(sampSATPoint, Coords.zw, 0, CoordsOffset);
    
    // Clamping
    // Disable this for now... only really necessary when we want exact reproduction
    // around the x|y == 0 lines. Otherwise just wastes some time.
    /*
    bool4 InBounds = (Coords >= 0);
    bool4 Mask = InBounds.xxzz && InBounds.ywyw;
    nn *= Mask.x;
    np *= Mask.y;
    pn *= Mask.z;
    pp *= Mask.w;
    */
    
    // Return the sum of the area within
    return (pp - pn - np + nn);
}

// As above, but average instead of sum
float4 SampleSAT(Texture2D Texture, float4 Coords, float2 TexSize)
{
    float2 TexelSize = 1 / TexSize;
    
    // Work out normalized coordinates and area
    float4 RealCoords = (Coords - TexelSize.xyxy);
    float2 Dims = (Coords.zw - Coords.xy) * TexSize;
    
    // Sample sum and divide to get average
    return SampleSATSum(Texture, RealCoords, int2(0, 0)) / (Dims.x * Dims.y);
}


//--------------------------------------------------------------------------------------
// Bilinear Sampling Functions
//--------------------------------------------------------------------------------------
void SampleSATSumBilinear(Texture2D Texture, float4 Coords, float2 TexSize,
                          out float4 X, out float4 Y, out float4 Z, out float4 W)
{
    float2 TexelSize = 1 / TexSize;
    float4 RealCoords = (Coords - TexelSize.xyxy);
       
    // Sample the four rectangles
    X = SampleSATSum(Texture, RealCoords, int2(1, 0));
    Y = SampleSATSum(Texture, RealCoords, int2(0, 1));
    Z = SampleSATSum(Texture, RealCoords, int2(1, 1));
    W = SampleSATSum(Texture, RealCoords, int2(0, 0));
}

// As above, but average instead of sum
void SampleSATBilinear(Texture2D Texture, float4 Coords, float2 TexSize,
                       out float4 X, out float4 Y, out float4 Z, out float4 W)
{
    SampleSATSumBilinear(Texture, Coords, TexSize, X, Y, Z, W);
    
    // Average
    float2 Dims = (Coords.zw - Coords.xy) * TexSize;
    float InvArea = 1 / float(Dims.x * Dims.y);
    X *= InvArea;
    Y *= InvArea;
    Z *= InvArea;
    W *= InvArea;
}
