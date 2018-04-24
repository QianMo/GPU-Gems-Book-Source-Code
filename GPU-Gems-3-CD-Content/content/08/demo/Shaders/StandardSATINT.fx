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
uint4 GenerateSATRD_PS(uniform int Samples, PostProcess_PSIn Input) : SV_Target
{
    int2 IntCoord = (Input.TexCoord * g_PPSourceSize);
    
    uint4 t = 0;
    [unroll] for (int i = 0; i < Samples; ++i) {
        int2 Coord = IntCoord - i * g_SATPassOffset;
        // TODO: Do we need this boundary check? Don't seem to, but that might be G80
        // specific... D3D10 docs don't define the out-of-bounds behaviour of Load.
        //[flatten] if (all(Coord >= 0)) {
            t += texPPSourceUint.Load(int3(Coord, 0));
        //}
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
uint4 SampleSATSum(UintTex Texture, int4 Coords, int2 CoordsOffset)
{
    // Modify inclusion to be on upper left (not lower right)
    Coords -= 1;

    // Sample four SAT corners
    uint4 nn = Texture.Load(int3(Coords.xy, 0), CoordsOffset);
    uint4 np = Texture.Load(int3(Coords.xw, 0), CoordsOffset);
    uint4 pn = Texture.Load(int3(Coords.zy, 0), CoordsOffset);
    uint4 pp = Texture.Load(int3(Coords.zw, 0), CoordsOffset);
    
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
    
    // Return the sum of the area within (computation order matters here!)
    return (((pp - pn) - np) + nn);
}

// As above, but average instead of sum
float4 SampleSAT(UintTex Texture, float4 Coords, float2 TexSize)
{    
    // Work out unnormalized coordinates and area
    int4 RealCoords = Coords * TexSize.xyxy;
    int2 Dims = RealCoords.zw - RealCoords.xy;
    int Area = Dims.x * Dims.y;
    
    // Sample sum and divide to get average
    uint4 Sum = SampleSATSum(Texture, RealCoords, int2(0, 0));
    return float4(Sum) / float(Area);
}


//--------------------------------------------------------------------------------------
// Bilinear Sampling Functions
//--------------------------------------------------------------------------------------
void SampleSATSumBilinear(UintTex Texture, int4 Coords, float2 TexSize,
                          out float4 X, out float4 Y, out float4 Z, out float4 W)
{
    // Sample the four rectangles
    X = float4(SampleSATSum(Texture, Coords, int2(1, 0)));
    Y = float4(SampleSATSum(Texture, Coords, int2(0, 1)));
    Z = float4(SampleSATSum(Texture, Coords, int2(1, 1)));
    W = float4(SampleSATSum(Texture, Coords, int2(0, 0)));
}

// As above, but average instead of sum
void SampleSATBilinear(UintTex Texture, float4 Coords, float2 TexSize,
                       out float4 X, out float4 Y, out float4 Z, out float4 W)
{
    // Work out unnormalized coordinates and area
    int4 RealCoords = Coords * TexSize.xyxy;
    int2 Dims = RealCoords.zw - RealCoords.xy;
    int Area = Dims.x * Dims.y;
    
    SampleSATSumBilinear(Texture, RealCoords, TexSize, X, Y, Z, W);
    
    // Average
    float InvArea = 1 / float(Area);
    X *= InvArea;
    Y *= InvArea;
    Z *= InvArea;
    W *= InvArea;
}
