//--------------------------------------------------------------------------------------
// Misc Utilities
//--------------------------------------------------------------------------------------
// Don't know why this isn't in the standard library...
float linstep(float min, float max, float v)
{
    return clamp((v - min) / (max - min), 0, 1);
}

// Rescale into [0, 1]
float RescaleDistToLight(float Distance)
{
    return linstep(g_LightLinNearFar.x, g_LightLinNearFar.y, Distance);
}

float2 GetFPBias()
{
    return float2(0.5, 0);
}

// Distribute float precision
// NOTE: We want cheap reconstruction, so do most of the work here
// Moments may be already biased here, so have to also handle negatives.
float4 DistributeFP(float2 Value)
{
    float FactorInv = 1 / g_DistributeFPFactor;
    
    // Split precision
    float2 IntPart;
    float2 FracPart = modf(Value * g_DistributeFPFactor, IntPart);
    
    // Compose outputs to make it cheap to recombine
    return float4(IntPart * FactorInv, FracPart);
}

// Recombine distributed floats (inverse of the above)
float2 RecombineFP(float4 Value)
{
    float FactorInv = 1 / g_DistributeFPFactor;
    return (Value.zw * FactorInv + Value.xy);
}

// Compute the filter size in pixels
float2 GetFilterSize(float2 dx, float2 dy, float2 TexSize)
{
    return 2 * (abs(dx) + abs(dy)) * TexSize;
}

// Compute the upper left and size of the filter tile
// MinFilterWidth, MaxSizeDerivatives and TexSize given in texels
// Rest of the parameters and returns are in normalized coordinates
// NOTE: Can provide an upper bound for the size (in texels) computed via derivatives.
// This is necessary since GPU's finite differencing screws up in some cases,
// returning rediculous sizes here. For operations that loop on the filter area
// this is a big problem...
float2 GetFilterTile(float2 tc, float2 dx, float2 dy, float2 TexSize,
                     float2 MinFilterWidth, float2 MaxSizeDerivatives,
                     out float2 Size)
{
    float2 TexelSize = 1 / TexSize;

    // Compute the filter size based on derivatives
    float2 SizeDerivatives = min(GetFilterSize(dx, dy, TexSize),
                                 MaxSizeDerivatives);
    
    // Force an integer tile size (in pixels) so that bilinear weights are consistent
    Size = round(max(SizeDerivatives, MinFilterWidth)) * TexelSize;
    
    // Compute upper left corner of the tile
    return (tc - 0.5 * (Size - TexelSize));
}

// Returns coordinates for the four pixels surround a given fragment.
// Given and returned Coords are normalized
// These are given by (in Fetch4 order) - where "R" is the returned value:
//   - R + (1, 0)
//   - R + (0, 1)
//   - R + (1, 1)
//   - R
// Also returns bilinear weights in the output parameter.
float2 GetBilCoordsAndWeights(float2 Coords, float2 TexSize, out float4 Weights)
{
    float2 TexelSize = 1 / TexSize;
    float2 TexelCoords = Coords * TexSize;
    
    // Compute weights
    Weights.xy = frac(TexelCoords + 0.5);
    Weights.zw = 1 - Weights.xy;
    Weights = Weights.xzxz * Weights.wyyw;
    
    // Compute upper-left pixel coordinates
    // NOTE: D3D texel alignment...
    return (floor(TexelCoords - 0.5) + 0.5) * TexelSize;
}

// Computes Chebyshev's Inequality
// Returns an upper bound given the first two moments and mean
float ChebyshevUpperBound(float2 Moments, float Mean, float MinVariance)
{
    // Standard shadow map comparison
    float p = (Mean <= Moments.x);
    
    // Compute variance
    float Variance = Moments.y - (Moments.x * Moments.x);
    Variance = max(Variance, MinVariance);
    
    // Compute probabilistic upper bound
    float d     = Mean - Moments.x;
    float p_max = Variance / (Variance + d*d);
    
    return max(p, p_max);
}

// Light bleeding reduction
float LBR(float p)
{
    // Lots of options here if we don't care about being an upper bound.
    // Use whatever falloff function works well for your scene.
    return linstep(g_LBRAmount, 1, p);
    //return smoothstep(g_LBRAmount, 1, p);
}
