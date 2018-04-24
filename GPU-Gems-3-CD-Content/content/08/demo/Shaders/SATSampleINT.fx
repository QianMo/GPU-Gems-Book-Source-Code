#include "StandardSATINT.fx"


//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
UintTex texSAT;


//--------------------------------------------------------------------------------------
// Summed area tables
// NOTE: Could generalize these to take a Texture2D if we have more than one SAT
//--------------------------------------------------------------------------------------
// Sample and return a single fragment
float2 SampleMomentsPoint(float4 Coords)
{
    float4 Value = SampleSAT(texSAT, Coords, g_ShadowTextureSize);
    float2 V = Value.xy;

    // Re-normalize
    float g_SATUINTToNormalizedFloat = 1 / g_NormalizedFloatToSATUINT;
    V *= g_SATUINTToNormalizedFloat;

    return V;
}

// Returns into the X, Y, Z, W coordinates the four rectangle averages
void SampleMomentsBilinear(float4 Coords,
                           out float2 X, out float2 Y, out float2 Z, out float2 W)
{
    float4 X1, Y1, Z1, W1;
    SampleSATBilinear(texSAT, Coords, g_ShadowTextureSize, X1, Y1, Z1, W1);
    X = X1.xy;
    Y = Y1.xy;
    Z = Z1.xy;
    W = W1.xy;
    
    // Re-normalize
    float g_SATUINTToNormalizedFloat = 1 / g_NormalizedFloatToSATUINT;
    X *= g_SATUINTToNormalizedFloat;
    Y *= g_SATUINTToNormalizedFloat;
    Z *= g_SATUINTToNormalizedFloat;
    W *= g_SATUINTToNormalizedFloat;
}


