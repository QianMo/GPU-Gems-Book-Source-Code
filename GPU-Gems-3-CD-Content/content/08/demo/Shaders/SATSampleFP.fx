#include "StandardSATFP.fx"


//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
Texture2D texSAT;


//--------------------------------------------------------------------------------------
// Summed area tables
// NOTE: Could generalize these to take a Texture2D if we have more than one SAT
//--------------------------------------------------------------------------------------
// Sample and return a single fragment
float2 SampleMomentsPoint(float4 Coords)
{
    float4 Value = SampleSAT(texSAT, Coords, g_ShadowTextureSize);
    
    float2 V;
    if (g_DistributePrecision) {
        V = RecombineFP(Value);
    } else {
        V = Value.xy;
    }

    // Unbias
    V += GetFPBias();
    return V;
}

// Returns into the X, Y, Z, W coordinates the four rectangle averages
void SampleMomentsBilinear(float4 Coords,
                           out float2 X, out float2 Y, out float2 Z, out float2 W)
{
    float4 X1, Y1, Z1, W1;
    SampleSATBilinear(texSAT, Coords, g_ShadowTextureSize, X1, Y1, Z1, W1);

    if (g_DistributePrecision) {
        X = RecombineFP(X1);
        Y = RecombineFP(Y1);
        Z = RecombineFP(Z1);
        W = RecombineFP(W1);
    } else {
        X = X1.xy;
        Y = Y1.xy;
        Z = Z1.xy;
        W = W1.xy;
    }
    
    // Unbias
    float2 FPBias = GetFPBias();
    X += FPBias;
    Y += FPBias;
    Z += FPBias;
    W += FPBias;
}

