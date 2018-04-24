//--------------------------------------------------------------------------------------
// VSM using summed area tables
//--------------------------------------------------------------------------------------
// Sample and apply Chebyshev's inequality
float PointChebyshev(float4 Coords, float Distance, float MinVariance)
{
    float2 Moments = SampleMomentsPoint(Coords);
    return ChebyshevUpperBound(Moments, Distance, MinVariance);
}

// Sample a bilinear tile and compute a Chebyshev upper bound
float BilinearChebyshev(float4 Coords, float4 BilWeights, float Distance,
                        float MinVariance)
{
    float2 X, Y, Z, W;
    SampleMomentsBilinear(Coords, X, Y, Z, W);
    
    // Manual bilinear interpolation
    float2 I;
    I.x = dot(BilWeights, float4(X.x, Y.x, Z.x, W.x));
    I.y = dot(BilWeights, float4(X.y, Y.y, Z.y, W.y));
    
    return ChebyshevUpperBound(I, Distance, MinVariance);
        
    /*
    // Compute an upper bound for all four samples
    // This could be vectorized a bit more with rotated data, but again it would
    // be a waste on the G80, and barely a benefit on other architectures.
    float4 Factors;
    Factors.x = ChebyshevUpperBound(X, Distance, MinVariance);
    Factors.y = ChebyshevUpperBound(Y, Distance, MinVariance);
    Factors.z = ChebyshevUpperBound(Z, Distance, MinVariance);
    Factors.w = ChebyshevUpperBound(W, Distance, MinVariance);
    
    // Combine results using bilinear weights
    return dot(BilWeights, Factors);
    */
}

float ShadowContribSAT(float2 tc,
                       float2 dx,
                       float2 dy,
                       float Distance)
{
    float2 TexelSize = 1 / g_ShadowTextureSize;
    
    // Compute the filter tile information
    // Don't clamp the filter area since we have constant-time filtering!
    float2 Size;
    float2 CoordsUL = GetFilterTile(tc, dx, dy, g_ShadowTextureSize,
                                    g_MinFilterWidth, g_MaxFilterWidth, Size);

    // Compute bilinear weights and coordinates
    float4 BilWeights;
    float2 BilCoordsUL = GetBilCoordsAndWeights(CoordsUL, g_ShadowTextureSize, BilWeights);
    float4 Tile = BilCoordsUL.xyxy + float4(0, 0, Size.xy);
    
    // Read the moments and compute a Chebyshev upper bound
    float ShadowContrib = BilinearChebyshev(Tile, BilWeights, Distance, g_VSMMinVariance);
    
    // Use this instead if hardware bilinear is enabled for the SAT
    // We currently don't do this since it causes more precision problems...
    //float4 Tile = CoordsUL.xyxy + float4(0, 0, Size.xy);
    //float ShadowContrib = PointChebyshev(Tile, Distance, g_VSMMinVariance);
    
    [flatten] if (g_LBR) {
        ShadowContrib = LBR(ShadowContrib);
    }
    
    return ShadowContrib;
}

float3 SpotLightShaderSAT(float3 SurfacePosition,
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
    float RescaledDist = RescaleDistToLight(DistToLight);
        
    // NOTE: Could branch here as in PCF, but SAT is much more efficient per-pixel,
    // and so the gain when most framebuffer pixels (like when using deferred lighting)
    // is insignificant, and the branching hurts us more.
    return LightContrib * ShadowContribSAT(tc, dx, dy, RescaledDist);
}

float4 ShadingSAT_PS(Shading_PSIn Input) : SV_Target
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
    float3 LightContrib = SpotLightShaderSAT(Input.PosWorld, Normal,
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
technique10 Shading
{
    pass p0
    {
        SetVertexShader(CompileShader(vs_4_0, Shading_VS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, ShadingSAT_PS()));
        
        SetRasterizerState(rsNormal);
        SetDepthStencilState(dsNormal, 0);
        SetBlendState(bsNormal, float4(0, 0, 0, 0), 0xFFFFFFFF);
    }
}
