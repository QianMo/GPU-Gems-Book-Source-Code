//--------------------------------------------------------------------------------------
// Depth and Moments Rendering Utilities
//--------------------------------------------------------------------------------------
struct Depth_VSIn
{
    float3 Position      : POSITION;     // Object space position
};

struct Depth_PSIn
{
     float4 Position     : SV_Position;  // Homogenious position
     float3 PosView      : PosView;      // View space position
};

Depth_PSIn Depth_VS(Shading_VSIn Input)
{
    Depth_PSIn Output;
    
    Output.Position = mul(float4(Input.Position, 1), g_WorldViewProjMatrix);
    Output.PosView  = mul(float4(Input.Position, 1), g_WorldViewMatrix);
    
    return Output;
}

float Depth_PS(Depth_PSIn Input) : SV_Target
{
    float Depth = RescaleDistToLight(length(Input.PosView));
    return Depth;
}

// Utility function
float2 ComputeMoments(float Depth)
{
    // Compute first few moments of depth
    float2 Moments;
    Moments.x = Depth;
    Moments.y = Depth * Depth;
    
    // Ajust the variance distribution to include the whole pixel if requested
    // NOTE: Disabled right now as a min variance clamp takes care of all problems
    // and doesn't risk screwy hardware derivatives.
    //float dx = ddx(Depth);
    //float dy = ddy(Depth);
    //float Delta = 0.25 * (dx*dx + dy*dy);
    // Perhaps clamp maximum Delta here
    //Moments.y += Delta;

    return Moments;
}

// Use centroid interpolation to avoid problems with the depth being outside of
// the real polygon extents when multisampling.
float2 MomentsFP_PS(centroid Depth_PSIn Input) : SV_Target
{
    float Depth = Depth_PS(Input);
    float2 Moments = ComputeMoments(Depth) - GetFPBias();
    return Moments;
}

// Distribute moments into four components
float4 MomentsDistributeFP_PS(centroid Depth_PSIn Input) : SV_Target
{
    float2 Moments = MomentsFP_PS(Input);
    return DistributeFP(Moments);
}

// Same as above, except outputting to an int32 texture
uint2 MomentsINT_PS(centroid Depth_PSIn Input) : SV_Target
{
    float Depth = Depth_PS(Input);
    float2 Moments = ComputeMoments(Depth);
    uint2 MomentsINT = uint2(round(Moments * g_NormalizedFloatToSATUINT));
    return MomentsINT;
}
