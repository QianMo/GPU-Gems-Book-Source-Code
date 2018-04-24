//--------------------------------------------------------------------------------------
// Uniforms
// NOTE: Generally if a value is set here, it is not set in the code, but may be at
// some point in the future.
//--------------------------------------------------------------------------------------

// Transform parameters
cbuffer changing
{
    float4x4 g_WorldMatrix;                   // Transform into world space
    float4x4 g_WorldViewMatrix;               // Transform into view space
    float4x4 g_WorldViewProjMatrix;           // Transform into view space and project
    
    float4x4 g_LightViewProjMatrix;           // Transform world to light space and project
    float3   g_ViewPos;                       // World space view origin
    
    float3   g_LightPosition;                 // World space light position
    float3   g_LightDirection;                // World space light direction
    float    g_LightFOV;                      // Light field of view (radians)
    float2   g_LightLinNearFar;               // Distances to use for rescaling shadows
    float2   g_LightDistFalloff;              // Light distance falloff
    float2   g_LightAngleFalloff = {0.8, 1};  // Light angular falloff (relative to FOV)
}

// Lighting parameters
float    g_AmbientIntensity;                  // Ambient lighting
float3   g_LightColor;                        // Light color/intensity

// Surface parameters
float    g_SpecularPower;
float3   g_SpecularColor;

// Shadow parameters
float2   g_ShadowTextureSize;                 // Size of the shadow map texture
float    g_DepthBias;                         // Point/PCF depth bias (in [0, 1] units)
float    g_VSMMinVariance = 0.000001;         // Minimum variance for VSM

// UI-related parameters
bool     g_LightingOnly;                      // Whether to only show the shadow
float    g_MinFilterWidth;                    // Minimum filter width
bool     g_LBR;                               // Enable/disable light bleeding reduction
float    g_LBRAmount;                         // Aggressiveness of light bleeding reduction
float    g_DistributePrecision;               // Distribute precision on/off flag


//--------------------------------------------------------------------------------------
// Constants (can be moved to uniforms easily)
//--------------------------------------------------------------------------------------
// Factor to use to distribute FP precision
// TODO: Perhaps make this dependent on shadow map size and z-scale at least
// Really we're just waiting for GPU doubles though...
static const float g_DistributeFPFactor = 128;

// Converting normalized floats to UINTs for storage in a SAT
// NOTE: Can compute a "safe" value for this from texture size and filter width ranges,
// but this value works quite well for our demo, and trial and error will arguably 
// produce better results anyways.
// However, most applications won't need gigantic filters, in which case this value can
// be raised to obtain even better numeric precision.
static const uint g_SATUINTPrecisionBits = 18;
static const float g_SATUINTMaxFilterWidth = 1 << ((32 - g_SATUINTPrecisionBits) / 2);
static const float g_NormalizedFloatToSATUINT = 1 << g_SATUINTPrecisionBits;


//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
Texture2D texDiffuse;


//--------------------------------------------------------------------------------------
// Sampling
//--------------------------------------------------------------------------------------
// Basic point sampling
SamplerState sampPoint
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = MIN_MAG_MIP_POINT;
};

// Basic bilinear sampling
SamplerState sampBilinear
{
    AddressU = Clamp;
    AddressV = Clamp;
    Filter = MIN_MAG_LINEAR_MIP_POINT;
};

// Same as above, except with periodic boundary conditions
SamplerState sampAnisotropicWrap
{
    AddressU = Wrap;
    AddressV = Wrap;
    Filter = ANISOTROPIC;
    MaxAnisotropy = 16;
};


//--------------------------------------------------------------------------------------
// State
//--------------------------------------------------------------------------------------
RasterizerState rsNormal
{
    CullMode = Back;
    FillMode = Solid;
    MultisampleEnable = true;
    ScissorEnable = false;
};

DepthStencilState dsNormal
{
    DepthEnable = true;
    DepthFunc = Less_Equal;
    DepthWriteMask = All;
    StencilEnable = false;
};

BlendState bsNormal
{
};


//--------------------------------------------------------------------------------------
// Utilities
//--------------------------------------------------------------------------------------
// Utilities (may use above uniforms, hence their inclusion here!)
#include "Util.fx"


//--------------------------------------------------------------------------------------
// Scene Shading
//--------------------------------------------------------------------------------------
struct Shading_VSIn
{
    float3 Position      : POSITION;     // Object space position
    float3 Normal        : NORMAL;       // Object space normal
    float2 TexCoord      : TEXCOORD;     // Texture coordinates
};

struct Shading_PSIn
{
     float4 Position     : SV_Position;  // Homogenious position
     float2 TexCoord     : TEXCOORD;     // Model texture coordinates
     float3 PosWorld     : PosWorld;     // World space position
     float3 Normal       : NORMAL;       // World space normal
     float4 PosLight     : PosLight;     // Light space homogenious position
};

Shading_PSIn Shading_VS(Shading_VSIn Input)
{
    Shading_PSIn Output;
    
    // Transform
    Output.Position = mul(float4(Input.Position, 1), g_WorldViewProjMatrix);
    float4 PosWorld = mul(float4(Input.Position, 1), g_WorldMatrix);
    Output.PosWorld = PosWorld;
    Output.Normal   = mul(float4(Input.Normal, 0), g_WorldMatrix);   // Assume orthogonal
    Output.PosLight = mul(PosWorld, g_LightViewProjMatrix);
    
    Output.TexCoord = Input.TexCoord;
    
    return Output;
}

float3 SpotLightShader(float3 SurfacePosition,
                       float3 SurfaceNormal,
                       out float DistToLight,
                       out float3 DirToLight)
{
    DirToLight = g_LightPosition - SurfacePosition;
    DistToLight = length(DirToLight);
    DirToLight /= DistToLight;
    
    // Distance attenuation
    float DistFactor = 1 - linstep(g_LightDistFalloff.x, g_LightDistFalloff.y, DistToLight);

    // Radial attenuation
    float2 CosAngleFalloff = cos(0.5 * g_LightFOV * g_LightAngleFalloff);
    float CosAngle = dot(-DirToLight, g_LightDirection);
    float AngleFactor = 1 - linstep(CosAngleFalloff.x, CosAngleFalloff.y, CosAngle);

    return DistFactor * AngleFactor * g_LightColor;
}

float3 PerPixelLighting(float3 Position,
                        float3 Normal,
                        float2 TexCoord,
                        float3 DirToLight,
                        float3 LightContrib)
{
    // Fetch the diffuse color
    float3 DiffuseColor = texDiffuse.Sample(sampAnisotropicWrap, TexCoord);
    float3 View = normalize(g_ViewPos - Position);
    
    // Blinn-Phong BRDF
    float3 HalfVector = normalize(DirToLight + View);
    float NdotL = dot(Normal, DirToLight);
    float DiffuseAmount  = max(0, NdotL);
    float SpecularAmount = pow(max(0, dot(Normal, HalfVector)), g_SpecularPower);

    // Combine ambient, diffuse, specular and external attenuation
    float3 Lit = DiffuseColor * g_AmbientIntensity +
                 LightContrib * (DiffuseColor * DiffuseAmount + 
                                 g_SpecularColor * SpecularAmount);
    
    return g_LightingOnly ? (LightContrib * DiffuseAmount) : Lit;
}


//--------------------------------------------------------------------------------------
// Include other utilities
// NOTE: Order is important here!
//--------------------------------------------------------------------------------------
#include "PostProcess.fx"
