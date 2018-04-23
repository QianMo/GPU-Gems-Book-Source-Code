#define PARABOLOID_SAMPLES 64

texture CubeMap;

float3 DirectionVec;

sampler CubeMapSampler = sampler_state
{
    Texture = <CubeMap>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

struct PS_INPUT
{
  float2 position : VPOS;
};

float4 Cubemap_To_Paraboloid_PS(PS_INPUT IN) : COLOR
{
  float3 parab_normal;
  parab_normal.xy = ((IN.position + float2(0.5, 0.5))/PARABOLOID_SAMPLES) * 2.f - 1.f; 
  
  //  optional (paraboloid only exists on x^+y^2<=1, but provide some padding for bilinear filtering)
  //  clip( 1.1 - dot(parab_normal.xy, parab_normal.xy) );
  
  parab_normal.z = 1.0;
  parab_normal = normalize(parab_normal);
  return texCUBE(CubeMapSampler, reflect(DirectionVec, parab_normal));
}

technique ConvertHemisphere
{
    pass P0
    {
        ZEnable = false;
        CullMode = none;
        PixelShader  = compile ps_3_0 Cubemap_To_Paraboloid_PS( );
    }
}