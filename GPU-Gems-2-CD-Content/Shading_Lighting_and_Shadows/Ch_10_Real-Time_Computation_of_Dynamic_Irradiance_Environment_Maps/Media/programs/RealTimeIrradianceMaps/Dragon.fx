texture DiffuseTexture;
texture DiffuseIrradiance;
texture SpecularIrradiance;
texture EnvironmentReflection;

sampler DiffuseMap = sampler_state
{
    texture = <DiffuseTexture>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

sampler DiffuseCube = sampler_state
{
    texture = <DiffuseIrradiance>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

sampler SpecularCube = sampler_state
{
    texture = <SpecularIrradiance>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

sampler EnvironmentCube = sampler_state
{
    texture = <EnvironmentReflection>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = LINEAR;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

float4x4 cWorldIT;
float4x4 cWorld;
float4x4 cViewProjection;
float4   cEyePosition;

struct VS_INPUT
{
    float4 Position : POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
};

struct VS_OUTPUT
{
    float4 Position : POSITION;
    float2 TexCoord : TEXCOORD0;
    float3 Normal   : TEXCOORD1;
    float3 View     : TEXCOORD2;
};


//---------------------------------------------------------------------------

VS_OUTPUT Transform_VS( VS_INPUT In )
{
    VS_OUTPUT Out = (VS_OUTPUT) 0;
    float4 worldPos = mul(In.Position, cWorld);
    Out.Position = mul(worldPos, cViewProjection );
    Out.Normal   = mul(In.Normal, (float3x3)cWorldIT );
    Out.View     = cEyePosition - worldPos;
    Out.TexCoord = In.TexCoord;
    return Out;
}

float4 IlluminateCubemap_PS( VS_OUTPUT In ) : COLOR
{
    float4 diffuse = texCUBE(DiffuseCube, In.Normal);
    float3 normal = normalize(In.Normal);
    float3 view   = normalize(In.View);
    float3 reflection = reflect(view, normal);
    float4 specular = texCUBE(SpecularCube, reflection);
    float4 mirror   = texCUBE(EnvironmentCube, reflection);
    
    float fresnel = 0.9*pow(saturate(1 - abs(dot(normal, view))), 8.0) + 0.1;
    
    specular = lerp(specular, mirror, fresnel);
    
    return tex2D(DiffuseMap, In.TexCoord)*diffuse + specular;
}

float4 IlluminateDirectional_PS( VS_OUTPUT In ) : COLOR
{
    const float3 lightDirection = float3( 0.66667, 0.66667, -0.33333 );   
    const float4 diffuseColor = float4(0.945,0.714,0.517,1.0);
    const float4 specularColor = float4(0.227,0.175,0.129, 1.0);
    
    float3 normal = normalize(In.Normal);
    float3 view   = normalize(In.View);
    float3 reflection = reflect(view, normal);

    float ndl = dot(lightDirection,normal);
    float4 diffuse = diffuseColor * ndl;
    float4 specular = (ndl>0) ? pow(saturate(dot(reflection,lightDirection)),9.0) * specularColor : 0;

    float4 mirror   = texCUBE(EnvironmentCube, reflection);
    
    float fresnel = 0.9*pow(saturate(1 - abs(dot(normal, view))), 8.0) + 0.1;
    
    specular = lerp(specular, mirror, fresnel);
    
    return tex2D(DiffuseMap, In.TexCoord)*diffuse + specular;
}


technique RenderCubemap
{
    Pass P0
    {
        ZEnable = TRUE;
        ZWriteEnable = TRUE;
        ColorWriteEnable = Red | Blue | Green | Alpha;
        CullMode = NONE;
        VertexShader = compile vs_2_0 Transform_VS();
        PixelShader = compile ps_2_0 IlluminateCubemap_PS();
    }
}

technique RenderDirectional
{
    Pass P0
    {
        ZEnable = TRUE;
        ZWriteEnable = TRUE;
        ColorWriteEnable = Red | Blue | Green | Alpha;
        CullMode = NONE;
        VertexShader = compile vs_2_0 Transform_VS();
        PixelShader = compile ps_2_0 IlluminateDirectional_PS();
    }
}