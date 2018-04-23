//
// Hemisphere Lighting Model
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// Note: This effect file works with EffectEdit.
//


string XFile = "SkullOcc.x";                // model
int    BCLR  = 0xff202080;                  // background

// light directions (view space)
float3 DirFromLight < string UIName = "Light Direction"; > = {0.577, -0.577, 0.577};

// direction of light from sky (view space)
float3 DirFromSky < string UIName = "Direction from Sky"; > = { 0.0f, -1.0f, 0.0f };            

// light intensity
float4 I_a < string UIName = "Ambient Intensity";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The light intensity of the ambient lights";     
        > = { 0.5f, 0.5f, 0.5f, 1.0f };    // ambient
float4 I_b < string UIName = "Ground color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The color of the ground";     
        > = { 0.1f, 0.0f, 0.0f, 1.0f };    // ground
float4 I_c < string UIName = "Sky Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The color of the sky";     
        > = { 0.9f, 0.9f, 1.0f, 1.0f };    // sky
float4 I_d < string UIName = "Diffuse Intensity";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The light intensity of the diffuse lights";     
        > = { 1.0f, 0.9f, 0.8f, 1.0f };    // diffuse
float4 I_s < string UIName = "Specular Intensity";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "The light intensity of the specular lights";     
        > = { 1.0f, 1.0f, 1.0f, 1.0f };    // specular

// material reflectivity
float4 k_a < string UIName = "Ambient Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "Material Ambient Color";     
        > = { 0.8f, 0.8f, 0.8f, 1.0f };    // ambient
float4 k_d < string UIName = "Diffuse Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "Material Diffuse Color";     
        > = { 0.4f, 0.4f, 0.4f, 1.0f };    // diffuse
float4 k_s < string UIName = "Specular Color";
             string UIWidget = "Color";
             float UIStep = 0.05;
             string UIHelp = "Material Specular Color";     
        > = { 0.1f, 0.1f, 0.1f, 1.0f };    // specular
float  n  < string UIName = "Specular Power";
             string UIWidget = "Numeric";
             float UIStep = 1;
             string UIHelp = "Specular Power";     
        > = 32.0f;                         // power


// transformations
float4x4 WorldView  : WORLDVIEW;
float4x4 Projection : PROJECTION;

struct VS_OUTPUT
{
    float4 Pos  : POSITION;
    float4 Diff : COLOR0;
    float4 Spec : COLOR1;
};

VS_OUTPUT VS(
    float3 Pos  : POSITION, 
    float3 Norm : NORMAL, 
    float  Occ  : TEXCOORD0,
    uniform bool bHemi, 
    uniform bool bDiff,
    uniform bool bSpec)
{
    VS_OUTPUT Out = (VS_OUTPUT)0;

    float3 L = -DirFromLight;                               // diffuse direction
    float3 Y = -DirFromSky;                                 // hemisphere up axis
    float3 P = mul(float4(Pos, 1), (float4x3)WorldView);    // position (view space)
    float3 N = normalize(mul(Norm, (float3x3)WorldView));   // normal (view space)
    float3 R = normalize(2 * dot(N, L) * N - L);            // reflection vector (view space)
    float3 V = -normalize(P);                               // view direction (view space)

    float4 Amb  = k_a * I_a;
    float4 Hemi = k_a * lerp(I_b, I_c, (dot(N, Y) + 1) / 2) * (1 - Occ);
    float  temp = 1 - max(0, dot(N, L));
    float4 Diff = k_d * I_d * (1 - temp * temp);
    float4 Spec = k_s * I_s * pow(max(0, dot(R, V)), n/4);
    float4 Zero = 0;

    Out.Pos  = mul(float4(P, 1), Projection);               // position (projected)
    Out.Diff = (bDiff ? Diff : 0)
             + (bHemi ? Hemi : Amb);                        // diffuse + ambient/hemisphere
    Out.Spec = (bSpec ? Spec : 0);                          // specular

    return Out;
}

technique THemisphereDiffuseSpecular
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS(true, true, true);

        SpecularEnable = TRUE;
    }
}

technique THemisphereDiffuse
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS(true, true, false);
    }
}


technique THemisphere
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS(true, false, false);
    }
}


technique TAmbient
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS(false, false, false);
    }
}

technique TAmbientDiffuse
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS(false, true, false);
    }
}

technique TAmbientDiffuseSpecular
{
    pass P0
    {
        VertexShader = compile vs_1_1 VS(false, true, true);

        SpecularEnable = TRUE;
    }
}

