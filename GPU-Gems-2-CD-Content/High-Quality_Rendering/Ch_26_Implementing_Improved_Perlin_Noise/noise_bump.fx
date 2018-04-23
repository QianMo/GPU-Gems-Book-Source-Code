/*********************************************************************NVMH3****
File:  $Id: //sw/devrel/SDK/MEDIA/HLSL/noise_bumpfx $

Copyright NVIDIA Corporation 2004
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Comments:
	Procedural bump mapping using Perlin noise.
	Based on BumpPlastic.fx

******************************************************************************/

#include "inoise.fxh"

float Script : STANDARDSGLOBAL <
    string UIWidget = "none";
    string ScriptClass = "object";
    string ScriptOrder = "standard";
    string ScriptOutput = "color";
    string Script = "Technique=dx9textured;";
> = 0.8;

/************* TWEAKABLES **************/

float4x4 WorldIT : WorldInverseTranspose < string UIWidget="None"; >;
float4x4 WorldViewProj : WorldViewProjection < string UIWidget="None"; >;
float4x4 World : World < string UIWidget="None"; >;
float4x4 ViewInv : ViewInverse < string UIWidget="None"; >;

float4 LightPos : Position
<
    string Object = "PointLight";
    string Space = "World";
> = {100.0f, 100.0f, -100.0f, 0.0f};

float4 LightColor
<
	string UIWidget = "Color";
> = {1.0f, 1.0f, 1.0f, 1.0f};

float4 AmbiColor : Ambient
<
    string UIName =  "Ambient Light Color";
> = {0.07f, 0.07f, 0.07f, 1.0f};

float4 SurfColor : DIFFUSE
<
    string UIName =  "Surface Color";
    string UIWidget = "Color";
> = {1.0f, 1.0f, 1.0f, 1.0f};

float SpecExpon : SpecularPower
<
    string UIWidget = "slider";
    float UIMin = 1.0;
    float UIMax = 128.0;
    float UIStep = 1.0;
    string UIName =  "specular power";
> = 12.0;

float Bumpy
<
    string UIWidget = "slider";
    float UIMin = 0.0;
    float UIMax = 10.0;
    float UIStep = 0.01;
    string UIName =  "bump scale";
> = 0.5;

float NoiseScale
<
    string UIWidget = "slider";
    float UIMin = 0.0;
    float UIMax = 100.0;
    float UIStep = 0.1;
    string UIName =  "noise scale";
> = 20.0;

//////////

texture colorTexture : DIFFUSE
<
    string ResourceName = "default_color.dds";
    string ResourceType = "2D";
>;

texture normalTexture : NORMAL
<
    string ResourceName = "default_bump_normal.dds";
    string ResourceType = "2D";
>;

sampler2D colorSampler = sampler_state
{
	Texture = <colorTexture>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = Linear;
};

sampler2D normalSampler = sampler_state
{
	Texture = <normalTexture>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = Linear;
};

/************* DATA STRUCTS **************/

/* data from application vertex buffer */
struct appdata {
    float3 Position	: POSITION;
    float4 UV		: TEXCOORD0;
    float4 Normal	: NORMAL;
    float4 Tangent	: TANGENT0;
    float4 Binormal	: BINORMAL0;
};

/* data passed from vertex shader to pixel shader */
struct vertexOutput {
    float4 HPosition	: POSITION;
    float4 TexCoord	: TEXCOORD0;
    float3 LightVec	: TEXCOORD1;
    float3 WorldNormal	: TEXCOORD2;
    float3 WorldEyeVec	: TEXCOORD3;
    float3 WorldTangent	: TEXCOORD4;
    float3 WorldBinorm	: TEXCOORD5;
    float3 WorldPos : TEXCOORD6;
};

/*********** vertex shader ******/

vertexOutput mainVS(appdata IN)
{
    vertexOutput OUT;
    OUT.WorldNormal = mul(IN.Normal, WorldIT).xyz;
    OUT.WorldTangent = mul(IN.Tangent, WorldIT).xyz;
    OUT.WorldBinorm = mul(IN.Binormal, WorldIT).xyz;
    float4 tempPos = float4(IN.Position.x,IN.Position.y,IN.Position.z,1.0);
    float3 worldSpacePos = mul(tempPos, World).xyz;
    OUT.WorldPos = worldSpacePos * NoiseScale;
    OUT.LightVec = LightPos - worldSpacePos;
    OUT.TexCoord = IN.UV;
    OUT.WorldEyeVec = normalize(ViewInv[3].xyz - worldSpacePos);
    OUT.HPosition = mul(tempPos, WorldViewProj);
    return OUT;
}

/********* pixel shader ********/

float4 mainPS(vertexOutput IN) : COLOR
{
    float4 map = tex2D(colorSampler,IN.TexCoord.xy);
    float3 Ln = normalize(IN.LightVec);
    float3 Nn = normalize(IN.WorldNormal);
    float3 Tn = normalize(IN.WorldTangent);
    float3 Bn = normalize(IN.WorldBinorm);
    float3 dF = inoiseGradient(IN.WorldPos, 0.1);
	float3 Nb = normalize(Nn + dF*Bumpy);
    Nb = normalize(Nb);
    float3 Vn = normalize(IN.WorldEyeVec);
    float3 Hn = normalize(Vn + Ln);
    float4 lighting = lit(dot(Ln,Nb),dot(Hn,Nb),SpecExpon);
    float hdn = lighting.z;
    float ldn = lighting.y;
    float diffComp = ldn;
    float4 diffContrib = SurfColor * map * (diffComp*LightColor + AmbiColor);
    float4 specContrib = hdn * LightColor;
    float4 result = AmbiColor + diffContrib + specContrib;
    return result;
}

/*************/

technique dx9textured <
	string Script = "Pass=p0;";
> {
	pass p0  <
		string Script = "Draw=geometry;";
	> {		
		// same vertex shader for both DX9 techniques
		VertexShader = compile vs_2_0 mainVS();
		ZEnable = true;
		ZWriteEnable = true;
		CullMode = None;
		PixelShader = compile ps_2_a mainPS();
	}
}

/***************************** eof ***/
