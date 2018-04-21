// This is the internal shader produced
// by C4Dfx for three light sources with shadows.
// Normally it is not written to a file.

float4x4 wvp;
float4x4 wit;
float4x4 w;
float4x4 vit;
float4 diffCol;
float4 lumiCol;
texture diffuseTexture;
float bumpHeight;
texture normalTexture;
float4 enviCol;
texture enviTexture;
texture specShapeTexture;
float4 specCol;
float4 lightPos0;
float4 lightCol0;
float4 lightParams0;
float4 lightUp0;
float4 lightDir0;
float4 lightSide0;
float4 lightPos1;
float4 lightCol1;
float4 lightParams1;
float4 lightUp1;
float4 lightDir1;
float4 lightSide1;
float4 lightPos2;
float4 lightCol2;
float4 lightParams2;
float4 lightUp2;
float4 lightDir2;
float4 lightSide2;
texture depthTexture0;
texture depthTexture1;
texture depthTexture2;
sampler2D diffuseSampler = sampler_state
{
	Texture = <diffuseTexture>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = Linear;
};
sampler1D specShapeSampler = sampler_state
{
	Texture = <specShapeTexture>;
 	AddressU = Clamp;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = None;
};
sampler2D normalSampler = sampler_state
{
	Texture = <normalTexture>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = Linear;
};
samplerCUBE envMapSampler = sampler_state
{
	Texture = <enviTexture>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = Linear;
};
sampler2D depthSampler0 = sampler_state
{
	Texture = <depthTexture0>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = None;
	AddressU = Border;
	AddressV = Border;
	BorderColor = {0.0f, 0.0f, 0.0f, 1.0f};
};
sampler2D depthSampler1 = sampler_state
{
	Texture = <depthTexture1>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = None;
	AddressU = Border;
	AddressV = Border;
	BorderColor = {0.0f, 0.0f, 0.0f, 1.0f};
};
sampler2D depthSampler2 = sampler_state
{
	Texture = <depthTexture2>;
	MinFilter = Linear;
	MagFilter = Linear;
	MipFilter = None;
	AddressU = Border;
	AddressV = Border;
	BorderColor = {0.0f, 0.0f, 0.0f, 1.0f};
}; 
struct appdata
{
	float4 position : POSITION;
	float4 norm : NORMAL;
	float4 uv: TEXCOORD0;
	float4 tang : TEXCOORD1;
	float4 binorm : TEXCOORD2;
};
struct vertexOutput
{
	float4 hPos : POSITION;
	float4 uv : TEXCOORD0;
	float3 norm : TEXCOORD1;
	float3 tang : TEXCOORD2;
	float3 binorm : TEXCOORD3;
	float3 view : TEXCOORD4;
	float3 wPos : TEXCOORD5;
};
struct pixelOutput
{
	float3 col : COLOR;
};
vertexOutput mainVS(
	appdata IN,
	uniform float4x4 wvp,
	uniform float4x4 wit,
	uniform float4x4 w,
	uniform float4x4 vit)
{
	vertexOutput OUT;
	OUT.uv = IN.uv;
	OUT.hPos = mul(wvp, IN.position);
	OUT.norm = mul(wit, IN.norm).xyz;
	OUT.tang = mul(w, IN.tang).xyz;
	OUT.binorm = mul(w, IN.binorm).xyz;
	float3 pW = mul(w, IN.position).xyz;
	OUT.wPos = pW;
	OUT.view = normalize(pW - vit[3].xyz);
	return OUT;
}
pixelOutput mainPS(
	vertexOutput IN,
	uniform sampler2D diffuseSampler,
	uniform sampler1D specShapeSampler,
	uniform sampler2D normalSampler,
	uniform samplerCUBE enviSampler,
	uniform sampler2D depthSampler0,
	uniform sampler2D depthSampler1,
	uniform sampler2D depthSampler2,
	uniform float4 lumiCol,
	uniform float4 diffCol,
	uniform float bumpHeight,
	uniform float4 enviCol,
	uniform float4 specCol,
	uniform float4 lightPos0,
	uniform float4 lightCol0,
	uniform float4 lightParams0,
	uniform float4 lightUp0,
	uniform float4 lightDir0,
	uniform float4 lightSide0,
	uniform float4 lightPos1,
	uniform float4 lightCol1,
	uniform float4 lightParams1,
	uniform float4 lightUp1,
	uniform float4 lightDir1,
	uniform float4 lightSide1,
	uniform float4 lightPos2,
	uniform float4 lightCol2,
	uniform float4 lightParams2,
	uniform float4 lightUp2,
	uniform float4 lightDir2,
	uniform float4 lightSide2)
{
	pixelOutput OUT;
 	float3 Vn = normalize(IN.view);
	float3 Nn = normalize(IN.norm);
	float3 tangn = normalize(IN.tang);
	float3 binormn = normalize(IN.binorm);
	float2 bumps = bumpHeight*(tex2D(normalSampler, IN.uv.xy).xy * 2.0 - float2(1.0, 1.0));
	float3 Nb = normalize(bumps.x*tangn + bumps.y*binormn + Nn);
	float3 env = texCUBE(enviSampler, reflect(Vn, Nb)).rgb;
	float3 colorSum = lumiCol.rgb + env*enviCol.rgb;
	float3 baseDiffCol = diffCol.rgb + tex2D(diffuseSampler, IN.uv.xy).rgb; 
	{
		float3 Ld = lightPos0.xyz - IN.wPos;
		float3 Ln = normalize(Ld);
		float3 baseCol = max(0.0, dot(Ln, Nb))*baseDiffCol;
		float spec = tex1D(specShapeSampler, dot(Vn, reflect(Ln, Nb))).r;
		baseCol += specCol.rgb*spec;
		float3 L1 = (Ln/dot(Ln, lightDir0.xyz) - lightDir0.xyz)*lightParams0.z;
		float shadowFactor = max(lightParams0.x, smoothstep(1.0, lightParams0.w, length(L1)));
		float d = dot(Ld, lightDir0.xyz);
		float z = 10.1010101/d + 1.01010101;
		float2 depthUV = float2(0.5, 0.5) + 0.5*float2(dot(L1, lightSide0.xyz), dot(L1, lightUp0.xyz));
		shadowFactor *= max(lightParams0.y, tex2Dproj(depthSampler0, float4(depthUV.x, depthUV.y, z-0.00009, 1.0)).x);
		colorSum += shadowFactor*baseCol*lightCol0.rgb;
	} 
	{
		float3 Ld = lightPos1.xyz - IN.wPos;
		float3 Ln = normalize(Ld);
		float3 baseCol = max(0.0, dot(Ln, Nb))*baseDiffCol;
		float spec = tex1D(specShapeSampler, dot(Vn, reflect(Ln, Nb))).r;
		baseCol += specCol.rgb*spec;
		float3 L1 = (Ln/dot(Ln, lightDir1.xyz) - lightDir1.xyz)*lightParams1.z;
		float shadowFactor = max(lightParams1.x, smoothstep(1.0, lightParams1.w, length(L1)));
		float d = dot(Ld, lightDir1.xyz);
		float z = 10.1010101/d + 1.01010101;
		float2 depthUV = float2(0.5, 0.5) + 0.5*float2(dot(L1, lightSide1.xyz), dot(L1, lightUp1.xyz));
		shadowFactor *= max(lightParams1.y, tex2Dproj(depthSampler1, float4(depthUV.x, depthUV.y, z-0.00009, 1.0)).x);
		colorSum += shadowFactor*baseCol*lightCol1.rgb;
	} 
	{
		float3 Ld = lightPos2.xyz - IN.wPos;
		float3 Ln = normalize(Ld);
		float3 baseCol = max(0.0, dot(Ln, Nb))*baseDiffCol;
		float spec = tex1D(specShapeSampler, dot(Vn, reflect(Ln, Nb))).r;
		baseCol += specCol.rgb*spec;
		float3 L1 = (Ln/dot(Ln, lightDir2.xyz) - lightDir2.xyz)*lightParams2.z;
		float shadowFactor = max(lightParams2.x, smoothstep(1.0, lightParams2.w, length(L1)));
		float d = dot(Ld, lightDir2.xyz);
		float z = 10.1010101/d + 1.01010101;
		float2 depthUV = float2(0.5, 0.5) + 0.5*float2(dot(L1, lightSide2.xyz), dot(L1, lightUp2.xyz));
		shadowFactor *= max(lightParams2.y, tex2Dproj(depthSampler2, float4(depthUV.x, depthUV.y, z-0.00009, 1.0)).x);
		colorSum += shadowFactor*baseCol*lightCol2.rgb;
	} 
	OUT.col = colorSum;
	return OUT;
}
technique t0
{
	pass p0
	{
		VertexShader = compile vs_2_x mainVS(wvp, wit, w, vit);
		ZEnable = true;
		ZWriteEnable = true;
		CullMode = None;
		PixelShader = compile ps_2_x mainPS(
			diffuseSampler, specShapeSampler, normalSampler, envMapSampler,
			depthSampler0,
			depthSampler1,
			depthSampler2,
			lumiCol, diffCol, bumpHeight, enviCol, specCol,
			lightPos0, lightCol0, lightParams0, lightUp0, lightDir0, lightSide0,
			lightPos1, lightCol1, lightParams1, lightUp1, lightDir1, lightSide1,
			lightPos2, lightCol2, lightParams2, lightUp2, lightDir2, lightSide2);
	}
} 