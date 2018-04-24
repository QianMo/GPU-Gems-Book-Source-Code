float4x4 g_mViewProj;
float4x4 g_mWorld;

void VS_RenderShadowMap(
  in float4 vPos : POSITION,
  out float4 vPosOut : POSITION,
  out float4 vPixelOut : TEXCOORD0)
{
  // transform vertex
  vPosOut = mul(vPos, g_mWorld);
  vPosOut = mul(vPosOut, g_mViewProj);
  // output position to pixel shader
  vPixelOut = vPosOut;
}

float4 PS_RenderShadowMap(float4 vPixelPos : TEXCOORD0): COLOR
{
  // write depth to texture
  return vPixelPos.z / vPixelPos.w;
}


// This technique is used when rendering meshes to the shadowmap
// 
technique RenderShadowMap
{
  pass p0
  {
    CullMode = CW;
    VertexShader = compile vs_2_0 VS_RenderShadowMap();
    PixelShader = compile ps_2_0 PS_RenderShadowMap();
  }
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

float3 g_vLightDir;
float3 g_vLightColor;
float3 g_vAmbient;
float g_fShadowMapSize;
float g_fShadowMapTexelSize;
static const int g_iNumSplits = 4;
float4x4 g_mTextureMatrix[g_iNumSplits];
float4x4 g_mView;
float g_fSplitPlane[g_iNumSplits];

sampler2D g_samShadowMap[g_iNumSplits];

//////////////////////////////////////////////////////////////////
// Helper functions:

float SamplePCF(sampler2D sam, float4 vTexCoord)
{
  // project texture coordinates
  vTexCoord.xyz /= vTexCoord.w;

  // 2x2 PCF Filtering
  // 
  float fShadow[4];
  fShadow[0] = (vTexCoord.z < tex2D(sam, vTexCoord + float2(                    0,                     0)));
  fShadow[1] = (vTexCoord.z < tex2D(sam, vTexCoord + float2(g_fShadowMapTexelSize,                     0)));
  fShadow[2] = (vTexCoord.z < tex2D(sam, vTexCoord + float2(                    0, g_fShadowMapTexelSize)));
  fShadow[3] = (vTexCoord.z < tex2D(sam, vTexCoord + float2(g_fShadowMapTexelSize, g_fShadowMapTexelSize)));

  float2 vLerpFactor = frac(g_fShadowMapSize * vTexCoord.xy);

  return lerp( lerp(fShadow[0], fShadow[1], vLerpFactor.x),
               lerp(fShadow[2], fShadow[3], vLerpFactor.x),
                    vLerpFactor.y);
}

float3 GouradShading(float3 vNormal)
{
  return g_vLightColor * saturate(dot(-g_vLightDir, normalize(mul(vNormal, (float3x3)g_mWorld))));
}

//////////////////////////////////////////////////////////////////
//
// DX9-level

void VS_RenderShadows(
  in float4 vPos : POSITION,
  in float3 vNormal : NORMAL,
  in float3 vColorIn : COLOR0,
  out float4 vPosOut : POSITION,
  out float4 vTexCoord[g_iNumSplits+1] : TEXCOORD,
  out float3 vLighting : COLOR0,
  out float3 vColorOut : COLOR1)
{
  // calculate world position
  float4 vPosWorld = mul(vPos, g_mWorld);
  // transform vertex
  vPosOut = mul(vPosWorld, g_mViewProj);
  
  // store view space position
  vTexCoord[0] = mul(vPosWorld, g_mView);

  // coordinates for shadow maps
  [unroll] for(int i=0;i<g_iNumSplits;i++)
  {
    vTexCoord[i+1] = mul(vPosWorld, g_mTextureMatrix[i]);
  }

  // calculate per vertex lighting
  vLighting = GouradShading(vNormal);
  vColorOut = vColorIn;
}

float4 PS_RenderShadows(
  float4 vTexCoord[g_iNumSplits+1] : TEXCOORD,
  float3 vLighting : COLOR0,
  float3 vColorIn : COLOR1) : COLOR
{
  float fLightingFactor = 1;
  float fDistance = vTexCoord[0].z;

  for(int i=0; i < g_iNumSplits; i++)
  {
    if(fDistance < g_fSplitPlane[i])
    {
      fLightingFactor = SamplePCF(g_samShadowMap[i], vTexCoord[i+1]);
      break;
    }
  }

  // final color
  float4 vColor=1;
  vColor.rgb = vColorIn * saturate(g_vAmbient.xyz + vLighting.xyz * fLightingFactor);
  return vColor;
}

// This technique is used to render the final shadowed meshes
//
technique RenderShadows
{
  pass p0
  {
	  VertexShader = compile vs_3_0 VS_RenderShadows();
	  PixelShader = compile ps_3_0 PS_RenderShadows();
  }
}


//////////////////////////////////////////////////////////////////
//
// Multi-pass

void VS_RenderShadows_MP(
  in float4 vPos : POSITION,
  in float3 vNormal : NORMAL,
  in float3 vColorIn : COLOR0,
  out float4 vPosOut : POSITION,
  out float4 vTexCoord : TEXCOORD,
  out float3 vLighting : COLOR0,
  out float3 vColorOut : COLOR1)
{
  // calculate world position
  float4 vPosWorld = mul(vPos, g_mWorld);
  // transform vertex
  vPosOut = mul(vPosWorld, g_mViewProj);

  // coordinates for shadow map
  vTexCoord = mul(vPosWorld, g_mTextureMatrix[0]);

  // calculate per vertex lighting
  vLighting = GouradShading(vNormal);
  vColorOut = vColorIn;
}

float4 PS_RenderShadows_MP(
  float4 vTexCoord : TEXCOORD0,
  float3 vLighting : COLOR0,
  float3 vColorIn : COLOR1) : COLOR
{
  float fLightingFactor = SamplePCF(g_samShadowMap[0], vTexCoord);

  // final color
  float4 vColor=1;
  vColor.rgb = vColorIn * saturate(g_vAmbient.xyz + vLighting.xyz * fLightingFactor);
  return vColor;
}


float4 PS_RenderShadows_MP2(
  float4 vTexCoord : TEXCOORD0,
  float3 vLighting : COLOR0,
  float3 vColorIn : COLOR1) : COLOR
{
  return SamplePCF(g_samShadowMap[0], vTexCoord);
}


// This technique is used to render the final shadowed meshes
//
technique RenderShadows_MP
{
  pass p0
  {
    CullMode = CCW;
	  VertexShader = compile vs_2_0 VS_RenderShadows_MP();
	  PixelShader = compile ps_2_0 PS_RenderShadows_MP();
  }
}