static const int g_iNumSplits = 4;

int g_iFirstSplit;
int g_iLastSplit;
matrix g_mWorld;
matrix g_mViewProj;
matrix g_mCropMatrix[g_iNumSplits];

RasterizerState ReverseCulling
{
  CullMode = FRONT;
};

struct VS_INPUT_SHADOWMAP
{
  float4 vPos : POSITION;
};

struct VS_OUTPUT_SHADOWMAP
{
  float4 vPos : SV_POSITION;
};

struct GS_OUTPUT_SHADOWMAP
{
  float4 vPos : SV_POSITION;
  uint RTIndex : SV_RenderTargetArrayIndex;
};

/////////////////////////////////////////////////////////////////////////////////
//
// GS Cloning method

VS_OUTPUT_SHADOWMAP VS_RenderShadowMap( VS_INPUT_SHADOWMAP i )
{
  VS_OUTPUT_SHADOWMAP o = (VS_OUTPUT_SHADOWMAP)0;
  o.vPos = mul(i.vPos, g_mWorld);
  o.vPos = mul(o.vPos, g_mViewProj);
  return o;
}

[maxvertexcount(12)]
void GS_RenderShadowMap_GSC( triangle VS_OUTPUT_SHADOWMAP In[3], inout TriangleStream<GS_OUTPUT_SHADOWMAP> TriStream )
{
  // for each split
  for(int iSplit = g_iFirstSplit; iSplit <= g_iLastSplit; iSplit++)
  {
    GS_OUTPUT_SHADOWMAP Out;
    Out.RTIndex = iSplit;
    // for each vertex
    [unroll] for(int iVertex = 0; iVertex < 3; iVertex++)
    {
      // transform with split specific projection matrix
      Out.vPos = mul( In[iVertex].vPos, g_mCropMatrix[iSplit] );
      // append vertex to stream
      TriStream.Append(Out);
    }
    // mark end of triangle
    TriStream.RestartStrip();
  }
}

technique10 RenderShadowMap_GSC
{
  pass P0
  {
    SetRasterizerState(ReverseCulling);
    SetVertexShader(CompileShader(vs_4_0, VS_RenderShadowMap()));
    SetGeometryShader(CompileShader(gs_4_0, GS_RenderShadowMap_GSC()));
    SetPixelShader(NULL);
  }
}

// Standard shadow map rendering
technique10 RenderShadowMap_Standard
{
  pass P0
  {
    SetRasterizerState(ReverseCulling);
    SetVertexShader(CompileShader(vs_4_0, VS_RenderShadowMap()));
    SetGeometryShader(NULL);
    SetPixelShader(NULL);
  }
}

/////////////////////////////////////////////////////////////////////////////////
//
// Instacing method

struct VS_INPUT_SHADOWMAP_INSTANCING
{
  float4 vPos : POSITION;
  uint iInstance : SV_InstanceID;
};

struct VS_OUTPUT_SHADOWMAP_INSTANCING
{
  float4 vPos : POSITION;
  uint iSplit : TEXTURE0;
};

VS_OUTPUT_SHADOWMAP_INSTANCING VS_RenderShadowMap_Inst( VS_INPUT_SHADOWMAP_INSTANCING i )
{
  VS_OUTPUT_SHADOWMAP_INSTANCING o = (VS_OUTPUT_SHADOWMAP_INSTANCING)0;
  o.vPos = mul(i.vPos, g_mWorld);
  o.vPos = mul(o.vPos, g_mViewProj);
  // determine split index from instance ID
  o.iSplit = g_iFirstSplit + i.iInstance;
  // transform with split specific projection matrix
  o.vPos = mul( o.vPos, g_mCropMatrix[o.iSplit] );
  return o;
}

[maxvertexcount(3)]
void GS_RenderShadowMap_Inst( triangle VS_OUTPUT_SHADOWMAP_INSTANCING In[3], inout TriangleStream<GS_OUTPUT_SHADOWMAP> TriStream )
{
  GS_OUTPUT_SHADOWMAP Out;
  // set render target index
  Out.RTIndex = In[0].iSplit;
  // pass vertices through
  Out.vPos = In[0].vPos;
  TriStream.Append(Out);
  Out.vPos = In[1].vPos;
  TriStream.Append(Out);
  Out.vPos = In[2].vPos;
  TriStream.Append(Out);
  TriStream.RestartStrip();
}

technique10 RenderShadowMap_Inst
{
  pass P0
  {
    SetRasterizerState(ReverseCulling);
    SetVertexShader(CompileShader(vs_4_0, VS_RenderShadowMap_Inst()));
    SetGeometryShader(CompileShader(gs_4_0, GS_RenderShadowMap_Inst()));
    SetPixelShader(NULL);
  }
}

/////////////////////////////////////////////////////////////////////////////////
//
// Rendering shadows

float3 g_vLightDir;
float3 g_vLightColor;
float3 g_vAmbient;
TextureCube g_txShadowMapArray;
matrix g_mView;
matrix g_mTextureMatrix[g_iNumSplits];
float g_fSplitPlane[g_iNumSplits];

float3 GouradShading(float3 vNormal)
{
  return g_vLightColor * saturate(dot(-g_vLightDir, normalize(mul(vNormal, (float3x3)g_mWorld))));
}

// for use with SampleCmpLevelZero
SamplerComparisonState g_samShadowMapArray
{
  ComparisonFunc = Less;
  Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
};

RasterizerState DefaultCulling
{
  CullMode = BACK;
};

struct VS_INPUT_FINAL
{
  float4 vPos : POSITION;
  float3 vNormal : NORMAL;
  float3 vColor : COLOR0;
};

struct PS_INPUT_FINAL
{
  float4 vPos : SV_POSITION;
  float4 vTexCoord[g_iNumSplits + 1] : TEXCOORD;
  float3 vLighting : COLOR0;
  float3 vColor : COLOR1;
};

PS_INPUT_FINAL VS_RenderShadows(VS_INPUT_FINAL In)
{
  PS_INPUT_FINAL o = (PS_INPUT_FINAL)0;
  // calculate world position
  float4 vPosWorld = mul(In.vPos, g_mWorld);
  // transform vertex
  o.vPos = mul(vPosWorld, g_mViewProj);
  
  // store view space position
  o.vTexCoord[0] = mul(vPosWorld, g_mView);

  // coordinates for shadow maps
  [unroll] for(int i=0;i<g_iNumSplits;i++)
  {
    o.vTexCoord[i+1] = mul(vPosWorld, g_mTextureMatrix[i]);
  }

  // calculate per vertex lighting
  o.vLighting = GouradShading(In.vNormal);
  o.vColor = In.vColor;
  return o;
}

static const float3 _vConstantOffset[6] = {
  float3(0.5, 0.5, 0.5),
  float3(-0.5, 0.5, -0.5),
  float3(-0.5, 0.5, -0.5),
  float3(-0.5, -0.5, 0.5),
  float3(-0.5, 0.5, 0.5),
  float3(0.5, 0.5, -0.5)
};

static const float3 _vPosXMultiplier[6] = {
  float3(0, 0, -1),
  float3(0, 0, 1),
  float3(1, 0, 0),
  float3(1, 0, 0),
  float3(1, 0, 0),
  float3(-1, 0, 0)
};

static const float3 _vPosYMultiplier[6] = {
  float3(0, -1, 0),
  float3(0, -1, 0),
  float3(0, 0, 1),
  float3(0, 0, -1),
  float3(0, -1, 0),
  float3(0, -1, 0)
};

float4 PS_RenderShadows(PS_INPUT_FINAL i) : SV_Target
{
  float fLightingFactor = 1;
  float fDistance = i.vTexCoord[0].z;

  for(int iSplit = 0; iSplit < g_iNumSplits; iSplit++)
  {      
    if(fDistance < g_fSplitPlane[iSplit])
    {
      float4 pos = i.vTexCoord[iSplit + 1];
      pos.xyz /= pos.w;

      float3 vCubeCoords;
      vCubeCoords = _vConstantOffset[iSplit] + _vPosXMultiplier[iSplit] * pos.x + _vPosYMultiplier[iSplit] * pos.y;

      // border clamping not possible with TextureCubes
      // so we must simply avoid sampling outside the borders
      //if(pos.x > 0 && pos.y > 0 && pos.x < 1 && pos.y < 1)
      if(min(pos.x, pos.y) > 0 && max(pos.x, pos.y) < 1)
        fLightingFactor = g_txShadowMapArray.SampleCmpLevelZero(g_samShadowMapArray, vCubeCoords, pos.z);
      break;
    }
  }
  float4 vColor=1;
  vColor.rgb = i.vColor * saturate(g_vAmbient.xyz + i.vLighting.xyz*fLightingFactor);
  return vColor;
}

technique10 RenderShadows
{
  pass P0
  {
    SetRasterizerState(DefaultCulling);
    SetVertexShader(CompileShader(vs_4_0, VS_RenderShadows()));
    SetGeometryShader(NULL);
    SetPixelShader(CompileShader(ps_4_0, PS_RenderShadows()));
  }
}

//////////////////////////////////////////////////////////////////
//
// DX9-level

SamplerComparisonState g_samShadowMap
{
  ComparisonFunc = Less;
  Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
  AddressU = Border;
  AddressV = Border;
  BorderColor = float4(1,1,1,1);
};

Texture2D<float> g_txShadowMap[g_iNumSplits];

float4 PS_RenderShadows_DX9(PS_INPUT_FINAL i) : SV_Target
{
  float fLightingFactor = 1;
  float fDistance = i.vTexCoord[0].z;

  [unroll] for(int iSplit = 0; iSplit < g_iNumSplits; iSplit++)
  {      
    if(fDistance < g_fSplitPlane[iSplit])
    {
      float4 pos = i.vTexCoord[iSplit + 1];
      fLightingFactor = g_txShadowMap[iSplit].SampleCmpLevelZero(g_samShadowMap, pos.xy/pos.w, pos.z/pos.w).x;
      break;
    }
  }

  // final color
  float4 vColor=1;
  vColor.rgb = i.vColor * saturate(g_vAmbient.xyz + i.vLighting.xyz*fLightingFactor);
  return vColor;
}

technique10 RenderShadows_DX9
{
  pass P0
  {
    SetRasterizerState(DefaultCulling);
    SetVertexShader(CompileShader(vs_4_0, VS_RenderShadows()));
    SetGeometryShader(NULL);
    SetPixelShader(CompileShader(ps_4_0, PS_RenderShadows_DX9()));
  }
}

//////////////////////////////////////////////////////////////////
//
// Multi-pass


float4 PS_RenderShadows_MP(PS_INPUT_FINAL i) : SV_Target
{
  float4 pos = i.vTexCoord[1];
  float fLightingFactor = g_txShadowMap[0].SampleCmpLevelZero(g_samShadowMap, pos.xy/pos.w, pos.z/pos.w);

  // final color
  float4 vColor=1;
  vColor.rgb = i.vColor * saturate(g_vAmbient.xyz + i.vLighting.xyz*fLightingFactor);
  return vColor;
}

technique10 RenderShadows_MP
{
  pass P0
  {
    SetRasterizerState(DefaultCulling);
    SetVertexShader(CompileShader(vs_4_0, VS_RenderShadows()));
    SetGeometryShader(NULL);
    SetPixelShader(CompileShader(ps_4_0, PS_RenderShadows_MP()));
  }
}