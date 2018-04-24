#include "master.h"

struct a2vConnector {
  uint index : TEX2;
  uint nVertexID : SV_VERTEXID;
};

struct v2fConnector {
  float4 POSITION             : POSITION;
  float4 wsCoordAmbo          : TEXCOORD;
  float3 wsNormal             : TEXCOORD1;
};

Buffer<float4> vb_worldCoordAmbo;
Buffer<float4> vb_worldNorm;

cbuffer ShaderCB {
  float4x4 view;
  float4x4 viewProj;
  float4 g_screenSize = float4(1024,768,1.0/1024.0,1.0/768.0);
  float4 g_worldEyePosIn = float4(0,0,0,0);
  float4 time;
  float  zbias;
}

v2fConnector main(a2vConnector a2v)
{
  float4 worldCoordAmbo = vb_worldCoordAmbo.Load(a2v.index);
  float3 worldNorm      = vb_worldNorm.Load(a2v.index).xyz;

  float3 worldCoord = worldCoordAmbo.xyz;
  float3 worldCoordForProj = worldCoord;
    
    // apply z bias to prioritize drawing of higher LODs:
    // careful - this method can cause divide-by-zero for close polys.
    float3 wsVecToPnt = normalize(worldCoord - g_worldEyePosIn.xyz);
    worldCoordForProj += wsVecToPnt * zbias * 3;

    // LOD   zbias
    //  lo     2
    //  med    1
    //  hi     0
    
  float4 projCoord = mul(viewProj, float4(worldCoordForProj,1));
    
    // apply z bias to prioritize drawing of higher LODs:
    // (but only for verts that are facing us)
    //float3 wsViewDir = normalize(view._13_23_33);
    //projCoord.z += zbias * -0.02;//* (3-2*abs(dot(worldNorm, wsViewDir)));
  
  v2fConnector v2f;
  v2f.POSITION    = projCoord;
  v2f.wsCoordAmbo = worldCoordAmbo;
  v2f.wsNormal    = worldNorm;
  
  return v2f;
}

