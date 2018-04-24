#include "master.h"

// input here is some # of instances of quadProxy_geo (a big quad to be rasterized)

struct a2vConnector {
  float3 objCoord    : POSITION;   // -1..1, -1..1, 0.5 [quadproxy_geo]
  float2 tex         : TEXCOORD;   // 0..1, 0..1 for the slice
  uint   nInstanceID : SV_InstanceID;   // each slice is an instance
};

struct v2gConnector {
  float4 projCoord    : POSITION;
  float4 wsCoord      : TEXCOORD;
  float3 chunkCoord   : TEXCOORD1;
  uint   nInstanceID  : BLAH;
};

cbuffer MyCB {
  float4 viewportDim = float4(8,6,1.0/8.0,1.0/6.0);
}

// 'ChunkCB' is updated each time we want to build a chunk:
cbuffer ChunkCB {
  float3 wsChunkPos = float3(0,0,0); //wsCoord of lower-left corner
  float  opacity = 1;
}

float3 rot(float3 coord, float4x4 mat)
{
  return float3( dot(mat._11_12_13, coord),   // 3x3 transform,
                 dot(mat._21_22_23, coord),   // no translation
                 dot(mat._31_32_33, coord) );
}

#include "LodCB.h"

v2gConnector main(a2vConnector a2v)
{
    v2gConnector v2g;
    float4 projCoord = float4(a2v.objCoord.xy, 0.5, 1);
      projCoord.y *= -1;               // flip Y coords for DX
    
    // chunkCoord is in [0..1] range
    float3 chunkCoord = float3( a2v.tex.xy,
                                a2v.nInstanceID * InvVoxelDimPlusMargins.x
                                ); 
      // HACK #1 - because in DX, when you render a quad with uv's in [0..1],
      //           the upper left corner will be a 0 and the lower right
      //           corner will be at 1 (it should be at 63/64, or whatever).
      chunkCoord.xyz *= VoxelDim.x*InvVoxelDimMinusOne.x;  
      //chunkCoord.xy *= VoxelDimPlusMargins.x*InvVoxelDimPlusMarginsMinusOne.x;  // HACK #1 

    // extChunkCoord goes outside that range, so we also compute
    // some voxels outside of the chunk
    float3 extChunkCoord = (chunkCoord*VoxelDimPlusMargins.x - Margin)*InvVoxelDim.x;
                                
    float3 ws = wsChunkPos + extChunkCoord*wsChunkSize;

    v2g.projCoord  = projCoord;
    v2g.wsCoord    = float4(ws,1);
    v2g.nInstanceID = a2v.nInstanceID;
    v2g.chunkCoord = chunkCoord;
             
    return v2g;
}
