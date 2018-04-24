//input: vert_list_geom

#include "LodCB.h"

struct a2vConnector {
  uint z8_y8_x8_null4_edgeNum4 : TEX2;
  uint nVertexID               : SV_VERTEXID;
};

struct v2gConnector {
  float4 projCoord          : POSITION;
  uint2  vertexID_and_slice : TEX2;
};

v2gConnector main(a2vConnector a2v)
{
  uint  edgeNum = a2v.z8_y8_x8_null4_edgeNum4 & 0x0F;
  int3 xyz = (int3)((a2v.z8_y8_x8_null4_edgeNum4.xxx >> uint3(8,16,24)) & 0xFF);

    // note: every vertex coming in here is on edge 3, 0, or 8.
    //       (lower-left edges of the cells...)
    xyz.x *= 3;
    if (edgeNum==3)
      xyz.x += 0;
    if (edgeNum==0)
      xyz.x += 1;
    if (edgeNum==8)
      xyz.x += 2;
    
  float2 uv = (float2)xyz.xy;
    // ALIGNMENT FIX: (for nearest-neighbor sampling)
    uv.x += 0.5*InvVoxelDim.x/3.0;
    uv.y += 0.5*InvVoxelDim.x/1.0;
  
  v2gConnector v2g;
  v2g.projCoord.x  = (uv.x*InvVoxelDim.x/3.0)*2 - 1;  //-1..1 range
  v2g.projCoord.y  = (uv.y*InvVoxelDim.x    )*2 - 1;  //-1..1 range
    // fix upside-down projection:
    v2g.projCoord.y *= -1;
  v2g.projCoord.z  = 0;
  v2g.projCoord.w  = 1;
  v2g.vertexID_and_slice.x = a2v.nVertexID;
  v2g.vertexID_and_slice.y = xyz.z;
    
  return v2g;
}