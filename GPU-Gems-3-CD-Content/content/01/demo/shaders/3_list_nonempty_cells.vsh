//input: dummy_cells

#include "master.h"

// input here is ~63x63 points, one for each *cell* we'll run marching cubes on.
// note that if VoxelDim is 65   (65 corners/64 cells)
// and Margin is 4
// then the points we actually get in here are [4..67]

struct a2vConnector {
  float2 uv_write    : POSITION;   //0..1 range
  float2 uv_read     : POSITION2;   //less - where to read the source texels - factors in margins.
  uint   nInstanceID : SV_InstanceID;   // each slice is an instance
};

struct v2gConnector {
  uint z8_y8_x8_case8 : TEX2;
};

// 'ChunkCB' is updated each time we want to build a chunk:
cbuffer ChunkCB {
  float3 wsChunkPos = float3(0,0,0); //wsCoord of lower-left corner
  float  opacity = 1;
}

#include "LodCB.h"

Texture3D density_vol;
SamplerState LinearClamp;
SamplerState NearestClamp;

v2gConnector main(a2vConnector a2v)
{
  int inst = a2v.nInstanceID;
  
  float3 chunkCoordRead  = float3( a2v.uv_read.x,//(a2v.uv.x*VoxelDimMinusOne + Margin)*InvVoxelDimPlusMarginsMinusOne.x , 
                                   a2v.uv_read.y,//(a2v.uv.y*VoxelDimMinusOne + Margin)*InvVoxelDimPlusMarginsMinusOne.x ,
                                   (a2v.nInstanceID + Margin)*InvVoxelDimPlusMargins.x );
  float3 chunkCoordWrite = float3( a2v.uv_write.x, 
                                   a2v.uv_write.y,
                                   a2v.nInstanceID * InvVoxelDim.x );
                                  
  float3 wsCoord = wsChunkPos + chunkCoordWrite*wsChunkSize;   //FIXME!
    
  // very important: ws_to_uvw() should subtract 1/2 a texel in XYZ, 
  // to prevent 1-bit float error from snapping to wrong cell every so often!  
  float3 uvw = chunkCoordRead + InvVoxelDimPlusMarginsMinusOne.xxx*0.125;
    // HACK #2
    uvw.xy *= ((VoxelDimPlusMargins.x-1)*InvVoxelDimPlusMargins.x).xx;
  
  float4 field0123;
  float4 field4567;

  field0123.x = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.yyy, 0).x;
  field0123.y = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.yxy, 0).x;
  field0123.z = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.xxy, 0).x;
  field0123.w = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.xyy, 0).x;
  field4567.x = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.yyx, 0).x;
  field4567.y = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.yxx, 0).x;
  field4567.z = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.xxx, 0).x;
  field4567.w = density_vol.SampleLevel(NearestClamp, uvw + InvVoxelDimPlusMarginsMinusOne.xyx, 0).x;

  uint4 i0123 = (uint4)saturate(field0123*99999);
  uint4 i4567 = (uint4)saturate(field4567*99999);
  int cube_case = (i0123.x     ) | (i0123.y << 1) | (i0123.z << 2) | (i0123.w << 3) |
                  (i4567.x << 4) | (i4567.y << 5) | (i4567.z << 6) | (i4567.w << 7);

  v2gConnector v2f;
  uint3 uint3coord = uint3(a2v.uv_write.xy * VoxelDimMinusOne.xx, a2v.nInstanceID);
  
  v2f.z8_y8_x8_case8 = (uint3coord.z << 24) |
                       (uint3coord.y << 16) |
                       (uint3coord.x <<  8) |
                       (cube_case         );
  return v2f;
}
