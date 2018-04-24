#include "LodCB.h"

struct v2gConnector {
  uint z8_y8_x8_case8 : TEX2;
};

struct g2fConnector {
  uint index : TEX2;
};

cbuffer g_mc_lut {
  uint case_to_numpolys[256];
  float3 edge_start[12];
  float3 edge_dir[12];
  float3 edge_end[12];
  uint   edge_axis[12];  // 0 for x edges, 1 for y edges, 2 for z edges.
};

cbuffer g_mc_lut2 {
  int4  g_triTable[1280];   // 256*5 = 1024 (256 cases; up to 15 (0/3/6/9/12/15) verts output for each.)
};

Texture3D<uint> VertexIDVol;
SamplerState NearestClamp;


[maxvertexcount (15)]
void main(inout TriangleStream<g2fConnector> Stream, point v2gConnector input[1])
{
  uint cube_case = (input[0].z8_y8_x8_case8 & 0xFF);
  uint num_polys = case_to_numpolys[ cube_case ];
  int3 xyz = (int3)((input[0].z8_y8_x8_case8.xxx >> uint3(8,16,24)) & 0xFF);
 
  // don't generate polys in the final layer (in XY / YZ / ZX) of phantom cells.
  if (max(max(xyz.x, xyz.y), xyz.z) >= (uint)VoxelDimMinusOne.x)
    num_polys = 0;
  
  for (uint i=0; i<num_polys; i++) 
  {
    // range: 0-11
    int3 edgeNums_for_triangle = g_triTable[ cube_case*5 + i ].xyz;
    
    // now sample the 3D VertexIDVol texture to get the vertex IDs
    // for those vertices!
    
    int3 xyz_edge; 
    int3 VertexID;
    
    xyz_edge = xyz + (int3)edge_start[ edgeNums_for_triangle.x ].xyz;
     xyz_edge.x = xyz_edge.x*3 + edge_axis[ edgeNums_for_triangle.x ].x;
    VertexID.x = VertexIDVol.Load(int4(xyz_edge, 0)).x;

    xyz_edge = xyz + (int3)edge_start[ edgeNums_for_triangle.y ].xyz;
     xyz_edge.x = xyz_edge.x*3 + edge_axis[ edgeNums_for_triangle.y ].x;
    VertexID.y = VertexIDVol.Load(int4(xyz_edge, 0)).x;

    xyz_edge = xyz + (int3)edge_start[ edgeNums_for_triangle.z ].xyz;
     xyz_edge.x = xyz_edge.x*3 + edge_axis[ edgeNums_for_triangle.z ].x;
    VertexID.z = VertexIDVol.Load(int4(xyz_edge, 0)).x;

    // if none of the IDs are zero, there were no invalid indices,
    //   so let's add the triangle.
    // NOTE: if VertexIDVol is in good shape, we don't need this check.
    //if (VertexID.x*VertexID.y*VertexID.z != 0)  
    { 
      g2fConnector output;
      output.index = VertexID.x;
      Stream.Append(output);
      output.index = VertexID.y;
      Stream.Append(output);
      output.index = VertexID.z;
      Stream.Append(output);
      Stream.RestartStrip();
    }
  }
}