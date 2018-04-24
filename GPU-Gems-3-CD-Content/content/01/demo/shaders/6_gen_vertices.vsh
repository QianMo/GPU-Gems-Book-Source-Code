//input: vert_list_geom

struct a2vConnector {
  uint z8_y8_x8_null4_edgeNum4 : TEX2;
};

struct v2fConnector {
  float4 wsCoordAmbo : COORD1;
  float3 wsNormal    : NORM1;
};

// 'ChunkCB' is updated each time we want to build a chunk:
cbuffer ChunkCB {
  float3 wsChunkPos = float3(0,0,0); //wsCoord of lower-left corner
  float  opacity = 1;
}

Texture3D density_vol;
Texture3D noiseVol0;
Texture3D noiseVol1;
Texture3D noiseVol2;
Texture3D noiseVol3;
Texture3D packedNoiseVol0;
Texture3D packedNoiseVol1;
Texture3D packedNoiseVol2;
Texture3D packedNoiseVol3;
SamplerState LinearClamp;
SamplerState NearestClamp;
SamplerState LinearRepeat;
SamplerState NearestRepeat;

#include "master.h"
#include "LodCB.h"
#include "density_NoMips_cheap.h"

cbuffer g_mc_lut {
  uint case_to_numpolys[256];
  float3 edge_start[12];
  float3 edge_dir[12];
  float3 edge_end[12];
  uint   edge_axis[12];  // 0 for x edges, 1 for y edges, 2 for z edges.
};

#define AMBO_STEPS 16
cbuffer g_ambo_lut {
  float  ambo_dist[16];
  float4 occlusion_amt[16];
  float3 g_ray_dirs_32[32];  // 32 rays w/a good poisson distrib.
  float3 g_ray_dirs_64[64];  // 64 rays w/a good poisson distrib.
  float3 g_ray_dirs_256[256];  // 256 rays w/a good poisson distrib.
};
#if (AMBO_RAYS==32)
  #define g_ray_dirs g_ray_dirs_32
#elif (AMBO_RAYS==64)
  #define g_ray_dirs g_ray_dirs_64
#elif (AMBO_RAYS==128)
  #define g_ray_dirs g_ray_dirs_256
#else
  ERROR
#endif

struct vertex {
  float4 worldCoordAmbo  : POSITION;     // .w = occlusion
  float3 worldNormalMisc : NORMAL;       
};


vertex PlaceVertOnEdge( float3 wsCoord_LL, float3 uvw_LL, int edgeNum )
{
  vertex output;
  
  // get the density values at the two ends of this edge of the cell,
  // then interpolate to find the point (t in 0..1) along the edge 
  // where the density value hits zero.  
  float str0 = density_vol.SampleLevel(NearestClamp, uvw_LL + InvVoxelDimPlusMarginsMinusOne.xxx*edge_start[edgeNum], 0).x;
  float str1 = density_vol.SampleLevel(NearestClamp, uvw_LL + InvVoxelDimPlusMarginsMinusOne.xxx*edge_end  [edgeNum], 0).x;
  float t = saturate( str0/(str0 - str1) );  // 'saturate' keeps occasional crazy stray triangle from appearing @ edges

  // reconstruct the interpolated point & place a vertex there.
  float3 pos_within_cell = edge_start[edgeNum] + t.xxx*edge_dir[edgeNum];  //0..1
  float3 wsCoord = wsCoord_LL + pos_within_cell*wsVoxelSize.xxx;
  float3 uvw     = uvw_LL     + pos_within_cell*InvVoxelDimPlusMarginsMinusOne.xxx;

  output.worldCoordAmbo.xyz  = wsCoord.xyz;
  
  
  // generate ambient occlusion for this vertex
  float ambo;
  {
    const float cells_to_skip_at_ray_start = 1.25;
    
    //float AMBO_RAY_DIST_CELLS = min(MAX_AMBO_RAY_DIST_CELLS, VoxelDimPlusMargins*0.18);
    float AMBO_RAY_DIST_CELLS = VoxelDimPlusMargins*0.25;
  
    // so that ambo looks the same if we change the voxel dim:
    float3 inv_voxelDim_tweaked = InvVoxelDimPlusMargins.xxx * VoxelDimPlusMargins/160.0;
        
    for (int i=0; i<AMBO_RAYS; i++) 
    {
      // cast a ray through uvw space
      float3 ray_dir = g_ray_dirs[ i ];
      float3 ray_start = uvw;
      float3 ray_now = ray_start + ray_dir*InvVoxelDimPlusMargins.xxx*cells_to_skip_at_ray_start;  // start a little out along the ray
      float3 ray_delta =           ray_dir*inv_voxelDim_tweaked*AMBO_RAY_DIST_CELLS.xxx;
      
      
      float ambo_this = 1;
      
      // SHORT RANGE:
      //  -step along the ray at AMBO_STEPS points,
      //     sampling the density volume texture
      //  -occlusion_amt[] LUT makes closer occlusions have more weight than far ones
      //  -start sampling a few cells away from the vertex, to reduce noise.
      ray_delta *= (1.0/(AMBO_STEPS));
      for (int j=0; j<AMBO_STEPS; j++) {   
        ray_now += ray_delta;
        float t = density_vol.SampleLevel(LinearClamp, ray_now, 0);
        ambo_this = lerp(ambo_this, 0, saturate(t*6) 
                                       * occlusion_amt[j].z//* pow(1-j/(float)AMBO_STEPS,0.4)
                         );
      }      
      
      // LONG RANGE: 
      //   also take a few samples far away,
      //   using the density *function*.
      for (int j=0; j<5; j++) {   
        // be sure to start some distance away, otherwise same vertex
        // in different LODs might have different brightness!  
        // (due to density function LOD bias)
        float distance = (j+2)/5.0;
        distance = pow(distance, 1.8);
        distance *= 40;
        float t = DENSITY(wsCoord + ray_dir*distance);
        const float shadow_hardness = 0.5;
        ambo_this *= 0.1 + 0.9*saturate(-t*shadow_hardness + 0.3);
      }      
      
      ambo_this *= 1.4;
      
      ambo += ambo_this;
    }
    ambo *= (1.0/(AMBO_RAYS));
  }  
  output.worldCoordAmbo.w = ambo;
  
  
  
  
  // figure out the normal vector for this vertex
  float3 grad;
  grad.x =   density_vol.SampleLevel(LinearClamp, uvw + InvVoxelDimPlusMargins.xyy, 0)
           - density_vol.SampleLevel(LinearClamp, uvw - InvVoxelDimPlusMargins.xyy, 0);
  grad.y =   density_vol.SampleLevel(LinearClamp, uvw + InvVoxelDimPlusMargins.yxy, 0)
           - density_vol.SampleLevel(LinearClamp, uvw - InvVoxelDimPlusMargins.yxy, 0);
  grad.z =   density_vol.SampleLevel(LinearClamp, uvw + InvVoxelDimPlusMargins.yyx, 0)
           - density_vol.SampleLevel(LinearClamp, uvw - InvVoxelDimPlusMargins.yyx, 0);
  output.worldNormalMisc.xyz = -normalize(grad);
  
 
  return output;
}

v2fConnector main(a2vConnector a2v)
{
  uint3 unpacked_coord;
  unpacked_coord.x = (a2v.z8_y8_x8_null4_edgeNum4 >>  8) & 0xFF;
  unpacked_coord.y = (a2v.z8_y8_x8_null4_edgeNum4 >> 16) & 0xFF;
  unpacked_coord.z = (a2v.z8_y8_x8_null4_edgeNum4 >> 24) & 0xFF;
  float3 chunkCoordWrite = (float3)unpacked_coord * InvVoxelDimMinusOne.xxx;
  float3 chunkCoordRead  = (Margin + VoxelDimMinusOne*chunkCoordWrite)*InvVoxelDimPlusMarginsMinusOne.xxx;

  float3 wsCoord = wsChunkPos + chunkCoordWrite*wsChunkSize;
    
  // very important: ws_to_uvw() should subtract 1/2 a texel in XYZ, 
  // to prevent 1-bit float error from snapping to wrong cell every so often!  
  float3 uvw = chunkCoordRead + InvVoxelDimPlusMarginsMinusOne.xxx*0.25;
    // HACK #2
    uvw.xyz *= (VoxelDimPlusMargins.x-1)*InvVoxelDimPlusMargins.x;

  // generate a vertex along this edge.
  int edgeNum = (a2v.z8_y8_x8_null4_edgeNum4 & 0x0F);
  vertex v = PlaceVertOnEdge( wsCoord, uvw, edgeNum );     

  // send it to the GS for stream out.
  v2fConnector v2f;
  v2f.wsCoordAmbo = v.worldCoordAmbo;
  v2f.wsNormal    = v.worldNormalMisc;
  return v2f;
}