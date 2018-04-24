#define MAX_INSTANCES      256
#define NOISE_LATTICE_SIZE 16

#define FOG_COLOR float3(0.161,0.322,0.588) // must match mainDrawPass2 in models\main.nma!

#define INV_LATTICE_SIZE (1.0/(float)(NOISE_LATTICE_SIZE))

#define HEED_THRESH 160 //0.5 wasn't quite enough - some chunks got missed

#define AMBO_RAYS            32 //64 //DO NOT EXCEED 64 (size of table). (it's a nice 64-point poisson distribution.)

// v- should match BUFFER_SLICES in src\blobs\BlobbyConstants.h:
// v- should match BUFFER_SLICES in src\blobs\BlobbyConstants.h:
#define MAX_AMBO_RAY_DIST_CELLS  24 
// ^- should match BUFFER_SLICES in src\blobs\BlobbyConstants.h!
// ^- should match BUFFER_SLICES in src\blobs\BlobbyConstants.h!
  
// also tweak values in these LUT's:
//   occlusion_amt[]
//   ambo_dist[]

cbuffer g_GlobalRockCB {
  //uint3    voxelDim;         // 128, 448, 128
  //float4   inv_voxelDim;     // 1/128, 1/448, 1/128, 0
  //float2   wsVoxelSize;      // 2/(128-1), (128-1)/2 [inverse], 0, 0
  //float2   uvw_to_ws_y;      // ~7, ~0.14
    // 'uvw_to_ws_y' helps you map between UVW space [0..1] for sampling the 3d texture,
    //   and world space Y [-1..1, 0..6..7ish, -1..1]
    // Note that in world space, Y is up; in the 3d texture, Z is up! (slices~z)
    // See "uvw_to_ws()" and "uvw_to_ws()" helper functions below.
  float4x4 octaveMat0;
  float4x4 octaveMat1;
  float4x4 octaveMat2;
  float4x4 octaveMat3;
  float4x4 octaveMat4;
  float4x4 octaveMat5;
  float4x4 octaveMat6;
  float4x4 octaveMat7;
  
  // use this for uvw stepping within a cell.  
  // notice it's only minus one on .xz!!:
  //float4   inv_voxelDimMinusOne; // 1/127, 1/448, 1/127, 0
  
  float4   timeValues;   // .x=anim time, .y=real time, .zw=0
  float3   wsEyePos;
  float3   wsLookAt;
};

//float3 uvw_to_ws(float3 uvw) {
//  // note: this is a one to many mapping, so your absolute WS y coord
//  //       might not be accurate.
//  // uvw -> ws:   ws.xzy = uvw.xyz*float3(2.0,2.0,uvw_to_ws_y.x) + float3(-1,-1,0);
//  return (uvw.xyz*float3(2.0,2.0,uvw_to_ws_y.x) + float3(-1,-1,0)).xzy;
//}
//
//float3 ws_to_uvw(float3 ws) {
//  // note: use results with 'repeat' sampling on Y in order to work!
//  // ws -> uvw:   uvw.xyz = ws.xzy*float3(0.5,0.5,uvw_to_ws_y.y) + float3(0.5,0.5,0);
//  float3 uvw = (ws.xzy*float3(0.5,0.5,uvw_to_ws_y.y) + float3(0.5,0.5,0)).xyz;
//  
//  // note that any half-voxel translations you do here 
//  // (in Y, for example) will affect both the rock AND the
//  // vines/water; they are both built from the field, and 
//  // ws_to_uvw is used for sampling field values from the field.
//  // so it's quite useless to shift ANYTHING here.
//  
//  // HOWEVER... we used to subtract 1/2 a voxel in Y here, to fix the
//  // nearest-neighbor sampling in build_rock.vsh, which would 
//  // occasionally (due to 1-bit float error) snap to the 
//  // wrong Y slice!  We now use bilinear sampling (and a 1/2 texel
//  // offset all around), though, so it's unnecessary.
//  
//  //uvw.z -= 0.5*inv_voxelDimMinusOne.y;
//  
//  // So... the only reason to move things here is so that the 
//  // rock geom generation hits the right spots to get good 
//  // trilinear interpolation.  ALL geom (rock, vines, water)
//  // will shift together, so no big deal there.
//  // It turns out that the field sampling for building the 
//  // rock polys needs a shift of +0.5 cells in XY, 
//  // but no shift in Z, to get good interpolation.
//  uvw.xy += 0.5*inv_voxelDimMinusOne.xz;
//  
//  return uvw;
//}

float3 vecMul(float4x4 m, float3 v) {
  return float3(dot(m._11_12_13, v), dot(m._21_22_23, v), dot(m._31_32_33, v));
}
float  smoothy(float  t) { return t*t*(3-2*t); }
float2 smoothy(float2 t) { return t*t*(3-2*t); }
float3 smoothy(float3 t) { return t*t*(3-2*t); }
float4 smoothy(float4 t) { return t*t*(3-2*t); }
