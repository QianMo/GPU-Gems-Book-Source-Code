cbuffer LodCB {
  float  VoxelDim = 65;          // # of cell corners
  float  VoxelDimMinusOne = 64;  // # of cells
  float2 wsVoxelSize = float2(1.0/64.0, 0);  // could be 1/63, 1/31, 1/15 depending on LOD
  float  wsChunkSize = 4.0;                  // 1.0, 2.0, or 4.0 depending on LOD
  float2 InvVoxelDim = float2(1.0/65.0, 0); 
  float2 InvVoxelDimMinusOne = float2(1.0/64.0, 0); 
  float  Margin                      = 4;
  float  VoxelDimPlusMargins         = 73;
  float  VoxelDimPlusMarginsMinusOne = 72;
  float2 InvVoxelDimPlusMargins         = float2(1.0/73.0, 0);
  float2 InvVoxelDimPlusMarginsMinusOne = float2(1.0/72.0, 0);
}

float3 ChunkCoord_To_ExtChunkCoord(float3 chunkCoord)
{ 
  // if VoxelDim is 65 
  // (i.e. 65 corners / 64 cells per block)
  // then chunkCoord should be in [0..64/65]
  // and extChunkCoord (returned) will be outside that range.
  return (chunkCoord*VoxelDimPlusMargins.xxx - Margin.xxx)*InvVoxelDim.xxx;
}

float3 ExtChunkCoord_To_ChunkCoord(float3 extChunkCoord)
{ 
  return (extChunkCoord*VoxelDim.xxx + Margin.xxx)*InvVoxelDimPlusMargins.xxx;
}