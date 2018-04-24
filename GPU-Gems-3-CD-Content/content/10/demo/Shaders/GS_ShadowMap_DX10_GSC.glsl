// Geometry shader
//
// Render shadow map with GS cloning (DX10-level)
//

#version 120

#extension ARB_draw_buffers : require
#extension EXT_gpu_shader4 : require
#extension EXT_geometry_shader4 : require

static const int g_iNumSplits = 4;

uniform mat4 g_mCropMatrix[g_iNumSplits];
uniform int g_iFirstSplit;
uniform int g_iLastSplit;

void main()
{	
  // for each split
  for(int iSplit = g_iFirstSplit; iSplit <= g_iLastSplit; iSplit++)
  {
    gl_Layer = iSplit;
    // for each vertex
    for(int iVertex = 0; iVertex < 3; iVertex++)
    {
      // transform with split specific projection matrix
      gl_Position = g_mCropMatrix[iSplit] * gl_PositionIn[iVertex];
      // append vertex to stream
      EmitVertex();
    }
    // mark end of triangle
    EndPrimitive();
  }
}