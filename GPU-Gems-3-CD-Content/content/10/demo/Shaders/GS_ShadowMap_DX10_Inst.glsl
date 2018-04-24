// Geometry shader
//
// Render shadow map with instancing (DX10-level)
//

#version 120

#extension ARB_draw_buffers : require
#extension EXT_gpu_shader4 : require
#extension EXT_geometry_shader4 : require

void main()
{
  // get split index from texcoord
  gl_Layer = int(gl_TexCoordIn[0][0].x);
  // pass vertices through
  gl_Position = gl_PositionIn[0];
  EmitVertex();
  gl_Position = gl_PositionIn[1];
  EmitVertex();
  gl_Position = gl_PositionIn[2];
  EmitVertex();
  EndPrimitive();
}