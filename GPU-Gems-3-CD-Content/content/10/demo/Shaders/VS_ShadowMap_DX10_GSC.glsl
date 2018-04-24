// Vertex shader
//
// Render shadow map with GS cloning (DX10-level)
//

#version 120

uniform mat4 g_mWorld;
uniform mat4 g_mViewProj;

void main()
{	
  vec4 vWorldPos = g_mWorld * gl_Vertex;
  gl_Position = g_mViewProj * vWorldPos;
}
