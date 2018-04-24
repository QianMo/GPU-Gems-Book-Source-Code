// Vertex shader
//
// Render shadow map with instancing (DX10-level)
//

#version 120
#extension EXT_gpu_shader4 : require

uniform mat4 g_mWorld;
uniform mat4 g_mViewProj;

static const int g_iNumSplits = 4;

uniform mat4 g_mCropMatrix[g_iNumSplits];
uniform int g_iFirstSplit;
uniform int g_iLastSplit;

void main()
{	
  vec4 vWorldPos = g_mWorld * gl_Vertex;
  gl_Position = g_mViewProj * vWorldPos;
  // determine split index from instance ID
  int iSplit = g_iFirstSplit + gl_InstanceID;
  // transform with split specific projection matrix
  gl_Position = g_mCropMatrix[iSplit] * gl_Position;
  // store split index in texcoord
  gl_TexCoord[0].x = float(iSplit);
}
