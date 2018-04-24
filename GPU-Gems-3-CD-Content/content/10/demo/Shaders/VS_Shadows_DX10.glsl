// Vertex shader
//
// Render shadows using texture array (DX10-level)
//

#version 120

static const int g_iNumSplits = 4;

uniform mat4 g_mWorld;
uniform mat4 g_mView;
uniform mat4 g_mViewProj;
uniform mat4 g_mTextureMatrix[g_iNumSplits];
uniform vec3 g_vLightDir;
uniform vec3 g_vLightColor;

varying vec3 vLighting;
varying vec3 vColor;
varying vec4 vTexCoord[g_iNumSplits + 1];

void main()
{	
  vec4 vWorldPos = g_mWorld * gl_Vertex;
  gl_Position = g_mViewProj * vWorldPos;

  // lighting
  vLighting = g_vLightColor * clamp(dot(-g_vLightDir,normalize(mat3(g_mWorld) * gl_Normal) ), 0, 1);
  vColor = gl_Color.rgb;

  // store view-space position
  vTexCoord[0] = g_mView * vWorldPos;

  // store coordinates for shadow map
  for(int i=0;i<g_iNumSplits;i++)
  {
    vTexCoord[i+1] = g_mTextureMatrix[i] * vWorldPos;
  }
}