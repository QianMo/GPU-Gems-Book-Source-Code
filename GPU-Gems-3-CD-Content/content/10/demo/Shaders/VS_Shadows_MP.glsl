// Vertex shader
//
// Render shadows (multi-pass)
//
uniform mat4 g_mWorld;
uniform mat4 g_mViewProj;
uniform mat4 g_mTextureMatrix;
uniform vec3 g_vLightDir;
uniform vec3 g_vLightColor;

varying vec3 vLighting;
varying vec3 vColor;
varying vec4 vTexCoord;

void main()
{	
  vec4 vWorldPos = g_mWorld * gl_Vertex;
  gl_Position = g_mViewProj * vWorldPos;

  // lighting
  vLighting = g_vLightColor * clamp(dot(-g_vLightDir,normalize(mat3(g_mWorld) * gl_Normal) ), 0, 1);
  vColor = gl_Color.rgb;

  vTexCoord = g_mTextureMatrix * vWorldPos;
}
