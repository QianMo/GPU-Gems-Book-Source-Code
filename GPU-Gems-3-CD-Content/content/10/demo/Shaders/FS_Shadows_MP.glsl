// Fragment shader
//
// Render shadows (multi-pass)
//
uniform vec3 g_vAmbient;
uniform sampler2DShadow g_samShadowMap;

varying vec3 vLighting;
varying vec3 vColor;
varying vec4 vTexCoord;

void main()
{
  float fLightingFactor = shadow2DProj(g_samShadowMap, vTexCoord).x;
  gl_FragColor.xyz = vColor * clamp(g_vAmbient + vLighting.xyz * fLightingFactor, vec3(0,0,0), vec3(1,1,1));
  gl_FragColor.a = 1.0;
}
