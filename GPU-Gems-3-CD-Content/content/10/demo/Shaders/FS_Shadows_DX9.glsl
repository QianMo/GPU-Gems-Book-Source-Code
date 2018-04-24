// Fragment shader
//
// Render shadows (DX9-level)
//
static const int g_iNumSplits = 4;

uniform vec3 g_vAmbient;
uniform float g_fSplitPlane[g_iNumSplits];
uniform sampler2DShadow g_samShadowMap[g_iNumSplits];

varying vec3 vLighting;
varying vec3 vColor;
varying vec4 vTexCoord[g_iNumSplits + 1];

void main()
{
  float fLightingFactor = 1.0;
  float fDistance = vTexCoord[0].z;

  if(fDistance < g_fSplitPlane[0])
    fLightingFactor = shadow2DProj(g_samShadowMap[0], vTexCoord[0+1]).x;
  else if(fDistance < g_fSplitPlane[1])
    fLightingFactor = shadow2DProj(g_samShadowMap[1], vTexCoord[1+1]).x;
  else if(fDistance < g_fSplitPlane[2])
    fLightingFactor = shadow2DProj(g_samShadowMap[2], vTexCoord[2+1]).x;
  else
    fLightingFactor = shadow2DProj(g_samShadowMap[3], vTexCoord[3+1]).x;

  gl_FragColor.xyz = vColor * clamp(g_vAmbient + vLighting.xyz * fLightingFactor, vec3(0,0,0), vec3(1,1,1));
  gl_FragColor.a = 1.0;
}
