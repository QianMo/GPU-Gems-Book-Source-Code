// Fragment shader
//
// Render shadows (DX10-level)
//
#version 120

#extension GL_EXT_gpu_shader4 : require
static const int g_iNumSplits = 4;

uniform vec3 g_vAmbient;
uniform float g_fSplitPlane[g_iNumSplits];
uniform sampler2DArrayShadow g_samShadowMap;

varying vec3 vLighting;
varying vec3 vColor;
varying vec4 vTexCoord[g_iNumSplits + 1];

void main()
{
  float fLightingFactor = 1.0;
  float fDistance = vTexCoord[0].z;

  for(int i=0; i < g_iNumSplits; i++)
  {
    if(fDistance < g_fSplitPlane[i])
    {
      vec4 vCoord = vTexCoord[i+1].xyzz / vTexCoord[i+1].w;
      vCoord.z = float(i);
      fLightingFactor = shadow2DArray(g_samShadowMap, vCoord).x;
      break;
    }
  }

	gl_FragColor.xyz = vColor * clamp(g_vAmbient + vLighting.xyz * fLightingFactor, vec3(0,0,0), vec3(1,1,1));
  gl_FragColor.a = 1.0;
}
