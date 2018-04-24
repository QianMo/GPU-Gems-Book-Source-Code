/*! \file fragmentOcclusion.glsl
 *  \author Jared Hoberock
 *  \brief Fragment shader to compute disc-to-fragment occlusion.
 */

// we're using rectangular textures
#extension GL_ARB_texture_rectangle : enable

#define PI 3.14159265

// textures
uniform sampler2DRect discCenters;
uniform sampler2DRect discNormalsAndAreas;
uniform sampler2DRect discOcclusion;
uniform sampler2DRect tree;

// global parameters
uniform vec2 treeRoot;
uniform float epsilon;

float solidAngle(vec3 v, float d2, vec3 receiverNormal,
                 vec3 emitterNormal, float emitterArea)
{
  float result = emitterArea
    * saturate(dot(emitterNormal, -v))
    * saturate(dot(receiverNormal, v))
    / (d2 + emitterArea / PI);
  return result / PI;
} // end solidAngle()

vec4 computeOcclusion(vec3 rPosition,
                      vec3 rNormal)
{
  float eArea;
  vec3 ePosition;
  vec4 eNormal;
  float eOcclusion;
  vec3 v;
  float result;
  vec4 eIndex = treeRoot.xyxx;
  vec3 bentNormal = rNormal;
  float contribution;
  float d2;

  while(eIndex.x != 0.0)
  {
    ePosition = texture2DRect(discCenters, eIndex.xy).xyz;
    eNormal = texture2DRect(discNormalsAndAreas, eIndex.xy);
    eOcclusion = texture2DRect(discOcclusion, eIndex.xy).x;
    eArea = eNormal.w;

    // get the next index
    eIndex = texture2DRect(tree, eIndex.xy).xyzw;
    v = ePosition - rPosition;
    d2 = dot(v,v) + 1e-16;

    // is receiver close enough to parent element?
    // note: non-leaves have negative area
    if(d2 < -epsilon * eArea)
    {
      // ignore this element
      eArea = 0.0;

      // traverse deeper into hierarchy
      eIndex.xy = eIndex.zw;
    } // end if

    // normalize v
    v *= rsqrt(d2);

    contribution = solidAngle(v, d2, rNormal, eNormal.xyz, abs(eArea));

    // modulate contribution by the emitter's occlusion
    contribution *= eOcclusion;

    result += contribution;

    bentNormal -= contribution * v;
  } // end while

  bentNormal = normalize(bentNormal);

  result = saturate(1.0 - result);
  return vec4(result, result, result, result);
} // end computeOcclusion()

void main(void)
{
  // world position passed on texcoord 0
  vec3 position = gl_TexCoord[0].xyz;

  // normal passed on texcoord 1
  vec3 normal = gl_TexCoord[1].xyz;

  // pass off
  gl_FragColor = computeOcclusion(position, normal);
} // end main()


