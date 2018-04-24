/*! \file discAccessibility.cg
 *  \author Jared Hoberock
 *  \brief A fragment program which computes
 *         per-disc accessibility.
 */

#extension GL_ARB_texture_rectangle : enable

#define PI 3.14159265

// textures
uniform sampler2DRect discCenters;
uniform sampler2DRect discNormalsAndAreas;
uniform sampler2DRect discBentNormalsAndAccessibility;
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

vec4 computeAccessibility(vec3 rPosition,
                          vec3 rNormal)
{
  vec3 ePosition;
  vec4 eNormal;
  vec3 v;
  vec4 eIndex = treeRoot.xyxx;
  vec3 bentNormal = rNormal;
  float eArea;
  float eAccessibility;
  float contribution;
  float d2;
  float occlusion;
  float accessibility;

  while(eIndex.x != 0.0)
  while(eIndex.x != 0.0)
  {
    ePosition = texture2DRect(discCenters, eIndex.xy).xyz;
    eNormal = texture2DRect(discNormalsAndAreas, eIndex.xy);
    eAccessibility = texture2DRect(discBentNormalsAndAccessibility, eIndex.xy).w;
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
      eArea = 0;

      // traverse deeper into hierarchy
      eIndex.xy = eIndex.zw;
    } // end if

    // normalize v
    v *= rsqrt(d2);

    contribution = solidAngle(v, d2, rNormal, eNormal.xyz, abs(eArea));

    // modulate contribution by the emitter's accessibility (visibility)
    contribution *= eAccessibility;

    occlusion += contribution;
    bentNormal -= contribution * v;
  } // end while

  bentNormal = normalize(bentNormal);

  // return the result as accessibility, not occlusion
  accessibility = saturate(1.0 - occlusion);
  return vec4(bentNormal.x, bentNormal.y, bentNormal.z, accessibility);
} // end computeAccessibility()

void main(void)
{
  vec2 discLocation = gl_FragCoord.xy;

  // look up the disc's position and normal
  vec3 position = texture2DRect(discCenters, discLocation).xyz;
  vec3 normal = texture2DRect(discNormalsAndAreas, discLocation).xyz;

  // don't waste time operating on garbage space at the end of
  // the texture
  if(isnan(position.x)) discard;

  // pass off
  gl_FragColor = computeAccessibility(position, normal);
} // end main()

