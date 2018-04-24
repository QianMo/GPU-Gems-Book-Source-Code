/*! \file discScattering.cg
 *  \author Yuntao Jia
 *  \brief A fragment program which computes
 *         per-vertex (disc) scattering.
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
uniform vec3 viewPosition;
uniform vec3 lightPosition;
uniform vec3 albedo_p;
uniform vec3 sig_tr;
uniform vec3 zr;
uniform vec3 zv;
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

vec3 multipleScattering(float d2, vec3 zr, vec3 zv, 
				vec3 sig_tr)
{
   vec3 r2  = vec3(d2, d2, d2);
   vec3 dr1 = rsqrt(r2+(zr*zr));
   vec3 dv1 = rsqrt(r2+(zv*zv));
   vec3 C1  = zr*(sig_tr+dr1);
   vec3 C2  = zv*(sig_tr+dv1);
   vec3 dL = C1*exp(-sig_tr/dr1)*dr1*dr1 + C2*exp(-sig_tr/dv1)*dv1*dv1;
   return dL;
} // end multipleScattering()

vec4 computeScattering(vec3 rPosition,
                      vec3 rNormal)
{
  float eArea;
  float d2;
  vec3 ePosition;
  vec4 eNormal;
  float eAccessibility;
  vec3 v;
  vec4 eIndex = treeRoot.xyxx;
  vec3 bentNormal;
  vec4 contribution;
  vec4 result;
  vec4 irradiance;

  // light & view direction
  vec3 viewDir  = normalize(viewPosition - rPosition);
  vec3 lightDir = normalize(lightPosition - rPosition);

  while(eIndex.x != 0.0)
  while(eIndex.x != 0.0)
  {
    ePosition = texture2DRect(discCenters, eIndex.xy).xyz;
    eNormal = texture2DRect(discNormalsAndAreas, eIndex.xy);
    bentNormal = texture2DRect(discBentNormalsAndAccessibility, eIndex.xy).xyz;
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

    contribution.w = solidAngle(v, d2, rNormal, eNormal.xyz, abs(eArea));
    contribution.xyz = multipleScattering(d2, zr, zv, sig_tr)/4/PI/PI*
                            albedo_p*abs(eArea)*
                            saturate(dot(bentNormal, lightDir));

    // modulate contribution by the emitter's accessibility
    contribution *= eAccessibility;

    irradiance += contribution;    
  } // end while  

  result = irradiance;   
  return result;
} // end computeScattering()

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
  gl_FragColor = computeScattering(position, normal);
} // end main()

