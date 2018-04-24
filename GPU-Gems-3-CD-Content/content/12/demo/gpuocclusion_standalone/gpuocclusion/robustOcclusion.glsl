/*! \file robustOcclusion.glsl
 *  \author Jared Hoberock
 *  \brief Fragment shader to compute robust occlusion from
 *         discs & triangles.
 */

#extension GL_ARB_texture_rectangle : enable

#define PI 3.14159265
#define INV_PI 0.318309886

// textures
uniform sampler2DRect discCenters;
uniform sampler2DRect discNormalsAndAreas;
uniform sampler2DRect discOcclusion;
uniform sampler2DRect tree;
uniform sampler2DRect triangles0;
uniform sampler2DRect triangles1;
uniform sampler2DRect vertices;

// global parameters
uniform vec2 treeRoot;
uniform float epsilon;
uniform float distanceAttenuation;
uniform float triangleAttenuation;
uniform float zoneRadius = 0.1;

void visibleQuad(vec3 p, vec3 n,
                 vec3 v0, vec3 v1, vec3 v2,
                 inout vec3 q0, inout vec3 q1, inout vec3 q2, inout vec3 q3)
{
  // increase this value to control sporadic noise
  // resulting from the robustness of this operation
  // poor quality shading normals tend to reduce the robustness here
  const float epsilon = 1e-6;
  float d = dot(n,p);

  // Compute the signed distances from the vertices to the plane.
  float sd[3];
  sd[0] = dot(n,v0) - d;
  if(abs(sd[0]) <= epsilon) sd[0] = 0.0;
  sd[1] = dot(n,v1) - d;
  if(abs(sd[1]) <= epsilon) sd[1] = 0.0;
  sd[2] = dot(n,v2) - d;
  if(abs(sd[2]) <= epsilon) sd[2] = 0.0;

  if(sd[0] > 0.0)
  {
    if(sd[1] > 0.0)
    {
      if(sd[2] < 0.0)
      {
        // ++-
        q0 = v0;
        q1 = v1;
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
      }
      else
      {
        // +++ and ++0
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
    else if(sd[1] < 0.0)
    {
      if(sd[2] > 0.0)
      {
        // +-+
        q0 = v0;
        q1 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = v2;
      }
      else if(sd[2] < 0.0)
      {
        // +--
        q0 = v0;
        q1 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q2 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q3 = q2;
      }
      else
      {
        // +-0
        q0 = v0;
        q1 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q2 = v2;
        q3 = q2;
      }
    }
    else
    {
      if(sd[2] < 0.0)
      {
        // +0-
        q0 = v0;
        q1 = v1;
        q2 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q3 = q2;
      }
      else
      {
        // +0+ and +00
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
  }
  else if(sd[0] < 0.0)
  {
    if(sd[1] > 0.0)
    {
      if(sd[2] > 0.0)
      {
        // -++
        q0 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q1 = v1;
        q2 = v2;
        q3 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
      }
      else if(sd[2] < 0.0)
      {
        // -+-
        q0 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q1 = v1;
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = q2;
      }
      else
      {
        // -+0
        q0 = v0+(sd[0]/(sd[0]-sd[1]))*(v1-v0);
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
    else if(sd[1] < 0.0)
    {
      if(sd[2] > 0.0)
      {
        // --+
        q0 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q1 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q2 = v2;
        q3 = q2;
      }
      else
      {
        // --- and --0
        q0 = q1 = q2 = q3 = p;
      }
    }
    else
    {
      if(sd[2] > 0.0)
      {
        // -0+
        q0 = v0+(sd[0]/(sd[0]-sd[2]))*(v2-v0);
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else
      {
        // -0- and -00
        q0 = q1 = q2 = q3 = p;
      }
    }
  }
  else
  {
    if(sd[1] > 0.0)
    {
      if(sd[2] < 0.0)
      {
        // 0+-
        q0 = v0;
        q1 = v1;
        q2 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q3 = q2;
      }
      else
      {
        // 0+0 and 0+-
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
    }
    else if(sd[1] < 0.0)
    {
      if(sd[2] > 0.0)
      {
        // 0-+
        q0 = v0;
        q1 = v1+(sd[1]/(sd[1]-sd[2]))*(v2-v1);
        q2 = v2;
        q3 = q2;
      }
      else
      {
        // 0-- and 0-0
        q0 = q1 = q2 = q3 = p;
      } // end else
    }
    else
    {
      if(sd[2] > 0.0)
      {
        // 00+
        q0 = v0;
        q1 = v1;
        q2 = v2;
        q3 = q2;
      }
      else
      {
        // 00- and 000
        q0 = q1 = q2 = q3 = p;
      }
    }
  }
} // end visibleQuad()

float myClamp(float m, float M, float val)
{
  return max(m, min(M, val));
}

float computeFormFactor(vec3 p, vec3 n,
                        vec3 q0, vec3 q1, vec3 q2, vec3 q3)
{
  vec3 r0 = q0 - p;
  r0 = normalize(r0);

  vec3 r1 = q1 - p;
  r1 = normalize(r1);

  vec3 r2 = q2 - p;
  r2 = normalize(r2);

  vec3 r3 = q3 - p;
  r3 = normalize(r3);

  vec3 g0 = normalize(cross(r1,r0));
  vec3 g1 = normalize(cross(r2,r1));
  vec3 g2 = normalize(cross(r3,r2));
  vec3 g3 = normalize(cross(r0,r3));

  float a = acos(myClamp(-1.0, 1.0, dot(r0,r1)));
  float d = clamp(dot(n,g0), -1.0, 1.0);
  float contrib = a * d;
  float result = contrib;

  a = acos(myClamp(-1.0, 1.0, dot(r1,r2)));
  d = clamp(dot(n,g1), -1.0, 1.0);
  contrib = a * d;
  result += contrib;

  a = acos(myClamp(-1.0, 1.0, dot(r2,r3)));
  d = clamp(dot(n,g2), -1.0, 1.0);
  contrib = a * d;
  result += contrib;

  a = acos(myClamp(-1.0, 1.0, dot(r3,r0)));
  d = clamp(dot(n,g3), -1.0, 1.0);
  contrib = a * d;
  result += contrib;

  result *= 0.5;
  result *= INV_PI;

  // use max for only front facing triangles
  return max(0.0, result);
} // end computeFormFactor()

float computeFormFactor(vec3 p, vec3 n, vec2 triIndex)
{
  // look up the vertex positions
  vec2 i0, i1, i2;
  vec3 tri0 = texture2DRect(triangles0, triIndex).xyz;
  vec3 tri1 = texture2DRect(triangles1, triIndex).xyz;
  i0 = tri0.xy;
  i1 = vec2(tri0.z, tri1.x);
  i2 = tri1.yz;

  vec3 v0 = texture2DRect(vertices, i0).xyz;
  vec3 v1 = texture2DRect(vertices, i1).xyz;
  vec3 v2 = texture2DRect(vertices, i2).xyz;

  vec3 q0,q1,q2,q3;
  visibleQuad(p, n,
              v0, v1, v2,
              q0, q1, q2, q3);

  return computeFormFactor(p,n,q0,q1,q2,q3);
} // end computeFormFactor()

float solidAngle(vec3 v, float d2, vec3 receiverNormal,
                 vec3 emitterNormal, float emitterArea)
{
  float result = emitterArea
    // only front facing emitters contribute shadow
    * saturate(dot(emitterNormal, -v))
    * saturate(dot(receiverNormal, v))
    / (d2 + emitterArea / PI);
 return result / PI;
} // end solidAngle()

vec4 computeRobustOcclusion(vec3 rPosition,
                            vec3 rNormal)
{
  float eArea;
  vec3 ePosition;
  vec4 eNormal;
  float eOcclusion;
  vec3 v;
  float result;
  vec4 eIndex = treeRoot.xyxx;
  vec2 thisIndex;
  vec3 bentNormal = rNormal;
  float contribution;
  float d2;
  float parentWeight = 0.0;
  float childrenWeight;
  float parentArea = 1.0;
  float parentContribution;
  float tooClose;
  vec2 parentNext;
  float reciprocalDistance;

  while(eIndex.x != 0.0)
  {
    // note this index
    thisIndex = eIndex.xy;

    ePosition = texture2DRect(discCenters, eIndex.xy).xyz;
    eNormal = texture2DRect(discNormalsAndAreas, eIndex.xy);
    eOcclusion = texture2DRect(discOcclusion, eIndex.xy).x;
    eArea = eNormal.w;

    // get the next index
    eIndex = texture2DRect(tree, eIndex.xy).xyzw;
    v = ePosition - rPosition;
    d2 = dot(v,v) + 1e-16;

    reciprocalDistance = rsqrt(d2);

    v *= reciprocalDistance;

    contribution = solidAngle(v, d2, rNormal, eNormal.xyz, abs(eArea));
    contribution *= eOcclusion;

    // attenuate by distance
    contribution /= (1.0 + distanceAttenuation * d2);

    tooClose = -epsilon * eArea;
    if(d2 < tooClose * (1.0 + zoneRadius))
    {
      // record the next index
      parentNext = eIndex.xy;

      // traverse deeper into hierarchy
      eIndex.xy = eIndex.zw;

      parentContribution = contribution;
      parentArea = eArea;
      parentWeight = saturate((d2 - (1.0 - zoneRadius)*tooClose) / (2.0 * zoneRadius * tooClose));
    } // end if
    else
    {
      if(eArea > 0.0)
      {
        // we're at a triangle
        contribution = 0;

        // only call the expensive procedure if the triangle
        // faces the point
        if(dot(eNormal.xyz, -v) >= 0)
        {
          contribution = computeFormFactor(rPosition, rNormal, thisIndex);

          // modulate triangles' contribution by their occlusion
          // low values for triangleAttenuation will emphasize small features such as creases and cracks
          // high values will compensate for multiple levels of occlusion and lessen the influence of
          // large far away triangles that probably aren't visible
          contribution *= pow(eOcclusion, triangleAttenuation);

          // attenuate by distance
          contribution /= (1.0 + distanceAttenuation * d2);
        } // end if
      } // end if

      bentNormal -= contribution * v;

      // blend contribution from ourself and a fraction of our parent's
      // simple linear blend
      childrenWeight = 1.0 - parentWeight;

      // blend contribution
      // parent's contribution is modulated by the ratio of the child's area to that of the parent
      result += childrenWeight*contribution + parentWeight*parentContribution*abs(eArea/parentArea);

      // clear the parent weight if we do not have a right brother
      // note if our next index is the same as our parent's, we don't have a right brother
      parentWeight = (eIndex.x == parentNext.x && eIndex.y == parentNext.y) ? 0.0 : parentWeight;
    } // end if
  } // end while

  result = saturate(1.0 - result);
  return vec4(result, result, result, result);
} // end computeRobustOcclusion()

void main(void)
{
  // world position passed on texcoord 0
  vec3 position = gl_TexCoord[0].xyz;

  // normal passed on texcoord 1
  vec3 normal = normalize(gl_TexCoord[1].xyz);

  // pass off
  gl_FragColor = computeRobustOcclusion(position, normal);
} // end main()

