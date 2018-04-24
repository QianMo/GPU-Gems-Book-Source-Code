/*! \file weightedMinimum.glsl
 *  \author Jared Hoberock
 *  \brief This fragment shader computes a weighted minimum
 *         of two textures.
 */

// we're using rectangular textures
#extension GL_ARB_texture_rectangle : enable

// textures
uniform sampler2DRect texture0;
uniform sampler2DRect texture1;

// global parameters
uniform float minScale;
uniform float maxScale;

void main(void)
{
  vec4 t0 = texture2DRect(texture0, gl_FragCoord.xy);
  vec4 t1 = texture2DRect(texture1, gl_FragCoord.xy);

  float4 m = min(t0,t1);
  float4 M = max(t0,t1);

  // weighted blend
  gl_FragColor = minScale * m + maxScale * M;
} // end main()

