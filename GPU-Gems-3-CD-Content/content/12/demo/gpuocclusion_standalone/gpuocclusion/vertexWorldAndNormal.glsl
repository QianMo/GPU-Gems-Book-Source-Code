/*! \file vertexWorldAndNormal.glsl
 *  \author Jared Hoberock
 *  \brief Vertex shader passes world position and normal on texture coordinates.
 */

void main(void)
{
  // transform
  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

  // pass world position
  gl_TexCoord[0].xyz = gl_Vertex.xyz;

  // pass normal
  gl_TexCoord[1].xyz = gl_Normal.xyz;

  // pass color
  gl_FrontColor = gl_Color;
} // end main()

