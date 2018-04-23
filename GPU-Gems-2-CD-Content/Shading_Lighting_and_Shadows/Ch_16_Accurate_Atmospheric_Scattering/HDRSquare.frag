//
// Atmospheric scattering fragment shader
//
// Author: Sean O'Neil
//
// Copyright (c) 2004 Sean O'Neil
//

uniform sampler2D s2Test;
uniform float fExposure;


void main (void)
{
	vec4 f4Color = texture2D(s2Test, gl_TexCoord[0].st);
	gl_FragColor = 1.0 - exp(f4Color * -fExposure);
}
