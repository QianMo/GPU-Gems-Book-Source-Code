//
// Atmospheric scattering fragment shader
//
// Author: Sean O'Neil
//
// Copyright (c) 2004 Sean O'Neil
//

uniform sampler2DRect s2Test;	// RECTANGLE textures supported in GLSL?
uniform float fExposure;


void main (void)
{
	vec4 f4Color = texture2DRect(s2Test, gl_TexCoord[0].st * 1024.0);
	gl_FragColor = 1.0 - exp(f4Color * -fExposure);
}
