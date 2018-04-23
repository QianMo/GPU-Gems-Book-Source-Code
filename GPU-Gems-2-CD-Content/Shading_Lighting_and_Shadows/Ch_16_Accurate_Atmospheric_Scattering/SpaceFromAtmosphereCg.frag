//
// Atmospheric scattering fragment shader
//
// Author: Sean O'Neil
//
// Copyright (c) 2004 Sean O'Neil
//

float4 main(in float4 gl_Color : COLOR0,
			in float4 gl_SecondaryColor : COLOR1,
			in float4 gl_TexCoord0 : TEXCOORD0,
			uniform sampler2D s2Test) : COLOR

{
	return  gl_SecondaryColor * tex2D(s2Test, gl_TexCoord0.st);
}
