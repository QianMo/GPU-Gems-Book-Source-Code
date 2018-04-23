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
			uniform samplerRECT s2Test) : COLOR
{
	float4 v4DiffuseColor = texRECT(s2Test, gl_TexCoord0.st);
	return gl_Color + v4DiffuseColor * gl_SecondaryColor;
}
