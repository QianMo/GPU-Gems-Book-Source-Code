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
			uniform float fExposure,
			uniform samplerRECT s2Test) : COLOR
{
	half4 f4Color = texRECT(s2Test, gl_TexCoord0.xy * 1024.0);
	return 1.0 - exp(f4Color * -fExposure);
}
