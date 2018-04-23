//
// Atmospheric scattering vertex shader
//
// Author: Sean O'Neil
//
// Copyright (c) 2004 Sean O'Neil
//

struct vertout
{
	float4 pos : POSITION;
	float4 t0 : TEXCOORD0;
};


vertout main(float4 gl_Vertex : POSITION,
			 float4 gl_MultiTexCoord0 : TEXCOORD0,
			 uniform float4x4 gl_ModelViewProjectionMatrix)
{
	vertout OUT;
	OUT.pos = mul(gl_ModelViewProjectionMatrix, gl_Vertex);
	OUT.t0 = gl_MultiTexCoord0;
	return OUT;
}
