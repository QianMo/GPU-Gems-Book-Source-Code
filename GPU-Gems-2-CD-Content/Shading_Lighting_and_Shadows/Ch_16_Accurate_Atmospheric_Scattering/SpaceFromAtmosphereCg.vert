//
// Atmospheric scattering vertex shader
//
// Author: Sean O'Neil
//
// Copyright (c) 2004 Sean O'Neil
//

#include "Common.cg"


vertout main(float4 gl_Vertex : POSITION,
			 float4 gl_MultiTexCoord0 : TEXCOORD0,
			 uniform float4x4 gl_ModelViewProjectionMatrix,
			 uniform float3 v3CameraPos,		// The camera's current position
			 uniform float3 v3LightPos,			// The direction vector to the light source
			 uniform float3 v3InvWavelength,	// 1 / pow(wavelength, 4) for the red, green, and blue channels
			 uniform float fCameraHeight,		// The camera's current height
			 uniform float fCameraHeight2,		// fCameraHeight^2
			 uniform float fOuterRadius,		// The outer (atmosphere) radius
			 uniform float fOuterRadius2,		// fOuterRadius^2
			 uniform float fInnerRadius,		// The inner (planetary) radius
			 uniform float fInnerRadius2,		// fInnerRadius^2
			 uniform float fKrESun,				// Kr * ESun
			 uniform float fKmESun,				// Km * ESun
			 uniform float fKr4PI,				// Kr * 4 * PI
			 uniform float fKm4PI,				// Km * 4 * PI
			 uniform float fScale,				// 1 / (fOuterRadius - fInnerRadius)
			 uniform float fScaleOverScaleDepth)// fScale / fScaleDepth
{
	// Get the ray from the camera to the vertex and its length (which is the far point of the ray passing through the atmosphere)
	float3 v3Pos = gl_Vertex.xyz;
	float3 v3Ray = v3Pos - v3CameraPos;
	float fFar = length(v3Ray);
	v3Ray /= fFar;

	// Calculate the farther intersection of the ray with the outer atmosphere (which is the far point of the ray passing through the atmosphere)
	fFar = getFarIntersection(v3CameraPos, v3Ray, fCameraHeight2, fOuterRadius2);

	// Calculate attenuation from the camera to the top of the atmosphere toward the vertex
	float3 v3Start = v3CameraPos;
	float fHeight = length(v3Start);
	float fDepth = exp(fScaleOverScaleDepth * (fInnerRadius - fCameraHeight));
	float fAngle = dot(v3Ray, v3Start) / fHeight;
	float fScatter = fDepth*scale(fAngle);

	vertout OUT;
	OUT.pos = mul(gl_ModelViewProjectionMatrix, gl_Vertex);
	OUT.c1.rgb = exp(-fScatter * (v3InvWavelength * fKr4PI + fKm4PI));
	OUT.t0 = gl_MultiTexCoord0;
	return OUT;
}
