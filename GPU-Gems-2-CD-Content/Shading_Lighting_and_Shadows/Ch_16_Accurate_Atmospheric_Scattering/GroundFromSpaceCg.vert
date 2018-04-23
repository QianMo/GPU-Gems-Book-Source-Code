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
	v3Pos = normalize(v3Pos);
	float fFar = length(v3Ray);
	v3Ray /= fFar;

	// Calculate the closest intersection of the ray with the outer atmosphere (which is the near point of the ray passing through the atmosphere)
	float fNear = getNearIntersection(v3CameraPos, v3Ray, fCameraHeight2, fOuterRadius2);

	// Calculate the ray's starting position, then calculate its scattering offset
	float3 v3Start = v3CameraPos + v3Ray * fNear;
	fFar -= fNear;
	float fDepth = exp((fInnerRadius - fOuterRadius) * fInvScaleDepth);
	float fCameraAngle = dot(-v3Ray, v3Pos);
	float fLightAngle = dot(v3LightPos, v3Pos);
	float fCameraScale = scale(fCameraAngle);
	float fLightScale = scale(fLightAngle);
	float fCameraOffset = fDepth*fCameraScale;
	float fTemp = (fLightScale + fCameraScale);

	// Initialize the scattering loop variables
	//gl_FrontColor = vec4(0.0, 0.0, 0.0, 0.0);
	float fSampleLength = fFar / fSamples;
	float fScaledLength = fSampleLength * fScale;
	float3 v3SampleRay = v3Ray * fSampleLength;
	float3 v3SamplePoint = v3Start + v3SampleRay * 0.5;

	// Now loop through the sample rays
	float3 v3FrontColor = float3(0.0, 0.0, 0.0);
	float3 v3Attenuate;
	for(int i=0; i<nSamples; i++)
	{
		float fHeight = length(v3SamplePoint);
		float fDepth = exp(fScaleOverScaleDepth * (fInnerRadius - fHeight));
		float fScatter = fDepth*fTemp - fCameraOffset;
		v3Attenuate = exp(-fScatter * (v3InvWavelength * fKr4PI + fKm4PI));
		v3FrontColor += v3Attenuate * (fDepth * fScaledLength);
		v3SamplePoint += v3SampleRay;
	}

	vertout OUT;
	OUT.pos = mul(gl_ModelViewProjectionMatrix, gl_Vertex);
	OUT.c0.rgb = v3FrontColor * (v3InvWavelength * fKrESun + fKmESun);
	OUT.c1.rgb = v3Attenuate;

	// Calculate spherical texture coordinates (yes, I know these are wrong)
	// (Most programs will just pass in their own texture coordinates)
	v3Pos.z += 1.0;
	float p = 2.0 * sqrt(dot(v3Pos, v3Pos));
	OUT.t0.x = 500.0 * (v3Pos.x/p + 0.5);
	OUT.t0.y = 500.0 * (-v3Pos.y/p + 0.5);
	return OUT;
}
