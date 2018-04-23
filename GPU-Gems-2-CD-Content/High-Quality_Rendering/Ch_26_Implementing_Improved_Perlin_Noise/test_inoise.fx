/*********************************************************************NVMH3****
File:  $Id: //sw/devrel/SDK/MEDIA/HLSL/test_inoise.fx#2 $

Copyright NVIDIA Corporation 2004
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Comments:
	Simple example to test procedural noise functions

******************************************************************************/

#include "inoise.fxh"

//------------------------------------
float4x4 worldViewProj : WorldViewProjection;
float4x4 world : World;

string description = "Perlin noise test";

float noiseScale
<
    string UIWidget = "slider";
    string UIName = "noise scale";
    float UIMin = 0.0; float UIMax = 20.0; float UIStep = 0.01;
> = 5.0;

float anim
<
    string UIWidget = "slider";
    string UIName = "animate";
    float UIMin = 0.0; float UIMax = 10.0; float UIStep = 0.01;
> = 0.0;

float lacunarity
<
    string UIWidget = "slider";
    string UIName = "lacunarity";
    float UIMin = 0.0; float UIMax = 10.0; float UIStep = 0.01;
> = 2.0;

float gain
<
    string UIWidget = "slider";
    string UIName = "gain";
    float UIMin = 0.0; float UIMax = 1.0; float UIStep = 0.01;
> = 0.5;

float threshold
<
    string UIWidget = "slider";
    string UIName = "threshold";
    float UIMin = 0.0; float UIMax = 1.0; float UIStep = 0.01;
> = 0.5;

float transition_width
<
    string UIWidget = "slider";
    string UIName = "transition width";
    float UIMin = 0.0; float UIMax = 1.0; float UIStep = 0.01;
> = 0.05;

float4 color1 : DIFFUSE
<
    string UIName = "color 1";
> = float4(0.0, 0.0, 0.5, 1.0);

float4 color2 : DIFFUSE
<
    string UIName = "color 2";
> = float4(0.0, 0.7, 0.0, 1.0);

//------------------------------------
struct vertexInput {
    float4 position		: POSITION;
    float2 texcoord     : TEXCOORD;
};

struct vertexOutput {
   float4 hPosition		: POSITION;
   float2 texcoord      : TEXCOORD0;
   float3 wPosition		: TEXCOORD1;
};


//------------------------------------
vertexOutput VS(vertexInput IN) 
{
    vertexOutput OUT;
    OUT.hPosition = mul(IN.position, worldViewProj);
    OUT.texcoord = IN.texcoord * noiseScale;
    OUT.wPosition = mul(IN.position, world).xyz * noiseScale;
    return OUT;
}

//-----------------------------------
float4 PS_inoise(vertexOutput IN): COLOR
{
	float3 p = IN.wPosition;
//	return abs(inoise(p));
//	return inoise(p);
	return inoise(p)*0.5+0.5;
//	return inoise(float3(IN.texcoord, 0.0))*0.5+0.5;
}

float4 PS_inoise4d(vertexOutput IN): COLOR
{
	float3 p = IN.wPosition;
	return inoise(float4(p, anim))*0.5+0.5;	
//	return abs(inoise(float4(p, anim)));
}

float4 PS_fBm(vertexOutput IN): COLOR
{
	float3 p = IN.wPosition;
	return fBm(p, 4, lacunarity, gain)*0.5+0.5;
}

float4 PS_earth(vertexOutput IN): COLOR
{
	float3 p = IN.wPosition;
	float n = fBm(p, 4, lacunarity, gain)*0.5+0.5;
	return lerp(color1, color2, smoothstep(threshold-transition_width, threshold+transition_width, n));
}

float4 PS_turbulence(vertexOutput IN): COLOR
{
	float3 p = IN.wPosition;
	return turbulence(p, 4, lacunarity, gain);
}

float4 PS_ridgedmf(vertexOutput IN): COLOR
{
	float3 p = IN.wPosition;
	return ridgedmf(p, 4, lacunarity, gain);
}

//-----------------------------------
technique inoise
{
    pass p0 
    {		
		VertexShader = compile vs_1_1 VS();
		PixelShader  = compile ps_2_a PS_inoise();
    }
}

technique inoise4d
{
    pass p0 
    {		
		VertexShader = compile vs_1_1 VS();
		PixelShader  = compile ps_2_a PS_inoise4d();
    }
}

technique fBm
{
    pass p0 
    {		
		VertexShader = compile vs_1_1 VS();
		PixelShader  = compile ps_2_a PS_fBm();
    }
}

technique earth
{
    pass p0 
    {		
		VertexShader = compile vs_1_1 VS();
		PixelShader  = compile ps_2_a PS_earth();
    }
}

technique turbulence
{
    pass p0 
    {		
		VertexShader = compile vs_1_1 VS();
		PixelShader  = compile ps_2_a PS_turbulence();
    }
}

technique ridgedmf
{
    pass p0 
    {		
		VertexShader = compile vs_1_1 VS();
		PixelShader  = compile ps_2_a PS_ridgedmf();
    }
}
