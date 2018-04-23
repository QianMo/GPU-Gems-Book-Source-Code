/************************************************************
 *															*
 * decr     : shaders for the clCrNiVector Class			*
 * version  : 1.1											*
 * author   : Jens Krüger									*
 * date     : 15.04.2003									*
 * modified	: 15.04.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

/////////////////////////////////////////////////////
// VARIABLES
/////////////////////////////////////////////////////

texture tLast;
texture tCurrent;
float fPreFac;
float4 f4Shift;

/////////////////////////////////////////////////////
// STRUCTURES
/////////////////////////////////////////////////////

struct app2Vertex {
	float4 Position   : POSITION;
	float4 TexCoords  : TEXCOORD0;
};

struct vertex2pixel {
	float4 Position   : POSITION;
	float4 TexCoords  : TEXCOORD0;
};

sampler sLast = sampler_state {
	Texture = (tLast);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Clamp;
	AddressV  = Clamp;
};

sampler sCurrent = sampler_state {
	Texture = (tCurrent);

	MinFilter = Point;
	MagFilter = Point;
	AddressU  = Clamp;
	AddressV  = Clamp;
};


/////////////////////////////////////////////////////
// PIXEL- AND VERTEX-SHADER
/////////////////////////////////////////////////////

vertex2pixel vsCompRHS(app2Vertex IN) {
	return IN;
}

float4 psCompRHS(vertex2pixel v) : COLOR0 {

	float Uij    = tex2D(sCurrent,v.TexCoords).r;
	float Uijtm1 = tex2D(sLast,v.TexCoords).r;
	float Uip1j  = tex2D(sCurrent,float2(v.TexCoords.x+f4Shift.z,v.TexCoords.y)).r;
	float Uim1j  = tex2D(sCurrent,float2(v.TexCoords.x-f4Shift.z,v.TexCoords.y)).r;
	float Uijp1  = tex2D(sCurrent,float2(v.TexCoords.x,v.TexCoords.y+f4Shift.w)).r;
	float Uijm1  = tex2D(sCurrent,float2(v.TexCoords.x,v.TexCoords.y-f4Shift.w)).r;

	float result = fPreFac*(Uip1j+Uim1j+Uijp1+Uijm1)+(2-4*fPreFac)*Uij-Uijtm1;
	
	return result;
}

/////////////////////////////////////////////////////
// TECHNIQUES
/////////////////////////////////////////////////////

technique tCompRHS {
	pass P0 {
		VertexShader = compile vs_2_0 vsCompRHS();
		PixelShader  = compile PS_PROFILE psCompRHS();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}