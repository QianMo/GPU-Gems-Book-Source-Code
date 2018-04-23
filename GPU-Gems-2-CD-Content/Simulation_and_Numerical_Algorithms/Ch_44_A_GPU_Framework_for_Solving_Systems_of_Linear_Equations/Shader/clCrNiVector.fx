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
	float4 TexCoords0 : TEXCOORD0;
	float4 TexCoords1 : TEXCOORD1;
	float4 TexCoords2 : TEXCOORD2;
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
	vertex2pixel OUT;

	OUT.Position   = IN.Position;
	OUT.TexCoords0 = IN.TexCoords;

	OUT.TexCoords1.xy = float2(IN.TexCoords.x+f4Shift.z,IN.TexCoords.y);
	OUT.TexCoords1.zw = float2(IN.TexCoords.x-f4Shift.z,IN.TexCoords.y);
	OUT.TexCoords2.xy = float2(IN.TexCoords.x,IN.TexCoords.y+f4Shift.w);
	OUT.TexCoords2.zw = float2(IN.TexCoords.x,IN.TexCoords.y-f4Shift.w);

	return OUT;
}

float4 psCompRHS(vertex2pixel v) : COLOR0 {

	float Uij    = tex2D(sCurrent,v.TexCoords0).r;
	float Uijtm1 = tex2D(sLast,v.TexCoords0).r;
	float Uip1j  = tex2D(sCurrent,v.TexCoords1.xy).r;
	float Uim1j  = tex2D(sCurrent,v.TexCoords1.zw).r;
	float Uijp1  = tex2D(sCurrent,v.TexCoords2.xy).r;
	float Uijm1  = tex2D(sCurrent,v.TexCoords2.zw).r;

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