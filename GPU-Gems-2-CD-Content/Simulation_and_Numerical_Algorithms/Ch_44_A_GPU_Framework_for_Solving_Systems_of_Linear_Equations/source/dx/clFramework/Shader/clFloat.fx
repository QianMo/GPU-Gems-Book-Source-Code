/************************************************************
 *															*
 * decr     : shaders for the clVector Classes				*
 * version  : 1.3											*
 * author   : Jens Krüger									*
 * date     : 15.04.2003									*
 * modified	: 18.11.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

/////////////////////////////////////////////////////
// VARIABLES
/////////////////////////////////////////////////////

texture tFloatA;
texture tFloatB;
float4 fScalar;

/////////////////////////////////////////////////////
// STRUCTURES
/////////////////////////////////////////////////////

struct app2Vertex {
	float4 Position   : POSITION;
};

struct vertex2pixel {
	float4 Position   : POSITION;
};

sampler sFloatA = sampler_state {
	Texture = (tFloatA);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

sampler sFloatB = sampler_state {
	Texture = (tFloatB);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};


/////////////////////////////////////////////////////
// PIXEL- AND VERTEX-SHADER
/////////////////////////////////////////////////////

vertex2pixel vsAll(app2Vertex IN) {
	return IN;
}

float4 psInvert(vertex2pixel v) : COLOR0 {
	return fScalar.x/(tex2D(sFloatA,float2(0,0)).x*fScalar.y);
}

float4 psMultilyScalar(vertex2pixel v) : COLOR0 {
	return tex2D(sFloatA,float2(0,0)).x*fScalar.x;
}

float4 psAddScalar(vertex2pixel v) : COLOR0 {
	return tex2D(sFloatA,float2(0,0)).x+fScalar.x;
}

float4 psAddClFloat(vertex2pixel v) : COLOR0 {
	float2 texCoords = float2(0,0);
	return dot(float2(tex2D(sFloatA,texCoords).r, tex2D(sFloatB,texCoords).r), fScalar.xy);
}

float4 psMulClFloat(vertex2pixel v) : COLOR0 {
	float2 texCoords = float2(0,0);
	return tex2D(sFloatA,texCoords).x*tex2D(sFloatB,texCoords).x*fScalar.x;
}

float4 psDivClFloat(vertex2pixel v) : COLOR0 {
	float2 texCoords = float2(0,0);
	return (tex2D(sFloatA,texCoords).x*fScalar.x)/(tex2D(sFloatB,texCoords).x*fScalar.y);
}

float4 psDivZClFloat(vertex2pixel v) : COLOR0 {
	float2 texCoords = float2(0,0);
	float b = tex2D(sFloatB,texCoords).x;
	float a;
	if (b==0.0f) {
		a = 0; b = 0.1;
	} else {
		a = tex2D(sFloatA,texCoords).x;
	}
	return (a*fScalar.x)/(b*fScalar.y); 	
}

/////////////////////////////////////////////////////
// TECHNIQUES
/////////////////////////////////////////////////////

technique tInvert {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psInvert();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tMultilyScalar {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psMultilyScalar();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tAddScalar {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psAddScalar();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tAddClFloat {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psAddClFloat();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tMulClFloat {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psMulClFloat();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tDivClFloat {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psDivClFloat();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tDivZClFloat {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psDivZClFloat();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}