/************************************************************
 *															*
 * decr     : shaders for the clMatrix Classes				*
 * version  : 1.1											*
 * author   : Jens Krüger									*
 * date     : 16.09.2003									*
 * modified	: 30.09.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

/////////////////////////////////////////////////////
// VARIABLES
/////////////////////////////////////////////////////

texture tVector;
texture tLastPass;

/////////////////////////////////////////////////////
// STRUCTURES
/////////////////////////////////////////////////////

struct app2Vertex {
	float4 Position    : POSITION;
	float2 TexCoords0  : TEXCOORD0;
	float2 TexCoords1  : TEXCOORD1;
	float2 TexCoords2  : TEXCOORD2;
	float2 TexCoords3  : TEXCOORD3;
	float4 vertexVals  : TEXCOORD4;
	float2 TexCoordPos : TEXCOORD5;
};

struct vertex2pixel {
	float4 Position    : POSITION;
	float2 TexCoords0  : TEXCOORD0;
	float2 TexCoords1  : TEXCOORD1;
	float2 TexCoords2  : TEXCOORD2;
	float2 TexCoords3  : TEXCOORD3;
	float4 vertexVals  : TEXCOORD4;
	float2 TexCoordPos : TEXCOORD5;
};

sampler sVector = sampler_state {
	Texture = (tVector);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

sampler sLastPass = sampler_state {
	Texture = (tLastPass);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

/////////////////////////////////////////////////////
// HELPER FUNCTIONS
/////////////////////////////////////////////////////

// NONE YET

/////////////////////////////////////////////////////
// PIXEL- AND VERTEX-SHADER
/////////////////////////////////////////////////////

vertex2pixel vsAll(app2Vertex IN) {
	return IN;
}

float4 psMatrixMultiply(vertex2pixel v) : COLOR0 {
	return tex2D(sLastPass,v.TexCoordPos).r+
		   dot(v.vertexVals,float4(tex2D(sVector,v.TexCoords0).r,tex2D(sVector,v.TexCoords1).r,tex2D(sVector,v.TexCoords2).r,tex2D(sVector,v.TexCoords3).r));
}

float4 psMatrixMultiplyNoLast(vertex2pixel v) : COLOR0 {
	return dot(v.vertexVals,float4(tex2D(sVector,v.TexCoords0).r,tex2D(sVector,v.TexCoords1).r,tex2D(sVector,v.TexCoords2).r,tex2D(sVector,v.TexCoords3).r));
}


/////////////////////////////////////////////////////
// TECHNIQUES
/////////////////////////////////////////////////////

technique tMatrixMultiply {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psMatrixMultiply();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;
		PointSize			= 1;   
    }   
}

technique tMatrixMultiplyNoLast {
	pass P0 {
		VertexShader = compile vs_2_0 vsAll();
		PixelShader  = compile PS_PROFILE psMatrixMultiplyNoLast();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
		PointSize			= 1;
    }   
}
