//
// Project:
//
//   DX9 "Omnidirectional Shadow Mapping" Demo
//
// Effect Version: 
//
//   DirectX 9.0
//
// Shader Profile:
//
//   Pixel Shaders 2.0 ( Pixel Shader Assembler )
//
//   Vertex Shaders 2.0 ( Vertex Shader Assembler )
//

//
// Global Variables
//

#if HALF_PRECISION
   #define PHG_FLOAT    half        
   #define PHG_FLOAT2   vector<half, 2> 
   #define PHG_FLOAT3   vector<half, 3> 
   #define PHG_FLOAT4   vector<half, 4> 
   #define PHG_FLOAT4x4 matrix<half, 4, 4> 
#else 
   #define PHG_FLOAT    float 
   #define PHG_FLOAT2   vector<float, 2> 
   #define PHG_FLOAT3   vector<float, 3> 
   #define PHG_FLOAT4   vector<float, 4> 
   #define PHG_FLOAT4x4 matrix<float, 4, 4> 
#endif

PHG_FLOAT4 vZero = { 0.0, 0.0, 0.0, 0.0 };
PHG_FLOAT4 vOne  = { 1.0, 1.0, 1.0, 1.0 };

PHG_FLOAT  fGeometryScale;

PHG_FLOAT4x4 mW;   // World Matrix
PHG_FLOAT4x4 mWVP; // World * View * Projection Matrix

PHG_FLOAT4 vLightPosition; // Light Position

//
// Shaders
//

struct VS_OUTPUT
{
    PHG_FLOAT4 vPosition      : POSITION;
    PHG_FLOAT3 vVertexToLight : TEXCOORD0;
};

// Default Vertex Shader

VS_OUTPUT DefaultVertexShader( PHG_FLOAT4 vPosition : POSITION )
{
	VS_OUTPUT o;

	o.vPosition = mul( vPosition, mWVP );
	o.vVertexToLight =  mul( vPosition, mW ).xyz * fGeometryScale - vLightPosition.xyz;

	return o;
}

// Default Pixel Shader

PHG_FLOAT4 DefaultPixelShader ( PHG_FLOAT3 vVertexToLight : TEXCOORD0 ) : COLOR
{

#if PACKED_DEPTH
	PHG_FLOAT fDepth = dot( vVertexToLight, vVertexToLight );
	return PHG_FLOAT4(  floor(fDepth) / 256.0, frac(fDepth), frac(fDepth), frac(fDepth) );
#else
	return dot( vVertexToLight, vVertexToLight );
#endif

}

// Default Technique

technique TDefault
{   
	pass p0
	{
		// Common States

		// CullMode	    = CCW;
		// AlphaBlendEnable = False;
		// ZWriteEnable	    = True;

		// Shaders

		PixelShader = compile ps_2_0 DefaultPixelShader();
		vertexshader = compile vs_2_0 DefaultVertexShader();

	}
}

// End of file

// End of file
