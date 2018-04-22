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
//   Pixel Shaders 2.0 ( Hight-Level Shader Language )
//
//   Vertex Shaders 2.0 ( Hight-Level Shader Language )
//
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

// Textures

texture tDiffuseMap;
texture tSpecularMap;
texture tNormalMap;
texture tShadowMap;

// Helpers

PHG_FLOAT4 vZero = { 0.0, 0.0, 0.0, 0.0 };
PHG_FLOAT4 vOne  = { 1.0, 1.0, 1.0, 1.0 };

// Camera Position

PHG_FLOAT4 vCameraPosition;

// Transforms

PHG_FLOAT4x4 mW;   // World Matrix
PHG_FLOAT4x4 mWVP; // World * View * Projection Matrix

//
// Samplers
//

sampler DiffuseMapSampler = sampler_state
{
   Texture = (tDiffuseMap);

   MinFilter = Linear;
   MagFilter = Linear;
   MipFilter = Linear;

   AddressU  = Wrap;
   AddressV  = Wrap;

   MaxAnisotropy = 16;
};

//
// Shaders
//

struct VS_OUTPUT
{
    PHG_FLOAT4 vPosition : POSITION;
};

struct VS_OUTPUT1
{
    PHG_FLOAT4 vPosition : POSITION;
    PHG_FLOAT2 vTexCoord : TEXCOORD0;
};

// Default Vertex Shader

VS_OUTPUT DefaultVertexShader( PHG_FLOAT4 vPosition : POSITION )
{
	VS_OUTPUT o;

	o.vPosition = mul( vPosition, mWVP );

	return o;
}

// Default Technique

technique TDefault
{   
	pass p0
	{
		// Common States

		// CullMode = CCW;

		// Shaders

		PixelShader  = NULL;
		vertexshader = compile vs_2_0 DefaultVertexShader();
	}
}

// End of file
