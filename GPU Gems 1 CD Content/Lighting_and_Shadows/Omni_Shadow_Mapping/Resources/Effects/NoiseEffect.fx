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

texture    tNoise;
PHG_FLOAT  fTimer;

//
// Shaders
//

struct VS_INPUT
{
    PHG_FLOAT4 vPosition : POSITION0;
    PHG_FLOAT2 vTexCoord : TEXCOORD0;
};

struct VS_OUTPUT
{
    PHG_FLOAT4 vPosition  : POSITION;
    PHG_FLOAT3 vTexCoord  : TEXCOORD0;
};

sampler NoiseSampler = sampler_state
{
   Texture = (tNoise);

   MinFilter = Linear;
   MagFilter = Linear;
   MipFilter = Linear;

   AddressU  = Wrap;
   AddressV  = Wrap;
   AddressW  = Wrap;

   MaxAnisotropy = 16;
};

// Default Vertex Shader

VS_OUTPUT DefaultVertexShader( const VS_INPUT v )
{
	VS_OUTPUT o;

	o.vPosition.xyz = v.vPosition.xyz;
	o.vPosition.w = 1.0;

	o.vTexCoord.xy = v.vTexCoord.xy + fTimer;
	o.vTexCoord.z = fTimer;

	return o;
}

// Default Pixel Shader

PHG_FLOAT4 DefaultPixelShader ( PHG_FLOAT3 vTexCoord  : TEXCOORD0 ) : COLOR
{
	PHG_FLOAT4 vOutColor;	 

	vOutColor.xyz = 0.1 * tex3D( NoiseSampler, vTexCoord.xyz );
	vOutColor.w = 0.1;

	return vOutColor;
}

// Default Technique

technique TDefault
{   
	pass p0
	{
		// Common States

		CullMode		 = None;

		AlphaBlendEnable         = True;
		ZWriteEnable		 = False;
		ZEnable                  = False;
		SrcBlend		 = One;
		DestBlend		 = One;

		PixelShader  = compile ps_2_0 DefaultPixelShader();
		VertexShader = compile vs_2_0 DefaultVertexShader();
	}
}

// End of file
