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

PHG_FLOAT4   vPointSize;
PHG_FLOAT4   vDirection;
PHG_FLOAT    fTimer;
PHG_FLOAT    fSpeed;

// Transforms

PHG_FLOAT4x4 mW;
PHG_FLOAT4x4 mWVP;

// Textures

texture  tDiffuseMap;

//
// Shaders
//

struct VS_INPUT
{
    PHG_FLOAT4 vPosition : POSITION0;
    PHG_FLOAT4 vNormal   : NORMAL0;
    PHG_FLOAT4 vDiffuse  : COLOR0;
};

struct VS_OUTPUT
{
    PHG_FLOAT4 vPosition  : POSITION;
    PHG_FLOAT4 vDiffuse   : COLOR0;
    PHG_FLOAT  fPointSize : PSIZE;
};

// Default Vertex Shader

VS_OUTPUT ShaderAmbientLightingAnimation( const VS_INPUT v )
{
	VS_OUTPUT o;

	// Animation

	PHG_FLOAT fA = tan( fTimer + v.vDiffuse.x * 3.14 );

	PHG_FLOAT4 vPos = v.vPosition;

	vPos.x += -2.0 * fA - fSpeed * abs(fA) * vDirection.x;
	vPos.y += -2.0 * fA - fSpeed * abs(fA) * vDirection.y;
	vPos.z += -2.0 * fA - fSpeed * abs(fA) * vDirection.z;

	PHG_FLOAT4 vPosition = mul( vPos, mWVP  );

	// Finalize

	o.vPosition  = vPosition;
	o.vDiffuse   = max( 0, (1.0 - abs(fA/20.0)));
	o.fPointSize = vPointSize.x;

	return o;
}

// Default Technique

technique TDefault
{   
	pass p0
	{
		// Common States

		CullMode		 = CCW;
		PointSpriteEnable	 = True;

		SrcBlend		 = One;
		DestBlend		 = One;

		// Texture Stages

		Texture[0]		 = <tDiffuseMap>;

		MinFilter[0]		 = Point;
		MagFilter[0]		 = Point;
		MipFilter[0]		 = None;

		ColorOp[ 0 ]             = Modulate;
		ColorArg1[ 0 ]           = Diffuse;
		ColorArg2[ 0 ]           = Texture;

		AlphaOp[ 0 ]             = Modulate;
	        AlphaArg1[ 0 ] 		 = Diffuse;
	        AlphaArg2[ 0 ] 		 = Texture;
        
	        ColorOp[ 1 ]   		 = Disable;
        	AlphaOp[ 1 ]   		 = Disable;

		VertexShader = compile vs_2_0 ShaderAmbientLightingAnimation();
	}
}

// End of file
