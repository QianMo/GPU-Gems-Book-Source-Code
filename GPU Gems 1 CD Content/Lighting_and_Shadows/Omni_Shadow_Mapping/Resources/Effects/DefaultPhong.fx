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

// Textures

texture tDiffuseMap;
texture tSpecularMap;
texture tNormalMap;
texture tShadowMap;
texture tNormalizationCubeMap; // Normalization cubemap

// Helpers

PHG_FLOAT4 vZero  = { 0.0,  0.0, 0.0, 0.0 };
PHG_FLOAT4 vOne   = { 1.0,  1.0, 1.0, 1.0 };
PHG_FLOAT4 vHalf  = { 0.5,  0.5, 0.5, 0.5 };
PHG_FLOAT4 vBias  = { 2.0, -1.0, 0.5, 0.5 };

PHG_FLOAT4 vGreyScale = { 0.3, 0.59, 0.11, 0.0 };

// Light & material

PHG_FLOAT4 vLightPosition;

PHG_FLOAT4 vLightDiffuse;   // Light diffuse * material diffuse
PHG_FLOAT4 vLightSpecular;  // Light specular * material specular

PHG_FLOAT fLightAttenuation = 0.02;

PHG_FLOAT fMaterialPower = 16.0;
PHG_FLOAT fBumpScale     = 1.5;

// Camera Position

PHG_FLOAT4 vCameraPosition;

// Transforms

PHG_FLOAT4x4 mW;   // World Matrix
PHG_FLOAT4x4 mV;   // View Matrix
PHG_FLOAT4x4 mP;   // Projection Matrix
PHG_FLOAT4x4 mWV;  // World * View Matrix
PHG_FLOAT4x4 mWVP; // World * View * Projection Matrix

// Shadow Mapping

PHG_FLOAT3 vUnpack = { 255.0, 1.0, 0.0 };

PHG_FLOAT  fGeometryScale;
PHG_FLOAT  fDepthBias = 0.98;

PHG_FLOAT3 vFilter[8];

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

   MaxAnisotropy = 4;
};

sampler NormalMapSampler = sampler_state
{
   Texture = (tNormalMap);

   MinFilter = Linear;
   MagFilter = Linear;
   MipFilter = Linear;

   AddressU  = Wrap;
   AddressV  = Wrap;

   MaxAnisotropy = 4;
};

sampler SpecularMapSampler = sampler_state
{
   Texture = (tSpecularMap);

   MinFilter = Linear;
   MagFilter = Linear;
   MipFilter = Linear;

   AddressU  = Wrap;
   AddressV  = Wrap;

   MaxAnisotropy = 4;
};

sampler ShadowMapSampler = sampler_state
{
   Texture = (tShadowMap);

   MinFilter = Point;
   MagFilter = Point;
   MipFilter = None;

   AddressU  = Clamp;
   AddressV  = Clamp;
   AddressW  = Clamp;

   MaxAnisotropy = 1;
};

sampler NormalizationCubeMapSampler = sampler_state
{
   Texture = (tNormalizationCubeMap);

   MinFilter = Linear;
   MagFilter = Linear;
   MipFilter = Linear;

   AddressU  = Clamp;
   AddressV  = Clamp;

   MaxAnisotropy = 1;
};

//
// Shaders
//

struct VS_OUTPUT
{
    PHG_FLOAT4 vPosition    : POSITION;

    PHG_FLOAT2 vTexCoord    : TEXCOORD0;
    PHG_FLOAT3 vTangent     : TEXCOORD1;
    PHG_FLOAT3 vBinormal    : TEXCOORD2;
    PHG_FLOAT3 vNormal      : TEXCOORD3;
    PHG_FLOAT3 vEye         : TEXCOORD4;
    PHG_FLOAT4 vLight[3]    : TEXCOORD5;
};

// Default Pixel Shader

PHG_FLOAT4 BlinnPixelShader ( PHG_FLOAT2 vTexCoord  : TEXCOORD0,
			      PHG_FLOAT3 vTangent   : TEXCOORD1,
			      PHG_FLOAT3 vBinormal  : TEXCOORD2,
			      PHG_FLOAT3 vNormal    : TEXCOORD3,
			      PHG_FLOAT3 vEye       : TEXCOORD4,
			      PHG_FLOAT4 vLight[3]  : TEXCOORD5 ) : COLOR
{
	// Light Direction ( xyz ) , Attenuation Factor ( w )

	PHG_FLOAT  fDistSquared    = dot( vLight[0].xyz, vLight[0].xyz );
	PHG_FLOAT4 vLightDirection = -vLight[0] / sqrt( fDistSquared );

	// Bump Mapping

	PHG_FLOAT2 vBump = tex2D( NormalMapSampler, vTexCoord );

#if NORMALIZATION_CUBEMAP
        PHG_FLOAT3 vN = texCUBE( NormalizationCubeMapSampler, vNormal + ( vBump.x * vTangent + vBump.y * vBinormal ) );
#else
        PHG_FLOAT3 vN = normalize( vNormal + ( vBump.x * vTangent + vBump.y * vBinormal ) );
#endif

	// Reflection vector

	PHG_FLOAT3 vReflect = 2.0f * vN.xyz * dot( -vLightDirection.xyz, -vN.xyz ) - vLightDirection.xyz;

	// Dot products

	PHG_FLOAT fNdotL   = saturate( dot( vLightDirection.xyz, vN.xyz ) );
	PHG_FLOAT fRDotE   = saturate( dot(vReflect, normalize( vEye.xyz ) ) );

	// Diffuse Color 

	PHG_FLOAT4 vDiffuse = vLightDiffuse * fNdotL * tex2D( DiffuseMapSampler, vTexCoord );

	// Self-Shadowing

	PHG_FLOAT SelfShadow = saturate( 4.0f * fNdotL );

	// Specular Color

	PHG_FLOAT4 vSpecular = SelfShadow * vLightSpecular * pow( fRDotE, fMaterialPower ) * tex2D( SpecularMapSampler, vTexCoord );

	// Shadow Mapping

	PHG_FLOAT fDepth = fDistSquared * fDepthBias;

#if !SOFT_SHADOWS
	PHG_FLOAT3 vShadowSample = texCUBE( ShadowMapSampler, vLight[0].xyz );
#if PACKED_DEPTH
	PHG_FLOAT fShadow = ( fDepth - dot( vUnpack.xyz, vShadowSample.xyz ) < 0.0 ) ? 1.0 : 0.0;
#else
	PHG_FLOAT fShadow = ( fDepth - vShadowSample.x < 0.0 ) ? 1.0 : 0.0;
#endif
#else
	PHG_FLOAT fShadow = 0.0;
	for( int i = 0; i < 3; i++ )
	{
		PHG_FLOAT3 vShadowSample = texCUBE( ShadowMapSampler, vLight[i].xyz );
#if PACKED_DEPTH
		fShadow += ( fDepth - dot( vUnpack.xyz, vShadowSample.xyz ) < 0.0 ) ? 0.333 : 0.0;
#else
		fShadow += ( fDepth - vShadowSample.x < 0.0) ? 0.333 : 0.0;
#endif
	}
#endif	
	// Finalize

#if !GREYSCALE
	return ( vDiffuse + vSpecular ) * vLightDirection.w * fShadow;
#else
	return dot(( vDiffuse + vSpecular ) * vLightDirection.w * fShadow, vGreyScale );
#endif
}

// Default Vertex Shader

VS_OUTPUT BlinnVertexShader( PHG_FLOAT4 vPosition : POSITION,
    			     PHG_FLOAT4 vNormal   : NORMAL,
			     PHG_FLOAT2 vTexCoord : TEXCOORD0,
			     PHG_FLOAT3 vTangent  : TEXCOORD1,
			     PHG_FLOAT3 vBinormal : TEXCOORD2 )
{
	VS_OUTPUT o;

	// Vertex Position -> Clip Space

	o.vPosition = mul( vPosition, mWVP );

	// Texture Coords

	o.vTexCoord.xy  = vTexCoord.xy;

	// Vertex Position -> World Space

	PHG_FLOAT4 vPosWorld = mul( vPosition, mW );

	// Eye Vector

	o.vEye = ( vCameraPosition.xyz - vPosWorld.xyz ) * fGeometryScale;

	// Light Vector

	o.vLight[0].xyz   = -(vLightPosition.xyz - vPosWorld.xyz * fGeometryScale);
	o.vLight[0].w     = -fLightAttenuation;

	o.vLight[1].xyz   = o.vLight[0].xyz + vFilter[0];
	o.vLight[1].w     = -fLightAttenuation;

	o.vLight[2].xyz   = o.vLight[0].xyz + vFilter[1];
	o.vLight[2].w     = -fLightAttenuation;

	// Tangent Space

	o.vTangent  = fBumpScale * mul( vTangent.xyz,  mW );
	o.vBinormal = fBumpScale * mul( vBinormal.xyz, mW );
	o.vNormal   = mul( vNormal.xyz,   mW );

	// Finalize

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

		PixelShader  = compile ps_2_0 BlinnPixelShader();
		vertexshader = compile vs_2_0 BlinnVertexShader();
	}
}

// End of file
