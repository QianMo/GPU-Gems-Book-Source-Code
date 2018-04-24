float4x4 World;					///< World matrix for the current object
float4x4 WorldIT;				///< World matrix IT (inverse transposed) to transform surface normals of the current object 
float4x4 WorldView;
float4x4 WorldViewIT;
float4x4 WorldViewProj;
float4x4 View;
float4x4 ViewIT;
float3 referencePos;
float3 eyePos;
float3 F0; // Freshnel factor
float N0; // Refraction coefficient
float3 lightPos = float3(0,1.0,0);
float ambient = 0.1;
float diffuse = 0.5;


texture colorMap;
sampler colorSampler = sampler_state 
{
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;    
    Texture   = <colorMap>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

texture envCube;
sampler envCubeSampler = sampler_state 
{
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;    
    Texture   = <envCube>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

//-------------------------------------------------------------------------
// Technique: write texture color and zero refraction index
//-------------------------------------------------------------------------
	struct Textured_VS_OUT
	{
		float4 hPosition :POSITION0;
		float2 texCoord :TEXCOORD0;
	};

	Textured_VS_OUT texturedVS(float4 position : POSITION0,
							float2 texCoord : TEXCOORD0)
	{
		Textured_VS_OUT OUT;
		OUT.hPosition = mul(position, WorldViewProj);
		OUT.texCoord = texCoord;
		return OUT;
	}

	float4 texturedPS(Textured_VS_OUT IN) : COLOR0
	{
		return float4(tex2D(colorSampler, IN.texCoord).rgb, 0);
	}

	technique Textured										
	{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 texturedVS();		
				PixelShader  = compile ps_2_0 texturedPS();	
			}															
	}
	
//-------------------------------------------------------------------------
// Technique: write phong shaded color and zero refraction index
//-------------------------------------------------------------------------
	struct Shaded_VS_OUT
	{
		float4 hPosition :POSITION0;
		float3 V		 :TEXCOORD1;
		float3 L		 :TEXCOORD3;
		float3 cNormal   :TEXCOORD2;
		float2 texCoord :TEXCOORD0;
	};

	Shaded_VS_OUT ShadedVS(float4 position : POSITION0,
							float3 normal  : NORMAL,
							float2 texCoord : TEXCOORD0)
	{
		Shaded_VS_OUT OUT;
		OUT.hPosition = mul(position, WorldViewProj);
		OUT.V = -mul(position, WorldView);
		OUT.cNormal = mul(normal, WorldViewIT);
		OUT.L = mul(lightPos, View) + OUT.V;
		OUT.texCoord = texCoord;
		return OUT;
	}

	float4 ShadedPS(Shaded_VS_OUT IN) : COLOR0
	{
		float3 N = normalize(IN.cNormal);
		float3 V = normalize(IN.V);
		float3 L = normalize(IN.L);
		float3 H = normalize(V + L);
		float d = length(IN.L);
		float4 lighting = lit(dot(N, L), dot(N, H), 20);
		return float4(tex2D(colorSampler, IN.texCoord).rgb * (ambient + diffuse * lighting.y / d * d) + lighting .z / d * d, 0);
	}

	technique Shaded										
	{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 ShadedVS();		
				PixelShader  = compile ps_2_0 ShadedPS();	
			}															
	}

//-------------------------------------------------------------------------
// Technique: Classical environment mapping
//-------------------------------------------------------------------------
	struct EnvMapped_VS_OUT
	{
		float4 hPosition : POSITION0;
		float2 texCoord  : TEXCOORD0;
		float3 wNormal   : TEXCOORD1; //world normal
		float3 cmPos     : TEXCOORD2; //cubemap position
		float3 V		 : TEXCOORD3; //view direction in world space
	};

	EnvMapped_VS_OUT EnvMappedVS(float4 position : POSITION0,
							float2 texCoord : TEXCOORD0,
							float3 normal : NORMAL)
	{
		EnvMapped_VS_OUT OUT;
		OUT.hPosition = mul(position, WorldViewProj);
		OUT.wNormal = mul(normal, WorldIT);
		OUT.cmPos = mul(position, World).xyz;
		OUT.V = OUT.cmPos - eyePos;
		OUT.cmPos -= referencePos;
		OUT.texCoord = texCoord;
		return OUT;
	}
	
	float4 EnvMappedPS(EnvMapped_VS_OUT IN) : COLOR0
	{
		float3 N = normalize(IN.wNormal);
		float3 V = normalize(IN.V);
		float3 R;
		float3 I = 1;
		
		float3 F = F0 + pow(1-dot(N, -V), 5) * (1 - F0);
		if (N0 <= 0) // reflective material
		{
      		R = reflect(V, N);
      		I *= F; // Fresnel reflection
      	}
      	else //refractive material
      	{
      		R = refract(V, N, N0);
      		if (dot(R, R) == 0)	// no refraction direction exits
      			R = reflect(V, N); // total reflection  			
			else
     			I *= (1 - F);      // Fresnel refraction
      	}	
		return texCUBE(envCubeSampler, R) * float4(I,1);
	}

	technique EnvMapped										
	{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 EnvMappedVS();		
				PixelShader  = compile ps_2_0 EnvMappedPS();	
			}															
	}
		
//-------------------------------------------------------------------------
// Technique: write Fresnel and refraction index
//-------------------------------------------------------------------------
   
    struct FresnelRefIndex_VS_OUT
	{
		float4 hPosition : POSITION0;
	};
	
	FresnelRefIndex_VS_OUT FresnelRefIndexVS(float4 position : POSITION0,
							float2 texCoord : TEXCOORD0)
	{
		FresnelRefIndex_VS_OUT OUT;
		OUT.hPosition = mul(position, WorldViewProj);
		return OUT;
	}
	
	float4 FresnelRefIndexPS(FresnelRefIndex_VS_OUT IN) : COLOR0
	{
		float4 color;
		color.rgb = F0;
		color.a = N0;
		return color;		
	}

	technique FresnelRefIndex									
	{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 FresnelRefIndexVS();		
				PixelShader  = compile ps_2_0 FresnelRefIndexPS();	
			}															
	}

//-------------------------------------------------------------------------
// Technique: write normal and distance from camera
//-------------------------------------------------------------------------
   
    struct NormalDist_VS_OUT
	{
		float4 hPosition : POSITION0;
		float2 texCoord  : TEXCOORD0;
		float3 wNormal   : TEXCOORD1; //world normal position		
		float3 cPos	     : TEXCOORD2; //view space position
	};
	
	NormalDist_VS_OUT NormalDistVS(float4 position : POSITION0,
							float2 texCoord : TEXCOORD0,
							float3 normal : NORMAL)
	{
		NormalDist_VS_OUT OUT;
		OUT.hPosition = mul(position, WorldViewProj);
		OUT.wNormal = mul(normal, WorldIT).xyz;
		OUT.cPos = mul(position, WorldView).xyz;
		OUT.texCoord = texCoord;
		return OUT;
	}
	
	float4 NormalDistPS(NormalDist_VS_OUT IN) : COLOR0
	{
		float4 color; 
		color.rgb = normalize(IN.wNormal);
		color.a = length(IN.cPos);				
		return color;		
	}

	technique NormalDistance										
	{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 NormalDistVS();		
				PixelShader  = compile ps_2_0 NormalDistPS();	
			}															
	}