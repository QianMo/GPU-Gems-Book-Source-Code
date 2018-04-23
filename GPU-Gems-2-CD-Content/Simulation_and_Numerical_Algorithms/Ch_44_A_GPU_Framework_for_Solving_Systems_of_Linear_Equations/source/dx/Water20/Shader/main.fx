/************************************************************
 *															*
 * decr     : display shader								*
 * version  : 1.0											*
 * author   : Jens Krüger									*
 * date     : 15.04.2003									*
 * modified	: 15.04.2003									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

/////////////////////////////////////////////////////
// VARIABLES
/////////////////////////////////////////////////////

texture tHeightMap;
texture tRefractMap;
texture tWaterTexture;
float4 f4Position;
float4 f4StepSize;
float fScale;

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

struct vertex2pixelMain {
	float4 Position   : POSITION;
	float4 TexCoords  : TEXCOORD0;
	float4 UVOffsetY  : TEXCOORD1;
};

sampler sWater = sampler_state {
	Texture = (tWaterTexture);

	MinFilter = Linear;		// Point, Linear
	MagFilter = Linear;
	AddressU  = Clamp;
	AddressV  = Clamp;
};

sampler sHeightField = sampler_state {
	Texture = (tHeightMap);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Clamp;
	AddressV  = Clamp;
};

sampler sRefraction = sampler_state {
	Texture = (tRefractMap);

	MinFilter = Linear;		// Point, Linear
	MagFilter = Linear;
	AddressU  = Clamp;
	AddressV  = Clamp;
};


/////////////////////////////////////////////////////
// HELPER FUNCTIONS
/////////////////////////////////////////////////////


// NONE so far


/////////////////////////////////////////////////////
// PIXEL- AND VERTEX-SHADER
/////////////////////////////////////////////////////

float4 psDrop(vertex2pixel IN) : COLOR0 {
	return saturate(0.3-length(IN.TexCoords-f4Position.xy)*40)+tex2D(sHeightField,IN.TexCoords);
}

vertex2pixel vsDrop(app2Vertex IN) {
    return IN;
}

vertex2pixelMain vsMain(app2Vertex IN) {
    vertex2pixelMain OUT;
    OUT.Position = IN.Position;
    float4 uv = float4(IN.TexCoords.x,1-IN.TexCoords.y, 0, 0);
    OUT.TexCoords = float4(uv.x, uv.y, uv.x - f4StepSize.x, uv.x + f4StepSize.x);
    OUT.UVOffsetY = float4(uv.x, uv.y, uv.y - f4StepSize.y, uv.y + f4StepSize.y);
	return OUT;
}

ps_2_a 
float4 psMain(vertex2pixelMain v) : COLOR0 {
	// viewing position
	half3 viewPos      = half3(0,1,0);

    half2 uv          = v.TexCoords.xy;
    half4 f4Neigbours = half4(tex2D(sHeightField, v.TexCoords.wy).r, //right
                              tex2D(sHeightField, v.UVOffsetY.xw).r, //top
                              tex2D(sHeightField, v.TexCoords.zy).r, //left
                              tex2D(sHeightField, v.UVOffsetY.xz).r);//bottom
                              
    // surface normal
    half3 normal = normalize(half3(f4Neigbours.x-f4Neigbours.z,2,f4Neigbours.w-f4Neigbours.y));

	// load height value
	half fHeight        = tex2D(sHeightField, uv.xy).r;
	
    // compute position from texture coordinates
	uv                -= 0.5;
	half3 position     = half3(uv.x, fHeight, uv.y);

	// set light pos and material properties
	half3 lightPos     = normalize(half3( 0.2,  1,  0.5 ));

	half3 mtrlAmbient  = half3( 0.1,  0.1,  0.1 );
	half3 mtrlDiffuse  = half3( 0.4,  0.4,  0.4 );
	half3 mtrlSpecular = half3( 1.0,  1.0,  1.0 );

	// phong illumination model  
	half3 halfVec      = normalize(lightPos + viewPos);
	half  diffuse      = dot(normal.xyz, lightPos);
	half  specular     = dot(normal.xyz, halfVec);
	half3 lighting     = lit(diffuse, specular, 32);
    
	half3 light        = dot(lighting, half3(0.1,0.4,1.0));

	// refraction
	half4 incident = (half3(uv.x*4, 0, uv.y*4)-viewPos).rgbb; incident.a = 1;
	half4 normal4  = half4(normal.rgb,1) * 0.707106781187; 

	half3 refraction   = refract(incident, normalize(normal4), 0.5);
	half4 refracttexel = texCUBE(sRefraction, refraction);

	return half4(light*refracttexel.rgb,1);
}

ps_2_0 
float4 psMain(vertex2pixelMain v) : COLOR0 {
	// viewing position
	float3 viewPos      = float3(0,1,0);

    float2 uv           = v.TexCoords.xy;
    half4 f4Neigbours = half4(tex2D(sHeightField, v.TexCoords.wy).r, //right
                              tex2D(sHeightField, v.UVOffsetY.xw).r, //top
                              tex2D(sHeightField, v.TexCoords.zy).r, //left
                              tex2D(sHeightField, v.UVOffsetY.xz).r);//bottom

    // surface normal
    float3 normal = normalize(float3(f4Neigbours.x-f4Neigbours.z,2,f4Neigbours.w-f4Neigbours.y));

	// load height value
	float fHeight        = tex2D(sHeightField, uv).r;

    // compute position from texture coordinates
	uv                 -= 0.5;
	float3 position     = float3(uv.x, fHeight, uv.y);

	// set light pos and material properties
	float3 lightPos     = normalize(float3( 0.2,  1,  0.5 ));

	float3 mtrlAmbient  = float3( 0.1,  0.1,  0.1 );
	float3 mtrlDiffuse  = float3( 0.4,  0.4,  0.4 );
	float3 mtrlSpecular = float3( 1.0,  1.0,  1.0 );

	// phong illumination model  
	float3 halfVec      = normalize(lightPos + viewPos);
	float  diffuse      = dot(normal.xyz, lightPos);
	float  specular     = dot(normal.xyz, halfVec);
	float3 lighting     = lit(diffuse, specular, 32);
    
	float3 light        = dot(lighting, float3(0.1,0.4,1.0));

	// refraction
	float4 incident = (float3(uv.x*4, 0, uv.y*4)-viewPos).rgbb; incident.a = 1;
	float4 normal4  = float4(normal.rgb,1) * 0.707106781187; 

	float3 refraction   = refract(incident, normalize(normal4), 0.5);
	float4 refracttexel = texCUBE(sRefraction, refraction);

	return float4(light*refracttexel.rgb,1);
}

/////////////////////////////////////////////////////
// TECHNIQUES
/////////////////////////////////////////////////////

technique tInsertDrop {
	pass P0 {
		VertexShader = compile vs_2_0 vsDrop();
		PixelShader  = compile PS_PROFILE psDrop();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tMainPass {
	pass P0 {
		VertexShader = compile vs_2_0 vsMain();
		PixelShader  = compile PS_PROFILE psMain();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;
    }   
}
